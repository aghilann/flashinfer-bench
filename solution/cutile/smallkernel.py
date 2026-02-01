# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
MoE Kernel for FlashInfer Competition - DeepSeek-V3 MoE Layer.

OPTIMIZED FOR SMALL SEQUENCE LENGTHS (1-100 tokens):
- All experts use CuTile GEMM (no cuBLAS fallback)
- Smaller tile sizes for better occupancy on small batches
- Aggressive autotuning for small M dimensions
- Batched SwiGLU for reduced kernel launch overhead
"""

from __future__ import annotations
import cuda.tile as ct
import torch
import random
import functools
import threading
import logging
from contextlib import contextmanager
from typing import Tuple, Any, Iterable, Callable, Sequence
from types import SimpleNamespace
from cuda.tile._exception import TileCompilerTimeoutError, TileCompilerExecutionError
from cuda.tile._cext import default_tile_context

logger = logging.getLogger(__name__)

ConstInt = ct.Constant[int]

# ============================================================================
# Autotuner (from cuda.tile_experimental)
# ============================================================================

_MAX_SEARCH_ITEMS = 10_000
_autotune_lock = threading.RLock()


def _with_autotune_lock(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _autotune_lock:
            return func(*args, **kwargs)
    return wrapper


def _shape_dtype_stride(arg: Any):
    shape = tuple(arg.shape)
    dtype = arg.dtype
    stride = None
    if hasattr(arg, "stride"):
        s = arg.stride() if callable(arg.stride) else arg.stride
        stride = tuple(int(x) for x in s)
    elif hasattr(arg, "strides"):
        itemsize = getattr(arg, "itemsize", 1)
        stride = tuple(int(b // itemsize) for b in arg.strides)
    return shape, dtype, stride


def _default_key(args):
    tinfo = []
    for arg in args:
        if hasattr(arg, "shape") and hasattr(arg, "dtype"):
            shape, dtype, stride = _shape_dtype_stride(arg)
            tinfo.append((shape, dtype, stride))
        else:
            tinfo.append(type(arg).__name__)
    return tuple(tinfo)


def _time_ms(run_once, *, get_args, stream, warmup=2, rep=10):
    stream.synchronize()
    for _ in range(warmup):
        run_once(get_args())
    args_per_run = [get_args() for _ in range(rep)]
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for i in range(rep):
        run_once(args_per_run[i])
    end.record(stream)
    end.synchronize()
    ms = start.elapsed_time(end)
    return ms / max(1, rep)


class TunedResult:
    def __init__(self, tuned_config, grid, kernel, tuning_record, cache_hit):
        self.tuned_config = tuned_config
        self.grid = grid
        self.kernel = kernel
        self.tuning_record = tuning_record
        self.cache_hit = cache_hit


class _CacheEntry:
    def __init__(self, best_cfg, best_grid, best_kernel, tuning_record):
        self.best_cfg = best_cfg
        self.best_grid = best_grid
        self.best_kernel = best_kernel
        self.tuning_record = tuning_record


@contextmanager
def compiler_timeout(timeout_sec: int):
    old_timeout = default_tile_context.config.compiler_timeout_sec
    default_tile_context.config.compiler_timeout_sec = timeout_sec
    try:
        yield
    finally:
        default_tile_context.config.compiler_timeout_sec = old_timeout


def _reservoir_sample(iterable: Iterable[Any], k: int, *, rng: random.Random, max_items: int):
    reservoir = []
    n_seen = 0
    for item in iterable:
        n_seen += 1
        if n_seen > max_items:
            break
        if len(reservoir) < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, n_seen - 1)
            if j < k:
                reservoir[j] = item
    return reservoir


def autotune_launch(stream, grid_fn, kernel,
                    args_fn: Callable[[Any], tuple[Any, ...]],
                    launch_args_fn: Callable[[Any], tuple[Any, ...]] | None = None,
                    hints_fn: Callable[[Any], dict[str, Any]] | None = None,
                    *,
                    search_space: Iterable[Any] | Callable[[], Iterable[Any]],
                    key: Any | None = None,
                    max_iter: int = 60,
                    compiler_time_limit_sec: int = 10,
                    seed: int | None = None,
                    force_retune: bool = False) -> TunedResult:
    if callable(search_space):
        search_space = search_space()

    rng = random.Random(seed)
    search_space = _reservoir_sample(search_space, k=max_iter, rng=rng, max_items=_MAX_SEARCH_ITEMS)
    if len(search_space) == 0:
        raise ValueError("Search space must contain at least 1 configuration")

    with _autotune_lock:
        autotune_cache = default_tile_context.autotune_cache
        if autotune_cache is None:
            autotune_cache = {}
            default_tile_context.autotune_cache = autotune_cache

        kernel_key = kernel._pyfunc
        per_kernel = autotune_cache.get(kernel_key)
        if per_kernel is None:
            per_kernel = {}
            autotune_cache[kernel_key] = per_kernel

        if key is None:
            arg_key = _default_key(args_fn(search_space[0]))
        else:
            arg_key = key

        tuning_entries = []
        cache_hit = False

        if not force_retune and arg_key in per_kernel:
            cache_hit = True
        else:
            indices = list(range(len(search_space)))
            rng.shuffle(indices)

            best_time_ms, best_cfg, best_kernel = float("inf"), None, None
            for i, cfg_idx in enumerate(indices):
                cfg = search_space[cfg_idx]
                grid = grid_fn(cfg)
                hints = hints_fn(cfg) if hints_fn else {}
                updated_kernel = ct.kernel(kernel._pyfunc, **hints)

                def run_once(args):
                    ct.launch(stream, grid, updated_kernel, args)

                try:
                    with compiler_timeout(compiler_time_limit_sec):
                        time_ms = _time_ms(run_once, get_args=lambda: args_fn(cfg), stream=stream)
                except TileCompilerTimeoutError:
                    continue
                except TileCompilerExecutionError:
                    continue

                if time_ms < best_time_ms:
                    best_time_ms = time_ms
                    best_cfg, best_grid, best_kernel = cfg, grid, updated_kernel
                tuning_entries.append((cfg, time_ms))

            if best_cfg is None:
                raise ValueError("No valid config found")
            per_kernel[arg_key] = _CacheEntry(best_cfg, best_grid, best_kernel, tuning_entries)

        cache_entry = per_kernel[arg_key]

    best_args = launch_args_fn(cache_entry.best_cfg) if launch_args_fn else args_fn(cache_entry.best_cfg)
    ct.launch(stream, cache_entry.best_grid, cache_entry.best_kernel, best_args)

    return TunedResult(
        tuned_config=cache_entry.best_cfg,
        grid=cache_entry.best_grid,
        kernel=cache_entry.best_kernel,
        tuning_record=cache_entry.tuning_record,
        cache_hit=cache_hit
    )


# ============================================================================
# Constants for DeepSeek-V3/R1 MoE
# ============================================================================
BLOCK = 128  # FP8 block scale block size
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4


# ============================================================================
# FP8 Dequantization
# ============================================================================

def dequantize_hidden_states(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 hidden states with block scales to float32."""
    data_f32 = data.to(torch.float32)
    scales_t = scales.permute(1, 0).contiguous()
    scales_expanded = scales_t.repeat_interleave(BLOCK, dim=1)
    return data_f32 * scales_expanded


def dequantize_weights_batched(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 weight matrices with block scales to float32."""
    data_f32 = data.to(torch.float32)
    scales_exp = scales.repeat_interleave(BLOCK, dim=1)
    scales_exp = scales_exp.repeat_interleave(BLOCK, dim=2)
    return data_f32 * scales_exp


# ============================================================================
# Routing
# ============================================================================

def deepseek_routing(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    routed_scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek-V3 no-aux-loss routing."""
    T, E_global = routing_logits.shape
    
    bias = routing_bias.to(torch.float32)
    s = torch.sigmoid(routing_logits)
    s_with_bias = s + bias
    
    group_size = E_global // N_GROUP
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)
    
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)
    
    scores_pruned = torch.where(score_mask > 0, s_with_bias, torch.finfo(torch.float32).min)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)
    
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor
    
    return weights, topk_idx


# ============================================================================
# CuTile GEMM Kernel with Autotuning - OPTIMIZED FOR SMALL M
# ============================================================================

@ct.kernel
def small_gemm_kernel(
    A_ptr,          # [M, K]
    B_ptr,          # [N, K] (transposed, so we compute A @ B.T)
    C_ptr,          # [M, N]
    M: int,
    N: ConstInt,
    K: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
):
    """GEMM kernel optimized for small M: C = A @ B.T"""
    bid = ct.bid(0)
    num_n_tiles = ct.cdiv(N, TILE_N)
    bid_m = bid // num_n_tiles
    bid_n = bid % num_n_tiles
    
    acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    num_k_tiles = ct.cdiv(K, TILE_K)
    
    for k in range(num_k_tiles):
        ta = ct.load(A_ptr, index=(bid_m, k), shape=(TILE_M, TILE_K), padding_mode=ct.PaddingMode.ZERO)
        tb = ct.load(B_ptr, index=(bid_n, k), shape=(TILE_N, TILE_K), padding_mode=ct.PaddingMode.ZERO)
        tb = ct.transpose(tb)
        ta = ct.astype(ta, ct.tfloat32)
        tb = ct.astype(tb, ct.tfloat32)
        acc = ct.mma(ta, tb, acc)
    
    acc = ct.astype(acc, C_ptr.dtype)
    ct.store(C_ptr, index=(bid_m, bid_n), tile=acc)


def _small_gemm_autotune_configs():
    """Autotune configurations for small M GEMM - smaller tiles for better occupancy."""
    gpu_capability = torch.cuda.get_device_capability()
    if gpu_capability in [(12, 0), (12, 1)]:
        # sm120, sm121 - smaller tiles for small batches
        yield SimpleNamespace(TILE_M=16, TILE_N=64, TILE_K=64, num_ctas=1, occupancy=4)
        yield SimpleNamespace(TILE_M=32, TILE_N=64, TILE_K=64, num_ctas=1, occupancy=4)
        yield SimpleNamespace(TILE_M=16, TILE_N=128, TILE_K=64, num_ctas=1, occupancy=2)
        yield SimpleNamespace(TILE_M=32, TILE_N=128, TILE_K=64, num_ctas=1, occupancy=2)
        yield SimpleNamespace(TILE_M=64, TILE_N=64, TILE_K=64, num_ctas=1, occupancy=2)
    else:
        # Blackwell - smaller tiles for small batches
        yield SimpleNamespace(TILE_M=16, TILE_N=128, TILE_K=64, num_ctas=1, occupancy=4)
        yield SimpleNamespace(TILE_M=32, TILE_N=128, TILE_K=64, num_ctas=1, occupancy=2)
        yield SimpleNamespace(TILE_M=32, TILE_N=256, TILE_K=64, num_ctas=1, occupancy=2)
        yield SimpleNamespace(TILE_M=64, TILE_N=128, TILE_K=64, num_ctas=1, occupancy=2)
        yield SimpleNamespace(TILE_M=64, TILE_N=256, TILE_K=64, num_ctas=1, occupancy=1)


def cutile_small_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """CuTile GEMM optimized for small M: C = A @ B.T"""
    M, K = A.shape
    N = B.shape[0]
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    stream = torch.cuda.current_stream()
    A_cont = A.contiguous()
    B_cont = B.contiguous()
    
    autotune_launch(
        stream,
        grid_fn=lambda cfg: (((M + cfg.TILE_M - 1) // cfg.TILE_M) * ((N + cfg.TILE_N - 1) // cfg.TILE_N),),
        kernel=small_gemm_kernel,
        args_fn=lambda cfg: (A_cont, B_cont, C, M, N, K, cfg.TILE_M, cfg.TILE_N, cfg.TILE_K),
        hints_fn=lambda cfg: {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy},
        search_space=_small_gemm_autotune_configs,
    )
    return C


# ============================================================================
# CuTile Fused SwiGLU Kernel
# ============================================================================

@ct.kernel
def fused_swiglu_kernel(
    G1_ptr,          # [M, 2*I] - GEMM1 output
    C_ptr,           # [M, I] - SwiGLU output
    I: ConstInt,     # intermediate_size
    TILE_SIZE: ConstInt,
):
    """Fused SwiGLU: C = silu(G1[:, I:]) * G1[:, :I]"""
    bid = ct.bid(0)
    
    # Load both halves in one kernel
    x1 = ct.load(G1_ptr, index=(bid, 0), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.ZERO)
    x2 = ct.load(G1_ptr, index=(bid, 1), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.ZERO)
    
    x1_f32 = ct.astype(x1, ct.float32)
    x2_f32 = ct.astype(x2, ct.float32)
    
    # SiLU(x2) = x2 * sigmoid(x2)
    sigmoid_x2 = 1.0 / (1.0 + ct.exp(-x2_f32))
    silu_x2 = x2_f32 * sigmoid_x2
    
    # Result = silu(x2) * x1
    result = silu_x2 * x1_f32
    result = ct.astype(result, C_ptr.dtype)
    
    ct.store(C_ptr, index=(bid, 0), tile=result)


def cutile_swiglu(G1: torch.Tensor, I: int) -> torch.Tensor:
    """CuTile fused SwiGLU: silu(G1[:, I:]) * G1[:, :I]"""
    M = G1.shape[0]
    if M == 0:
        return torch.empty((0, I), dtype=G1.dtype, device=G1.device)
    C = torch.empty((M, I), dtype=G1.dtype, device=G1.device)
    
    ct.launch(
        torch.cuda.current_stream(),
        (M,),
        fused_swiglu_kernel,
        (G1.contiguous(), C, I, I),
    )
    return C


# ============================================================================
# Token Sorting (Vectorized)
# ============================================================================

def sort_tokens_by_expert(
    topk_idx: torch.Tensor,     # [T, TOP_K]
    weights: torch.Tensor,      # [T, E_global]
    E_local: int,
    local_expert_offset: int,
    E_global: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Vectorized token sorting by expert assignment."""
    device = topk_idx.device
    T, K = topk_idx.shape
    local_start = local_expert_offset
    local_end = local_expert_offset + E_local
    
    token_indices = torch.arange(T, device=device).unsqueeze(1).expand(T, K)
    flat_experts = topk_idx.reshape(-1)
    flat_tokens = token_indices.reshape(-1)
    
    local_mask = (flat_experts >= local_start) & (flat_experts < local_end)
    
    if not local_mask.any():
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
            torch.zeros(E_local + 1, dtype=torch.int64, device=device),
            0
        )
    
    local_experts = flat_experts[local_mask] - local_start
    local_tokens = flat_tokens[local_mask]
    local_global_experts = flat_experts[local_mask]
    local_weights = weights[local_tokens, local_global_experts]
    
    total_count = local_tokens.shape[0]
    expert_counts = torch.bincount(local_experts, minlength=E_local)
    
    expert_offsets = torch.zeros(E_local + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)
    
    sort_idx = torch.argsort(local_experts, stable=True)
    sorted_token_ids = local_tokens[sort_idx]
    sorted_weights = local_weights[sort_idx]
    
    return sorted_token_ids, sorted_weights, expert_offsets, total_count


# ============================================================================
# Main Entry Point - OPTIMIZED FOR SMALL SEQUENCE LENGTHS
# ============================================================================

@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    output: torch.Tensor,
):
    """
    MoE kernel OPTIMIZED FOR SMALL SEQUENCE LENGTHS (1-100 tokens):
    1. Always uses CuTile GEMM (no cuBLAS fallback)
    2. Smaller tile sizes for better occupancy
    3. Batched SwiGLU for reduced kernel launch overhead
    """
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]
    device = hidden_states.device

    # Step 1: FP8 Dequantization
    A = dequantize_hidden_states(hidden_states, hidden_states_scale)
    W13 = dequantize_weights_batched(gemm1_weights, gemm1_weights_scale)
    W2 = dequantize_weights_batched(gemm2_weights, gemm2_weights_scale)

    # Step 2: DeepSeek-V3 Routing
    weights, topk_idx = deepseek_routing(
        routing_logits, routing_bias, routed_scaling_factor
    )

    # Step 3: Token sorting (vectorized)
    sorted_token_ids, sorted_weights, expert_offsets, total_count = sort_tokens_by_expert(
        topk_idx, weights, E_local, local_expert_offset, E_global
    )

    # Step 4: Expert Computation - ALL CuTile for small batches
    result = torch.zeros((T, H), dtype=torch.float32, device=device)

    if total_count > 0:
        expert_counts = expert_offsets[1:] - expert_offsets[:-1]
        active_experts = (expert_counts > 0).nonzero(as_tuple=True)[0]
        
        # Phase 1: GEMM1 for all experts using CuTile
        G1_list = []
        expert_info = []
        
        for le in active_experts.tolist():
            start = expert_offsets[le].item()
            end = expert_offsets[le + 1].item()
            
            token_idx = sorted_token_ids[start:end]
            w_tok = sorted_weights[start:end]
            A_e = A[token_idx]
            W13_e = W13[le]
            
            # Always use CuTile GEMM for small batches
            G1 = cutile_small_gemm(A_e, W13_e)
            G1_list.append(G1)
            expert_info.append((le, token_idx, w_tok, G1.shape[0]))
        
        # Phase 2: Batched SwiGLU (single kernel call for all tokens)
        if G1_list:
            G1_cat = torch.cat(G1_list, dim=0)
            C_cat = cutile_swiglu(G1_cat, I)
            
            # Phase 3: GEMM2 for all experts using CuTile
            offset = 0
            for le, token_idx, w_tok, size in expert_info:
                C = C_cat[offset:offset + size]
                offset += size
                
                W2_e = W2[le]
                # Always use CuTile GEMM for small batches
                O = cutile_small_gemm(C, W2_e)
                
                result.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    # Step 5: Write output (DPS)
    output.copy_(result.to(torch.bfloat16))
