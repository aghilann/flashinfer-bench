# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
MoE Kernel for FlashInfer Competition - DeepSeek-V3 MoE Layer.

VERSION: v3 - CuTile Optimization with Autotuning:
1. Optimized silu_and_mul kernel from TileGym
2. Autotuned Group GEMM for batching expert matrix multiplications
3. Vectorized token sorting
"""

# ============================================================================
# Constants for DeepSeek-V3/R1 MoE
# ============================================================================
BLOCK = 128  # FP8 block scale block size
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4

import cuda.tile as ct
import torch
import random
import functools
import threading
import logging
from contextlib import contextmanager
from math import ceil
from types import SimpleNamespace
from typing import Tuple, List, Any, Iterable, Callable, Sequence
from cuda.tile._numeric_semantics import RoundingMode as RMd
from cuda.tile._exception import TileCompilerTimeoutError, TileCompilerExecutionError
from cuda.tile._cext import default_tile_context

logger = logging.getLogger(__name__)

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


def _reservoir_sample(iterable: Iterable[Any], k: int, *, rng: random.Random, max_items: int) -> list[Any]:
    reservoir: list[Any] = []
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

        tuning_entries: list[tuple[Any, float]] = []
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

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


def next_power_of_2(n):
    """Return the next power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


# ============================================================================
# FP8 Dequantization - Fused CuTile Kernels (optimized, no repeat_interleave)
# ============================================================================

@ct.kernel
def dequant_hidden_kernel(
    data,       # [T, H] FP8
    scales,     # [num_blocks, T] float32
    output,     # [T, H] float32
    TILE_H: ConstInt,
    H: ConstInt,
):
    """Fused FP8 dequantization for hidden states - compute scale index on-the-fly."""
    row_idx = ct.bid(0)  # token index
    col_offsets = ct.arange(TILE_H, dtype=torch.int32)
    
    # Load FP8 data for this row
    data_tile = ct.gather(data, (row_idx, col_offsets), check_bounds=True)
    data_f32 = ct.astype(data_tile, torch.float32)
    
    # Compute block indices: col_offsets // BLOCK -> which scale block
    # scales is [num_blocks, T], we need scales[block_idx, row_idx]
    block_indices = col_offsets // 128  # BLOCK = 128
    
    # Gather scales for each element based on its block
    scale_vals = ct.gather(scales, (block_indices, row_idx), check_bounds=True)
    
    # Multiply
    result = ct.mul(data_f32, scale_vals, flush_to_zero=True)
    
    ct.scatter(output, (row_idx, col_offsets), result, check_bounds=True)


def dequantize_hidden_states(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 hidden states with block scales to float32 - FUSED."""
    T, H = data.shape
    
    if T == 0:
        return torch.empty((0, H), dtype=torch.float32, device=data.device)
    
    output = torch.empty((T, H), dtype=torch.float32, device=data.device)
    TILE_H = next_power_of_2(H)
    
    # scales is [num_blocks, T], keep it as-is (no permute needed)
    scales_cont = scales.contiguous()
    
    ct.launch(
        torch.cuda.current_stream(),
        (T,),  # one block per row/token
        dequant_hidden_kernel,
        (data.contiguous(), scales_cont, output, TILE_H, H),
    )
    return output


@ct.kernel  
def dequant_weights_kernel(
    data,       # [out_dim, in_dim] FP8
    scales,     # [num_out_blocks, num_in_blocks] float32
    output,     # [out_dim, in_dim] float32
    TILE_K: ConstInt,
    in_dim: ConstInt,
):
    """Fused FP8 dequantization for weight matrix - compute scale indices on-the-fly."""
    row_idx = ct.bid(0)  # output dimension index
    col_offsets = ct.arange(TILE_K, dtype=torch.int32)
    
    # Load FP8 data for this row
    data_tile = ct.gather(data, (row_idx, col_offsets), check_bounds=True)
    data_f32 = ct.astype(data_tile, torch.float32)
    
    # Compute block indices
    row_block = row_idx // 128  # which output block
    col_blocks = col_offsets // 128  # which input block for each element
    
    # Gather scales: scales[row_block, col_block]
    scale_vals = ct.gather(scales, (row_block, col_blocks), check_bounds=True)
    
    # Multiply
    result = ct.mul(data_f32, scale_vals, flush_to_zero=True)
    
    ct.scatter(output, (row_idx, col_offsets), result, check_bounds=True)


def dequantize_weights_batched(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 weight matrices with block scales to float32 - FUSED."""
    E_local, out_dim, in_dim = data.shape
    
    if E_local == 0:
        return torch.empty((0, out_dim, in_dim), dtype=torch.float32, device=data.device)
    
    output = torch.empty((E_local, out_dim, in_dim), dtype=torch.float32, device=data.device)
    TILE_K = next_power_of_2(in_dim)
    
    # Process each expert separately (could be batched further if needed)
    for e in range(E_local):
        data_e = data[e].contiguous()
        scales_e = scales[e].contiguous()
        output_e = output[e]
        
        ct.launch(
            torch.cuda.current_stream(),
            (out_dim,),  # one block per row
            dequant_weights_kernel,
            (data_e, scales_e, output_e, TILE_K, in_dim),
        )
    
    return output


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
# Token Sorting (Vectorized)
# ============================================================================

def sort_tokens_by_expert(
    topk_idx: torch.Tensor,
    weights: torch.Tensor,
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
# CuTile SiLU and Mul Kernel (from TileGym silu_and_mul.py - OPTIMIZED)
# ============================================================================

@ct.kernel
def silu_and_mul_kernel_row_wise(
    input,
    output,
    TILE_SIZE: ConstInt,
    hidden_size: ConstInt,
):
    """Optimized fused SiLU and Mul: silu(input[:, hidden:]) * input[:, :hidden]"""
    bid = ct.bid(0)
    offsets = ct.arange(TILE_SIZE, dtype=torch.int32)

    row_idx = bid
    a_col_idx = offsets + hidden_size
    b_col_idx = offsets

    a_tile = ct.gather(input, (row_idx, a_col_idx), check_bounds=True)
    b_tile = ct.gather(input, (row_idx, b_col_idx), check_bounds=True)
    a_tile = ct.astype(a_tile, torch.float32)
    b_tile = ct.astype(b_tile, torch.float32)

    denom = ct.add(1, ct.exp(-a_tile), flush_to_zero=True)
    sigmoid_a = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=RMd.APPROX)
    silu_a = ct.mul(a_tile, sigmoid_a, flush_to_zero=True)
    result = ct.mul(silu_a, b_tile, flush_to_zero=True)
    result = ct.astype(result, output.dtype)

    out_col_idx = offsets
    ct.scatter(output, (row_idx, out_col_idx), result, check_bounds=True)


def cutile_silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    """CuTile fused SiLU and Mul: silu(input[:, hidden:]) * input[:, :hidden]"""
    batch_size, double_hidden = input.shape
    hidden_size = double_hidden // 2
    
    if batch_size == 0:
        return torch.empty((0, hidden_size), dtype=input.dtype, device=input.device)
    
    output = torch.empty((batch_size, hidden_size), dtype=input.dtype, device=input.device)
    TILE_SIZE = next_power_of_2(hidden_size)
    
    ct.launch(
        torch.cuda.current_stream(),
        (batch_size,),
        silu_and_mul_kernel_row_wise,
        (input.contiguous(), output, TILE_SIZE, hidden_size),
    )
    return output


# ============================================================================
# CuTile Group GEMM Kernel with Autotuning (from TileGym group_gemm.py)
# ============================================================================

@ct.kernel
def group_gemm_kernel(
    As,
    Bs,
    Cs,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    num_sm: ConstInt,
    transpose_b: ConstBool,
):
    """Persistent group GEMM kernel for batching multiple matrix multiplications."""
    tile_idx = ct.bid(0)
    last_problem_end = 0
    group_size = len(As)
    zero_pad = ct.PaddingMode.ZERO

    for g in range(group_size):
        Ai = As[g]
        Bi = Bs[g]
        Ci = Cs[g]

        num_m_tiles = ct.num_tiles(Ai, 0, (TILE_M, TILE_K))
        num_k_tiles = ct.num_tiles(Ai, 1, (TILE_M, TILE_K))
        if transpose_b:
            num_n_tiles = ct.num_tiles(Bi, 0, (TILE_N, TILE_K))
        else:
            num_n_tiles = ct.num_tiles(Bi, 1, (TILE_K, TILE_N))

        num_tiles = num_m_tiles * num_n_tiles

        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)

            for kk in range(num_k_tiles):
                ta = ct.load(Ai, (tile_m_idx, kk), shape=(TILE_M, TILE_K), padding_mode=zero_pad)

                if transpose_b:
                    tb = ct.load(Bi, (tile_n_idx, kk), shape=(TILE_N, TILE_K), padding_mode=zero_pad)
                    tb = ct.transpose(tb)
                else:
                    tb = ct.load(Bi, (kk, tile_n_idx), shape=(TILE_K, TILE_N), padding_mode=zero_pad)

                ta = ct.astype(ta, ct.tfloat32)
                tb = ct.astype(tb, ct.tfloat32)
                acc = ct.mma(ta, tb, acc)

            acc = ct.astype(acc, Ci.dtype)
            ct.store(Ci, (tile_m_idx, tile_n_idx), tile=acc)

            tile_idx += num_sm

        last_problem_end = last_problem_end + num_tiles


def _group_gemm_autotune_configs():
    """Autotune configurations for group GEMM kernel."""
    gpu_capability = torch.cuda.get_device_capability()
    if gpu_capability in [(12, 0), (12, 1)]:
        yield SimpleNamespace(TILE_M=64, TILE_N=128, TILE_K=128, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_M=128, TILE_N=128, TILE_K=128, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_M=128, TILE_N=128, TILE_K=64, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_M=64, TILE_N=64, TILE_K=64, num_ctas=1, occupancy=2)
    else:
        yield SimpleNamespace(TILE_M=256, TILE_N=256, TILE_K=64, num_ctas=2, occupancy=1)
        yield SimpleNamespace(TILE_M=128, TILE_N=256, TILE_K=64, num_ctas=2, occupancy=1)
        yield SimpleNamespace(TILE_M=128, TILE_N=128, TILE_K=64, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_M=64, TILE_N=128, TILE_K=64, num_ctas=1, occupancy=2)


def cutile_group_gemm(group_A: List[torch.Tensor], group_B: List[torch.Tensor], transpose_b=True) -> List[torch.Tensor]:
    """CuTile autotuned group GEMM for batching multiple matrix multiplications."""
    if not group_A or not group_B:
        return []
    
    device = group_A[0].device
    dtype = group_A[0].dtype
    
    # Create output tensors
    group_C = []
    for A, B in zip(group_A, group_B):
        M, K = A.shape
        N = B.shape[0] if transpose_b else B.shape[1]
        C = torch.empty((M, N), device=device, dtype=dtype)
        group_C.append(C)
    
    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    stream = torch.cuda.current_stream()
    
    # Prepare contiguous inputs
    group_A_cont = [a.contiguous() for a in group_A]
    group_B_cont = [b.contiguous() for b in group_B]
    
    autotune_launch(
        stream,
        grid_fn=lambda cfg: (NUM_SMS // cfg.num_ctas * cfg.occupancy, 1, 1),
        kernel=group_gemm_kernel,
        args_fn=lambda cfg: (
            group_A_cont,
            group_B_cont,
            group_C,
            cfg.TILE_M, cfg.TILE_N, cfg.TILE_K,
            NUM_SMS // cfg.num_ctas * cfg.occupancy,
            transpose_b,
        ),
        hints_fn=lambda cfg: {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy},
        search_space=_group_gemm_autotune_configs,
    )
    return group_C


# ============================================================================
# Threshold Configuration
# ============================================================================

GROUP_GEMM_MIN_EXPERTS = 4


# ============================================================================
# Main Entry Point
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
    MoE kernel v3 - CuTile Optimization with Autotuning:
    1. Vectorized token sorting
    2. CuTile silu_and_mul kernel (optimized activation)
    3. Autotuned Group GEMM for batching multiple expert GEMMs
    4. cuBLAS for individual expert GEMMs when few experts active
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

    # Step 4: Expert Computation
    result = torch.zeros((T, H), dtype=torch.float32, device=device)

    if total_count > 0:
        expert_counts = expert_offsets[1:] - expert_offsets[:-1]
        active_experts = (expert_counts > 0).nonzero(as_tuple=True)[0]
        num_active = len(active_experts)
        
        if num_active > 0:
            expert_token_info = []
            for le in active_experts.tolist():
                start = expert_offsets[le].item()
                end = expert_offsets[le + 1].item()
                token_idx = sorted_token_ids[start:end]
                expert_token_info.append((start, end, le, token_idx))
            
            if num_active >= GROUP_GEMM_MIN_EXPERTS:
                group_A = [A[info[3]] for info in expert_token_info]
                group_W13 = [W13[info[2]] for info in expert_token_info]
                group_W2 = [W2[info[2]] for info in expert_token_info]
                
                gemm1_outputs = cutile_group_gemm(group_A, group_W13, transpose_b=True)
                swiglu_outputs = [cutile_silu_and_mul(G1) for G1 in gemm1_outputs]
                gemm2_outputs = cutile_group_gemm(swiglu_outputs, group_W2, transpose_b=True)
                
                for i, (start, end, le, token_idx) in enumerate(expert_token_info):
                    w_tok = sorted_weights[start:end]
                    result.index_add_(0, token_idx, gemm2_outputs[i] * w_tok.unsqueeze(1))
            else:
                for start, end, le, token_idx in expert_token_info:
                    A_e = A[token_idx]
                    W13_e = W13[le]
                    W2_e = W2[le]
                    
                    G1 = torch.mm(A_e, W13_e.t())
                    C = cutile_silu_and_mul(G1)
                    O = torch.mm(C, W2_e.t())
                    
                    w_tok = sorted_weights[start:end]
                    result.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    # Step 5: Write output
    output.copy_(result.to(torch.bfloat16))

if __name__ == "__main__":
    # Constants matching DeepSeek-V3/R1 benchmark spec
    NUM_EXPERTS = 256
    NUM_LOCAL_EXPERTS = 32
    HIDDEN_SIZE = 7168
    INTERMEDIATE_SIZE = 2048
    GEMM1_OUT_SIZE = 4096
    NUM_HIDDEN_BLOCKS = 56      # hidden_size // 128
    NUM_INTERMEDIATE_BLOCKS = 16  # intermediate_size // 128
    NUM_GEMM1_OUT_BLOCKS = 32   # gemm1_out_size // 128
    
    # Hardcoded seq_len
    T = 64
    device = "cuda"
    
    print(f"Creating test tensors: seq_len={T}")
    print(f"  Fixed: num_experts={NUM_EXPERTS}, num_local_experts={NUM_LOCAL_EXPERTS}")
    print(f"         hidden_size={HIDDEN_SIZE}, intermediate_size={INTERMEDIATE_SIZE}")
    print(f"         gemm1_out_size={GEMM1_OUT_SIZE}")
    
    # routing_logits: float32 [seq_len, num_experts]
    routing_logits = torch.randn(T, NUM_EXPERTS, dtype=torch.float32, device=device)
    
    # routing_bias: bfloat16 [num_experts]
    routing_bias = torch.randn(NUM_EXPERTS, dtype=torch.bfloat16, device=device) * 0.1
    
    # hidden_states: float8_e4m3fn [seq_len, hidden_size]
    hidden_states = torch.randn(T, HIDDEN_SIZE, dtype=torch.float32, device=device) * 0.5
    hidden_states = hidden_states.to(torch.float8_e4m3fn)
    
    # hidden_states_scale: float32 [num_hidden_blocks, seq_len]
    hidden_states_scale = torch.ones(NUM_HIDDEN_BLOCKS, T, dtype=torch.float32, device=device) * 0.5
    
    # gemm1_weights: float8_e4m3fn [num_local_experts, gemm1_out_size, hidden_size]
    gemm1_weights = torch.randn(NUM_LOCAL_EXPERTS, GEMM1_OUT_SIZE, HIDDEN_SIZE, dtype=torch.float32, device=device) * 0.1
    gemm1_weights = gemm1_weights.to(torch.float8_e4m3fn)
    
    # gemm1_weights_scale: float32 [num_local_experts, num_gemm1_out_blocks, num_hidden_blocks]
    gemm1_weights_scale = torch.ones(
        NUM_LOCAL_EXPERTS, NUM_GEMM1_OUT_BLOCKS, NUM_HIDDEN_BLOCKS, dtype=torch.float32, device=device
    ) * 0.3
    
    # gemm2_weights: float8_e4m3fn [num_local_experts, hidden_size, intermediate_size]
    gemm2_weights = torch.randn(NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=torch.float32, device=device) * 0.1
    gemm2_weights = gemm2_weights.to(torch.float8_e4m3fn)
    
    # gemm2_weights_scale: float32 [num_local_experts, num_hidden_blocks, num_intermediate_blocks]
    gemm2_weights_scale = torch.ones(
        NUM_LOCAL_EXPERTS, NUM_HIDDEN_BLOCKS, NUM_INTERMEDIATE_BLOCKS, dtype=torch.float32, device=device
    ) * 0.2
    
    # local_expert_offset: int32 scalar
    local_expert_offset = 0
    
    # routed_scaling_factor: float32 scalar
    routed_scaling_factor = 2.5
    
    # Pre-allocate output: bfloat16 [seq_len, hidden_size]
    output = torch.zeros(T, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    
    print(f"Tensors created. GPU Memory: ~{torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print("Running kernel...")
    
    run(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        local_expert_offset,
        routed_scaling_factor,
        output,
    )
    
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")
    print(f"Output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.float().mean().item():.4f}")