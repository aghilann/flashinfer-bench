# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
MoE Kernel for FlashInfer Competition - DeepSeek-V3 MoE Layer.

VERSION: v4-hardcoded - CuTile with Hardcoded Optimal Tile Sizes:
1. Optimized silu_and_mul kernel from TileGym
2. Group GEMM with hardcoded optimal tile configurations (no autotuning)
3. Vectorized token sorting
4. Tile size selection based on workload size (small vs large)

Optimal tile sizes determined by autotuning on NVIDIA B200:
- Small workloads (tokens_per_expert < 100): TILE_M=64, TILE_N=128, TILE_K=64, occupancy=2
- Large workloads (tokens_per_expert >= 100): TILE_M=256, TILE_N=256, TILE_K=64, num_ctas=2
"""

# ============================================================================
# Constants for DeepSeek-V3/R1 MoE
# ============================================================================
BLOCK = 128  # FP8 block scale block size
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4

# Threshold for switching between small and large tile configs
# Based on autotuning results: small config better when M < 100
LARGE_WORKLOAD_THRESHOLD = 100

import cuda.tile as ct
import torch
from typing import Tuple, List
from cuda.tile._numeric_semantics import RoundingMode as RMd

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
# CuTile Group GEMM Kernel - Hardcoded Optimal Tile Sizes
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


# Pre-compiled kernel handles for small and large workloads
_small_kernel = ct.kernel(group_gemm_kernel._pyfunc, num_ctas=1, occupancy=2)
_large_kernel = ct.kernel(group_gemm_kernel._pyfunc, num_ctas=2, occupancy=1)


def cutile_group_gemm(group_A: List[torch.Tensor], group_B: List[torch.Tensor], transpose_b=True) -> List[torch.Tensor]:
    """CuTile group GEMM with hardcoded optimal tile configurations.
    
    Tile sizes determined by autotuning on NVIDIA B200:
    - Small workloads (avg M < 100): TILE_M=64, TILE_N=128, TILE_K=64, occupancy=2
    - Large workloads (avg M >= 100): TILE_M=256, TILE_N=256, TILE_K=64, num_ctas=2
    """
    if not group_A or not group_B:
        return []
    
    device = group_A[0].device
    dtype = group_A[0].dtype
    
    # Create output tensors
    group_C = []
    total_m = 0
    for A, B in zip(group_A, group_B):
        M, K = A.shape
        N = B.shape[0] if transpose_b else B.shape[1]
        C = torch.empty((M, N), device=device, dtype=dtype)
        group_C.append(C)
        total_m += M
    
    # Calculate average M dimension per expert
    avg_m = total_m / len(group_A)
    
    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    stream = torch.cuda.current_stream()
    
    # Prepare contiguous inputs
    group_A_cont = [a.contiguous() for a in group_A]
    group_B_cont = [b.contiguous() for b in group_B]
    
    # Select tile configuration based on workload size
    if avg_m < LARGE_WORKLOAD_THRESHOLD:
        # Small workload config (best for seq_len 1-80)
        TILE_M, TILE_N, TILE_K = 64, 128, 64
        num_ctas, occupancy = 1, 2
        kernel = _small_kernel
    else:
        # Large workload config (best for seq_len 901+)
        TILE_M, TILE_N, TILE_K = 256, 256, 64
        num_ctas, occupancy = 2, 1
        kernel = _large_kernel
    
    grid = (NUM_SMS // num_ctas * occupancy, 1, 1)
    num_sm = NUM_SMS // num_ctas * occupancy
    
    ct.launch(
        stream,
        grid,
        kernel,
        (group_A_cont, group_B_cont, group_C, TILE_M, TILE_N, TILE_K, num_sm, transpose_b),
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
    MoE kernel v4-hardcoded - CuTile with Hardcoded Optimal Tile Sizes:
    1. Vectorized token sorting
    2. CuTile silu_and_mul kernel (optimized activation)
    3. Group GEMM with hardcoded optimal tile sizes (no autotuning overhead)
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
