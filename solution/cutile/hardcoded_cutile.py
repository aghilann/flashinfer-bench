# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""
MoE Kernel - Pure CuTile Implementation (no Triton).
Replaces Triton dequant kernels with CuTile equivalents.
"""

BLOCK = 128
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
LARGE_WORKLOAD_THRESHOLD = 100
GROUP_GEMM_MIN_EXPERTS = 4

import cuda.tile as ct
import torch
from typing import Tuple, List
from cuda.tile._numeric_semantics import RoundingMode as RMd

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


def next_power_of_2(n):
    return 1 << (n - 1).bit_length() if n > 0 else 1


# ============================================================================
# FP8 Dequantization - CuTile Kernels
# ============================================================================

@ct.kernel
def dequant_hidden_kernel(data, scales, output, TILE_H: ConstInt, H: ConstInt):
    """CuTile FP8 dequantization for hidden states.
    data: [T, H] FP8, scales: [num_blocks, T], output: [T, H] float32
    """
    row_idx = ct.bid(0)
    col_offsets = ct.arange(TILE_H, dtype=torch.int32)
    
    # Load FP8 data and convert to float32
    data_tile = ct.gather(data, (row_idx, col_offsets), check_bounds=True)
    data_f32 = ct.astype(data_tile, torch.float32)
    
    # Compute block indices: col_offsets // 128
    block_indices = col_offsets // 128
    
    # Gather scales: scales[block_idx, row_idx]
    scale_vals = ct.gather(scales, (block_indices, row_idx), check_bounds=True)
    
    # Multiply and store
    result = ct.mul(data_f32, scale_vals, flush_to_zero=True)
    ct.scatter(output, (row_idx, col_offsets), result, check_bounds=True)


def dequantize_hidden_states(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    T, H = data.shape
    if T == 0:
        return torch.empty((0, H), dtype=torch.float32, device=data.device)
    
    output = torch.empty((T, H), dtype=torch.float32, device=data.device)
    TILE_H = next_power_of_2(H)
    
    ct.launch(torch.cuda.current_stream(), (T,), dequant_hidden_kernel,
              (data.contiguous(), scales.contiguous(), output, TILE_H, H))
    return output


@ct.kernel
def dequant_weights_kernel(data, scales, output, TILE_K: ConstInt, in_dim: ConstInt):
    """CuTile FP8 dequantization for weight matrix.
    data: [out_dim, in_dim] FP8, scales: [num_out_blocks, num_in_blocks], output: [out_dim, in_dim] float32
    """
    row_idx = ct.bid(0)
    col_offsets = ct.arange(TILE_K, dtype=torch.int32)
    
    # Load FP8 data and convert to float32
    data_tile = ct.gather(data, (row_idx, col_offsets), check_bounds=True)
    data_f32 = ct.astype(data_tile, torch.float32)
    
    # Compute block indices
    row_block = row_idx // 128
    col_blocks = col_offsets // 128
    
    # Gather scales: scales[row_block, col_block]
    scale_vals = ct.gather(scales, (row_block, col_blocks), check_bounds=True)
    
    # Multiply and store
    result = ct.mul(data_f32, scale_vals, flush_to_zero=True)
    ct.scatter(output, (row_idx, col_offsets), result, check_bounds=True)


def dequantize_weights_batched(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    E_local, out_dim, in_dim = data.shape
    if E_local == 0:
        return torch.empty((0, out_dim, in_dim), dtype=torch.float32, device=data.device)
    
    output = torch.empty((E_local, out_dim, in_dim), dtype=torch.float32, device=data.device)
    TILE_K = next_power_of_2(in_dim)
    
    for e in range(E_local):
        ct.launch(torch.cuda.current_stream(), (out_dim,), dequant_weights_kernel,
                  (data[e].contiguous(), scales[e].contiguous(), output[e], TILE_K, in_dim))
    return output


# ============================================================================
# Routing
# ============================================================================

def deepseek_routing(routing_logits: torch.Tensor, routing_bias: torch.Tensor,
                     routed_scaling_factor: float) -> Tuple[torch.Tensor, torch.Tensor]:
    T, E_global = routing_logits.shape
    group_size = E_global // N_GROUP
    
    s = torch.sigmoid(routing_logits)
    s_with_bias = s + routing_bias.float()
    
    # Group selection
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
    group_scores = torch.topk(s_wb_grouped, k=2, dim=2).values.sum(dim=2)
    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1).indices
    
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)
    
    # Top-K selection
    scores_pruned = torch.where(score_mask > 0, s_with_bias, torch.finfo(torch.float32).min)
    topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1).indices
    
    M = torch.zeros_like(s).scatter_(1, topk_idx, 1.0)
    weights = (s * M) / ((s * M).sum(dim=1, keepdim=True) + 1e-20) * routed_scaling_factor
    return weights, topk_idx


# ============================================================================
# Token Sorting
# ============================================================================
def sort_tokens_by_expert(topk_idx: torch.Tensor, weights: torch.Tensor,
                          E_local: int, local_expert_offset: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    device = topk_idx.device
    T, K = topk_idx.shape
    
    token_indices = torch.arange(T, device=device).unsqueeze(1).expand(T, K).reshape(-1)
    flat_experts = topk_idx.reshape(-1)
    
    local_mask = (flat_experts >= local_expert_offset) & (flat_experts < local_expert_offset + E_local)
    if not local_mask.any():
        return (torch.empty(0, dtype=torch.int64, device=device),
                torch.empty(0, dtype=torch.float32, device=device),
                torch.zeros(E_local + 1, dtype=torch.int64, device=device), 0)
    
    local_experts = flat_experts[local_mask] - local_expert_offset
    local_tokens = token_indices[local_mask]
    local_weights = weights[local_tokens, flat_experts[local_mask]]
    
    sort_idx = torch.argsort(local_experts, stable=True)
    expert_offsets = torch.zeros(E_local + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = torch.bincount(local_experts, minlength=E_local).cumsum(0)
    
    return local_tokens[sort_idx], local_weights[sort_idx], expert_offsets, len(local_tokens)


# ============================================================================
# CuTile Kernels
# ============================================================================

@ct.kernel
def silu_and_mul_kernel(input, output, TILE_SIZE: ConstInt, hidden_size: ConstInt):
    row_idx = ct.bid(0)
    offsets = ct.arange(TILE_SIZE, dtype=torch.int32)
    
    a = ct.astype(ct.gather(input, (row_idx, offsets + hidden_size), check_bounds=True), torch.float32)
    b = ct.astype(ct.gather(input, (row_idx, offsets), check_bounds=True), torch.float32)
    
    silu_a = ct.mul(a, ct.truediv(1.0, ct.add(1, ct.exp(-a), flush_to_zero=True),
                                   flush_to_zero=True, rounding_mode=RMd.APPROX), flush_to_zero=True)
    result = ct.astype(ct.mul(silu_a, b, flush_to_zero=True), output.dtype)
    ct.scatter(output, (row_idx, offsets), result, check_bounds=True)


def cutile_silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    batch_size, double_hidden = input.shape
    hidden_size = double_hidden // 2
    if batch_size == 0:
        return torch.empty((0, hidden_size), dtype=input.dtype, device=input.device)
    
    output = torch.empty((batch_size, hidden_size), dtype=input.dtype, device=input.device)
    ct.launch(torch.cuda.current_stream(), (batch_size,), silu_and_mul_kernel,
              (input.contiguous(), output, next_power_of_2(hidden_size), hidden_size))
    return output


@ct.kernel
def group_gemm_kernel(As, Bs, Cs, TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt,
                      num_sm: ConstInt, transpose_b: ConstBool):
    tile_idx = ct.bid(0)
    last_problem_end = 0

    for g in range(len(As)):
        Ai, Bi, Ci = As[g], Bs[g], Cs[g]
        num_m_tiles = ct.num_tiles(Ai, 0, (TILE_M, TILE_K))
        num_k_tiles = ct.num_tiles(Ai, 1, (TILE_M, TILE_K))
        num_n_tiles = ct.num_tiles(Bi, 0 if transpose_b else 1, (TILE_N, TILE_K) if transpose_b else (TILE_K, TILE_N))
        num_tiles = num_m_tiles * num_n_tiles

        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            idx = tile_idx - last_problem_end
            tile_m_idx, tile_n_idx = idx // num_n_tiles, idx % num_n_tiles
            acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)

            for kk in range(num_k_tiles):
                ta = ct.load(Ai, (tile_m_idx, kk), shape=(TILE_M, TILE_K), padding_mode=ct.PaddingMode.ZERO)
                if transpose_b:
                    tb = ct.transpose(ct.load(Bi, (tile_n_idx, kk), shape=(TILE_N, TILE_K), padding_mode=ct.PaddingMode.ZERO))
                else:
                    tb = ct.load(Bi, (kk, tile_n_idx), shape=(TILE_K, TILE_N), padding_mode=ct.PaddingMode.ZERO)
                acc = ct.mma(ct.astype(ta, ct.tfloat32), ct.astype(tb, ct.tfloat32), acc)

            ct.store(Ci, (tile_m_idx, tile_n_idx), tile=ct.astype(acc, Ci.dtype))
            tile_idx += num_sm
        last_problem_end += num_tiles


_small_kernel = ct.kernel(group_gemm_kernel._pyfunc, num_ctas=1, occupancy=2)
_large_kernel = ct.kernel(group_gemm_kernel._pyfunc, num_ctas=1, occupancy=1)


def cutile_group_gemm(group_A: List[torch.Tensor], group_B: List[torch.Tensor], transpose_b=True) -> List[torch.Tensor]:
    if not group_A:
        return []
    
    device, dtype = group_A[0].device, group_A[0].dtype
    group_C = [torch.empty((A.shape[0], B.shape[0] if transpose_b else B.shape[1]), device=device, dtype=dtype)
               for A, B in zip(group_A, group_B)]
    
    avg_m = sum(A.shape[0] for A in group_A) / len(group_A)
    is_large = avg_m >= LARGE_WORKLOAD_THRESHOLD
    TILE_M, TILE_N, TILE_K = (128, 128, 64) if is_large else (64, 128, 64)
    kernel = _large_kernel if is_large else _small_kernel
    occupancy = 1 if is_large else 2
    
    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    num_sm = NUM_SMS * occupancy
    
    ct.launch(torch.cuda.current_stream(), (num_sm, 1, 1), kernel,
              ([a.contiguous() for a in group_A], [b.contiguous() for b in group_B],
               group_C, TILE_M, TILE_N, TILE_K, num_sm, transpose_b))
    return group_C


# ============================================================================
# Main Entry Point
# ============================================================================
@torch.no_grad()
def run(routing_logits: torch.Tensor, routing_bias: torch.Tensor,
        hidden_states: torch.Tensor, hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor, gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor, gemm2_weights_scale: torch.Tensor,
        local_expert_offset: int, routed_scaling_factor: float, output: torch.Tensor):
    
    E_local, T, H = gemm1_weights.shape[0], routing_logits.shape[0], 7168
    device = hidden_states.device

    # Dequantize (using CuTile kernels)
    A = dequantize_hidden_states(hidden_states, hidden_states_scale)
    W13 = dequantize_weights_batched(gemm1_weights, gemm1_weights_scale)
    W2 = dequantize_weights_batched(gemm2_weights, gemm2_weights_scale)

    # Route and sort
    weights, topk_idx = deepseek_routing(routing_logits, routing_bias, routed_scaling_factor)
    sorted_token_ids, sorted_weights, expert_offsets, total_count = sort_tokens_by_expert(
        topk_idx, weights, E_local, local_expert_offset)

    result = torch.zeros((T, H), dtype=torch.float32, device=device)
    
    if total_count > 0:
        expert_counts = expert_offsets[1:] - expert_offsets[:-1]
        active_experts = (expert_counts > 0).nonzero(as_tuple=True)[0].tolist()
        
        if active_experts:
            expert_info = [(expert_offsets[e].item(), expert_offsets[e+1].item(), e,
                           sorted_token_ids[expert_offsets[e]:expert_offsets[e+1]]) for e in active_experts]
            
            if len(active_experts) >= GROUP_GEMM_MIN_EXPERTS:
                group_A = [A[info[3]] for info in expert_info]
                gemm1_out = cutile_group_gemm(group_A, [W13[info[2]] for info in expert_info])
                swiglu_out = [cutile_silu_and_mul(g) for g in gemm1_out]
                gemm2_out = cutile_group_gemm(swiglu_out, [W2[info[2]] for info in expert_info])
                
                for i, (start, end, _, token_idx) in enumerate(expert_info):
                    result.index_add_(0, token_idx, gemm2_out[i] * sorted_weights[start:end].unsqueeze(1))
            else:
                for start, end, le, token_idx in expert_info:
                    O = torch.mm(cutile_silu_and_mul(torch.mm(A[token_idx], W13[le].t())), W2[le].t())
                    result.index_add_(0, token_idx, O * sorted_weights[start:end].unsqueeze(1))

    output.copy_(result.to(torch.bfloat16))
