# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
MoE Kernel for FlashInfer Competition - DeepSeek-V3 MoE Layer.

Optimizations:
1. Vectorized token sorting
2. CuTile fused SwiGLU kernel
"""

import cuda.tile as ct
import torch
from typing import Tuple

ConstInt = ct.Constant[int]

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
# CuTile GEMM Kernel - OPTIMIZATION 3 (for small batches)
# ============================================================================

@ct.kernel
def gemm_kernel(
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
    """Simple GEMM kernel: C = A @ B.T"""
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


def cutile_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """CuTile GEMM: C = A @ B.T"""
    M, K = A.shape
    N = B.shape[0]
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    TILE_M, TILE_N, TILE_K = 64, 64, 64
    num_m_tiles = (M + TILE_M - 1) // TILE_M
    num_n_tiles = (N + TILE_N - 1) // TILE_N
    grid = (num_m_tiles * num_n_tiles,)
    
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_kernel,
        (A.contiguous(), B.contiguous(), C, M, N, K, TILE_M, TILE_N, TILE_K),
    )
    return C


# Threshold: use CuTile GEMM for batch <= this, cuBLAS for larger
CUTILE_GEMM_THRESHOLD = 64


# ============================================================================
# CuTile Fused SwiGLU Kernel - OPTIMIZATION 2
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
# Token Sorting (Vectorized) - OPTIMIZATION 1
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
    Optimized MoE kernel.
    Optimizations:
    1. Vectorized token sorting
    2. CuTile fused SwiGLU kernel (batched)
    3. CuTile GEMM for small batches
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

    # Step 3: Token sorting (OPTIMIZATION 1)
    sorted_token_ids, sorted_weights, expert_offsets, total_count = sort_tokens_by_expert(
        topk_idx, weights, E_local, local_expert_offset, E_global
    )

    # Step 4: Expert Computation with Batched SwiGLU (OPTIMIZATION 3)
    result = torch.zeros((T, H), dtype=torch.float32, device=device)

    if total_count > 0:
        expert_counts = expert_offsets[1:] - expert_offsets[:-1]
        active_experts = (expert_counts > 0).nonzero(as_tuple=True)[0]
        
        # Phase 1: GEMM1 for all experts
        G1_list = []
        expert_info = []
        
        for le in active_experts.tolist():
            start = expert_offsets[le].item()
            end = expert_offsets[le + 1].item()
            
            token_idx = sorted_token_ids[start:end]
            w_tok = sorted_weights[start:end]
            A_e = A[token_idx]
            W13_e = W13[le]
            batch_size = A_e.shape[0]
            
            # OPTIMIZATION 3: CuTile GEMM for small batches
            if batch_size <= CUTILE_GEMM_THRESHOLD:
                G1 = cutile_gemm(A_e, W13_e)
            else:
                G1 = torch.mm(A_e, W13_e.t())
            G1_list.append(G1)
            expert_info.append((le, token_idx, w_tok, G1.shape[0]))
        
        # Phase 2: Batched SwiGLU (single kernel call)
        if G1_list:
            G1_cat = torch.cat(G1_list, dim=0)
            C_cat = cutile_swiglu(G1_cat, I)
            
            # Phase 3: GEMM2 for all experts
            offset = 0
            for le, token_idx, w_tok, size in expert_info:
                C = C_cat[offset:offset + size]
                offset += size
                
                W2_e = W2[le]
                # OPTIMIZATION 3: CuTile GEMM for small batches
                if size <= CUTILE_GEMM_THRESHOLD:
                    O = cutile_gemm(C, W2_e)
                else:
                    O = torch.mm(C, W2_e.t())
                
                result.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    # Step 5: Write output (DPS)
    output.copy_(result.to(torch.bfloat16))