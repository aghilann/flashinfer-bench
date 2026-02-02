Here's a comprehensive summary of the kernel:

---

# DeepSeek-V3 MoE Kernel - CuTile Optimized

## Overview

This kernel implements the **Mixture-of-Experts (MoE) layer** for DeepSeek-V3/R1 models, optimized using NVIDIA CuTile (CUDA Tile) for high-performance GPU execution.

### Model Configuration (Fixed)
- **256 total experts**, 32 local experts per node
- **Hidden size:** 7168
- **Intermediate size:** 2048
- **Top-K routing:** 8 experts per token
- **FP8 block quantization:** 128-element blocks

---

## Pipeline Stages

```
Input (FP8) → Dequantize → Route → Sort → Expert GEMMs → Aggregate → Output (BF16)
```

### Stage 1: FP8 Dequantization (OPTIMIZED)
Converts FP8-quantized tensors to FP32 with block-scale multiplication.

**Optimization:** Fused CuTile kernels that compute scale indices on-the-fly, eliminating expensive `repeat_interleave` operations.

```python
# Before: Created massive intermediate tensors
scales_expanded = scales.repeat_interleave(128, dim=1)  # 56 → 7168
result = data_f32 * scales_expanded

# After: Fused kernel with on-the-fly indexing
block_idx = col_offset // 128
scale = scales[block_idx, row_idx]  # Direct gather
result = data_f32 * scale
```

### Stage 2: DeepSeek-V3 Routing
Implements the no-auxiliary-loss routing algorithm:
1. Apply sigmoid to logits + bias
2. Group experts into 8 groups, select top-4 groups
3. Within selected groups, pick top-8 experts total
4. Normalize weights with scaling factor (2.5)

### Stage 3: Token Sorting (Vectorized)
Sorts tokens by expert assignment using fully vectorized PyTorch operations:
- `bincount` for expert counts
- `cumsum` for offsets
- `argsort` for stable sorting

### Stage 4: Expert Computation
Two-stage feed-forward with SwiGLU activation:

```
GEMM1: [tokens, 7168] × [4096, 7168]ᵀ → [tokens, 4096]
SwiGLU: silu([:, 2048:]) * [:, :2048] → [tokens, 2048]
GEMM2: [tokens, 2048] × [7168, 2048]ᵀ → [tokens, 7168]
```

**Adaptive strategy:**
- **≥4 active experts:** Use autotuned CuTile Group GEMM (batched)
- **<4 active experts:** Use cuBLAS individual GEMMs

### Stage 5: Output Aggregation
Weighted sum of expert outputs using `index_add_`, converted to BF16.

---

## CuTile Kernels

### 1. `dequant_hidden_kernel`
- Fused FP8→FP32 + scale multiplication for hidden states
- One thread block per token row
- Computes `block_idx = col // 128` on-the-fly

### 2. `dequant_weights_kernel`
- Fused FP8→FP32 + scale multiplication for weight matrices
- One thread block per output dimension row
- Computes both row and column block indices

### 3. `silu_and_mul_kernel_row_wise`
- Fused SwiGLU activation: `silu(x[:, I:]) * x[:, :I]`
- Uses approximate reciprocal for sigmoid
- One thread block per row

### 4. `group_gemm_kernel`
- Persistent group GEMM for batching multiple expert matrix multiplications
- Autotuned tile sizes (64-256) based on GPU capability
- Supports transposed B matrix

---

## Autotuning

The Group GEMM kernel uses runtime autotuning:
- **Search space:** Multiple tile configurations (TILE_M, TILE_N, TILE_K)
- **Caching:** Results cached by input shapes
- **GPU-specific:** Different defaults for Blackwell vs Hopper/Ampere

---

## Performance Results

| Seq Len | CuTile | PyTorch | Speedup |
|---------|--------|---------|---------|
| 1 | 2.5 ms | 10.1 ms | **4.1x** |
| 64 | 4.0 ms | 12.4 ms | **3.1x** |
| 1024 | 5.5 ms | 18.0 ms | **3.3x** |
| 16384 | 10.8 ms | 41.6 ms | **3.8x** |

---

## Key Optimizations Summary

1. **Fused FP8 Dequantization** - Eliminated ~12ms of `repeat_interleave` overhead
2. **CuTile SiLU+Mul Fusion** - Single kernel for activation (~7µs per call)
3. **Autotuned Group GEMM** - Batched expert computation with optimal tile sizes
4. **Vectorized Token Sorting** - No Python loops, pure tensor operations
5. **Adaptive Expert Dispatch** - Group GEMM for many experts, cuBLAS for few