#!/usr/bin/env python3
"""
Quick test script to compare CuTile vs PyTorch MoE implementations.

Usage:
    python test_cutile.py                    # Run correctness test
    python test_cutile.py --benchmark        # Run benchmark
    python test_cutile.py --benchmark --iterations 100
"""

import argparse
import time
from typing import Tuple

import torch

# ============================================================================
# Constants matching DeepSeek-V3/R1 benchmark spec (fixed values)
# ============================================================================
# Geometry
NUM_EXPERTS = 256           # num_experts (E_global)
NUM_LOCAL_EXPERTS = 32      # num_local_experts (E_local)
HIDDEN_SIZE = 7168          # hidden_size (H)
INTERMEDIATE_SIZE = 2048    # intermediate_size (I)
GEMM1_OUT_SIZE = 4096       # gemm1_out_size (2*I)

# Block sizes for FP8 quantization (block = 128)
BLOCK = 128
NUM_HIDDEN_BLOCKS = 56      # hidden_size // 128 = 7168 // 128
NUM_INTERMEDIATE_BLOCKS = 16  # intermediate_size // 128 = 2048 // 128
NUM_GEMM1_OUT_BLOCKS = 32   # gemm1_out_size // 128 = 4096 // 128

# Aliases for backward compatibility
H = HIDDEN_SIZE
I = INTERMEDIATE_SIZE
E_LOCAL = NUM_LOCAL_EXPERTS
E_GLOBAL = NUM_EXPERTS


def create_test_tensors(
    seq_len: int = 4,  # Variable: number of tokens
    device: str = "cuda",
) -> Tuple[torch.Tensor, ...]:
    """
    Create test tensors matching DeepSeek-V3 benchmark spec.
    
    Fixed dimensions:
    - num_experts = 256
    - num_local_experts = 32  
    - hidden_size = 7168
    - intermediate_size = 2048
    - gemm1_out_size = 4096
    - num_hidden_blocks = 56
    - num_intermediate_blocks = 16
    - num_gemm1_out_blocks = 32
    
    Variable:
    - seq_len (T)
    """
    T = seq_len
    print(f"Creating test tensors: seq_len={T}")
    print(f"  Fixed: num_experts={NUM_EXPERTS}, num_local_experts={NUM_LOCAL_EXPERTS}")
    print(f"         hidden_size={HIDDEN_SIZE}, intermediate_size={INTERMEDIATE_SIZE}")
    print(f"         gemm1_out_size={GEMM1_OUT_SIZE}")
    
    # ========================================================================
    # Inputs (matching benchmark spec exactly)
    # ========================================================================
    
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
    
    return (
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


def run_torch_reference(
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
) -> torch.Tensor:
    """
    Pure PyTorch reference implementation for correctness checking.
    This is a copy of the torch/kernel.py implementation.
    """
    from solution.torch.kernel import run as torch_run
    
    # Clone output to not modify the original
    output_ref = output.clone()
    torch_run(
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
        output_ref,
    )
    return output_ref


def run_cutile(
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
) -> torch.Tensor:
    """Run CuTile implementation."""
    from kernel import run as cutile_run
    
    output_cutile = output.clone()
    cutile_run(
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
        output_cutile,
    )
    return output_cutile


def check_correctness(
    output_ref: torch.Tensor,
    output_test: torch.Tensor,
    name: str = "CuTile",
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> bool:
    """Check if two outputs are close enough."""
    
    # Convert to float32 for comparison
    ref_f32 = output_ref.to(torch.float32)
    test_f32 = output_test.to(torch.float32)
    
    # Compute metrics
    abs_diff = torch.abs(ref_f32 - test_f32)
    rel_diff = abs_diff / (torch.abs(ref_f32) + 1e-8)
    
    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    
    # Check if close
    is_close = torch.allclose(ref_f32, test_f32, rtol=rtol, atol=atol)
    
    print(f"\n{'='*60}")
    print(f"Correctness Check: {name}")
    print(f"{'='*60}")
    print(f"Max absolute diff:  {max_abs_diff:.6e}")
    print(f"Mean absolute diff: {mean_abs_diff:.6e}")
    print(f"Max relative diff:  {max_rel_diff:.6e}")
    print(f"Tolerance: rtol={rtol}, atol={atol}")
    print(f"Result: {'✓ PASSED' if is_close else '✗ FAILED'}")
    print(f"{'='*60}\n")
    
    return is_close


def benchmark(
    func,
    args,
    name: str,
    warmup: int = 10,
    iterations: int = 50,
) -> float:
    """Benchmark a function and return average time in ms."""
    
    # Warmup
    for _ in range(warmup):
        func(*args)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_ms = (end - start) / iterations * 1000
    return avg_ms


def run_benchmark(seq_len: int = 16, iterations: int = 50, usesmall: bool = False):
    """Run benchmark comparing CuTile vs PyTorch."""
    
    kernel_name = "smallkernel.py" if usesmall else "kernel.py"
    
    print(f"\n{'='*70}")
    print(f"Benchmark: CuTile vs PyTorch MoE Kernel ({kernel_name})")
    print(f"seq_len={seq_len} (variable)")
    print(f"Fixed: num_experts={NUM_EXPERTS}, num_local_experts={NUM_LOCAL_EXPERTS}")
    print(f"       hidden_size={HIDDEN_SIZE}, intermediate_size={INTERMEDIATE_SIZE}")
    print(f"Iterations={iterations}")
    print(f"{'='*70}\n")
    
    # Create test data
    args = create_test_tensors(seq_len=seq_len)
    
    # Use importlib to explicitly load from correct files
    import importlib.util
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    
    if usesmall:
        cutile_kernel_path = os.path.join(repo_root, 'solution', 'cutile', 'smallkernel.py')
    else:
        cutile_kernel_path = os.path.join(repo_root, 'solution', 'cutile', 'kernel.py')
    torch_kernel_path = os.path.join(repo_root, 'solution', 'torch', 'kernel.py')
    
    def load_kernel(kernel_path, module_name):
        spec = importlib.util.spec_from_file_location(module_name, kernel_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    # Load CuTile kernel
    cutile_kernel = load_kernel(cutile_kernel_path, "cutile_kernel_bench")
    cutile_run = cutile_kernel.run
    
    # Try to load torch reference
    has_torch_ref = False
    torch_run = None
    try:
        if os.path.exists(torch_kernel_path):
            torch_kernel = load_kernel(torch_kernel_path, "torch_kernel_bench")
            torch_run = torch_kernel.run
            has_torch_ref = True
    except Exception as e:
        print(f"Warning: Could not import torch reference: {e}")
    
    # Benchmark CuTile
    cutile_time = benchmark(
        lambda *a: cutile_run(*a),
        args,
        "CuTile",
        iterations=iterations,
    )
    print(f"CuTile:  {cutile_time:.3f} ms")
    
    # Benchmark PyTorch reference
    if has_torch_ref:
        # Need to clone output for each run
        def torch_run_clone(*a):
            args_list = list(a)
            args_list[-1] = args_list[-1].clone()  # Clone output
            torch_run(*args_list)
        
        torch_time = benchmark(
            torch_run_clone,
            args,
            "PyTorch",
            iterations=iterations,
        )
        print(f"PyTorch: {torch_time:.3f} ms")
        print(f"Speedup: {torch_time / cutile_time:.2f}x")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Test CuTile MoE kernel")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length (number of tokens)")
    parser.add_argument("--tokens", type=int, default=None, help="Alias for --seq-len")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--skip-correctness", action="store_true", help="Skip correctness check")
    parser.add_argument("--vary-seq-len", action="store_true", help="Test multiple seq_len values")
    parser.add_argument("--usesmall", action="store_true", help="Use smallkernel.py optimized for small sequences")
    args = parser.parse_args()
    
    # Handle --tokens as alias for --seq-len
    seq_len = args.tokens if args.tokens is not None else args.seq_len

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    if args.usesmall:
        print("Using: smallkernel.py (optimized for small sequences)")
    else:
        print("Using: kernel.py")
    
    # Setup paths - works whether run from repo root or solution/cutile/
    import sys
    import os
    
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine repo root (script is at solution/cutile/test_cutile.py)
    # So repo root is 2 levels up
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Define solution directories
    cutile_dir = os.path.join(repo_root, 'solution', 'cutile')
    torch_dir = os.path.join(repo_root, 'solution', 'torch')
    
    # Use importlib to explicitly load from correct files (avoid sys.path confusion)
    import importlib.util
    
    def load_kernel(kernel_path, module_name):
        """Load a kernel module from a specific file path."""
        spec = importlib.util.spec_from_file_location(module_name, kernel_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    # Choose kernel file based on --usesmall flag
    if args.usesmall:
        cutile_kernel_path = os.path.join(cutile_dir, 'smallkernel.py')
    else:
        cutile_kernel_path = os.path.join(cutile_dir, 'kernel.py')
    torch_kernel_path = os.path.join(torch_dir, 'kernel.py')
    
    # Run correctness check
    if not args.skip_correctness:
        print("\n" + "="*70)
        print("Running Correctness Check")
        print("="*70)
        
        test_args = create_test_tensors(seq_len=seq_len)
        
        # Run CuTile - explicitly load from cutile/kernel.py
        cutile_kernel = load_kernel(cutile_kernel_path, "cutile_kernel")
        output_cutile = test_args[-1].clone()
        cutile_kernel.run(*test_args[:-1], output_cutile)
        
        # Run PyTorch reference
        try:
            if os.path.exists(torch_kernel_path):
                torch_kernel = load_kernel(torch_kernel_path, "torch_kernel")
                
                output_torch = test_args[-1].clone()
                torch_kernel.run(*test_args[:-1], output_torch)
                
                check_correctness(output_torch, output_cutile, "CuTile vs PyTorch")
            else:
                print(f"Warning: PyTorch reference not found at {torch_kernel_path}")
                print("Skipping correctness check against reference.")
        except Exception as e:
            print(f"Warning: Could not run PyTorch reference: {e}")
    
    # Run benchmark
    if args.benchmark:
        if args.vary_seq_len:
            # Test multiple seq_len values
            for sl in [1, 4, 16, 64, 256, 1024, 2048, 4096, 8192, 16384]:
                try:
                    run_benchmark(seq_len=sl, iterations=args.iterations, usesmall=args.usesmall)
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM at seq_len={sl}, stopping")
                    break
        else:
            run_benchmark(seq_len=seq_len, iterations=args.iterations, usesmall=args.usesmall)


if __name__ == "__main__":
    main()
