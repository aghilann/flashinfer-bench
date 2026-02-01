"""
Simple benchmark script for the torch MoE kernel.
Loads workloads from the trace set and benchmarks the torch implementation.
"""

import os
import sys
import time
import json
from pathlib import Path

import torch
import safetensors.torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from solution.torch.kernel import run as torch_moe_run


def get_trace_set_path() -> Path:
    """Get trace set path from environment variable."""
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Set it to the path of your flashinfer-trace dataset."
        )
    return Path(path)


def load_workload_inputs(trace_path: Path, workload: dict, device: str = "cuda") -> dict:
    """Load workload inputs, handling both safetensors and random inputs."""
    
    axes = workload["axes"]
    inputs_spec = workload["inputs"]
    
    T = axes["seq_len"]  # Number of tokens
    
    # Fixed DeepSeek-V3/R1 geometry
    H = 7168           # hidden_size
    I = 2048           # intermediate_size
    E_global = 256     # total experts
    E_local = 32       # local experts
    BLOCK = 128
    
    num_hidden_blocks = H // BLOCK          # 56
    num_intermediate_blocks = I // BLOCK    # 16
    num_gemm1_out_blocks = (2 * I) // BLOCK # 32
    
    tensors = {}
    
    for name, spec in inputs_spec.items():
        if spec["type"] == "safetensors":
            # Load from safetensors file
            st_path = trace_path / spec["path"]
            tensor_key = spec["tensor_key"]
            loaded = safetensors.torch.load_file(str(st_path), device=device)
            tensors[name] = loaded[tensor_key]
            
        elif spec["type"] == "random":
            # Generate random tensor based on expected shape
            # FP8 tensors: generate in float32 and convert
            if name == "hidden_states":
                tensors[name] = torch.randn(T, H, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
            elif name == "hidden_states_scale":
                tensors[name] = torch.rand(num_hidden_blocks, T, dtype=torch.float32, device=device) * 0.1 + 0.01
            elif name == "gemm1_weights":
                tensors[name] = torch.randn(E_local, 2 * I, H, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
            elif name == "gemm1_weights_scale":
                tensors[name] = torch.rand(E_local, num_gemm1_out_blocks, num_hidden_blocks, 
                                           dtype=torch.float32, device=device) * 0.1 + 0.01
            elif name == "gemm2_weights":
                tensors[name] = torch.randn(E_local, H, I, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
            elif name == "gemm2_weights_scale":
                tensors[name] = torch.rand(E_local, num_hidden_blocks, num_intermediate_blocks,
                                           dtype=torch.float32, device=device) * 0.1 + 0.01
            elif name == "routing_bias":
                tensors[name] = torch.randn(E_global, dtype=torch.float32, device=device) * 0.1
            elif name == "routing_logits":
                tensors[name] = torch.randn(T, E_global, dtype=torch.float32, device=device)
            else:
                print(f"Unknown random tensor: {name}")
                
        elif spec["type"] == "scalar":
            tensors[name] = spec["value"]
    
    return tensors


def benchmark_torch_kernel(tensors: dict, warmup: int = 3, iterations: int = 20) -> float:
    """Benchmark the torch MoE kernel and return average latency in ms."""
    
    # Extract inputs
    routing_logits = tensors["routing_logits"]
    routing_bias = tensors["routing_bias"]
    hidden_states = tensors["hidden_states"]
    hidden_states_scale = tensors["hidden_states_scale"]
    gemm1_weights = tensors["gemm1_weights"]
    gemm1_weights_scale = tensors["gemm1_weights_scale"]
    gemm2_weights = tensors["gemm2_weights"]
    gemm2_weights_scale = tensors["gemm2_weights_scale"]
    local_expert_offset = tensors["local_expert_offset"]
    routed_scaling_factor = tensors["routed_scaling_factor"]
    
    # Warmup
    for _ in range(warmup):
        _ = torch_moe_run(
            routing_logits, routing_bias,
            hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale,
            local_expert_offset, routed_scaling_factor
        )
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = torch_moe_run(
            routing_logits, routing_bias,
            hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale,
            local_expert_offset, routed_scaling_factor
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_ms = (end - start) / iterations * 1000
    return avg_ms


def main():
    trace_path = get_trace_set_path()
    
    # Load MoE workloads JSONL
    moe_jsonl = trace_path / "workloads" / "moe" / "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.jsonl"
    
    if not moe_jsonl.exists():
        print(f"MoE workloads not found at: {moe_jsonl}")
        return
    
    # Parse JSONL
    workloads = []
    with open(moe_jsonl) as f:
        for line in f:
            workloads.append(json.loads(line))
    
    print(f"Found {len(workloads)} MoE workloads")
    print("=" * 70)
    print(f"{'Workload UUID':<40} {'T':>6} {'Latency (ms)':>15}")
    print("-" * 70)
    
    results = []
    for w in workloads[:10]:  # Test first 10 workloads
        workload = w["workload"]
        uuid = workload["uuid"]
        T = workload["axes"]["seq_len"]
        
        try:
            tensors = load_workload_inputs(trace_path, workload)
            latency_ms = benchmark_torch_kernel(tensors, warmup=3, iterations=20)
            
            print(f"{uuid:<40} {T:>6} {latency_ms:>15.3f}")
            
            results.append({
                "uuid": uuid,
                "tokens": T,
                "latency_ms": latency_ms
            })
            
        except Exception as e:
            print(f"{uuid:<40} {T:>6} ERROR: {e}")
    
    print("=" * 70)
    
    if results:
        avg_latency = sum(r["latency_ms"] for r in results) / len(results)
        print(f"\nAverage latency across {len(results)} workloads: {avg_latency:.3f} ms")


if __name__ == "__main__":
    main()
