"""
FlashInfer-Bench Quick Benchmark Runner.

Fast iteration testing with reduced workloads and iterations.
Runs in ~30 seconds instead of ~10 minutes.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from scripts.pack_solution import pack_solution


def get_trace_set_path() -> str:
    """Get trace set path from environment variable."""
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your flashinfer-trace dataset."
        )
    return path


def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark locally and return results."""
    # QUICK CONFIG: 1 warmup, 10 iterations, 1 trial
    if config is None:
        config = BenchmarkConfig(warmup_runs=1, iterations=10, num_trials=1)

    trace_set_path = get_trace_set_path()
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    all_workloads = trace_set.workloads.get(solution.definition, [])

    if not all_workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    # QUICK: Only use first 3 workloads (small, medium, large batch sizes)
    # Sort by seq_len to get representative samples
    def get_seq_len(w):
        if hasattr(w, 'axes'):
            return w.axes.get("seq_len", 0)
        elif hasattr(w, 'workload') and hasattr(w.workload, 'axes'):
            return w.workload.axes.get("seq_len", 0)
        return 0
    
    sorted_workloads = sorted(all_workloads, key=get_seq_len)
    
    # Pick evenly spaced workloads across the size range
    # Change NUM_WORKLOADS to test more/fewer workloads
    NUM_WORKLOADS = int(os.environ.get("NUM_WORKLOADS", "3"))
    
    n = len(sorted_workloads)
    if NUM_WORKLOADS >= n:
        workloads = sorted_workloads
    else:
        # Evenly space across the range
        step = (n - 1) / (NUM_WORKLOADS - 1) if NUM_WORKLOADS > 1 else 0
        selected_indices = [int(i * step) for i in range(NUM_WORKLOADS)]
        workloads = [sorted_workloads[i] for i in selected_indices]
    
    print(f"Quick mode: Testing {len(workloads)} of {len(all_workloads)} workloads")
    for w in workloads:
        print(f"  - seq_len={get_seq_len(w)}")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=False)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{'='*70}")
        print(f"{def_name}:")
        print(f"{'='*70}")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                speedup = result['speedup_factor']
                emoji = "🚀" if speedup > 1.5 else "✓" if speedup >= 1.0 else "⚠️"
                print(f" | {speedup:.2f}x {emoji}", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | err={abs_err:.2e}", end="")

            print()


def main():
    """Pack solution and run quick benchmark."""
    num_workloads = int(os.environ.get("NUM_WORKLOADS", "3"))
    print("=" * 70)
    print("QUICK BENCHMARK MODE")
    print(f"  - {num_workloads} workloads (set NUM_WORKLOADS env var to change)")
    print("  - 1 warmup, 10 iterations, 1 trial")
    print(f"  - Expected time: ~{num_workloads * 10} seconds")
    print("=" * 70)
    
    print("\nPacking solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning quick benchmark...")
    results = run_benchmark(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)


if __name__ == "__main__":
    main()
