"""
Performance Benchmark - FP32 vs INT8
======================================
Measures latency, throughput, memory, and accuracy for both model variants.
Can be run standalone or called from the API /benchmark endpoint.
"""

import os
import sys
import time
import json
import tracemalloc
import numpy as np
import pandas as pd
import joblib

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def benchmark_model(model, X_test, y_test, n_requests=1000, model_type="fp32"):
    """
    Benchmark a single model on latency, throughput, memory, accuracy.

    Returns:
        dict with benchmark metrics
    """
    from sklearn.metrics import accuracy_score

    # - Accuracy -
    preds_full = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds_full)

    # - Latency (single-sample inference) -
    latencies = []
    sample_indices = np.random.choice(len(X_test), n_requests, replace=True)

    for i in sample_indices:
        row = X_test.iloc[i:i + 1]
        start = time.perf_counter()
        model.predict(row)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    latencies = np.array(latencies)

    # - Throughput (batch inference) -
    start = time.perf_counter()
    for i in sample_indices:
        model.predict(X_test.iloc[i:i + 1])
    total_time = time.perf_counter() - start
    throughput = n_requests / total_time

    # - Memory -
    tracemalloc.start()
    model.predict(X_test)
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "model_type": model_type,
        "total_requests": n_requests,
        "avg_latency_ms": round(float(np.mean(latencies)), 3),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 3),
        "throughput_rps": round(throughput, 1),
        "memory_mb": round(peak_memory / (1024 * 1024), 2),
        "accuracy": round(accuracy, 4),
    }


def run_comparison_benchmark(model_fp32=None, model_int8=None, n_requests=1000):
    """
    Run a full FP32 vs INT8 benchmark comparison.

    Can receive models directly (from API) or load them from artifacts.
    Returns a dict suitable for the BenchmarkResponse schema.
    """
    from mlops.train import load_and_preprocess, DATA_PATH
    from sklearn.model_selection import train_test_split

    # Load test data
    X, y, *_ = load_and_preprocess(DATA_PATH)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    # - FP32 -
    if model_fp32 is None:
        fp32_path = os.path.join(PROJECT_ROOT, "artifacts", "model_fp32.joblib")
        if os.path.exists(fp32_path):
            model_fp32 = joblib.load(fp32_path)

    if model_fp32 is not None:
        print("  Benchmarking FP32...")
        results["fp32"] = benchmark_model(model_fp32, X_test, y_test, n_requests, "fp32")

    # - INT8 -
    if model_int8 is None:
        int8_path = os.path.join(PROJECT_ROOT, "artifacts", "model_int8.joblib")
        if os.path.exists(int8_path):
            model_int8 = joblib.load(int8_path)

    if model_int8 is not None:
        print("  Benchmarking INT8...")
        results["int8"] = benchmark_model(model_int8, X_test, y_test, n_requests, "int8")

    # - Comparison -
    if "fp32" in results and "int8" in results:
        results["speedup"] = round(
            results["fp32"]["avg_latency_ms"] / max(results["int8"]["avg_latency_ms"], 0.001), 2
        )
        fp32_size = _get_size(model_fp32)
        int8_size = _get_size(model_int8)
        results["size_reduction"] = round(fp32_size / max(int8_size, 1), 2)
        results["accuracy_delta"] = round(
            results["int8"]["accuracy"] - results["fp32"]["accuracy"], 4
        )

    return results


def _get_size(model) -> int:
    """Get serialized model size."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        joblib.dump(model, f.name)
        size = os.path.getsize(f.name)
    os.unlink(f.name)
    return size


def main():
    """Run standalone benchmark from CLI."""
    print("=" * 60)
    print("  PERFORMANCE BENCHMARK - FP32 -> INT8")
    print("=" * 60)

    results = run_comparison_benchmark(n_requests=1000)

    if not results:
        print("\n  [FAIL] No models found. Run quantization first:")
        print("         python -m api.quantize")
        return

    # - Display Results -
    print("\n" + "-" * 70)
    header = f"  {'Metric':<25}"
    if "fp32" in results:
        header += f"{'FP32':<15}"
    if "int8" in results:
        header += f"{'INT8':<15}"
    print(header)
    print("  " + "-" * 65)

    metrics_to_show = [
        ("Requests", "total_requests"),
        ("Avg Latency (ms)", "avg_latency_ms"),
        ("P50 Latency (ms)", "p50_latency_ms"),
        ("P95 Latency (ms)", "p95_latency_ms"),
        ("P99 Latency (ms)", "p99_latency_ms"),
        ("Throughput (req/s)", "throughput_rps"),
        ("Memory (MB)", "memory_mb"),
        ("Accuracy", "accuracy"),
    ]

    for label, key in metrics_to_show:
        row = f"  {label:<25}"
        if "fp32" in results:
            row += f"{results['fp32'][key]:<15}"
        if "int8" in results:
            row += f"{results['int8'][key]:<15}"
        print(row)

    print("  " + "-" * 65)

    if "speedup" in results:
        print(f"\n  Speedup:          {results['speedup']}x")
        print(f"  Size reduction:   {results['size_reduction']}x")
        print(f"  Accuracy delta:   {results['accuracy_delta']}")

    # Save results
    os.makedirs(os.path.join(PROJECT_ROOT, "artifacts"), exist_ok=True)
    report_path = os.path.join(PROJECT_ROOT, "artifacts", "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Report saved to {report_path}")

    print("\n" + "=" * 60)
    print("  BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
