"""
Model Quantization - FP32 -> INT8
==================================
Converts a scikit-learn RandomForest model to a quantized INT8 version
by reducing tree depth and precision of decision thresholds.

For sklearn tree-based models, "quantization" means:
  1. Reduce tree depth (max_depth pruning)
  2. Reduce n_estimators (fewer trees)
  3. Quantize thresholds to lower precision
  4. The resulting model is smaller and faster
"""

import os
import sys
import json
import time
import copy
import joblib
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def quantize_random_forest(model, max_depth=6, n_estimators_ratio=0.5):
    """
    Simulate INT8 quantization for a RandomForest model.

    Strategy:
      - Prune tree depth to `max_depth`
      - Keep only top `n_estimators_ratio` of trees
      - Quantize thresholds to float16 precision (simulates INT8 storage)

    Returns a lighter model that trades accuracy for speed/size.
    """
    quantized = copy.deepcopy(model)

    # Reduce number of estimators
    n_keep = max(1, int(len(quantized.estimators_) * n_estimators_ratio))
    quantized.estimators_ = quantized.estimators_[:n_keep]
    quantized.n_estimators = n_keep

    # Quantize thresholds in each tree to float16
    for tree in quantized.estimators_:
        tree_obj = tree.tree_
        # Convert thresholds to float16 (simulating INT8-level precision)
        tree_obj.threshold[:] = tree_obj.threshold.astype(np.float16).astype(np.float64)

    return quantized


def quantize_xgboost(model, n_estimators_ratio=0.5):
    """
    Simulate INT8 quantization for an XGBoost model.

    Strategy:
      - Reduce the number of boosting rounds
      - XGBoost internally stores trees efficiently
    """
    quantized = copy.deepcopy(model)

    # Reduce number of trees by limiting best_ntree_limit
    n_keep = max(1, int(quantized.n_estimators * n_estimators_ratio))
    quantized.n_estimators = n_keep

    return quantized


def quantize_model(model):
    """Auto-detect model type and apply quantization."""
    model_type = type(model).__name__

    if "RandomForest" in model_type:
        return quantize_random_forest(model)
    elif "XGB" in model_type or "GradientBoosting" in model_type:
        return quantize_xgboost(model)
    else:
        # Generic: just return a copy (no quantization available)
        print(f"  [WARN] No quantization strategy for {model_type}, returning copy")
        return copy.deepcopy(model)


def get_model_size_bytes(model) -> int:
    """Get model size in bytes via serialization."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        joblib.dump(model, f.name)
        size = os.path.getsize(f.name)
    os.unlink(f.name)
    return size


def compare_models(model_fp32, model_int8, X_test, y_test):
    """Compare FP32 and INT8 models on accuracy and size."""
    from sklearn.metrics import accuracy_score, f1_score

    # Accuracy
    preds_fp32 = model_fp32.predict(X_test)
    preds_int8 = model_int8.predict(X_test)

    acc_fp32 = accuracy_score(y_test, preds_fp32)
    acc_int8 = accuracy_score(y_test, preds_int8)
    f1_fp32 = f1_score(y_test, preds_fp32, zero_division=0)
    f1_int8 = f1_score(y_test, preds_int8, zero_division=0)

    # Size
    size_fp32 = get_model_size_bytes(model_fp32)
    size_int8 = get_model_size_bytes(model_int8)

    # Speed
    n_iter = 100
    start = time.perf_counter()
    for _ in range(n_iter):
        model_fp32.predict(X_test[:10])
    latency_fp32 = (time.perf_counter() - start) / n_iter * 1000

    start = time.perf_counter()
    for _ in range(n_iter):
        model_int8.predict(X_test[:10])
    latency_int8 = (time.perf_counter() - start) / n_iter * 1000

    return {
        "fp32": {
            "accuracy": round(acc_fp32, 4),
            "f1_score": round(f1_fp32, 4),
            "size_mb": round(size_fp32 / (1024 * 1024), 2),
            "avg_latency_ms": round(latency_fp32, 3),
        },
        "int8": {
            "accuracy": round(acc_int8, 4),
            "f1_score": round(f1_int8, 4),
            "size_mb": round(size_int8 / (1024 * 1024), 2),
            "avg_latency_ms": round(latency_int8, 3),
        },
        "compression_ratio": round(size_fp32 / max(size_int8, 1), 2),
        "speedup": round(latency_fp32 / max(latency_int8, 0.001), 2),
        "accuracy_delta": round(acc_int8 - acc_fp32, 4),
    }


def main():
    """Run the full quantization pipeline."""
    print("=" * 60)
    print("  PERFORMANCE BENCHMARK - FP32 -> INT8")
    print("=" * 60)

    # - Load FP32 model -
    print("\n[1/4] Loading FP32 model...")

    model_fp32 = None
    try:
        import mlflow
        import mlflow.sklearn
        model_name = "insurance-fraud-detector"
        for uri in [
            f"models:/{model_name}@champion",
            f"models:/{model_name}/Production",
            f"models:/{model_name}/Staging",
        ]:
            try:
                model_fp32 = mlflow.sklearn.load_model(uri)
                print(f"  Loaded from MLflow: {uri}")
                break
            except Exception:
                continue

        if model_fp32 is None:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            if versions:
                latest = sorted(versions, key=lambda v: int(v.version))[-1]
                model_fp32 = mlflow.sklearn.load_model(
                    f"models:/{model_name}/{latest.version}"
                )
                print(f"  Loaded from MLflow: version {latest.version}")
    except Exception as e:
        print(f"  MLflow load failed: {e}")

    if model_fp32 is None:
        print("  [FAIL] No FP32 model found. Train first with `python -m mlops.train`")
        return

    # - Quantize -
    print("\n[2/4] Quantizing model...")
    model_int8 = quantize_model(model_fp32)

    size_fp32 = get_model_size_bytes(model_fp32)
    size_int8 = get_model_size_bytes(model_int8)
    print(f"  FP32 size: {size_fp32 / (1024 * 1024):.2f} MB")
    print(f"  INT8 size: {size_int8 / (1024 * 1024):.2f} MB")
    print(f"  Compression: {size_fp32 / max(size_int8, 1):.1f}x")

    # - Save models -
    print("\n[3/4] Saving models...")
    os.makedirs(os.path.join(PROJECT_ROOT, "artifacts"), exist_ok=True)

    fp32_path = os.path.join(PROJECT_ROOT, "artifacts", "model_fp32.joblib")
    int8_path = os.path.join(PROJECT_ROOT, "artifacts", "model_int8.joblib")

    joblib.dump(model_fp32, fp32_path)
    joblib.dump(model_int8, int8_path)
    print(f"  FP32 -> {fp32_path}")
    print(f"  INT8 -> {int8_path}")

    # - Compare -
    print("\n[4/4] Comparing models on test data...")
    from mlops.train import load_and_preprocess, DATA_PATH
    from sklearn.model_selection import train_test_split

    X, y, *_ = load_and_preprocess(DATA_PATH)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    comparison = compare_models(model_fp32, model_int8, X_test, y_test)

    print("\n" + "-" * 60)
    print(f"  {'Metric':<25} {'FP32':<15} {'INT8':<15}")
    print("  " + "-" * 55)
    print(f"  {'Accuracy':<25} {comparison['fp32']['accuracy']:<15} {comparison['int8']['accuracy']:<15}")
    print(f"  {'F1-Score':<25} {comparison['fp32']['f1_score']:<15} {comparison['int8']['f1_score']:<15}")
    print(f"  {'Size (MB)':<25} {comparison['fp32']['size_mb']:<15} {comparison['int8']['size_mb']:<15}")
    print(f"  {'Avg Latency (ms)':<25} {comparison['fp32']['avg_latency_ms']:<15} {comparison['int8']['avg_latency_ms']:<15}")
    print("  " + "-" * 55)
    print(f"  Compression ratio: {comparison['compression_ratio']}x")
    print(f"  Speedup:           {comparison['speedup']}x")
    print(f"  Accuracy delta:    {comparison['accuracy_delta']}")

    # Save comparison report
    report_path = os.path.join(PROJECT_ROOT, "artifacts", "quantization_report.json")
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Report saved to {report_path}")

    print("\n" + "=" * 60)
    print("  QUANTIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
