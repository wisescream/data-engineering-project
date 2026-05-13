"""
Insurance Fraud Model — Deployment & Canary Release
=====================================================
Production deployment script:
  - Loads champion model from MLflow Registry
  - Simulates API prediction endpoint
  - Implements canary deployment: 5% → 25% → 50% → 100%
  - Rollback if accuracy degrades
"""

import os
import json
import time
import random
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.metrics import accuracy_score, f1_score


# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "insurance-fraud-detector"
CANARY_STAGES = [0.05, 0.25, 0.50, 1.00]  # traffic percentages
ROLLBACK_THRESHOLD = 0.70  # rollback if accuracy drops below
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'fraud_claims.csv')


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_production_model(tracking_uri: str = None):
    """Load the production model from MLflow Registry."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    # Try alias-based loading first (MLflow >= 2.9), then stage-based
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
        version_info = "alias=champion"
    except Exception:
        try:
            model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
            version_info = "stage=Production"
        except Exception:
            # Fall back to latest version
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            if not versions:
                raise RuntimeError(f"No versions found for model '{MODEL_NAME}'")
            latest = sorted(versions, key=lambda v: int(v.version))[-1]
            model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{latest.version}")
            version_info = f"version={latest.version}"

    print(f"[OK] Loaded model '{MODEL_NAME}' ({version_info})")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION API (Simulation)
# ══════════════════════════════════════════════════════════════════════════════

def predict_single(model, features: dict) -> dict:
    """Simulate a single prediction API call."""
    from mlops.train import FEATURE_COLS

    input_df = pd.DataFrame([features])
    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[FEATURE_COLS]
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    return {
        'prediction': int(prediction),
        'label': 'FRAUD' if prediction == 1 else 'LEGIT',
        'confidence': float(max(probability)),
        'fraud_probability': float(probability[1]),
    }


def predict_batch(model, X: pd.DataFrame) -> np.ndarray:
    """Batch prediction for evaluation."""
    return model.predict(X)


# ══════════════════════════════════════════════════════════════════════════════
# CANARY DEPLOYMENT
# ══════════════════════════════════════════════════════════════════════════════

def run_canary_deployment(model, X_test, y_test):
    """
    Simulate progressive canary deployment:
      5% → 25% → 50% → 100%
    Rolls back if accuracy drops below threshold.
    """
    print("\n" + "=" * 60)
    print("  CANARY DEPLOYMENT — Progressive Rollout")
    print("=" * 60)

    n = len(X_test)
    baseline_preds = model.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_preds)
    print(f"\n  Baseline accuracy (full test set): {baseline_acc:.4f}")

    deployment_log = []

    for stage_pct in CANARY_STAGES:
        stage_name = f"{int(stage_pct * 100)}%"
        sample_size = max(1, int(n * stage_pct))

        # Sample traffic
        indices = random.sample(range(n), sample_size)
        X_sample = X_test.iloc[indices]
        y_sample = y_test.iloc[indices]

        # Predict on canary traffic
        preds = model.predict(X_sample)
        acc = accuracy_score(y_sample, preds)
        f1 = f1_score(y_sample, preds, zero_division=0)

        stage_result = {
            'stage': stage_name,
            'traffic_pct': stage_pct,
            'sample_size': sample_size,
            'accuracy': round(acc, 4),
            'f1_score': round(f1, 4),
            'status': 'OK',
        }

        # Rollback check
        if acc < ROLLBACK_THRESHOLD:
            stage_result['status'] = 'ROLLBACK'
            deployment_log.append(stage_result)
            print(f"\n  [{stage_name}] acc={acc:.4f} f1={f1:.4f} — ROLLBACK TRIGGERED!")
            print(f"  Accuracy {acc:.4f} < threshold {ROLLBACK_THRESHOLD}")
            print("  Reverting to previous stable version.")
            break

        deployment_log.append(stage_result)
        print(f"  [{stage_name}] traffic={sample_size:>5} samples | acc={acc:.4f} | f1={f1:.4f} | OK")

        # Simulate delay between stages
        time.sleep(0.5)

    # Final status
    final_status = deployment_log[-1]['status']
    if final_status == 'OK' and deployment_log[-1]['traffic_pct'] == 1.0:
        print("\n  DEPLOYMENT COMPLETE — Model serving 100% traffic")
    elif final_status == 'ROLLBACK':
        print("\n  DEPLOYMENT ROLLED BACK — Model NOT promoted")

    # Save deployment log
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/canary_deployment_log.json', 'w') as f:
        json.dump(deployment_log, f, indent=2)

    return deployment_log


# ══════════════════════════════════════════════════════════════════════════════
# VERSION CONTROL
# ══════════════════════════════════════════════════════════════════════════════

def list_model_versions(tracking_uri: str = None):
    """List all registered versions of the model."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        print(f"\n  Model: {MODEL_NAME}")
        print(f"  {'Version':<10} {'Stage':<15} {'Status':<10} {'Created'}")
        print("  " + "-" * 60)
        for v in sorted(versions, key=lambda x: int(x.version)):
            stage = v.current_stage if hasattr(v, 'current_stage') else 'N/A'
            print(f"  v{v.version:<9} {stage:<15} {v.status:<10} {v.creation_timestamp}")
    except Exception as e:
        print(f"  No model versions found: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Full deployment simulation: load model, predict, canary deploy."""
    from mlops.train import load_and_preprocess
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("  INSURANCE FRAUD MODEL — DEPLOYMENT")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading production model...")
    model = load_production_model()

    # Load test data
    print("\n[2/4] Loading test data...")
    X, y, *_ = load_and_preprocess(DATA_PATH)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Test set: {len(X_test)} samples")

    # Single prediction demo
    print("\n[3/4] API Prediction Demo...")
    sample = X_test.iloc[0].to_dict()
    result = predict_single(model, sample)
    print(f"  Input:  {json.dumps({k: round(v, 2) for k, v in list(sample.items())[:5]}, indent=2)}")
    print(f"  Output: {json.dumps(result, indent=2)}")

    # Canary deployment
    print("\n[4/4] Running canary deployment...")
    run_canary_deployment(model, X_test, y_test)

    # Version info
    list_model_versions()


if __name__ == '__main__':
    main()
