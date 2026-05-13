"""
Insurance Fraud Detection — Training Pipeline
===============================================
MLOps-grade training script:
  - Loads & preprocesses insurance fraud data
  - Trains 3 models: RandomForest, XGBoost, SVM
  - Tracks everything in MLflow
  - Registers the champion model in the Model Registry
"""

import os
import json
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "insurance-fraud"
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'fraud_claims.csv')
ACCURACY_THRESHOLD = 0.75
RANDOM_STATE = 42

FEATURE_COLS = [
    'age', 'claim_amount', 'policy_annual_premium',
    'months_as_customer', 'num_previous_claims',
    'police_report_filed', 'witnesses',
    'injury_claim', 'vehicle_claim', 'total_claim_amount',
    'insurance_type_encoded', 'region_encoded',
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(data_path: str = DATA_PATH):
    """Load CSV and engineer features for fraud detection."""
    df = pd.read_csv(data_path)

    # Encode categoricals
    le_insurance = LabelEncoder()
    le_region = LabelEncoder()
    df['insurance_type_encoded'] = le_insurance.fit_transform(df['insurance_type'])
    df['region_encoded'] = le_region.fit_transform(df['region'])

    X = df[FEATURE_COLS].copy()
    y = df['fraud_reported'].copy()

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )

    return X_scaled, y, scaler, le_insurance, le_region


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_models():
    """Return dict of model_name → (model_instance, hyperparams_dict)."""
    return {
        'RandomForest': (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 5, 'class_weight': 'balanced'},
        ),
        'XGBoost': None,  # placeholder — resolved below
        'SVM': (
            SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=RANDOM_STATE,
            ),
            {'kernel': 'rbf', 'C': 10, 'gamma': 'scale', 'class_weight': 'balanced'},
        ),
    }


def _resolve_xgboost():
    """Import XGBoost and return (model, params). Falls back to GradientBoosting."""
    try:
        from xgboost import XGBClassifier
        fraud_ratio = 0.15
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=(1 - fraud_ratio) / fraud_ratio,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'engine': 'xgboost'}
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        )
        params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'engine': 'sklearn-gb'}
    return model, params


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING & TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, model_name: str) -> str:
    """Save confusion matrix plot and return the file path."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legit', 'Fraud'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()

    os.makedirs('artifacts', exist_ok=True)
    path = f'artifacts/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def train_single_model(model, params: dict, model_name: str,
                       X_train, X_test, y_train, y_test):
    """Train one model, log everything to MLflow, return metrics dict."""
    with mlflow.start_run(run_name=model_name, nested=True):
        # Log hyperparameters
        mlflow.log_param('model_type', model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param('train_size', len(X_train))
        mlflow.log_param('test_size', len(X_test))

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob),
        }

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, round(v, 4))

        # Confusion matrix artifact
        cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(cm_path, artifact_path='plots')

        # Classification report artifact
        report = classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])
        report_path = f'artifacts/report_{model_name.lower().replace(" ", "_")}.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        mlflow.log_artifact(report_path, artifact_path='reports')

        # Log model
        mlflow.sklearn.log_model(model, artifact_path='model')

        print(f"  [{model_name}]  acc={metrics['accuracy']:.4f}  f1={metrics['f1_score']:.4f}  auc={metrics['roc_auc']:.4f}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def register_champion(champion_name: str, champion_run_id: str):
    """Register the best model and promote it through stages."""
    client = MlflowClient()
    model_uri = f"runs:/{champion_run_id}/model"
    registered_name = "insurance-fraud-detector"

    # Register the model version
    mv = mlflow.register_model(model_uri, registered_name)
    version = mv.version
    print(f"\n  Registered model '{registered_name}' version {version}")

    # Transition: None → Staging → Production
    try:
        client.transition_model_version_stage(
            name=registered_name,
            version=version,
            stage="Staging",
        )
        print(f"  Transitioned v{version} → Staging")

        client.transition_model_version_stage(
            name=registered_name,
            version=version,
            stage="Production",
        )
        print(f"  Transitioned v{version} → Production")
    except Exception as e:
        # MLflow >= 2.9 uses aliases instead of stages
        try:
            client.set_registered_model_alias(registered_name, "champion", version)
            print(f"  Set alias 'champion' → v{version}")
        except Exception:
            print(f"  [WARN] Could not set stage/alias: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_training_pipeline(data_path: str = DATA_PATH, tracking_uri: str = None):
    """End-to-end training pipeline: preprocess → train 3 models → register best."""
    # ── MLflow setup ─────────────────────────────────────────────────────
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("=" * 60)
    print("  INSURANCE FRAUD DETECTION — TRAINING PIPELINE")
    print("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────
    print("\n[1/4] Loading & preprocessing data...")
    X, y, scaler, le_ins, le_reg = load_and_preprocess(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)} | Fraud rate: {y.mean():.2%}")

    # ── Train models ─────────────────────────────────────────────────────
    print("\n[2/4] Training models...")
    models = get_models()
    models['XGBoost'] = _resolve_xgboost()

    results = {}
    run_ids = {}

    with mlflow.start_run(run_name="training-pipeline") as parent_run:
        mlflow.log_param('dataset_rows', len(X))
        mlflow.log_param('fraud_rate', round(y.mean(), 4))
        mlflow.log_param('num_features', len(FEATURE_COLS))

        for name, (model, params) in models.items():
            metrics = train_single_model(
                model, params, name, X_train, X_test, y_train, y_test,
            )
            results[name] = metrics

            # Grab the child run ID
            runs = mlflow.search_runs(
                filter_string=f"tags.mlflow.runName = '{name}'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if not runs.empty:
                run_ids[name] = runs.iloc[0]['run_id']

        # ── Champion selection ───────────────────────────────────────────
        print("\n[3/4] Selecting champion model...")
        champion_name = max(results, key=lambda k: results[k]['f1_score'])
        champion_metrics = results[champion_name]
        champion_run_id = run_ids[champion_name]

        mlflow.log_param('champion_model', champion_name)
        mlflow.log_metric('champion_f1', champion_metrics['f1_score'])
        mlflow.log_metric('champion_accuracy', champion_metrics['accuracy'])

        print(f"\n  CHAMPION: {champion_name}")
        print(f"  Accuracy:  {champion_metrics['accuracy']:.4f}")
        print(f"  F1-Score:  {champion_metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {champion_metrics['roc_auc']:.4f}")

        # Accuracy gate
        if champion_metrics['accuracy'] < ACCURACY_THRESHOLD:
            print(f"\n  [FAIL] Accuracy {champion_metrics['accuracy']:.4f} < threshold {ACCURACY_THRESHOLD}")
            print("  Model NOT registered.")
            return results, None

        # ── Register ─────────────────────────────────────────────────────
        print("\n[4/4] Registering champion in Model Registry...")
        register_champion(champion_name, champion_run_id)

    # Save results summary
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/training_results.json', 'w') as f:
        json.dump({k: {mk: round(mv, 4) for mk, mv in v.items()} for k, v in results.items()}, f, indent=2)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)

    return results, champion_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train insurance fraud models')
    parser.add_argument('--data', default=DATA_PATH, help='Path to fraud_claims.csv')
    parser.add_argument('--tracking-uri', default=None, help='MLflow tracking URI')
    args = parser.parse_args()

    run_training_pipeline(data_path=args.data, tracking_uri=args.tracking_uri)
