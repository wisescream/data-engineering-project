"""
MLOps Tests — Insurance Fraud Detection
=========================================
- Unit tests: training functions, preprocessing, model selection
- Integration test: full pipeline end-to-end
- Model accuracy validation test
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def fraud_dataset():
    """Generate or load the fraud dataset for testing."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'fraud_claims.csv')
    if not os.path.exists(data_path):
        from mlops.generate_fraud_dataset import generate_fraud_dataset
        generate_fraud_dataset()
    return data_path


@pytest.fixture(scope='module')
def preprocessed_data(fraud_dataset):
    """Return preprocessed X, y."""
    from mlops.train import load_and_preprocess
    X, y, scaler, le_ins, le_reg = load_and_preprocess(fraud_dataset)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

class TestDataGeneration:
    """Tests for the fraud dataset generator."""

    def test_dataset_file_exists(self, fraud_dataset):
        assert os.path.exists(fraud_dataset), "fraud_claims.csv was not created"

    def test_dataset_row_count(self, fraud_dataset):
        df = pd.read_csv(fraud_dataset)
        assert len(df) == 5000, f"Expected 5000 rows, got {len(df)}"

    def test_dataset_columns(self, fraud_dataset):
        df = pd.read_csv(fraud_dataset)
        required = ['claim_id', 'age', 'claim_amount', 'insurance_type',
                     'region', 'fraud_reported']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_fraud_rate(self, fraud_dataset):
        df = pd.read_csv(fraud_dataset)
        fraud_rate = df['fraud_reported'].mean()
        assert 0.10 <= fraud_rate <= 0.20, f"Fraud rate {fraud_rate:.2%} outside [10%, 20%]"

    def test_no_null_values(self, fraud_dataset):
        df = pd.read_csv(fraud_dataset)
        assert df.isnull().sum().sum() == 0, "Dataset contains null values"

    def test_fraud_is_binary(self, fraud_dataset):
        df = pd.read_csv(fraud_dataset)
        assert set(df['fraud_reported'].unique()) == {0, 1}, "fraud_reported must be binary"


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessing:
    """Tests for data preprocessing pipeline."""

    def test_feature_count(self, preprocessed_data):
        X, y = preprocessed_data
        assert X.shape[1] == 12, f"Expected 12 features, got {X.shape[1]}"

    def test_scaled_values(self, preprocessed_data):
        X, y = preprocessed_data
        # After StandardScaler, means should be ~0
        means = X.mean()
        for col in X.columns:
            assert abs(means[col]) < 0.1, f"Column {col} not properly scaled (mean={means[col]:.4f})"

    def test_target_balance(self, preprocessed_data):
        X, y = preprocessed_data
        fraud_rate = y.mean()
        assert 0.10 <= fraud_rate <= 0.20, f"Target imbalance: fraud_rate={fraud_rate:.2%}"

    def test_no_nans_after_preprocessing(self, preprocessed_data):
        X, y = preprocessed_data
        assert X.isnull().sum().sum() == 0, "NaN values after preprocessing"
        assert y.isnull().sum() == 0, "NaN values in target"


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

class TestModelTraining:
    """Tests for individual model training functions."""

    def test_models_definition(self):
        from mlops.train import get_models
        models = get_models()
        assert 'RandomForest' in models
        assert 'SVM' in models
        assert 'XGBoost' in models  # will be None, resolved later

    def test_random_forest_trains(self, preprocessed_data):
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        X, y = preprocessed_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        assert acc > 0.60, f"RandomForest accuracy too low: {acc:.4f}"

    def test_svm_trains(self, preprocessed_data):
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score

        X, y = preprocessed_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        assert acc > 0.60, f"SVM accuracy too low: {acc:.4f}"

    def test_confusion_matrix_generation(self, preprocessed_data):
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from mlops.train import plot_confusion_matrix

        X, y = preprocessed_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        path = plot_confusion_matrix(y_test, preds, 'test_model')
        assert os.path.exists(path), "Confusion matrix image not created"


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST — FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegrationPipeline:
    """End-to-end integration test for the full training pipeline."""

    def test_full_training_pipeline(self, fraud_dataset):
        """Run the entire training pipeline and validate outputs."""
        from mlops.train import run_training_pipeline

        results, champion = run_training_pipeline(data_path=fraud_dataset)

        # All 3 models trained
        assert len(results) == 3, f"Expected 3 model results, got {len(results)}"
        assert 'RandomForest' in results
        assert 'XGBoost' in results
        assert 'SVM' in results

        # Each model has required metrics
        for name, metrics in results.items():
            assert 'accuracy' in metrics, f"{name} missing accuracy"
            assert 'f1_score' in metrics, f"{name} missing f1_score"
            assert 'roc_auc' in metrics, f"{name} missing roc_auc"
            assert metrics['accuracy'] > 0, f"{name} accuracy is 0"

        # Champion selected
        assert champion is not None, "No champion model selected"
        assert champion in results, f"Champion '{champion}' not in results"

    def test_artifacts_generated(self):
        """Verify training artifacts are saved."""
        assert os.path.exists('artifacts/training_results.json'), "training_results.json missing"

        with open('artifacts/training_results.json') as f:
            data = json.load(f)
        assert len(data) == 3, "training_results.json should have 3 models"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ACCURACY VALIDATION TEST
# ══════════════════════════════════════════════════════════════════════════════

class TestModelAccuracyValidation:
    """Validate that the champion model meets production thresholds."""

    def test_champion_accuracy_threshold(self):
        """Champion model must exceed 0.75 accuracy."""
        results_path = 'artifacts/training_results.json'
        if not os.path.exists(results_path):
            pytest.skip("Run full pipeline first")

        with open(results_path) as f:
            results = json.load(f)

        best_acc = max(v['accuracy'] for v in results.values())
        assert best_acc >= 0.75, f"Best accuracy {best_acc:.4f} < 0.75 threshold"

    def test_champion_f1_score(self):
        """Champion model must have reasonable F1 score."""
        results_path = 'artifacts/training_results.json'
        if not os.path.exists(results_path):
            pytest.skip("Run full pipeline first")

        with open(results_path) as f:
            results = json.load(f)

        best_f1 = max(v['f1_score'] for v in results.values())
        assert best_f1 >= 0.30, f"Best F1 {best_f1:.4f} too low for fraud detection"

    def test_champion_roc_auc(self):
        """Champion AUC must be above random (0.5)."""
        results_path = 'artifacts/training_results.json'
        if not os.path.exists(results_path):
            pytest.skip("Run full pipeline first")

        with open(results_path) as f:
            results = json.load(f)

        best_auc = max(v['roc_auc'] for v in results.values())
        assert best_auc >= 0.60, f"Best ROC-AUC {best_auc:.4f} too close to random"
