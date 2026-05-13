"""
Generate a synthetic insurance fraud detection dataset.
Builds on the existing claims data with fraud-specific features.

Usage:
    python mlops/generate_fraud_dataset.py
"""

import os
import random
import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_ROWS = 5000
FRAUD_RATE = 0.15  # 15% fraud — realistic for insurance
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'fraud_claims.csv')

INSURANCE_TYPES = ['Auto', 'Health', 'Home', 'Life', 'Travel']
REGIONS = ['Casablanca', 'Rabat', 'Tangier', 'Marrakech', 'Fes', 'Agadir']

random.seed(42)
np.random.seed(42)


def generate_fraud_dataset():
    """Generate synthetic fraud detection dataset with realistic patterns."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = {
        'claim_id': range(1, NUM_ROWS + 1),
        'customer_id': np.random.randint(1000, 9999, NUM_ROWS),
        'policy_id': np.random.randint(100000, 999999, NUM_ROWS),
        'age': np.random.randint(18, 75, NUM_ROWS),
        'insurance_type': np.random.choice(INSURANCE_TYPES, NUM_ROWS),
        'region': np.random.choice(REGIONS, NUM_ROWS),
        'claim_amount': np.round(np.random.lognormal(mean=8, sigma=1.2, size=NUM_ROWS), 2),
        'policy_annual_premium': np.round(np.random.uniform(500, 15000, NUM_ROWS), 2),
        'months_as_customer': np.random.randint(1, 240, NUM_ROWS),
        'num_previous_claims': np.random.poisson(lam=1.5, size=NUM_ROWS),
        'police_report_filed': np.random.choice([0, 1], NUM_ROWS, p=[0.4, 0.6]),
        'witnesses': np.random.choice([0, 1, 2, 3], NUM_ROWS, p=[0.3, 0.4, 0.2, 0.1]),
        'injury_claim': np.random.choice([0, 1], NUM_ROWS, p=[0.6, 0.4]),
        'vehicle_claim': np.random.choice([0, 1], NUM_ROWS, p=[0.5, 0.5]),
        'total_claim_amount': np.round(np.random.lognormal(mean=8.5, sigma=1, size=NUM_ROWS), 2),
    }

    df = pd.DataFrame(data)

    # ── Generate fraud labels with realistic correlations ─────────────────
    fraud_prob = np.zeros(NUM_ROWS)
    fraud_prob += (df['claim_amount'] > df['claim_amount'].quantile(0.85)).astype(float) * 0.15
    fraud_prob += (df['num_previous_claims'] >= 3).astype(float) * 0.12
    fraud_prob += (df['months_as_customer'] < 12).astype(float) * 0.10
    fraud_prob += (df['police_report_filed'] == 0).astype(float) * 0.08
    fraud_prob += (df['witnesses'] == 0).astype(float) * 0.10
    fraud_prob += (df['age'] < 25).astype(float) * 0.05
    fraud_prob = np.clip(fraud_prob + np.random.uniform(-0.05, 0.05, NUM_ROWS), 0, 1)

    # Calibrate to target fraud rate
    threshold = np.quantile(fraud_prob, 1 - FRAUD_RATE)
    df['fraud_reported'] = (fraud_prob >= threshold).astype(int)

    # Ensure exact fraud rate
    actual_fraud = df['fraud_reported'].sum()
    target_fraud = int(NUM_ROWS * FRAUD_RATE)
    if actual_fraud > target_fraud:
        fraud_indices = df[df['fraud_reported'] == 1].index
        drop_n = actual_fraud - target_fraud
        drop_idx = np.random.choice(fraud_indices, drop_n, replace=False)
        df.loc[drop_idx, 'fraud_reported'] = 0
    elif actual_fraud < target_fraud:
        non_fraud_indices = df[df['fraud_reported'] == 0].index
        add_n = target_fraud - actual_fraud
        add_idx = np.random.choice(non_fraud_indices, add_n, replace=False)
        df.loc[add_idx, 'fraud_reported'] = 1

    df.to_csv(OUTPUT_FILE, index=False)
    fraud_count = df['fraud_reported'].sum()
    print(f"[OK] Fraud dataset generated: {OUTPUT_FILE}")
    print(f"     {NUM_ROWS} rows | {fraud_count} frauds ({fraud_count/NUM_ROWS*100:.1f}%)")


if __name__ == '__main__':
    generate_fraud_dataset()
