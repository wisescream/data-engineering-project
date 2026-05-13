"""
Insurance Fraud Dataset Generator
==================================
Generates a synthetic dataset for fraud detection training.
"""

import os
import pandas as pd
import numpy as np

# Configuration
NUM_ROWS = 5000
FRAUD_RATE = 0.15
RANDOM_STATE = 42

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_PATH = os.path.join(DATA_DIR, 'fraud_claims.csv')


def generate_fraud_dataset():
    """Create a synthetic insurance fraud dataset with correlated features."""
    np.random.seed(RANDOM_STATE)

    # Base features
    data = {
        'claim_id': range(1000, 1000 + NUM_ROWS),
        'age': np.random.normal(40, 12, NUM_ROWS).astype(int),
        'months_as_customer': np.random.randint(1, 480, NUM_ROWS),
        'policy_annual_premium': np.random.normal(1200, 300, NUM_ROWS).round(2),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], NUM_ROWS),
        'insurance_type': np.random.choice(['Auto', 'Home', 'Life', 'Health'], NUM_ROWS),
        'num_previous_claims': np.random.poisson(0.5, NUM_ROWS),
        'police_report_filed': np.random.choice([0, 1], NUM_ROWS, p=[0.7, 0.3]),
        'witnesses': np.random.randint(0, 4, NUM_ROWS),
        'injury_claim': np.random.normal(5000, 10000, NUM_ROWS).clip(0).round(2),
        'vehicle_claim': np.random.normal(15000, 20000, NUM_ROWS).clip(0).round(2),
    }

    df = pd.DataFrame(data)
    df['total_claim_amount'] = df['injury_claim'] + df['vehicle_claim']
    df['claim_amount'] = df['total_claim_amount']  # alias

    # Generate target (fraud_reported) with some simple logic
    # Higher fraud risk if: many previous claims, no police report, high claim amount
    risk_score = (
        (df['num_previous_claims'] * 0.2)
        + ((1 - df['police_report_filed']) * 0.3)
        + (df['total_claim_amount'] / 100000 * 0.5)
        + (np.random.random(NUM_ROWS) * 0.2)
    )


    # Threshold for target fraud rate
    threshold = np.percentile(risk_score, (1 - FRAUD_RATE) * 100)
    df['fraud_reported'] = (risk_score >= threshold).astype(int)

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    fraud_count = df['fraud_reported'].sum()
    print(f"[OK] Fraud dataset generated: {os.path.abspath(OUTPUT_PATH)}")
    print(f"     {NUM_ROWS} rows | {fraud_count} frauds ({fraud_count / NUM_ROWS * 100:.1f}%)")


if __name__ == '__main__':
    generate_fraud_dataset()
