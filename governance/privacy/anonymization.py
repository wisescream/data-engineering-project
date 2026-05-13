import pandas as pd
import numpy as np
import hashlib
import os

def apply_differential_privacy(series, epsilon=1.0):
    """Adds Laplace noise to numerical data for differential privacy."""
    scale = 1.0 / epsilon
    noise = np.random.laplace(0, scale, len(series))
    return series + noise

def anonymize_pipeline(input_path, output_path):
    print("[RGPD] Starting Anonymization Pipeline...")
    
    if not os.path.exists(input_path):
        print("Input dataset not found. Generating mock data for demo...")
        df = pd.DataFrame({
            'customer_name': ['Alice Martin', 'Bob Durand', 'Charlie Smith', 'Diana Prince'],
            'email': ['alice@mail.com', 'bob@mail.com', 'charlie@mail.com', 'diana@mail.com'],
            'ip_address': ['192.168.1.1', '10.0.0.5', '172.16.0.10', '8.8.8.8'],
            'claim_amount': [1200.50, 45000.00, 320.00, 15000.00],
            'insurance_type': ['Auto', 'Life', 'Home', 'Auto'],
            'region': ['IDF', 'PACA', 'IDF', 'BRE'],
            'fraud_reported': [0, 1, 0, 0]
        })
    else:
        df = pd.read_csv(input_path)

    # 1. DATA MINIMIZATION: Drop direct identifiers
    df_clean = df.drop(columns=['customer_name', 'email'], errors='ignore')
    
    # 2. PSEUDONYMIZATION: Hash sensitive IDs (Salted SHA256)
    salt = "REDACTED_INSURANCE_SECRET_2026"
    df_clean['ip_address_hash'] = df_clean['ip_address'].apply(
        lambda x: hashlib.sha256((str(x) + salt).encode()).hexdigest()
    )
    df_clean = df_clean.drop(columns=['ip_address'])
    
    # 3. DIFFERENTIAL PRIVACY: Add noise to financial data
    df_clean['claim_amount'] = apply_differential_privacy(df_clean['claim_amount'])
    
    # Save anonymized dataset
    df_clean.to_csv(output_path, index=False)
    print(f"OK [RGPD] Anonymized dataset saved to: {output_path}")
    print(df_clean.head())

if __name__ == "__main__":
    anonymize_pipeline('raw_insurance_data.csv', 'governance/privacy/anonymized_dataset.csv')
