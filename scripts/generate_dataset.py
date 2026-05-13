"""
Generate a synthetic insurance claims dataset.
Use this if you cannot download the Kaggle dataset.

Usage:
    python scripts/generate_dataset.py
"""

import csv
import random
import os
from datetime import datetime, timedelta

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_ROWS = 1000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'sample_claims.csv')

INSURANCE_TYPES = ['Auto', 'Health', 'Home', 'Life', 'Travel']
REGIONS = [ 'Casablanca', 'Rabat', 'Tangier', 'Marrakech', 'Fes', 'Agadir']
STATUSES = ['Approved', 'Denied', 'Pending', 'Under Review', 'Settled']

# ── Helpers ───────────────────────────────────────────────────────────────────

def random_date(start_year: int = 2020, end_year: int = 2024) -> str:
    """Generate a random date string between start_year and end_year."""
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime('%Y-%m-%d')


def random_amount() -> float:
    """Generate a random claim amount with realistic distribution."""
    # 80% small claims, 15% medium, 5% large
    roll = random.random()
    if roll < 0.80:
        return round(random.uniform(100, 5000), 2)
    elif roll < 0.95:
        return round(random.uniform(5000, 25000), 2)
    else:
        return round(random.uniform(25000, 100000), 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_dataset():
    """Generate the synthetic claims CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'claim_id', 'customer_id', 'policy_id',
            'claim_date', 'claim_amount', 'insurance_type',
            'region', 'status'
        ])

        for i in range(1, NUM_ROWS + 1):
            writer.writerow([
                i,                                          # claim_id
                random.randint(1000, 9999),                 # customer_id
                random.randint(100000, 999999),             # policy_id
                random_date(),                              # claim_date
                random_amount(),                            # claim_amount
                random.choice(INSURANCE_TYPES),             # insurance_type
                random.choice(REGIONS),                     # region
                random.choice(STATUSES),                    # status
            ])

    print(f"[OK] Dataset generated: {OUTPUT_FILE} ({NUM_ROWS} rows)")


if __name__ == '__main__':
    generate_dataset()
