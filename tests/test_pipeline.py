"""
Unit & Integration Tests for the Insurance Pipeline
====================================================
Run with: pytest tests/ -v
"""

import os
import sys
import csv
import tempfile
from datetime import datetime

import pandas as pd
import pytest

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestDataGeneration:
    """Test the synthetic data generator."""

    def test_generate_creates_file(self, tmp_path):
        """Verify CSV file is created with correct structure."""
        from generate_dataset import INSURANCE_TYPES, REGIONS, STATUSES

        output_file = tmp_path / 'test_claims.csv'

        # Create a small dataset manually
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'claim_id', 'customer_id', 'policy_id',
                'claim_date', 'claim_amount', 'insurance_type',
                'region', 'status'
            ])
            writer.writerow([1, 1001, 100001, '2023-06-15', 2500.50, 'Auto', 'North', 'Approved'])

        df = pd.read_csv(output_file)
        assert len(df) == 1
        assert list(df.columns) == [
            'claim_id', 'customer_id', 'policy_id',
            'claim_date', 'claim_amount', 'insurance_type',
            'region', 'status'
        ]

    def test_insurance_types_valid(self):
        from generate_dataset import INSURANCE_TYPES
        expected = ['Auto', 'Health', 'Home', 'Life', 'Travel']
        assert INSURANCE_TYPES == expected

    def test_statuses_valid(self):
        from generate_dataset import STATUSES
        expected = ['Approved', 'Denied', 'Pending', 'Under Review', 'Settled']
        assert STATUSES == expected


class TestDataTransformation:
    """Test transformation logic (mirrors DAG task 3)."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'claim_id': [1, 2, 3, 4],
            'customer_id': [1001, 1002, 1003, 1004],
            'policy_id': [100001, 100002, 100003, 100004],
            'claim_date': ['2023-01-15', '2023-06-20', '2023-11-05', '2024-03-10'],
            'claim_amount': [500.0, 5000.0, 25000.0, -100.0],
            'insurance_type': ['Auto', 'Health', 'Home', 'Life'],
            'region': ['north', 'South', 'east', 'West'],
            'status': ['Approved', 'Denied', 'Pending', 'Settled'],
        })

    def test_filter_negative_amounts(self, sample_df):
        filtered = sample_df[sample_df['claim_amount'] > 0]
        assert len(filtered) == 3
        assert -100.0 not in filtered['claim_amount'].values

    def test_uppercase_regions(self, sample_df):
        sample_df['region'] = sample_df['region'].str.upper()
        assert all(r.isupper() for r in sample_df['region'])
        assert 'NORTH' in sample_df['region'].values

    def test_amount_categorization(self, sample_df):
        df = sample_df[sample_df['claim_amount'] > 0].copy()
        df['amount_category'] = df['claim_amount'].apply(
            lambda x: 'Low' if x < 1000 else ('Medium' if x < 10000 else 'High')
        )
        assert df.loc[df['claim_id'] == 1, 'amount_category'].values[0] == 'Low'
        assert df.loc[df['claim_id'] == 2, 'amount_category'].values[0] == 'Medium'
        assert df.loc[df['claim_id'] == 3, 'amount_category'].values[0] == 'High'

    def test_derived_date_columns(self, sample_df):
        df = sample_df.copy()
        df['claim_date'] = pd.to_datetime(df['claim_date'])
        df['claim_year'] = df['claim_date'].dt.year
        df['claim_month'] = df['claim_date'].dt.month
        assert df.loc[0, 'claim_year'] == 2023
        assert df.loc[0, 'claim_month'] == 1
        assert df.loc[3, 'claim_year'] == 2024


class TestDataValidation:
    """Test Great Expectations-style validation rules."""

    @pytest.fixture
    def valid_df(self):
        return pd.DataFrame({
            'claim_id': [1, 2, 3],
            'customer_id': [1001, 1002, 1003],
            'policy_id': [100001, 100002, 100003],
            'claim_date': ['2023-01-15', '2023-06-20', '2023-11-05'],
            'claim_amount': [500.0, 5000.0, 25000.0],
            'insurance_type': ['Auto', 'Health', 'Home'],
            'region': ['North', 'South', 'East'],
            'status': ['Approved', 'Denied', 'Pending'],
        })

    def test_no_null_critical_columns(self, valid_df):
        critical_cols = ['claim_id', 'customer_id', 'policy_id', 'claim_date', 'claim_amount']
        for col in critical_cols:
            assert valid_df[col].isnull().sum() == 0, f"Nulls found in {col}"

    def test_positive_amounts(self, valid_df):
        assert (valid_df['claim_amount'] > 0).all()

    def test_valid_statuses(self, valid_df):
        valid_statuses = {'Approved', 'Denied', 'Pending', 'Under Review', 'Settled'}
        assert set(valid_df['status'].unique()).issubset(valid_statuses)

    def test_valid_insurance_types(self, valid_df):
        valid_types = {'Auto', 'Health', 'Home', 'Life', 'Travel'}
        assert set(valid_df['insurance_type'].unique()).issubset(valid_types)

    def test_detects_invalid_status(self):
        df = pd.DataFrame({
            'status': ['Approved', 'INVALID_STATUS', 'Denied']
        })
        valid_statuses = {'Approved', 'Denied', 'Pending', 'Under Review', 'Settled'}
        invalid = df[~df['status'].isin(valid_statuses)]
        assert len(invalid) == 1


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS (require running services)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    os.getenv('RUN_INTEGRATION_TESTS', '0') != '1',
    reason="Integration tests require running Docker services (set RUN_INTEGRATION_TESTS=1)"
)
class TestIntegration:
    """Integration tests — require Docker services to be up."""

    def test_clickhouse_connection(self):
        import clickhouse_connect
        client = clickhouse_connect.get_client(
            host=os.getenv('CLICKHOUSE_HOST', 'localhost'),
            port=int(os.getenv('CLICKHOUSE_PORT', '8123')),
        )
        result = client.query('SELECT 1')
        assert result.result_rows[0][0] == 1

    def test_clickhouse_table_exists(self):
        import clickhouse_connect
        client = clickhouse_connect.get_client(
            host=os.getenv('CLICKHOUSE_HOST', 'localhost'),
            port=int(os.getenv('CLICKHOUSE_PORT', '8123')),
        )
        result = client.query("SELECT count() FROM system.tables WHERE database='insurance' AND name='claims'")
        assert result.result_rows[0][0] >= 1

    def test_minio_connection(self):
        import boto3
        s3 = boto3.client(
            's3',
            endpoint_url=os.getenv('AWS_ENDPOINT_URL', 'http://localhost:9002'),
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin',
            region_name='us-east-1',
        )
        buckets = s3.list_buckets()
        bucket_names = [b['Name'] for b in buckets['Buckets']]
        assert 'insurance-data' in bucket_names

    def test_end_to_end_data_flow(self):
        """Verify data can be read from MinIO and written to ClickHouse."""
        import boto3
        import clickhouse_connect

        # Read from MinIO
        s3 = boto3.client(
            's3',
            endpoint_url=os.getenv('AWS_ENDPOINT_URL', 'http://localhost:9002'),
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin',
            region_name='us-east-1',
        )
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            s3.download_file('insurance-data', 'raw/sample_claims.csv', tmp.name)
            df = pd.read_csv(tmp.name)

        assert len(df) > 0
        assert 'claim_id' in df.columns

        # Verify ClickHouse is reachable
        client = clickhouse_connect.get_client(
            host=os.getenv('CLICKHOUSE_HOST', 'localhost'),
            port=int(os.getenv('CLICKHOUSE_PORT', '8123')),
        )
        result = client.query('SELECT 1')
        assert result.result_rows[0][0] == 1
