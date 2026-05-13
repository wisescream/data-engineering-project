"""
Insurance Pipeline DAG
======================
Orchestrates the full ETL pipeline:
  1. Extract CSV from MinIO (S3)
  2. Validate data quality with Great Expectations
  3. Transform data with dbt
  4. Load into ClickHouse
  5. Send Slack notification
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import boto3
import clickhouse_connect
import great_expectations as gx
import pandas as pd
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Configuration ─────────────────────────────────────────────────────────────

MINIO_ENDPOINT = os.getenv('AWS_ENDPOINT_URL', 'http://minio:9000')
MINIO_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin')
MINIO_BUCKET = 'insurance-data'
MINIO_KEY = 'raw/sample_claims.csv'

CLICKHOUSE_HOST = os.getenv('CLICKHOUSE_HOST', 'clickhouse')
CLICKHOUSE_PORT = int(os.getenv('CLICKHOUSE_PORT', '8123'))
CLICKHOUSE_USER = os.getenv('CLICKHOUSE_USER', 'default')
CLICKHOUSE_PASSWORD = os.getenv('CLICKHOUSE_PASSWORD', '')

SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', 'https://hooks.slack.com/services/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')

DATA_DIR = '/opt/airflow/data'
LOCAL_CSV = os.path.join(DATA_DIR, 'extracted_claims.csv')
TRANSFORMED_CSV = os.path.join(DATA_DIR, 'transformed_claims.csv')

logger = logging.getLogger(__name__)

# ── DAG Default Args ─────────────────────────────────────────────────────────

default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'sla': timedelta(hours=1),
}

def sla_miss_callback(dag, task_list, blocking_task_list, slas, blocking_tis):
    """
    Sends a Slack alert if the SLA of 1 hour is missed.
    """
    message = {
        "text": f"🚨 *SLA Missed!* The DAG `{dag.dag_id}` took more than 1 hour to complete."
    }
    if SLACK_WEBHOOK_URL and 'YOUR' not in SLACK_WEBHOOK_URL:
        try:
            requests.post(
                SLACK_WEBHOOK_URL,
                json=message,
                headers={'Content-Type': 'application/json'},
                timeout=10,
            )
        except Exception as e:
            logger.error("❌ Failed to send SLA Slack alert: %s", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# TASK FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def extract_from_s3(**context):
    """
    Task 1 — Download the claims CSV from MinIO (S3-compatible).
    Stores the file locally and pushes row count to XCom.
    """
    logger.info("Connecting to MinIO at %s", MINIO_ENDPOINT)

    s3_client = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name='us-east-1',
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    s3_client.download_file(MINIO_BUCKET, MINIO_KEY, LOCAL_CSV)

    df = pd.read_csv(LOCAL_CSV)
    row_count = len(df)
    logger.info("✅ Extracted %d rows from s3://%s/%s", row_count, MINIO_BUCKET, MINIO_KEY)

    # Push metrics to XCom
    context['ti'].xcom_push(key='rows_extracted', value=row_count)
    context['ti'].xcom_push(key='columns', value=list(df.columns))

    return LOCAL_CSV


def validate_schema(**context):
    """
    Task 2 — Validate data quality using Great Expectations.
    Checks:
      - No null values in critical columns
      - claim_amount > 0
      - status in valid set
    """
    logger.info("Running Great Expectations validation...")

    df = pd.read_csv(LOCAL_CSV)

    # ── Build GX context and validator programmatically ───────────────────
    gx_context = gx.get_context()

    datasource = gx_context.sources.add_or_update_pandas(name="claims_datasource")
    data_asset = datasource.add_dataframe_asset(name="claims_asset")
    batch_request = data_asset.build_batch_request(dataframe=df)

    # Create expectation suite
    suite_name = "claims_validation_suite"
    try:
        suite = gx_context.add_expectation_suite(expectation_suite_name=suite_name)
    except Exception:
        suite = gx_context.get_expectation_suite(expectation_suite_name=suite_name)

    validator = gx_context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name,
    )

    # ── Expectations ──────────────────────────────────────────────────────

    # Null checks on critical columns
    for col in ['claim_id', 'customer_id', 'policy_id', 'claim_date', 'claim_amount']:
        validator.expect_column_values_to_not_be_null(column=col)

    # Positive amount check
    validator.expect_column_values_to_be_between(
        column='claim_amount',
        min_value=0.01,
        max_value=1_000_000,
    )

    # Status values validation
    validator.expect_column_values_to_be_in_set(
        column='status',
        value_set=['Approved', 'Denied', 'Pending', 'Under Review', 'Settled'],
    )

    # Insurance type validation
    validator.expect_column_values_to_be_in_set(
        column='insurance_type',
        value_set=['Auto', 'Health', 'Home', 'Life', 'Travel'],
    )

    # Run validation
    results = validator.validate()
    success = results.success

    # Compute quality metrics
    total_rows = len(df)
    null_counts = df[['claim_id', 'customer_id', 'claim_amount']].isnull().sum().sum()
    null_pct = round((null_counts / (total_rows * 3)) * 100, 2) if total_rows > 0 else 0
    invalid_status = df[~df['status'].isin(['Approved', 'Denied', 'Pending', 'Under Review', 'Settled'])].shape[0]
    invalid_pct = round((invalid_status / total_rows) * 100, 2) if total_rows > 0 else 0

    # Push metrics
    ti = context['ti']
    ti.xcom_push(key='validation_success', value=success)
    ti.xcom_push(key='null_percentage', value=null_pct)
    ti.xcom_push(key='invalid_percentage', value=invalid_pct)
    ti.xcom_push(key='total_expectations', value=results.statistics.get('evaluated_expectations', 0))
    ti.xcom_push(key='successful_expectations', value=results.statistics.get('successful_expectations', 0))

    if success:
        logger.info("✅ Data validation PASSED — %d expectations evaluated", results.statistics.get('evaluated_expectations', 0))
    else:
        logger.warning("⚠️ Data validation FAILED — check expectations results")
        # Log failures but don't raise — let downstream tasks decide
        for result in results.results:
            if not result.success:
                logger.warning("  FAILED: %s", result.expectation_config.expectation_type)

    return success


def transform_data(**context):
    """
    Task 3 — Transform the extracted data.
    Applies the same logic as the dbt model:
      - Uppercase regions
      - Filter claim_amount > 0
      - Add derived columns (year, month, amount_category)
    
    Also attempts to run dbt if available, with Python fallback.
    """
    import subprocess

    logger.info("Running data transformation...")

    # ── Try dbt first ────────────────────────────────────────────────────
    dbt_project_dir = '/opt/airflow/dbt'
    dbt_success = False

    if os.path.exists(os.path.join(dbt_project_dir, 'dbt_project.yml')):
        try:
            result = subprocess.run(
                ['dbt', 'run', '--project-dir', dbt_project_dir, '--profiles-dir', dbt_project_dir],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                logger.info("✅ dbt transformation completed successfully")
                dbt_success = True
            else:
                logger.warning("⚠️ dbt run failed, falling back to Python: %s", result.stderr)
        except Exception as e:
            logger.warning("⚠️ dbt not available, using Python fallback: %s", str(e))

    # ── Python fallback transformation ───────────────────────────────────
    df = pd.read_csv(LOCAL_CSV)

    # Apply transformations matching the dbt model
    df = df[df['claim_amount'] > 0].copy()
    df['region'] = df['region'].str.upper()
    df['claim_date'] = pd.to_datetime(df['claim_date'])
    df['claim_year'] = df['claim_date'].dt.year
    df['claim_month'] = df['claim_date'].dt.month
    df['amount_category'] = df['claim_amount'].apply(
        lambda x: 'Low' if x < 1000 else ('Medium' if x < 10000 else 'High')
    )

    # Save transformed data
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(TRANSFORMED_CSV, index=False)

    rows_transformed = len(df)
    logger.info("✅ Transformed %d rows → %s", rows_transformed, TRANSFORMED_CSV)

    # Push metrics
    ti = context['ti']
    ti.xcom_push(key='rows_transformed', value=rows_transformed)
    ti.xcom_push(key='dbt_used', value=dbt_success)

    return TRANSFORMED_CSV


def load_to_clickhouse(**context):
    """
    Task 4 — Load transformed data into ClickHouse.
    Uses clickhouse_connect for high-performance insertion.
    """
    logger.info("Loading data into ClickHouse at %s:%s", CLICKHOUSE_HOST, CLICKHOUSE_PORT)

    df = pd.read_csv(TRANSFORMED_CSV)
    df['claim_date'] = pd.to_datetime(df['claim_date']).dt.date

    # Connect to ClickHouse
    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
    )

    # Ensure database and table exist
    client.command('CREATE DATABASE IF NOT EXISTS insurance')
    client.command('''
        CREATE TABLE IF NOT EXISTS insurance.claims_transformed (
            claim_id         UInt32,
            customer_id      UInt32,
            policy_id        UInt32,
            claim_date       Date,
            claim_amount     Float32,
            insurance_type   String,
            region           String,
            status           String,
            claim_year       UInt16,
            claim_month      UInt8,
            amount_category  String
        )
        ENGINE = MergeTree()
        ORDER BY claim_id
    ''')

    # Truncate before inserting (idempotent loads)
    client.command('TRUNCATE TABLE insurance.claims_transformed')

    # Insert data
    columns = [
        'claim_id', 'customer_id', 'policy_id', 'claim_date',
        'claim_amount', 'insurance_type', 'region', 'status',
        'claim_year', 'claim_month', 'amount_category'
    ]

    data = df[columns].values.tolist()
    client.insert('insurance.claims_transformed', data, column_names=columns)

    # Verify
    count_result = client.query('SELECT COUNT(*) FROM insurance.claims_transformed')
    rows_loaded = count_result.result_rows[0][0]

    logger.info("✅ Loaded %d rows into insurance.claims_transformed", rows_loaded)

    # Push metrics
    context['ti'].xcom_push(key='rows_loaded', value=rows_loaded)

    return rows_loaded


def send_notification(**context):
    """
    Task 5 — Send Slack notification with pipeline results.
    Sends success or failure message with metrics summary.
    """
    ti = context['ti']

    # Collect metrics from upstream tasks
    rows_extracted = ti.xcom_pull(task_ids='extract_from_s3', key='rows_extracted') or 0
    rows_transformed = ti.xcom_pull(task_ids='transform_data', key='rows_transformed') or 0
    rows_loaded = ti.xcom_pull(task_ids='load_to_clickhouse', key='rows_loaded') or 0
    validation_success = ti.xcom_pull(task_ids='validate_schema', key='validation_success')
    null_pct = ti.xcom_pull(task_ids='validate_schema', key='null_percentage') or 0
    invalid_pct = ti.xcom_pull(task_ids='validate_schema', key='invalid_percentage') or 0
    dbt_used = ti.xcom_pull(task_ids='transform_data', key='dbt_used') or False

    # Compute pipeline duration
    dag_run = context.get('dag_run')
    duration = "N/A"
    if dag_run and dag_run.start_date:
        elapsed = datetime.utcnow() - dag_run.start_date.replace(tzinfo=None)
        duration = str(elapsed).split('.')[0]

    # Compute completeness
    completeness = round((rows_loaded / rows_extracted) * 100, 2) if rows_extracted > 0 else 0

    # Build message
    status_emoji = "✅" if validation_success else "⚠️"
    transform_method = "dbt + Python" if dbt_used else "Python (pandas)"

    message = {
        "text": f"""
{status_emoji} *Insurance Pipeline Complete*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 *Pipeline Metrics*
• Rows extracted: `{rows_extracted}`
• Rows transformed: `{rows_transformed}`
• Rows loaded: `{rows_loaded}`
• Completeness: `{completeness}%`
• Null %: `{null_pct}%`
• Invalid %: `{invalid_pct}%`
• Transform method: `{transform_method}`
• Duration: `{duration}`

📋 *Data Validation*: {"PASSED ✅" if validation_success else "FAILED ❌"}
🏗️ *DAG*: `{context.get('dag', {}).dag_id}`
📅 *Execution Date*: `{context.get('ds', 'N/A')}`
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    }

    # Send to Slack
    if SLACK_WEBHOOK_URL and 'YOUR' not in SLACK_WEBHOOK_URL:
        try:
            response = requests.post(
                SLACK_WEBHOOK_URL,
                json=message,
                headers={'Content-Type': 'application/json'},
                timeout=10,
            )
            if response.status_code == 200:
                logger.info("✅ Slack notification sent successfully")
            else:
                logger.warning("⚠️ Slack returned status %d: %s", response.status_code, response.text)
        except Exception as e:
            logger.error("❌ Failed to send Slack notification: %s", str(e))
    else:
        logger.info("ℹ️ Slack webhook not configured — notification logged locally:")
        logger.info(message['text'])

    # Always log the summary
    logger.info("Pipeline Summary: extracted=%d, transformed=%d, loaded=%d, duration=%s",
                rows_extracted, rows_transformed, rows_loaded, duration)

    return True


# ══════════════════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id='insurance_pipeline',
    default_args=default_args,
    description='Insurance claims ETL pipeline: MinIO → GE → dbt → ClickHouse → Slack',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['insurance', 'etl', 'production'],
    doc_md="""
    ## Insurance Claims Pipeline
    
    ### Overview
    End-to-end ETL pipeline for processing insurance claims data:
    1. **Extract** — Download CSV from MinIO (S3-compatible storage)
    2. **Validate** — Run data quality checks with Great Expectations
    3. **Transform** — Clean and enrich data using dbt/pandas
    4. **Load** — Insert into ClickHouse data warehouse
    5. **Notify** — Send Slack alert with pipeline metrics
    
    ### Connections Required
    - `aws_default` — MinIO/S3 credentials
    - `clickhouse_default` — ClickHouse connection
    
    ### SLA
    Pipeline must complete within 1 hour.
    """,
    sla_miss_callback=sla_miss_callback,
) as dag:

    t1_extract = PythonOperator(
        task_id='extract_from_s3',
        python_callable=extract_from_s3,
        provide_context=True,
    )

    t2_validate = PythonOperator(
        task_id='validate_schema',
        python_callable=validate_schema,
        provide_context=True,
    )

    t3_transform = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data,
        provide_context=True,
    )

    t4_load = PythonOperator(
        task_id='load_to_clickhouse',
        python_callable=load_to_clickhouse,
        provide_context=True,
    )

    t5_notify = PythonOperator(
        task_id='send_notification',
        python_callable=send_notification,
        provide_context=True,
        trigger_rule='all_done',  # Notify even on failure
    )

    # ── Pipeline Flow ────────────────────────────────────────────────────
    t1_extract >> t2_validate >> t3_transform >> t4_load >> t5_notify
