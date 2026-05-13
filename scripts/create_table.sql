-- ============================================================
-- ClickHouse Initialization Script
-- Runs automatically on container first start
-- ============================================================

CREATE DATABASE IF NOT EXISTS insurance;

-- Raw claims table (loaded from MinIO CSV)
CREATE TABLE IF NOT EXISTS insurance.claims (
    claim_id         UInt32,
    customer_id      UInt32,
    policy_id        UInt32,
    claim_date       Date,
    claim_amount     Float32,
    insurance_type   String,
    region           String,
    status           String
)
ENGINE = MergeTree()
ORDER BY claim_id;

-- Transformed claims table (output of dbt model)
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
ORDER BY claim_id;
