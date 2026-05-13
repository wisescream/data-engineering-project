-- ============================================================
-- dbt Model: claims_transformed
-- ============================================================
-- Transforms raw insurance claims data:
--   • Filters out invalid (non-positive) claim amounts
--   • Uppercases region names for consistency
--   • Adds derived analytics columns (year, month, category)
-- ============================================================

{{
    config(
        materialized='table',
        engine='MergeTree()',
        order_by='claim_id'
    )
}}

SELECT
    claim_id,
    customer_id,
    policy_id,
    claim_date,
    claim_amount,
    insurance_type,
    UPPER(region) AS region,
    status,

    -- Derived analytics columns
    toYear(claim_date)  AS claim_year,
    toMonth(claim_date) AS claim_month,

    -- Amount categorisation
    CASE
        WHEN claim_amount < 1000  THEN 'Low'
        WHEN claim_amount < 10000 THEN 'Medium'
        ELSE 'High'
    END AS amount_category

FROM {{ source('insurance', 'claims') }}
WHERE claim_amount > 0
