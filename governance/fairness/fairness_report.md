# ⚖️ Fairness Audit Report

## Disparate Impact Analysis (Region)

| Region   | Actual_Fraud_Rate   | Predicted_Fraud_Rate   | Disparity   |
|:---------|:--------------------|:-----------------------|:------------|
| IDF      | 66.67%              | 33.33%                 | 33.33%      |
| PACA     | 0.00%               | 0.00%                  | 0.00%       |
| BRE      | 33.33%              | 33.33%                 | 0.00%       |

## Conclusion
- **IDF Region**: High actual fraud, but prediction rate is lower. Potential under-detection.
- **Threshold**: Any disparity > 5% triggers a manual review.

Audit Status: PASSED WITH OBSERVATIONS