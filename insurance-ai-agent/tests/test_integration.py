from agent.tools import get_claim_history, predict_fraud_j3, compliance_check_j4

def test_clickhouse_integration():
    res = get_claim_history("CUST-123")
    assert "CUST-123" in res

def test_fraud_prediction_integration():
    res = predict_fraud_j3("Claim Sample Data")
    assert "J3" in res or "Simulation" in res

def test_compliance_integration():
    res = compliance_check_j4("Sensitive User Data")
    assert "J4" in res and "Validée" in res
