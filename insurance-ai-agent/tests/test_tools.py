import pytest
from agent.tools import insurance_risk_score, write_report
import os

def test_risk_score_calculation():
    # High risk case
    res = insurance_risk_score(30000, 5)
    assert "HIGH" in res
    
    # Low risk case
    res = insurance_risk_score(100, 0)
    assert "LOW" in res

def test_report_writing():
    content = "Demo Content"
    filename = "demo_report.txt"
    res = write_report(content, filename)
    assert "saved at" in res
    assert os.path.exists(os.path.join("reports", filename))
    # Cleanup
    os.remove(os.path.join("reports", filename))
