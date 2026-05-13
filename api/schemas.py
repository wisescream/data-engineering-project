"""
Pydantic Models for the Insurance Fraud Detection API
======================================================
Request and response schemas for all endpoints.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

class PredictionRequest(BaseModel):
    """Input features for fraud prediction."""
    age: float = Field(..., description="Age of the policyholder", example=35.0)
    claim_amount: float = Field(..., description="Total claim amount", example=15000.0)
    policy_annual_premium: float = Field(..., description="Annual premium", example=1200.0)
    months_as_customer: float = Field(..., description="Months as customer", example=120.0)
    num_previous_claims: float = Field(..., description="Number of previous claims", example=1.0)
    police_report_filed: float = Field(..., description="Police report filed (0 or 1)", example=1.0)
    witnesses: float = Field(..., description="Number of witnesses", example=2.0)
    injury_claim: float = Field(..., description="Injury claim amount", example=5000.0)
    vehicle_claim: float = Field(..., description="Vehicle claim amount", example=10000.0)
    total_claim_amount: float = Field(..., description="Total claim amount", example=15000.0)
    insurance_type_encoded: float = Field(..., description="Encoded insurance type", example=0.0)
    region_encoded: float = Field(..., description="Encoded region", example=2.0)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35.0,
                "claim_amount": 15000.0,
                "policy_annual_premium": 1200.0,
                "months_as_customer": 120.0,
                "num_previous_claims": 1.0,
                "police_report_filed": 1.0,
                "witnesses": 2.0,
                "injury_claim": 5000.0,
                "vehicle_claim": 10000.0,
                "total_claim_amount": 15000.0,
                "insurance_type_encoded": 0.0,
                "region_encoded": 2.0,
            }
        }


class PredictionResponse(BaseModel):
    """Fraud prediction result."""
    prediction: int = Field(..., description="0 = Legit, 1 = Fraud")
    label: str = Field(..., description="LEGIT or FRAUD")
    fraud_probability: float = Field(..., description="Probability of fraud")
    confidence: float = Field(..., description="Prediction confidence")
    model_type: str = Field(..., description="Model type used (fp32 or int8)")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    fp32_loaded: bool = Field(..., description="FP32 model loaded")
    int8_loaded: bool = Field(..., description="INT8 model loaded")
    version: str = Field(..., description="API version")


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    fp32: Optional[Dict[str, Any]] = Field(None, description="FP32 model info")
    int8: Optional[Dict[str, Any]] = Field(None, description="INT8 model info")
    compression_ratio: Optional[float] = Field(None, description="Size ratio FP32/INT8")


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

class BenchmarkRequest(BaseModel):
    """Benchmark configuration."""
    n_requests: int = Field(default=1000, ge=10, le=50000, description="Number of requests to simulate")


class BenchmarkMetrics(BaseModel):
    """Metrics for a single model benchmark run."""
    model_type: str
    total_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    memory_mb: float
    accuracy: Optional[float] = None


class BenchmarkResponse(BaseModel):
    """Complete benchmark comparison result."""
    fp32: Optional[BenchmarkMetrics] = None
    int8: Optional[BenchmarkMetrics] = None
    speedup: Optional[float] = Field(None, description="INT8 speedup vs FP32")
    size_reduction: Optional[float] = Field(None, description="Model size reduction ratio")
    accuracy_delta: Optional[float] = Field(None, description="Accuracy difference (INT8 - FP32)")
