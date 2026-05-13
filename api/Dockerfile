FROM python:3.11-slim

LABEL maintainer="insurance-fraud-api"
LABEL description="Production API for Insurance Fraud Detection"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy application code
COPY api/ /app/api/
COPY mlops/ /app/mlops/
COPY data/ /app/data/
COPY artifacts/ /app/artifacts/

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
