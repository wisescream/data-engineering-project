FROM apache/airflow:2.8.1-python3.11

USER root

# System dependencies for dbt and data processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy and install Python requirements
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
