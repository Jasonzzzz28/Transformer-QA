FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the evaluation code
COPY evaluation_monitor/ /app/evaluation_monitor/
COPY workspace/ /app/workspace/

# Set working directory
WORKDIR /app

# Create results directory
RUN mkdir -p /results 