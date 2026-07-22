# syntax=docker/dockerfile:1

FROM python:3.11-slim

WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libnuma1 \
    libelf1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# ROCm environment variables
ENV ROCM_HOME=/opt/rocm
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/amdgpu/lib/x86_64-linux-gnu
ENV PATH="/opt/rocm/bin:${PATH}"

# Copy application
COPY . .

ENTRYPOINT ["python3", "syntheticdatagen.py", "--type", "1"]
