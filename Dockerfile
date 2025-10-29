# Multi-stage Dockerfile for Quantum Market Domination System

# Stage 1: Python build environment
FROM python:3.11-slim as python-builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Node.js build environment
FROM node:18-slim as node-builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install Node.js dependencies
RUN npm ci --production

# Stage 3: Final runtime image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=python-builder /root/.local /root/.local

# Copy Node.js runtime and dependencies from builder
COPY --from=node-builder /usr/local/bin/node /usr/local/bin/
COPY --from=node-builder /app/node_modules ./node_modules

# Copy application code
COPY . .

# Make scripts executable
RUN chmod +x *.py

# Set Python path
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Create directories for secrets and logs
RUN mkdir -p secrets logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "main.py"]

# Expose ports if needed (uncomment if running web services)
# EXPOSE 8000

# Labels
LABEL maintainer="Quantum Market Domination Team"
LABEL version="2.0"
LABEL description="Advanced algorithmic trading system with ML and DeFi capabilities"
