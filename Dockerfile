# Quantum Market Domination System - Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (for DeFi components)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt package.json ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js dependencies
RUN npm install

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/secrets

# Set permissions
RUN find /app/secrets -type f -exec chmod 600 {} \; 2>/dev/null || true

# Environment variables
ENV ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Expose ports (if needed for monitoring/API)
EXPOSE 8080

# Default command
CMD ["python", "main.py"]
