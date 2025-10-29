#!/bin/bash
# Deployment script for Quantum Trading System

set -e

# Configuration
ENVIRONMENT=${1:-development}
VERSION=${2:-latest}

echo "========================================="
echo "Quantum Trading System Deployment"
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "========================================="

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose is required but not installed. Aborting." >&2; exit 1; }

# Validate environment
if [ "$ENVIRONMENT" != "development" ] && [ "$ENVIRONMENT" != "staging" ] && [ "$ENVIRONMENT" != "production" ]; then
    echo "Invalid environment. Use: development, staging, or production"
    exit 1
fi

# Load environment variables
if [ -f ".env.$ENVIRONMENT" ]; then
    echo "Loading environment variables from .env.$ENVIRONMENT"
    export $(cat .env.$ENVIRONMENT | grep -v '^#' | xargs)
else
    echo "Warning: .env.$ENVIRONMENT not found, using .env"
    if [ -f ".env" ]; then
        export $(cat .env | grep -v '^#' | xargs)
    fi
fi

# Build Docker images
echo "Building Docker images..."
docker-compose build --no-cache

# Run database migrations (if applicable)
# echo "Running database migrations..."
# docker-compose run --rm quantum-trader python manage.py migrate

# Stop existing containers
echo "Stopping existing containers..."
docker-compose down

# Start services
echo "Starting services..."
if [ "$ENVIRONMENT" = "production" ]; then
    docker-compose up -d
else
    docker-compose up -d
fi

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
sleep 10

# Check service health
echo "Checking service health..."
docker-compose ps

# Run health checks
echo "Running health checks..."
QUANTUM_TRADER_STATUS=$(docker inspect --format='{{.State.Status}}' quantum-trader 2>/dev/null || echo "not found")

if [ "$QUANTUM_TRADER_STATUS" = "running" ]; then
    echo "✓ Quantum Trader is running"
else
    echo "✗ Quantum Trader is not running"
    docker-compose logs quantum-trader
    exit 1
fi

# Display logs
if [ "$ENVIRONMENT" = "development" ]; then
    echo "Showing logs (Ctrl+C to exit)..."
    docker-compose logs -f
else
    echo "========================================="
    echo "Deployment completed successfully!"
    echo "========================================="
    echo "To view logs: docker-compose logs -f"
    echo "To stop: docker-compose down"
    echo "To restart: docker-compose restart"
fi
