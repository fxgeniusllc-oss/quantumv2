# Deployment Guide

## Overview

This guide covers deploying the Quantum Market Domination System in various environments.

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB+ available space
- **Python**: 3.11+
- **Node.js**: 18+ (for DeFi features)
- **Docker**: 20.10+ (optional, for containerized deployment)
- **Docker Compose**: 2.0+ (optional)

### API Keys Required
- Exchange API keys (Binance, Bybit, OKX, etc.)
- Blockchain RPC endpoints (Infura, Alchemy, etc.)
- Optional: Slack webhook, email SMTP

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/fxgeniusllc-oss/quantumv2.git
cd quantumv2
```

### 2. Run Setup Script
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit with your credentials
nano .env
```

### 4. Run Tests
```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest tests/ -v
```

### 5. Start Application
```bash
# For quantum trading
python main.py

# For DeFi strategies
python defi_main.py
```

## Deployment Methods

### Method 1: Local/Development Deployment

#### Step-by-step

1. **Install dependencies**
   ```bash
   ./setup.sh
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Run application**
   ```bash
   source venv/bin/activate
   python main.py
   ```

### Method 2: Docker Deployment (Recommended for Production)

#### Build and run with Docker Compose

1. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

2. **Build images**
   ```bash
   docker-compose build
   ```

3. **Start services**
   ```bash
   docker-compose up -d
   ```

4. **View logs**
   ```bash
   docker-compose logs -f
   ```

5. **Stop services**
   ```bash
   docker-compose down
   ```

#### Individual Docker commands

**Build image:**
```bash
docker build -t quantum-trading:latest .
```

**Run container:**
```bash
docker run -d \
  --name quantum-trader \
  -v $(pwd)/secrets:/app/secrets:ro \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  quantum-trading:latest
```

**View logs:**
```bash
docker logs -f quantum-trader
```

### Method 3: Automated Deployment Script

Use the included deployment script for streamlined deployment:

```bash
# Development deployment
./deploy.sh development

# Staging deployment
./deploy.sh staging

# Production deployment
./deploy.sh production latest
```

## Environment Configuration

### Required Variables

```bash
# Exchange API Keys
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET=your_binance_secret
BYBIT_API_KEY=your_bybit_key
BYBIT_SECRET=your_bybit_secret

# Blockchain Configuration
ETH_RPC_URL=https://mainnet.infura.io/v3/your_key
POLYGON_RPC_URL=https://polygon-rpc.com

# Risk Parameters
MAX_SINGLE_TRADE_RISK=0.05
TOTAL_PORTFOLIO_RISK=0.15
STOP_LOSS_THRESHOLD=0.10

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### Optional Variables

```bash
# Monitoring
SLACK_WEBHOOK_URL=your_slack_webhook
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_FROM=alerts@example.com
EMAIL_TO=admin@example.com

# Advanced Configuration
COMPRESSION_LEVEL=9
MAX_CONCURRENT_TRADES=10
WEBSOCKET_TIMEOUT=500
```

## Production Considerations

### Security

1. **Protect sensitive files**
   ```bash
   chmod 600 .env
   chmod 700 secrets/
   ```

2. **Use secure credential storage**
   - Consider using AWS Secrets Manager, HashiCorp Vault, or similar
   - Never commit `.env` or `secrets/` to version control

3. **Enable encryption**
   - All API credentials are encrypted using Fernet
   - Master keys stored in `secrets/master.key`

### Monitoring

1. **System monitoring**
   - CPU, memory, GPU usage tracked
   - Alerts sent on threshold breaches

2. **Performance tracking**
   - All trades logged
   - P&L, win rate, drawdown calculated
   - Export to CSV for analysis

3. **Compliance monitoring**
   - Position limits enforced
   - Trade frequency limits
   - Wash sale detection

### High Availability

For production deployments requiring high availability:

1. **Load balancing**
   - Deploy multiple instances behind load balancer
   - Use Redis for distributed locking

2. **Database replication**
   - Set up master-slave replication
   - Implement automatic failover

3. **Monitoring and alerting**
   - Use Prometheus + Grafana for metrics
   - Configure PagerDuty for critical alerts

## Scaling

### Horizontal Scaling

Deploy multiple instances with shared Redis:

```yaml
# docker-compose.scale.yml
services:
  quantum-trader:
    deploy:
      replicas: 3
      
  redis:
    deploy:
      replicas: 1
```

Start scaled deployment:
```bash
docker-compose -f docker-compose.yml -f docker-compose.scale.yml up -d
```

### Vertical Scaling

Allocate more resources to containers:

```yaml
services:
  quantum-trader:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          cpus: '2'
          memory: 8G
```

## Backup and Recovery

### Backup Strategy

1. **Configuration backups**
   ```bash
   # Backup .env and secrets
   tar -czf backup-$(date +%Y%m%d).tar.gz .env secrets/
   ```

2. **Data backups**
   ```bash
   # Backup trade data and logs
   tar -czf data-backup-$(date +%Y%m%d).tar.gz data/ logs/
   ```

3. **Automated backups**
   ```bash
   # Add to crontab
   0 2 * * * /path/to/backup.sh
   ```

### Recovery

1. **Restore configuration**
   ```bash
   tar -xzf backup-20231201.tar.gz
   ```

2. **Restart services**
   ```bash
   docker-compose restart
   ```

## Troubleshooting

### Common Issues

**1. Connection errors to exchanges**
- Check API keys in `.env`
- Verify IP whitelist on exchange
- Check network connectivity

**2. Out of memory errors**
- Increase container memory limits
- Reduce concurrent operations
- Optimize data retention

**3. Permission errors**
- Check file permissions: `chmod 600 .env`
- Ensure Docker has volume access
- Verify user permissions

**4. Test failures**
- Update dependencies: `pip install -r requirements.txt --upgrade`
- Clear pytest cache: `pytest --cache-clear`
- Check Python version: `python --version`

### Logs

**View application logs:**
```bash
# Docker
docker-compose logs -f quantum-trader

# Local
tail -f logs/quantum.log
```

**View system logs:**
```bash
tail -f logs/system_monitor.log
```

**View error logs:**
```bash
tail -f logs/errors.log
```

## Updating

### Pull Latest Changes
```bash
git pull origin main
```

### Update Dependencies
```bash
# Python
pip install -r requirements.txt --upgrade

# Node.js
npm update
```

### Rebuild Docker Images
```bash
docker-compose build --no-cache
docker-compose up -d
```

### Database Migrations
```bash
# If applicable
python manage.py migrate
```

## CI/CD Pipeline

The project includes GitHub Actions workflow for automated deployment:

- **Linting**: Code quality checks
- **Testing**: Unit and integration tests
- **Security**: Vulnerability scanning
- **Building**: Docker image creation
- **Deployment**: Automated deployment to staging/production

See `.github/workflows/ci-cd.yml` for details.

## Support

For issues or questions:
- Check documentation: `/docs`
- Review logs for errors
- Open issue on GitHub
- Contact support team

## License

Proprietary software. All rights reserved.

## Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. Cryptocurrency trading carries significant risk. Always conduct thorough testing before deploying with real funds. The developers are not responsible for any financial losses.
