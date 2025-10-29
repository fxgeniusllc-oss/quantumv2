# Quantum Market Domination System - Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Quantum Market Domination System in various environments.

> **New!** We now provide automated installation scripts for both Windows and Linux/Mac. See [INSTALL_DEPLOY.md](INSTALL_DEPLOY.md) for detailed instructions.

## Prerequisites

- Python 3.9 or higher
- Node.js 16 or higher (required for Yarn installation)
- Yarn (installed automatically by setup scripts if Node.js is present)
- 4GB+ RAM recommended
- Linux/macOS/Windows

## Quick Deployment

### Option 1: Automated Installation (Recommended)

#### Windows Users

```batch
# Run the complete installation and deployment
install-and-deploy.bat

# For production environment
install-and-deploy.bat production
```

#### Linux/Mac Users

```bash
# Run automated setup
./setup.sh
```

See [INSTALL_DEPLOY.md](INSTALL_DEPLOY.md) for complete documentation.

### Option 2: Manual Installation

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/fxgeniusllc-oss/quantumv2.git
cd quantumv2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
yarn install  # Use Yarn instead of npm
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env  # or use your preferred editor
```

Required environment variables:
- Exchange API keys (Binance, Bybit, OKX, etc.)
- Blockchain RPC endpoints
- Risk parameters
- System configuration

### 3. Validation

```bash
# Run tests
pytest tests/ -v

# Validate implementation
python validate_readme.py

# Final validation
python final_validation.py
```

### 4. Start Services

```bash
# Start quantum trading system
python main.py

# Or start DeFi system
python defi_main.py
```

## Production Deployment

### Docker Deployment (Recommended)

```bash
# Build Docker image
docker build -t quantum-trading:latest .

# Run container
docker run -d \
  --name quantum-trading \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  quantum-trading:latest
```

### Systemd Service (Linux)

Create service file `/etc/systemd/system/quantum-trading.service`:

```ini
[Unit]
Description=Quantum Market Domination Trading System
After=network.target

[Service]
Type=simple
User=trading
WorkingDirectory=/opt/quantumv2
Environment="PATH=/opt/quantumv2/venv/bin"
ExecStart=/opt/quantumv2/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable quantum-trading
sudo systemctl start quantum-trading
sudo systemctl status quantum-trading
```

### Process Manager (PM2 - Node.js)

```bash
# Install PM2
npm install -g pm2

# Start with PM2
pm2 start main.py --name quantum-trading --interpreter python3

# Enable auto-restart on system boot
pm2 startup
pm2 save
```

## Environment-Specific Configuration

### Development

```bash
export ENVIRONMENT=development
python main.py
```

- Verbose logging
- Lower position sizes
- Paper trading mode
- Debug features enabled

### Staging

```bash
export ENVIRONMENT=staging
python main.py
```

- Moderate logging
- Reduced position sizes
- Real API connections
- Performance monitoring

### Production

```bash
export ENVIRONMENT=production
python main.py
```

- Optimized logging
- Full position sizes
- High-frequency trading mode
- Complete monitoring

## Monitoring & Maintenance

### Health Checks

```bash
# Check system status
curl http://localhost:8080/health

# Check metrics
curl http://localhost:8080/metrics
```

### Log Management

Logs are stored in:
- `/var/log/quantum-trading/` (production)
- `./logs/` (development)

Rotate logs regularly:
```bash
# Using logrotate
sudo cp deploy/logrotate.conf /etc/logrotate.d/quantum-trading
```

### Backup Strategy

Regular backups:
```bash
# Backup configuration
./scripts/backup_config.sh

# Backup data
./scripts/backup_data.sh

# Backup to S3 (if configured)
./scripts/backup_to_s3.sh
```

## Security Best Practices

1. **Credential Management**
   - Never commit credentials to git
   - Use environment variables or encrypted vault
   - Rotate API keys regularly

2. **Network Security**
   - Use firewall rules
   - Limit API access to trusted IPs
   - Enable SSL/TLS for all connections

3. **Access Control**
   - Run with dedicated user account
   - Limit file permissions (chmod 600 for sensitive files)
   - Use sudo only when necessary

4. **Monitoring**
   - Enable real-time alerts
   - Monitor system resources
   - Track trading performance
   - Review logs regularly

## Scaling & Performance

### Horizontal Scaling

Deploy multiple instances for different strategies:

```bash
# Instance 1: Trend trading
python main.py --strategy=trend --config=config/trend.yml

# Instance 2: Arbitrage
python main.py --strategy=arbitrage --config=config/arbitrage.yml

# Instance 3: Market making
python main.py --strategy=market_making --config=config/mm.yml
```

### Optimization

1. **Database Optimization**
   - Use connection pooling
   - Implement caching
   - Optimize queries

2. **Network Optimization**
   - Use WebSocket connections
   - Implement connection pooling
   - Enable compression

3. **Computing Optimization**
   - Use multiprocessing for CPU-bound tasks
   - Implement async I/O
   - Optimize ML model inference

## Troubleshooting

### Common Issues

1. **Connection Errors**
   ```bash
   # Check network connectivity
   ping api.binance.com
   
   # Verify API credentials
   python -c "import ccxt; print(ccxt.binance().fetch_balance())"
   ```

2. **Memory Issues**
   ```bash
   # Check memory usage
   free -h
   
   # Increase swap space if needed
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

3. **Performance Issues**
   ```bash
   # Profile CPU usage
   python -m cProfile main.py
   
   # Monitor in real-time
   htop
   ```

### Getting Help

- Check logs: `tail -f logs/quantum-trading.log`
- Run diagnostics: `python diagnostics.py`
- Review documentation: `docs/`
- Report issues: GitHub Issues

## Rollback Procedure

If deployment fails:

```bash
# Stop service
sudo systemctl stop quantum-trading

# Restore previous version
git checkout <previous-tag>
pip install -r requirements.txt

# Restore configuration
./scripts/restore_config.sh

# Start service
sudo systemctl start quantum-trading
```

## Disaster Recovery

1. **System Failure**
   - Restore from backup
   - Verify configuration
   - Run validation tests
   - Gradually resume trading

2. **Data Loss**
   - Restore from latest backup
   - Verify data integrity
   - Reconcile with exchange records

3. **Security Breach**
   - Immediately revoke all API keys
   - Change all credentials
   - Review audit logs
   - Restore from clean backup

## Continuous Integration/Deployment

### GitHub Actions (Example)

```yaml
name: Deploy

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest tests/
      - name: Deploy to production
        run: ./scripts/deploy_production.sh
```

## Support

For additional support:
- Documentation: `README.md`
- Build guide: `BUILD_AND_TEST.md`
- Email: support@example.com

## License

Proprietary - All Rights Reserved
