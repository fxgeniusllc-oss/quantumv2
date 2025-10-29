# Quick Start Guide

Get the Quantum Market Domination System running in 5 minutes!

## Prerequisites

- Python 3.9+
- Node.js 16+ (optional, for DeFi features)
- 4GB RAM minimum
- Internet connection

## Easy Installation (Recommended)

### Windows Users

Use the automated installation and deployment script:

```batch
# Run the complete installation and setup
install-and-deploy.bat

# For production environment
install-and-deploy.bat production
```

The script will automatically:
- ✅ Check all prerequisites (Python, Node.js, Yarn, Git)
- ✅ Create Python virtual environment
- ✅ Install all Python dependencies
- ✅ Install all Node.js dependencies with Yarn
- ✅ Create necessary directories
- ✅ Set up configuration files
- ✅ Run system tests

### Linux/Mac Users

Use the automated setup script:

```bash
# 1. Clone repository
git clone https://github.com/fxgeniusllc-oss/quantumv2.git
cd quantumv2

# 2. Run automated setup (includes all dependencies and tests)
./setup.sh
```

The setup script will:
- ✅ Check Python and Node.js versions
- ✅ Create virtual environment
- ✅ Install Python dependencies
- ✅ Install Node.js dependencies with Yarn
- ✅ Create necessary directories
- ✅ Set up .env configuration
- ✅ Run initial test suite

## Manual Installation

If you prefer manual installation:

```bash
# 1. Clone repository
git clone https://github.com/fxgeniusllc-oss/quantumv2.git
cd quantumv2

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate.bat

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install Node.js dependencies (optional, for DeFi features)
yarn install

# 5. Configure environment
cp .env.example .env
# Edit .env with your API credentials

# 6. Run tests (verify installation)
pytest tests/ -v
```

## Building with Yarn

All Node.js operations now use Yarn for better dependency management:

```bash
# Install dependencies
yarn install

# Run build (if needed)
yarn build

# Run tests
yarn test

# Start Node.js engine
yarn start
```

## Quick Test Run

```bash
# Run the main trading system
python main.py
```

## What's Included

### ✅ Core Features
- **Multi-Exchange Integration**: 6+ exchanges
- **ML Intelligence**: Price prediction, volatility analysis, anomaly detection
- **Risk Management**: Dynamic position sizing, stop-loss, portfolio limits
- **Real-Time Monitoring**: System health, performance metrics, alerts
- **Compliance**: Regulatory checks and reporting

### ✅ Advanced Features (NEW!)
- **Correlation Engine**: Cross-market correlation analysis
- **Predictive Surface**: Multi-dimensional market surface mapping
- **Opportunity Evaluator**: AI-powered opportunity detection and scoring
- **Performance Tracker**: Comprehensive trading metrics (Sharpe, Sortino, etc.)
- **Alert System**: Multi-level real-time alerting
- **Compliance Checker**: Pattern day trader, position limits, leverage checks
- **Data Compression**: Efficient storage and transmission
- **Distributed Lock**: Thread and process-safe locking

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_advanced_features.py -v

# Run with coverage
pytest tests/ --cov=quantum_market_domination
```

## Configuration

Key environment variables in `.env`:

```bash
# Exchange API Keys
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret

# Risk Parameters
MAX_SINGLE_TRADE_RISK=0.05  # 5%
TOTAL_PORTFOLIO_RISK=0.15    # 15%
STOP_LOSS_THRESHOLD=0.10     # 10%

# Environment
ENVIRONMENT=development  # development, staging, production
```

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f
```

## Next Steps

1. **Configure Credentials**: Edit `.env` with your exchange API keys
2. **Review Documentation**: See `README.md` for detailed features
3. **Deploy**: See `DEPLOYMENT.md` for production deployment
4. **Monitor**: Check logs and performance metrics

## Common Commands

```bash
# Development
python main.py                    # Run trading system
python defi_main.py              # Run DeFi system
python validate_readme.py        # Validate implementation

# Testing
pytest tests/                    # Run all tests
pytest tests/ -k "test_name"     # Run specific test

# Deployment
./scripts/deploy.sh              # Deploy to production
docker-compose up -d             # Start with Docker
```

## Troubleshooting

**Issue**: Module not found
```bash
pip install -r requirements.txt
```

**Issue**: Permission denied
```bash
chmod +x scripts/*.sh
```

**Issue**: Tests failing
```bash
# Check Python version
python --version  # Should be 3.9+

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## Support

- **Documentation**: `README.md`, `BUILD_AND_TEST.md`, `DEPLOYMENT.md`
- **Tests**: 71 comprehensive tests
- **Issues**: GitHub Issues

## Safety Notice

⚠️ **Important**: This is a trading system that can lose money. Always:
- Start with paper trading
- Test thoroughly in development
- Never risk more than you can afford to lose
- Monitor your positions constantly
- Have proper risk management in place

## License

Proprietary - All Rights Reserved
