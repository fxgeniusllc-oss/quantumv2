# Installation and Deployment Guide

Complete guide for installing and deploying the Quantum Market Domination System across all platforms.

## Table of Contents

- [Overview](#overview)
- [Windows Installation](#windows-installation)
- [Linux/Mac Installation](#linux-mac-installation)
- [Deployment Options](#deployment-options)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Overview

This project provides automated installation and deployment scripts for both Windows and Linux/Mac systems. All scripts now use **Yarn** for Node.js dependency management instead of npm for better performance and reliability.

### Available Scripts

| Script | Platform | Purpose |
|--------|----------|---------|
| `install-and-deploy.bat` | Windows | Full system installation and deployment |
| `setup.sh` | Linux/Mac | Development environment setup |
| `deploy.sh` | Linux/Mac | Docker-based deployment |
| `scripts/deploy.sh` | Linux/Mac | Production deployment with systemd |

## Windows Installation

### Prerequisites

Before running the installation script, ensure you have:

- **Python 3.9+** - [Download](https://www.python.org/downloads/)
- **Node.js 16+** - [Download](https://nodejs.org/)
- **Git** - [Download](https://git-scm.com/download/win)
- Administrator privileges (recommended)

### Quick Start

1. **Clone the repository:**
   ```batch
   git clone https://github.com/fxgeniusllc-oss/quantumv2.git
   cd quantumv2
   ```

2. **Run the installation script:**
   ```batch
   install-and-deploy.bat
   ```

   Or for production environment:
   ```batch
   install-and-deploy.bat production
   ```

### What the Script Does

The `install-and-deploy.bat` script performs the following steps:

1. **Checks Prerequisites**
   - Verifies Python, Node.js, and Git installations
   - Installs Yarn globally if not present
   - Validates minimum version requirements

2. **Sets Up Python Environment**
   - Creates Python virtual environment
   - Upgrades pip to latest version
   - Installs all Python dependencies from `requirements.txt`

3. **Installs Node.js Dependencies**
   - Uses Yarn to install all Node.js packages
   - Handles all Node.js dependencies more efficiently than npm

4. **Creates Directory Structure**
   - `secrets/` - For encrypted credentials
   - `logs/` - For application logs
   - `data/` - For market data storage
   - `backups/` - For configuration backups

5. **Configuration Setup**
   - Copies `.env.example` to `.env`
   - Loads environment-specific settings if available

6. **Runs Tests**
   - Executes full test suite
   - Validates system integrity

7. **Builds Components**
   - Runs Yarn build if build script exists
   - Prepares system for deployment

### After Installation

After successful installation:

1. **Edit Configuration:**
   ```batch
   notepad .env
   ```
   Add your API keys and customize settings.

2. **Activate Virtual Environment:**
   ```batch
   venv\Scripts\activate.bat
   ```

3. **Run the System:**
   ```batch
   python main.py
   ```

## Linux/Mac Installation

### Prerequisites

- **Python 3.9+** - Usually pre-installed
- **Node.js 16+** - Install via package manager or [nvm](https://github.com/nvm-sh/nvm)
- **Git** - Usually pre-installed
- Bash shell

### Quick Start with Automated Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fxgeniusllc-oss/quantumv2.git
   cd quantumv2
   ```

2. **Run the setup script:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

### What the Setup Script Does

The `setup.sh` script performs:

1. **Environment Validation**
   - Checks Python and Node.js versions
   - Warns if Node.js is not found (optional)

2. **Virtual Environment Setup**
   - Creates Python virtual environment
   - Activates the environment
   - Upgrades pip

3. **Dependency Installation**
   - Installs Yarn globally if not present
   - Installs Python dependencies
   - Installs Node.js dependencies with Yarn

4. **Directory and Configuration Setup**
   - Creates necessary directories
   - Copies `.env.example` to `.env`

5. **Git Hooks**
   - Sets up pre-commit hooks
   - Ensures tests run before commits

6. **Testing**
   - Runs initial test suite
   - Validates installation

### After Setup

1. **Edit Configuration:**
   ```bash
   nano .env
   # or
   vim .env
   ```

2. **Activate Virtual Environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Run the System:**
   ```bash
   python main.py
   ```

## Deployment Options

### Option 1: Docker Deployment (Recommended for Production)

Use the Docker-based deployment script:

```bash
# Development environment
./deploy.sh development latest

# Staging environment
./deploy.sh staging v1.0.0

# Production environment
./deploy.sh production v1.0.0
```

**Features:**
- Isolated containerized environment
- Easy scaling and orchestration
- Automatic health checks
- Rolling updates support

**What it does:**
- Validates environment selection
- Loads environment-specific configuration
- Builds Docker images with no cache
- Stops existing containers
- Starts new containers
- Runs health checks
- Displays service status

### Option 2: Systemd Deployment (Linux Production)

For production Linux servers with systemd:

```bash
sudo ./scripts/deploy.sh
```

**Features:**
- Automatic service restart on failure
- System boot integration
- Log management via journalctl
- Resource control and monitoring

**What it does:**
- Creates backup of existing deployment
- Installs system dependencies
- Installs Yarn if not present
- Installs Python and Node.js dependencies with Yarn
- Sets up environment and directories
- Runs full test suite
- Creates systemd service
- Starts and enables service

**After deployment:**
```bash
# View logs
journalctl -u quantum-trading -f

# Check status
systemctl status quantum-trading

# Restart service
sudo systemctl restart quantum-trading

# Stop service
sudo systemctl stop quantum-trading
```

### Option 3: Direct Execution

For development or testing:

```bash
# Activate virtual environment
source venv/bin/activate

# Run main trading system
python main.py

# Run DeFi strategies
python defi_main.py
```

## Configuration

### Environment Variables

Create and edit `.env` file with your settings:

```bash
# Exchange API Keys
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET=your_binance_secret
BYBIT_API_KEY=your_bybit_api_key
BYBIT_SECRET=your_bybit_secret
# ... add other exchanges

# Blockchain RPC URLs (for DeFi features)
ETH_RPC_URL=https://mainnet.infura.io/v3/your_infura_key
POLYGON_RPC_URL=https://polygon-rpc.com

# Risk Parameters
MAX_SINGLE_TRADE_RISK=0.05        # 5% max per trade
TOTAL_PORTFOLIO_RISK=0.15         # 15% max total exposure
STOP_LOSS_THRESHOLD=0.10          # 10% stop-loss trigger
MAX_CONCURRENT_TRADES=10          # Maximum simultaneous trades

# System Configuration
ALERT_THRESHOLD=0.85              # 85% resource alert threshold
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
ENVIRONMENT=development           # development, staging, production
```

### Environment-Specific Configuration

Create environment-specific files:

- `.env.development` - Development settings
- `.env.staging` - Staging settings
- `.env.production` - Production settings

These will be automatically loaded by the deployment scripts.

## Yarn Commands

All Node.js operations use Yarn:

```bash
# Install dependencies
yarn install

# Add new dependency
yarn add package-name

# Add dev dependency
yarn add --dev package-name

# Remove dependency
yarn remove package-name

# Run scripts
yarn start        # Start Node.js engine
yarn test         # Run tests
yarn build        # Build project (if applicable)

# Check outdated packages
yarn outdated

# Upgrade packages
yarn upgrade

# Clean cache
yarn cache clean
```

## Troubleshooting

### Common Issues

#### Python Version Too Old
```bash
# Check Python version
python --version

# Install newer Python version
# Windows: Download from python.org
# Linux: Use package manager
# Mac: Use homebrew
```

#### Yarn Not Found
```bash
# Install Yarn globally
npm install -g yarn

# Verify installation
yarn --version
```

#### Permission Denied (Linux/Mac)
```bash
# Make script executable
chmod +x setup.sh
chmod +x deploy.sh

# Run with sudo if needed
sudo ./scripts/deploy.sh
```

#### Virtual Environment Issues
```bash
# Delete and recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Port Already in Use
```bash
# Find process using port
# Linux/Mac:
lsof -i :8080

# Windows:
netstat -ano | findstr :8080

# Kill the process or change port in configuration
```

#### Docker Issues
```bash
# Clean Docker system
docker system prune -a

# Rebuild images
docker-compose build --no-cache

# View container logs
docker-compose logs -f quantum-trader
```

#### Node.js Dependencies Fail to Install
```bash
# Clear Yarn cache
yarn cache clean

# Remove node_modules and reinstall
rm -rf node_modules
yarn install

# Try with verbose logging
yarn install --verbose
```

### Getting Help

1. **Check Logs:**
   - Windows: `type logs\output.log`
   - Linux/Mac: `tail -f logs/output.log`

2. **Run Tests:**
   ```bash
   pytest tests/ -v --tb=short
   ```

3. **Verify Installation:**
   ```bash
   python final_validation.py
   ```

4. **Check System Status:**
   ```bash
   # Docker deployment
   docker-compose ps

   # Systemd deployment
   systemctl status quantum-trading
   ```

## Best Practices

### For Development

1. Always work in a virtual environment
2. Use `.env.development` for local settings
3. Run tests frequently: `pytest tests/ -v`
4. Use `yarn install` for new dependencies
5. Keep dependencies up to date: `yarn upgrade`

### For Production

1. Use Docker or systemd deployment
2. Set `ENVIRONMENT=production` in `.env`
3. Use proper logging: `LOG_LEVEL=WARNING`
4. Enable monitoring and alerts
5. Create regular backups of configuration
6. Use environment-specific `.env.production` file
7. Review security settings in `.env`
8. Set proper file permissions on `.env` and `secrets/`

### Security

1. **Never commit `.env` files to Git**
2. **Protect API keys and secrets**
3. **Use proper file permissions:**
   ```bash
   chmod 600 .env
   chmod 600 secrets/*
   ```
4. **Regularly rotate API keys**
5. **Use read-only API keys where possible**
6. **Monitor for unusual activity**

## Next Steps

After successful installation and deployment:

1. **Configure API Keys** - Edit `.env` with your credentials
2. **Review Settings** - Customize risk parameters
3. **Run Tests** - Verify system functionality
4. **Start Trading** - Run `python main.py`
5. **Monitor System** - Check logs and alerts
6. **Read Documentation** - See `README.md` for detailed features

## Additional Resources

- **Main README**: Complete feature documentation
- **QUICKSTART.md**: 5-minute quick start guide
- **BUILD_AND_TEST.md**: Build and testing details
- **DEPLOYMENT.md**: Advanced deployment options
- **GitHub Issues**: Report bugs or request features

---

*Built with ⚡ by the Quantum Market Domination Team*
