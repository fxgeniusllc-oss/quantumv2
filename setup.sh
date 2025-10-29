#!/bin/bash
# System setup script for development environment

set -e

echo "========================================="
echo "Quantum Trading System Setup"
echo "========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required"
    exit 1
fi

# Check Node.js version
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo "Node.js version: $NODE_VERSION"
else
    echo "Warning: Node.js not found (optional for DeFi features)"
fi

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install Yarn if not present
if command -v node &> /dev/null; then
    if ! command -v yarn &> /dev/null; then
        echo "Installing Yarn..."
        npm install -g yarn
    fi
fi

# Install Node.js dependencies with Yarn
if command -v yarn &> /dev/null; then
    echo "Installing Node.js dependencies with Yarn..."
    yarn install
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p secrets logs data

# Copy environment template
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "Creating .env from template..."
        cp .env.example .env
        echo "Please edit .env with your configuration"
    else
        echo "Warning: .env.example not found"
    fi
fi

# Set up pre-commit hooks
echo "Setting up pre-commit hooks..."
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Run tests before commit
echo "Running tests..."
python -m pytest tests/ -v --tb=short
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
EOF
chmod +x .git/hooks/pre-commit

# Run initial tests
echo "Running initial tests..."
python -m pytest tests/ -v

echo "========================================="
echo "Setup completed successfully!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys and configuration"
echo "2. Run: python main.py (for Quantum Trading)"
echo "3. Run: python defi_main.py (for DeFi strategies)"
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
