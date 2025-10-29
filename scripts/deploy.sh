#!/bin/bash
# Quantum Market Domination System - Deployment Script

set -e  # Exit on error

echo "==================================="
echo "Quantum Trading System Deployment"
echo "==================================="

# Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
APP_DIR="/opt/quantumv2"
BACKUP_DIR="/opt/quantumv2_backup"
LOG_FILE="/var/log/quantum-deploy.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    error "Please run as root or with sudo"
fi

# Backup current deployment
backup_deployment() {
    log "Creating backup of current deployment..."
    if [ -d "$APP_DIR" ]; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        BACKUP_PATH="${BACKUP_DIR}_${TIMESTAMP}"
        cp -r "$APP_DIR" "$BACKUP_PATH"
        log "Backup created at $BACKUP_PATH"
    else
        warning "No existing deployment found, skipping backup"
    fi
}

# Install dependencies
install_dependencies() {
    log "Installing system dependencies..."
    if ! apt-get update; then
        error "Failed to update package lists"
    fi
    
    if ! apt-get install -y python3 python3-pip python3-venv nodejs npm git; then
        error "Failed to install system dependencies"
    fi
    
    # Install Yarn globally if not present
    if ! command -v yarn &> /dev/null; then
        log "Installing Yarn..."
        npm install -g yarn
    fi
    
    # Clean up
    apt-get clean && rm -rf /var/lib/apt/lists/*
    
    log "Installing Python dependencies..."
    if ! pip3 install -r "$APP_DIR/requirements.txt"; then
        error "Failed to install Python dependencies"
    fi
    
    log "Installing Node.js dependencies with Yarn..."
    cd "$APP_DIR" && yarn install
}

# Setup environment
setup_environment() {
    log "Setting up environment..."
    
    # Create necessary directories
    mkdir -p /var/log/quantum-trading
    mkdir -p "$APP_DIR/data"
    mkdir -p "$APP_DIR/logs"
    
    # Copy environment file if not exists
    if [ ! -f "$APP_DIR/.env" ]; then
        if [ -f "$APP_DIR/.env.example" ]; then
            cp "$APP_DIR/.env.example" "$APP_DIR/.env"
            warning ".env file created from template - please configure it!"
        else
            error ".env.example not found"
        fi
    fi
    
    # Set permissions
    chmod 600 "$APP_DIR/.env"
    chmod 600 "$APP_DIR/secrets/"* 2>/dev/null || true
    
    log "Environment setup complete"
}

# Run tests
run_tests() {
    log "Running tests..."
    
    # Run tests in a subshell to avoid directory change side effects
    if (cd "$APP_DIR" && python3 -m pytest tests/ -v); then
        log "All tests passed"
    else
        error "Tests failed! Aborting deployment."
    fi
}

# Setup systemd service
setup_service() {
    log "Setting up systemd service..."
    
    cat > /etc/systemd/system/quantum-trading.service <<EOF
[Unit]
Description=Quantum Market Domination Trading System
After=network.target

[Service]
Type=simple
User=trading
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
ExecStart=$APP_DIR/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/quantum-trading/output.log
StandardError=append:/var/log/quantum-trading/error.log

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable quantum-trading
    log "Systemd service configured"
}

# Start service
start_service() {
    log "Starting quantum trading service..."
    systemctl start quantum-trading
    sleep 5
    
    if systemctl is-active --quiet quantum-trading; then
        log "${GREEN}Service started successfully${NC}"
        systemctl status quantum-trading
    else
        error "Service failed to start"
    fi
}

# Main deployment flow
main() {
    log "Starting deployment to $ENVIRONMENT environment"
    
    # Backup
    backup_deployment
    
    # Install dependencies
    install_dependencies
    
    # Setup
    setup_environment
    
    # Test
    run_tests
    
    # Configure service
    setup_service
    
    # Start
    start_service
    
    log "==================================="
    log "Deployment completed successfully!"
    log "==================================="
    log ""
    log "Next steps:"
    log "1. Configure .env file with your credentials"
    log "2. Review logs: journalctl -u quantum-trading -f"
    log "3. Monitor status: systemctl status quantum-trading"
}

# Run deployment
main "$@"
