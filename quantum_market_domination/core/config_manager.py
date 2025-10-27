"""
QUANTUM CONFIGURATION MANAGEMENT SYSTEM
Handles secure, dynamic configuration across the entire trading ecosystem
"""

from .secret_vault import SecretVault
import os
from dotenv import load_dotenv


class QuantumConfigManager:
    """
    QUANTUM CONFIGURATION MANAGEMENT SYSTEM
    Handles secure, dynamic configuration across the entire trading ecosystem
    """
    
    MODULES = {
        'DATA_ACQUISITION': {
            'WEBSOCKET_TIMEOUT': 500,  # ms
            'MAX_RECONNECT_ATTEMPTS': 5,
            'COMPRESSION_LEVEL': 9  # Maximum compression
        },
        'INTELLIGENCE': {
            'ML_MODEL_REFRESH_INTERVAL': 3600,  # 1 hour
            'CORRELATION_DEPTH': 500,  # Historical data points
            'ANOMALY_SENSITIVITY': 0.95  # 95% confidence interval
        },
        'EXECUTION': {
            'MAX_CONCURRENT_TRADES': 10,
            'GLOBAL_RISK_LIMIT': 0.05,  # 5% total portfolio risk
            'LIQUIDATION_THRESHOLD': 0.02  # 2% drawdown trigger
        }
    }

    def __init__(self):
        load_dotenv()
        self.secret_vault = SecretVault()
        self.environment = self._detect_environment()

    def _detect_environment(self):
        """
        Dynamically detect and configure environment
        Supports: 
        - Development
        - Staging
        - Production
        - High-Frequency Trading Mode
        """
        env = os.getenv('ENVIRONMENT', 'development').lower()
        
        valid_environments = ['development', 'staging', 'production', 'hft']
        
        if env not in valid_environments:
            print(f"Warning: Unknown environment '{env}', defaulting to 'development'")
            env = 'development'
        
        return env

    def get_exchange_credentials(self, exchange_name):
        """
        Securely retrieve exchange credentials
        
        Args:
            exchange_name: Name of the exchange (e.g., 'binance', 'bybit')
            
        Returns:
            Dictionary with API credentials
        """
        # Try to get from vault first
        try:
            return self.secret_vault.get_credentials(exchange_name)
        except ValueError:
            # Fall back to environment variables
            api_key = os.getenv(f'{exchange_name.upper()}_API_KEY')
            secret = os.getenv(f'{exchange_name.upper()}_SECRET')
            
            if api_key and secret:
                return {
                    'exchange': exchange_name,
                    'api_key': api_key,
                    'secret': secret
                }
            
            raise ValueError(f"No credentials found for {exchange_name}")

    def get_module_config(self, module_name):
        """
        Retrieve specific module configuration
        
        Args:
            module_name: Name of the module (e.g., 'DATA_ACQUISITION')
            
        Returns:
            Dictionary with module configuration
        """
        return self.MODULES.get(module_name, {})

    def get_risk_parameters(self):
        """
        Retrieve risk management parameters
        
        Returns:
            Dictionary with risk parameters
        """
        return {
            'max_single_trade_risk': float(os.getenv('MAX_SINGLE_TRADE_RISK', '0.05')),
            'total_portfolio_risk': float(os.getenv('TOTAL_PORTFOLIO_RISK', '0.15')),
            'stop_loss_threshold': float(os.getenv('STOP_LOSS_THRESHOLD', '0.10'))
        }

    def get_blockchain_config(self, chain_name):
        """
        Retrieve blockchain configuration
        
        Args:
            chain_name: Name of the blockchain (e.g., 'ethereum', 'polygon')
            
        Returns:
            Dictionary with blockchain configuration
        """
        chain_configs = {
            'ethereum': {
                'rpc_url': os.getenv('ETH_RPC_URL', 'https://mainnet.infura.io/v3/'),
                'chain_id': 1,
                'max_gas_price': 500_000_000_000  # 500 gwei
            },
            'polygon': {
                'rpc_url': os.getenv('POLYGON_RPC_URL', 'https://polygon-rpc.com'),
                'chain_id': 137,
                'max_gas_price': 200_000_000_000
            },
            'bsc': {
                'rpc_url': os.getenv('BSC_RPC_URL', 'https://bsc-dataseed.binance.org'),
                'chain_id': 56,
                'max_gas_price': 10_000_000_000  # 10 gwei
            }
        }
        
        return chain_configs.get(chain_name.lower(), {})

    def is_production(self):
        """Check if running in production environment"""
        return self.environment == 'production'

    def is_hft_mode(self):
        """Check if running in high-frequency trading mode"""
        return self.environment == 'hft'

    def get_alert_threshold(self):
        """Get system alert threshold"""
        return float(os.getenv('ALERT_THRESHOLD', '0.85'))

    def get_log_level(self):
        """Get logging level"""
        return os.getenv('LOG_LEVEL', 'INFO')
