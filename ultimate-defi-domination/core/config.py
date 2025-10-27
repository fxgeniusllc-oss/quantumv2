"""
DEFI DOMINANCE CONFIGURATION
Configuration for multi-chain DeFi operations
"""

import os
from dotenv import load_dotenv


class DominanceConfig:
    """DeFi trading configuration"""
    
    def __init__(self):
        load_dotenv()

    # Blockchain Configurations
    CHAINS = {
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
            'max_gas_price': 10_000_000_000
        }
    }

    # Liquidity Pool Targets
    LIQUIDITY_POOLS = {
        'uniswap_v3': {
            'address': '0x1f98431c8ad98523631ae4a59f267346ea31f984',
            'min_liquidity': 1_000_000,
            'fee_tier': 0.01
        },
        'curve_3pool': {
            'address': '0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7',
            'min_liquidity': 10_000_000,
            'fee_tier': 0.001
        },
        'pancakeswap': {
            'address': '0x10ED43C718714eb63d5aA57B78B54704E256024E',
            'min_liquidity': 1_000_000,
            'fee_tier': 0.0025
        }
    }

    # Flashloan Configuration
    FLASHLOAN_CONFIGS = {
        'aave_v3': {
            'pool_address': '0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2',
            'max_loan_size': 10_000_000,  # USDC
            'supported_assets': ['USDC', 'USDT', 'DAI', 'WETH', 'WBTC']
        },
        'dydx': {
            'pool_address': '0x1E0447b19BB6EcFdAe1e4AE1694b0C3659614e4e',
            'max_loan_size': 5_000_000
        }
    }

    # Risk Parameters
    RISK_PARAMS = {
        'max_single_trade_risk': float(os.getenv('MAX_SINGLE_TRADE_RISK', '0.05')),
        'total_portfolio_risk': float(os.getenv('TOTAL_PORTFOLIO_RISK', '0.15')),
        'stop_loss_threshold': float(os.getenv('STOP_LOSS_THRESHOLD', '0.10'))
    }

    @staticmethod
    def get_private_key():
        """Get private key from environment"""
        return os.getenv('PRIVATE_KEY', '')

    @staticmethod
    def get_flashbots_auth_key():
        """Get Flashbots auth key"""
        return os.getenv('FLASHBOTS_AUTH_KEY', '')
