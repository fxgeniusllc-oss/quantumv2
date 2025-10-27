"""
Tests for Core Configuration Manager
"""

import pytest
import os
from quantum_market_domination.core.config_manager import QuantumConfigManager
from quantum_market_domination.core.secret_vault import SecretVault


class TestQuantumConfigManager:
    """Test configuration management"""
    
    def test_initialization(self):
        """Test config manager initialization"""
        config = QuantumConfigManager()
        assert config is not None
        assert hasattr(config, 'MODULES')
        assert hasattr(config, 'environment')

    def test_environment_detection(self):
        """Test environment detection"""
        config = QuantumConfigManager()
        assert config.environment in ['development', 'staging', 'production', 'hft']

    def test_module_config_retrieval(self):
        """Test retrieving module configuration"""
        config = QuantumConfigManager()
        
        data_config = config.get_module_config('DATA_ACQUISITION')
        assert 'WEBSOCKET_TIMEOUT' in data_config
        assert data_config['WEBSOCKET_TIMEOUT'] == 500
        
        intel_config = config.get_module_config('INTELLIGENCE')
        assert 'ML_MODEL_REFRESH_INTERVAL' in intel_config
        assert intel_config['ML_MODEL_REFRESH_INTERVAL'] == 3600

    def test_risk_parameters(self):
        """Test risk parameter retrieval"""
        config = QuantumConfigManager()
        risk_params = config.get_risk_parameters()
        
        assert 'max_single_trade_risk' in risk_params
        assert 'total_portfolio_risk' in risk_params
        assert 'stop_loss_threshold' in risk_params
        
        assert 0 < risk_params['max_single_trade_risk'] <= 1
        assert 0 < risk_params['total_portfolio_risk'] <= 1

    def test_blockchain_config(self):
        """Test blockchain configuration"""
        config = QuantumConfigManager()
        
        eth_config = config.get_blockchain_config('ethereum')
        assert 'rpc_url' in eth_config
        assert 'chain_id' in eth_config
        assert eth_config['chain_id'] == 1
        
        polygon_config = config.get_blockchain_config('polygon')
        assert polygon_config['chain_id'] == 137


class TestSecretVault:
    """Test secret vault functionality"""
    
    def test_vault_initialization(self):
        """Test vault initialization"""
        vault = SecretVault(vault_path='secrets/test_vault.encrypted')
        assert vault is not None
        assert hasattr(vault, '_cipher_suite')

    def test_store_and_retrieve_credentials(self):
        """Test storing and retrieving credentials"""
        vault = SecretVault(vault_path='secrets/test_vault.encrypted')
        
        test_credentials = {
            'exchange': 'test_exchange',
            'api_key': 'test_key_123',
            'secret': 'test_secret_456'
        }
        
        vault.store_credentials('test_exchange', test_credentials)
        retrieved = vault.get_credentials('test_exchange')
        
        assert retrieved['exchange'] == 'test_exchange'
        assert retrieved['api_key'] == 'test_key_123'
        assert retrieved['secret'] == 'test_secret_456'

    def test_list_exchanges(self):
        """Test listing exchanges"""
        vault = SecretVault(vault_path='secrets/test_vault.encrypted')
        exchanges = vault.list_exchanges()
        assert isinstance(exchanges, list)

    def test_invalid_exchange_retrieval(self):
        """Test retrieving non-existent exchange"""
        vault = SecretVault(vault_path='secrets/test_vault.encrypted')
        
        with pytest.raises(ValueError):
            vault.get_credentials('nonexistent_exchange')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
