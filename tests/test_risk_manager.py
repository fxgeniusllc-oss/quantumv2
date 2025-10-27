"""
Tests for Risk Manager
"""

import pytest
from quantum_market_domination.core.config_manager import QuantumConfigManager
from quantum_market_domination.execution.risk_manager import RiskManager, RiskLimits


class TestRiskManager:
    """Test risk management functionality"""
    
    @pytest.fixture
    def config(self):
        """Create config fixture"""
        return QuantumConfigManager()
    
    @pytest.fixture
    def risk_manager(self, config):
        """Create risk manager fixture"""
        return RiskManager(config, initial_capital=100000)

    def test_initialization(self, risk_manager):
        """Test risk manager initialization"""
        assert risk_manager is not None
        assert risk_manager.initial_capital == 100000
        assert risk_manager.current_capital == 100000
        assert isinstance(risk_manager.limits, RiskLimits)

    def test_position_size_calculation(self, risk_manager):
        """Test position size calculation"""
        position_size = risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            entry_price=50000,
            stop_loss_price=49000,
            confidence=1.0
        )
        
        assert position_size > 0
        # Risk should not exceed max single trade risk
        max_risk = risk_manager.current_capital * risk_manager.limits.max_single_trade_risk
        position_risk = position_size * (50000 - 49000)
        assert position_risk <= max_risk * 1.01  # Allow 1% tolerance

    def test_can_open_position(self, risk_manager):
        """Test position opening validation"""
        can_open = risk_manager.can_open_position(
            symbol='BTC/USDT',
            position_size=1.0,
            entry_price=50000
        )
        
        assert isinstance(can_open, bool)

    def test_open_position(self, risk_manager):
        """Test opening a position"""
        success = risk_manager.open_position(
            symbol='BTC/USDT',
            size=1.0,
            entry_price=50000,
            stop_loss=49000,
            take_profit=51000
        )
        
        assert success is True
        assert 'BTC/USDT' in risk_manager.open_positions
        
        position = risk_manager.open_positions['BTC/USDT']
        assert position['size'] == 1.0
        assert position['entry_price'] == 50000
        assert position['stop_loss'] == 49000

    def test_update_position(self, risk_manager):
        """Test updating position with current price"""
        risk_manager.open_position(
            symbol='BTC/USDT',
            size=1.0,
            entry_price=50000,
            stop_loss=49000
        )
        
        # Update with profitable price
        risk_manager.update_position('BTC/USDT', 50500)
        assert 'BTC/USDT' in risk_manager.open_positions
        
        position = risk_manager.open_positions['BTC/USDT']
        assert position['unrealized_pnl'] == 500

    def test_stop_loss_trigger(self, risk_manager):
        """Test stop loss triggering"""
        risk_manager.open_position(
            symbol='BTC/USDT',
            size=1.0,
            entry_price=50000,
            stop_loss=49000
        )
        
        # Trigger stop loss
        risk_manager.update_position('BTC/USDT', 48500)
        
        # Position should be closed
        assert 'BTC/USDT' not in risk_manager.open_positions
        assert len(risk_manager.closed_trades) == 1

    def test_close_position(self, risk_manager):
        """Test closing a position"""
        risk_manager.open_position(
            symbol='BTC/USDT',
            size=1.0,
            entry_price=50000,
            stop_loss=49000
        )
        
        initial_capital = risk_manager.current_capital
        
        risk_manager.close_position('BTC/USDT', 51000, reason='take_profit')
        
        assert 'BTC/USDT' not in risk_manager.open_positions
        assert risk_manager.current_capital > initial_capital
        assert len(risk_manager.closed_trades) == 1

    def test_performance_metrics(self, risk_manager):
        """Test performance metrics calculation"""
        # Execute some trades
        risk_manager.open_position('BTC/USDT', 1.0, 50000, 49000)
        risk_manager.close_position('BTC/USDT', 51000, 'manual')
        
        risk_manager.open_position('ETH/USDT', 10.0, 3000, 2900)
        risk_manager.close_position('ETH/USDT', 2950, 'stop_loss')
        
        metrics = risk_manager.get_performance_metrics()
        
        assert 'total_trades' in metrics
        assert 'winning_trades' in metrics
        assert 'win_rate' in metrics
        assert 'total_pnl' in metrics
        
        assert metrics['total_trades'] == 2
        assert 0 <= metrics['win_rate'] <= 1

    def test_concurrent_trades_limit(self, risk_manager):
        """Test maximum concurrent trades limit"""
        # Open positions up to limit
        for i in range(risk_manager.limits.max_concurrent_trades):
            risk_manager.open_position(
                symbol=f'SYMBOL{i}/USDT',
                size=0.1,
                entry_price=1000,
                stop_loss=950
            )
        
        # Try to open one more
        can_open = risk_manager.can_open_position(
            symbol='EXTRA/USDT',
            position_size=0.1,
            entry_price=1000
        )
        
        assert can_open is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
