"""
Tests for new advanced features
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Intelligence layer
from quantum_market_domination.intelligence.correlation_engine import CorrelationEngine
from quantum_market_domination.intelligence.predictive_surface import PredictiveSurface
from quantum_market_domination.intelligence.opportunity_evaluator import OpportunityEvaluator, OpportunityType

# Execution layer
from quantum_market_domination.execution.position_sizer import PositionSizer

# Monitoring layer
from quantum_market_domination.monitoring.performance_tracker import PerformanceTracker
from quantum_market_domination.monitoring.alert_system import AlertSystem, AlertLevel, AlertType
from quantum_market_domination.monitoring.compliance_checker import ComplianceChecker, ComplianceRule

# Utils
from quantum_market_domination.utils.compression import DataCompressor, CompressionMethod
from quantum_market_domination.utils.distributed_lock import DistributedLock, LockManager


class TestCorrelationEngine:
    """Test correlation engine"""
    
    def test_initialization(self):
        engine = CorrelationEngine(correlation_depth=100)
        assert engine.correlation_depth == 100
        assert len(engine.price_history) == 0
    
    def test_update_price(self):
        engine = CorrelationEngine()
        engine.update_price('BTC/USDT', 50000.0)
        assert 'BTC/USDT' in engine.price_history
        assert len(engine.price_history['BTC/USDT']) == 1
    
    def test_calculate_correlation_matrix(self):
        engine = CorrelationEngine()
        
        # Add correlated data
        for i in range(100):
            engine.update_price('BTC/USDT', 50000 + i * 100)
            engine.update_price('ETH/USDT', 3000 + i * 6)  # Correlated
        
        matrix = engine.calculate_correlation_matrix()
        assert not matrix.empty
        assert 'BTC/USDT' in matrix.columns
        assert 'ETH/USDT' in matrix.columns
    
    def test_get_highly_correlated_pairs(self):
        engine = CorrelationEngine()
        
        # Add correlated data
        for i in range(100):
            engine.update_price('BTC/USDT', 50000 + i * 100)
            engine.update_price('ETH/USDT', 3000 + i * 6)
        
        engine.calculate_correlation_matrix()
        pairs = engine.get_highly_correlated_pairs(threshold=0.5)
        
        assert isinstance(pairs, list)


class TestPredictiveSurface:
    """Test predictive surface mapper"""
    
    def test_initialization(self):
        surface = PredictiveSurface(horizon=200)
        assert surface.horizon == 200
        assert len(surface.surfaces) == 0
    
    def test_add_market_data(self):
        surface = PredictiveSurface()
        data = pd.DataFrame({
            'close': np.random.randn(100) + 50000,
            'volume': np.random.randn(100) + 1000
        })
        
        surface.add_market_data('BTC/USDT', data)
        assert 'BTC/USDT' in surface.historical_data
    
    def test_generate_price_volatility_surface(self):
        surface = PredictiveSurface()
        data = pd.DataFrame({
            'close': np.random.randn(100) + 50000,
            'volume': np.random.randn(100) + 1000
        })
        
        surface.add_market_data('BTC/USDT', data)
        vol_surface = surface.generate_price_volatility_surface('BTC/USDT')
        
        assert 'prices' in vol_surface
        assert 'returns' in vol_surface
    
    def test_predict_future_surface(self):
        surface = PredictiveSurface()
        data = pd.DataFrame({
            'close': np.linspace(50000, 51000, 100),
            'volume': np.random.randn(100) + 1000
        })
        
        surface.add_market_data('BTC/USDT', data)
        prediction = surface.predict_future_surface('BTC/USDT', steps_ahead=5)
        
        assert 'predicted_prices' in prediction
        assert len(prediction['predicted_prices']) == 5


class TestOpportunityEvaluator:
    """Test opportunity evaluator"""
    
    def setup_method(self):
        dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
        self.test_data = pd.DataFrame({
            'close': np.linspace(50000, 51000, 100),
            'high': np.linspace(50100, 51100, 100),
            'low': np.linspace(49900, 50900, 100),
            'volume': np.random.randn(100) * 100 + 1000
        }, index=dates)
    
    def test_initialization(self):
        evaluator = OpportunityEvaluator(min_score=0.5)
        assert evaluator.min_score == 0.5
    
    def test_evaluate_trend_opportunity(self):
        evaluator = OpportunityEvaluator(min_score=0.3)
        opportunity = evaluator.evaluate_trend_opportunity('BTC/USDT', self.test_data)
        
        assert 'valid' in opportunity
        assert 'type' in opportunity
        if opportunity['valid']:
            assert 'score' in opportunity
            assert 'direction' in opportunity
    
    def test_evaluate_breakout_opportunity(self):
        evaluator = OpportunityEvaluator(min_score=0.3)
        opportunity = evaluator.evaluate_breakout_opportunity('BTC/USDT', self.test_data)
        
        assert 'valid' in opportunity
        # Type may not be present if no breakout detected (which is valid)
    
    def test_evaluate_all_opportunities(self):
        evaluator = OpportunityEvaluator(min_score=0.3)
        opportunities = evaluator.evaluate_all_opportunities('BTC/USDT', self.test_data)
        
        assert isinstance(opportunities, list)


class TestPositionSizer:
    """Test position sizer"""
    
    def test_initialization(self):
        sizer = PositionSizer(max_position_pct=0.1, max_leverage=3.0)
        assert sizer.max_position_pct == 0.1
        assert sizer.max_leverage == 3.0
    
    def test_calculate_fixed_fractional(self):
        sizer = PositionSizer()
        position_size = sizer.calculate_fixed_fractional(10000, risk_per_trade=0.02)
        
        assert position_size > 0
        assert position_size <= 10000 * sizer.max_position_pct
    
    def test_calculate_kelly_criterion(self):
        sizer = PositionSizer()
        position_size = sizer.calculate_kelly_criterion(
            win_rate=0.6,
            avg_win=100,
            avg_loss=50,
            portfolio_value=10000
        )
        
        assert position_size >= 0
    
    def test_calculate_volatility_based(self):
        sizer = PositionSizer()
        position_units = sizer.calculate_volatility_based(
            portfolio_value=10000,
            volatility=0.20,
            target_volatility=0.15,
            price=50000
        )
        
        assert position_units >= 0
    
    def test_calculate_risk_based(self):
        sizer = PositionSizer()
        position_units = sizer.calculate_risk_based(
            portfolio_value=10000,
            entry_price=50000,
            stop_loss=49000,
            risk_amount=200
        )
        
        assert position_units >= 0
    
    def test_calculate_optimal_position(self):
        sizer = PositionSizer()
        position = sizer.calculate_optimal_position(
            portfolio_value=10000,
            entry_price=50000,
            stop_loss=49000,
            strategy='risk'
        )
        
        assert position['valid']
        assert position['units'] >= 0
        assert position['strategy'] == 'risk'


class TestPerformanceTracker:
    """Test performance tracker"""
    
    def test_initialization(self):
        tracker = PerformanceTracker()
        assert len(tracker.trades) == 0
    
    def test_record_trade(self):
        tracker = PerformanceTracker()
        trade = {
            'symbol': 'BTC/USDT',
            'entry_price': 50000,
            'exit_price': 51000,
            'quantity': 0.1,
            'pnl': 100
        }
        
        tracker.record_trade(trade)
        assert len(tracker.trades) == 1
    
    def test_get_win_rate(self):
        tracker = PerformanceTracker()
        
        # Add winning trade
        tracker.record_trade({
            'symbol': 'BTC/USDT',
            'entry_price': 50000,
            'exit_price': 51000,
            'quantity': 0.1,
            'pnl': 100
        })
        
        # Add losing trade
        tracker.record_trade({
            'symbol': 'ETH/USDT',
            'entry_price': 3000,
            'exit_price': 2900,
            'quantity': 1.0,
            'pnl': -100
        })
        
        win_rate = tracker.get_win_rate()
        assert win_rate == 0.5
    
    def test_get_performance_summary(self):
        tracker = PerformanceTracker()
        tracker.record_trade({
            'symbol': 'BTC/USDT',
            'entry_price': 50000,
            'exit_price': 51000,
            'quantity': 0.1,
            'pnl': 100
        })
        
        summary = tracker.get_performance_summary()
        assert 'total_trades' in summary
        assert 'win_rate' in summary
        assert summary['total_trades'] == 1


class TestAlertSystem:
    """Test alert system"""
    
    def test_initialization(self):
        alert_system = AlertSystem()
        assert len(alert_system.alerts) == 0
    
    def test_send_alert(self):
        alert_system = AlertSystem()
        alert_system.send_alert(
            AlertLevel.WARNING,
            AlertType.SYSTEM,
            "Test alert"
        )
        
        assert len(alert_system.alerts) == 1
        assert alert_system.alerts[0]['level'] == AlertLevel.WARNING.value
    
    def test_check_system_health(self):
        alert_system = AlertSystem()
        metrics = {
            'cpu_usage': 90.0,  # Above threshold
            'memory_usage': 80.0
        }
        
        alert_system.check_system_health(metrics)
        assert len(alert_system.alerts) >= 1
    
    def test_get_alert_summary(self):
        alert_system = AlertSystem()
        alert_system.send_alert(AlertLevel.INFO, AlertType.SYSTEM, "Test")
        
        summary = alert_system.get_alert_summary()
        assert summary['total_alerts'] == 1


class TestComplianceChecker:
    """Test compliance checker"""
    
    def test_initialization(self):
        checker = ComplianceChecker()
        assert len(checker.violations) == 0
    
    def test_check_position_limits(self):
        checker = ComplianceChecker()
        position = {
            'symbol': 'BTC/USDT',
            'value': 3000,  # 30% of portfolio
            'quantity': 0.1
        }
        
        result = checker.check_position_limits(position, portfolio_value=10000)
        assert 'status' in result
        assert result['status'] == 'violation'  # Exceeds 25% limit
    
    def test_check_leverage_limits(self):
        checker = ComplianceChecker()
        result = checker.check_leverage_limits(leverage=2.0)
        
        assert 'status' in result
        assert result['status'] == 'compliant'
    
    def test_record_trade(self):
        checker = ComplianceChecker()
        trade = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'value': 1000,
            'pnl': 100
        }
        
        checker.record_trade(trade)
        assert len(checker.trade_history) == 1
    
    def test_get_compliance_report(self):
        checker = ComplianceChecker()
        report = checker.get_compliance_report()
        
        assert 'total_violations' in report
        assert 'total_warnings' in report


class TestDataCompressor:
    """Test data compressor"""
    
    def test_initialization(self):
        compressor = DataCompressor()
        assert compressor.compression_level == 9
    
    def test_compress_decompress_string(self):
        compressor = DataCompressor()
        original = "Test data " * 100
        
        compressed = compressor.compress_string(original)
        decompressed = compressor.decompress_string(compressed)
        
        assert decompressed == original
        assert len(compressed) < len(original.encode('utf-8'))
    
    def test_compress_decompress_dict(self):
        compressor = DataCompressor()
        original = {
            'symbol': 'BTC/USDT',
            'prices': [50000, 50100, 50200],
            'volumes': [100, 110, 120]
        }
        
        compressed = compressor.compress_dict(original)
        decompressed = compressor.decompress_dict(compressed)
        
        assert decompressed == original
    
    def test_compare_methods(self):
        compressor = DataCompressor()
        data = "Test data " * 100
        
        results = compressor.compare_methods(data)
        
        assert CompressionMethod.ZLIB in results
        assert CompressionMethod.GZIP in results
        assert 'compression_ratio' in results[CompressionMethod.ZLIB]


class TestDistributedLock:
    """Test distributed lock"""
    
    def test_initialization(self):
        lock = DistributedLock('test_lock')
        assert lock.lock_name == 'test_lock'
        assert not lock._acquired
    
    def test_acquire_release(self):
        lock = DistributedLock('test_lock_1')
        
        acquired = lock.acquire(blocking=False)
        assert acquired
        assert lock._acquired
        
        lock.release()
        assert not lock._acquired
    
    def test_context_manager(self):
        lock = DistributedLock('test_lock_2')
        
        with lock:
            assert lock._acquired
        
        assert not lock._acquired
    
    def test_lock_manager(self):
        manager = LockManager()
        lock1 = manager.get_lock('test_lock_3')
        lock2 = manager.get_lock('test_lock_3')
        
        assert lock1 is lock2  # Same lock instance


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
