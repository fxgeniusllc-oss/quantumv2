"""
Integration tests for new modules
Tests the interaction between different system components
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quantum_market_domination.data_acquisition.websocket_manager import WebSocketManager, WebSocketConfig
from quantum_market_domination.data_acquisition.data_preprocessor import DataPreprocessor
from quantum_market_domination.intelligence.correlation_engine import CorrelationEngine
from quantum_market_domination.intelligence.predictive_surface import PredictiveSurface
from quantum_market_domination.intelligence.opportunity_evaluator import OpportunityEvaluator, OpportunityType
from quantum_market_domination.execution.position_sizer import PositionSizer
from quantum_market_domination.monitoring.performance_tracker import PerformanceTracker
from quantum_market_domination.monitoring.alert_system import AlertSystem, AlertSeverity
from quantum_market_domination.monitoring.compliance_checker import ComplianceChecker
from quantum_market_domination.utils.compression import CompressionEngine
from quantum_market_domination.utils.distributed_lock import LockManager


class TestDataPreprocessor:
    """Test data preprocessor functionality"""
    
    def test_preprocessor_initialization(self):
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
        assert preprocessor.sma_windows == [5, 10, 20, 50, 200]
        
    def test_clean_ohlcv_data(self):
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        
        preprocessor = DataPreprocessor()
        cleaned = preprocessor.clean_ohlcv_data(data)
        
        assert len(cleaned) > 0
        assert 'close' in cleaned.columns
        
    def test_calculate_technical_indicators(self):
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        data = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        
        preprocessor = DataPreprocessor()
        with_indicators = preprocessor.calculate_technical_indicators(data)
        
        # Check key indicators exist
        assert 'sma_20' in with_indicators.columns
        assert 'rsi' in with_indicators.columns
        assert 'bb_upper' in with_indicators.columns
        assert 'volatility_10' in with_indicators.columns


class TestCorrelationEngine:
    """Test correlation engine functionality"""
    
    def test_correlation_engine_initialization(self):
        engine = CorrelationEngine()
        assert engine is not None
        assert engine.correlation_depth == 500
        
    def test_add_price_data(self):
        engine = CorrelationEngine()
        
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        prices = pd.Series(np.random.uniform(100, 110, 100), index=dates)
        
        engine.add_price_data('BTC/USDT', prices)
        assert 'BTC/USDT' in engine.price_history
        
    def test_calculate_correlation(self):
        engine = CorrelationEngine()
        
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        prices1 = pd.Series(np.random.uniform(100, 110, 100), index=dates)
        prices2 = pd.Series(np.random.uniform(200, 220, 100), index=dates)
        
        engine.add_price_data('BTC/USDT', prices1)
        engine.add_price_data('ETH/USDT', prices2)
        
        corr = engine.get_correlation('BTC/USDT', 'ETH/USDT')
        assert isinstance(corr, float)
        assert -1 <= corr <= 1


class TestOpportunityEvaluator:
    """Test opportunity evaluator"""
    
    def test_evaluator_initialization(self):
        evaluator = OpportunityEvaluator()
        assert evaluator is not None
        assert evaluator.min_score_threshold == 0.6
        
    def test_evaluate_trend_opportunity(self):
        evaluator = OpportunityEvaluator()
        
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        price_data = pd.DataFrame({
            'timestamp': dates,
            'close': np.linspace(100, 110, 100),  # Uptrend
            'high': np.linspace(102, 112, 100),
            'low': np.linspace(98, 108, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        
        indicators = {
            'sma_20': 105,
            'sma_50': 103,
            'sma_200': 100,
            'atr': 2.0,
            'momentum_5': 2.0,
            'momentum_10': 3.0,
            'volume_ratio': 1.5
        }
        
        opportunity = evaluator.evaluate_trend_opportunity('BTC/USDT', price_data, indicators)
        
        if opportunity:
            assert opportunity.symbol == 'BTC/USDT'
            assert opportunity.opportunity_type == OpportunityType.TREND_FOLLOWING
            assert 0 <= opportunity.score <= 1


class TestPositionSizer:
    """Test position sizer"""
    
    def test_position_sizer_initialization(self):
        sizer = PositionSizer()
        assert sizer is not None
        assert sizer.max_single_trade_risk == 0.05
        
    def test_calculate_position_size_fixed(self):
        sizer = PositionSizer()
        
        result = sizer.calculate_position_size(
            capital=100000,
            entry_price=100,
            stop_loss=95,
            method='fixed'
        )
        
        assert 'shares' in result
        assert 'risk_amount' in result
        assert result['shares'] > 0
        
    def test_calculate_position_size_kelly(self):
        sizer = PositionSizer()
        
        result = sizer.calculate_position_size(
            capital=100000,
            entry_price=100,
            stop_loss=95,
            method='kelly',
            win_rate=0.6,
            win_loss_ratio=2.0
        )
        
        assert 'shares' in result
        assert result['method'] == 'kelly_criterion'


class TestPerformanceTracker:
    """Test performance tracker"""
    
    def test_tracker_initialization(self):
        tracker = PerformanceTracker({'initial_capital': 100000})
        assert tracker is not None
        assert tracker.initial_capital == 100000
        
    def test_record_trade(self):
        tracker = PerformanceTracker({'initial_capital': 100000})
        
        # Record entry
        trade = tracker.record_trade_entry(
            trade_id='TEST-001',
            symbol='BTC/USDT',
            side='buy',
            entry_price=100,
            shares=10,
            strategy='test'
        )
        
        assert trade is not None
        assert trade.trade_id == 'TEST-001'
        
        # Record exit
        completed = tracker.record_trade_exit('TEST-001', exit_price=105, fees=1.0)
        
        assert completed is not None
        assert completed.pnl > 0
        
    def test_get_performance_metrics(self):
        tracker = PerformanceTracker({'initial_capital': 100000})
        
        # Create some trades
        tracker.record_trade_entry('T1', 'BTC/USDT', 'buy', 100, 10)
        tracker.record_trade_exit('T1', 105)
        
        tracker.record_trade_entry('T2', 'ETH/USDT', 'buy', 200, 5)
        tracker.record_trade_exit('T2', 195)
        
        metrics = tracker.get_performance_metrics()
        
        assert 'total_trades' in metrics
        assert 'win_rate' in metrics
        assert metrics['total_trades'] == 2


class TestAlertSystem:
    """Test alert system"""
    
    def test_alert_system_initialization(self):
        alert_sys = AlertSystem()
        assert alert_sys is not None
        
    def test_send_alert(self):
        alert_sys = AlertSystem()
        
        alert = alert_sys.send_alert(
            severity=AlertSeverity.WARNING,
            category='test',
            message='Test alert',
            details={'key': 'value'}
        )
        
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == 'Test alert'
        
    def test_get_recent_alerts(self):
        alert_sys = AlertSystem()
        
        for i in range(5):
            alert_sys.send_alert(
                severity=AlertSeverity.INFO,
                category='test',
                message=f'Alert {i}'
            )
            
        recent = alert_sys.get_recent_alerts(n=3)
        assert len(recent) == 3


class TestComplianceChecker:
    """Test compliance checker"""
    
    def test_compliance_checker_initialization(self):
        checker = ComplianceChecker()
        assert checker is not None
        
    def test_check_position_compliance(self):
        checker = ComplianceChecker()
        
        result = checker.check_position_compliance(
            symbol='BTC/USDT',
            position_value=50000,
            total_portfolio_value=200000,
            open_positions=5
        )
        
        assert 'compliant' in result
        assert isinstance(result['compliant'], bool)
        
    def test_check_trade_frequency_compliance(self):
        checker = ComplianceChecker()
        
        now = datetime.now()
        trades = [
            {'timestamp': now - timedelta(seconds=30)},
            {'timestamp': now - timedelta(seconds=20)},
            {'timestamp': now - timedelta(seconds=10)}
        ]
        
        result = checker.check_trade_frequency_compliance(trades)
        
        assert 'compliant' in result
        assert 'trade_counts' in result


class TestCompressionEngine:
    """Test compression engine"""
    
    def test_compression_engine_initialization(self):
        engine = CompressionEngine()
        assert engine is not None
        
    def test_compress_decompress_zlib(self):
        engine = CompressionEngine()
        
        data = {'test': 'data', 'numbers': [1, 2, 3, 4, 5]}
        
        compressed = engine.compress(data, algorithm='zlib')
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
        
        decompressed = engine.decompress(compressed, algorithm='zlib', as_json=True)
        assert decompressed == data
        
    def test_benchmark_algorithms(self):
        engine = CompressionEngine()
        
        data = "Test data " * 100
        results = engine.benchmark_algorithms(data)
        
        assert 'zlib' in results
        assert 'gzip' in results
        assert results['zlib']['roundtrip_correct']


class TestLockManager:
    """Test distributed lock manager"""
    
    def test_lock_manager_initialization(self):
        manager = LockManager()
        assert manager is not None
        
    def test_get_lock(self):
        manager = LockManager()
        
        lock = manager.get_lock('test_lock')
        assert lock is not None
        assert lock.lock_name == 'test_lock'
        
    def test_lock_acquisition(self):
        manager = LockManager()
        
        lock = manager.get_lock('test_lock', timeout=10)
        
        # Acquire lock
        acquired = lock.acquire(blocking=False)
        assert acquired
        
        # Release lock
        lock.release()
        assert not lock.acquired


class TestIntegrationWorkflow:
    """Test integration of multiple modules"""
    
    def test_data_to_opportunity_pipeline(self):
        """Test pipeline from data preprocessing to opportunity evaluation"""
        
        # Step 1: Preprocess data
        preprocessor = DataPreprocessor()
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        raw_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        
        cleaned_data = preprocessor.clean_ohlcv_data(raw_data)
        data_with_indicators = preprocessor.calculate_technical_indicators(cleaned_data)
        
        # Step 2: Evaluate opportunities
        evaluator = OpportunityEvaluator()
        
        indicators = {
            'sma_20': data_with_indicators['sma_20'].iloc[-1] if 'sma_20' in data_with_indicators.columns else 105,
            'sma_50': 103,
            'atr': 2.0,
            'momentum_5': 1.0,
            'momentum_10': 2.0,
            'volume_ratio': 1.5
        }
        
        opportunity = evaluator.evaluate_trend_opportunity('BTC/USDT', data_with_indicators, indicators)
        
        # Step 3: Calculate position size if opportunity found
        if opportunity:
            sizer = PositionSizer()
            position = sizer.calculate_position_size(
                capital=100000,
                entry_price=opportunity.entry_price,
                stop_loss=opportunity.stop_loss
            )
            
            assert position['shares'] > 0
            
    def test_compliance_and_alerting_integration(self):
        """Test compliance checking with alerting"""
        
        checker = ComplianceChecker()
        alert_sys = AlertSystem()
        
        # Check position compliance
        result = checker.check_position_compliance(
            symbol='BTC/USDT',
            position_value=150000,  # Over limit
            total_portfolio_value=200000,
            open_positions=5
        )
        
        # Send alert if not compliant
        if not result['compliant']:
            alert_sys.alert_risk_breach(
                risk_type='position_limit',
                current=150000,
                limit=100000
            )
            
        # Verify alert was sent
        critical_alerts = alert_sys.get_critical_alerts()
        assert len(critical_alerts) > 0
        
    def test_performance_tracking_workflow(self):
        """Test complete trade workflow with performance tracking"""
        
        tracker = PerformanceTracker({'initial_capital': 100000})
        alert_sys = AlertSystem()
        
        # Execute some trades
        for i in range(3):
            trade_id = f'TRADE-{i:03d}'
            
            # Enter trade
            tracker.record_trade_entry(
                trade_id=trade_id,
                symbol='BTC/USDT',
                side='buy',
                entry_price=100 + i,
                shares=10,
                strategy='test'
            )
            
            # Exit trade with profit
            tracker.record_trade_exit(trade_id, exit_price=105 + i, fees=0.5)
            
        # Get performance metrics
        metrics = tracker.get_performance_metrics()
        
        assert metrics['total_trades'] == 3
        assert metrics['win_rate'] > 0
        
        # Alert on performance milestone
        if metrics['total_return_pct'] > 5:
            alert_sys.alert_performance_milestone('5% return', metrics['total_return_pct'])
