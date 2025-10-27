"""
Tests for ML Models
"""

import pytest
import numpy as np
import pandas as pd
from quantum_market_domination.intelligence.ml_models.price_predictor import PricePredictor
from quantum_market_domination.intelligence.ml_models.volatility_model import VolatilityModel
from quantum_market_domination.intelligence.ml_models.anomaly_detector import AnomalyDetector


class TestPricePredictor:
    """Test price prediction model"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=200, freq='1H')
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(200).cumsum() + 50000,
            'high': np.random.randn(200).cumsum() + 50500,
            'low': np.random.randn(200).cumsum() + 49500,
            'close': np.random.randn(200).cumsum() + 50000,
            'volume': np.random.randint(100, 1000, 200)
        })
        
        return data
    
    def test_predictor_initialization(self):
        """Test predictor initialization"""
        predictor = PricePredictor(n_estimators=100, max_depth=10)
        assert predictor is not None
        assert predictor.is_trained is False

    def test_feature_extraction(self, sample_data):
        """Test feature extraction"""
        predictor = PricePredictor()
        features = predictor._extract_features(sample_data)
        
        assert not features.empty
        assert 'returns' in features.columns
        assert 'sma_5' in features.columns
        assert 'volatility_5' in features.columns

    def test_model_training(self, sample_data):
        """Test model training"""
        predictor = PricePredictor(n_estimators=50, max_depth=5)
        score = predictor.train(sample_data, target_horizon=1)
        
        assert predictor.is_trained is True
        assert isinstance(score, float)
        assert -1 <= score <= 1  # R² score range

    def test_prediction(self, sample_data):
        """Test price prediction"""
        predictor = PricePredictor(n_estimators=50, max_depth=5)
        predictor.train(sample_data)
        
        prediction = predictor.predict(sample_data)
        
        assert isinstance(prediction, (int, float, np.number))
        assert prediction > 0  # Price should be positive

    def test_feature_importance(self, sample_data):
        """Test feature importance retrieval"""
        predictor = PricePredictor(n_estimators=50, max_depth=5)
        predictor.train(sample_data)
        
        importance = predictor.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0


class TestVolatilityModel:
    """Test volatility prediction model"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=200, freq='1H')
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(200).cumsum() + 50000,
            'high': np.random.randn(200).cumsum() + 50500,
            'low': np.random.randn(200).cumsum() + 49500,
            'close': np.random.randn(200).cumsum() + 50000,
            'volume': np.random.randint(100, 1000, 200)
        })
        
        return data

    def test_volatility_model_initialization(self):
        """Test volatility model initialization"""
        model = VolatilityModel(n_estimators=100, max_depth=10)
        assert model is not None
        assert model.is_trained is False

    def test_volatility_calculation(self, sample_data):
        """Test volatility calculation"""
        model = VolatilityModel()
        volatility = model._calculate_volatility(sample_data, window=20)
        
        assert not volatility.empty
        assert volatility.dtype in [np.float64, np.float32]

    def test_model_training(self, sample_data):
        """Test volatility model training"""
        model = VolatilityModel(n_estimators=50, max_depth=5)
        score = model.train(sample_data)
        
        assert model.is_trained is True
        assert isinstance(score, float)

    def test_volatility_prediction(self, sample_data):
        """Test volatility prediction"""
        model = VolatilityModel(n_estimators=50, max_depth=5)
        model.train(sample_data)
        
        prediction = model.predict(sample_data)
        
        assert isinstance(prediction, (int, float, np.number))
        assert prediction >= 0  # Volatility should be non-negative


class TestAnomalyDetector:
    """Test anomaly detection"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data with some anomalies"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=200, freq='1H')
        
        close_prices = np.random.randn(200).cumsum() + 50000
        # Add some anomalies
        close_prices[50] = close_prices[49] * 1.1  # 10% spike
        close_prices[100] = close_prices[99] * 0.9  # 10% drop
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices * 0.999,
            'high': close_prices * 1.001,
            'low': close_prices * 0.998,
            'close': close_prices,
            'volume': np.random.randint(100, 1000, 200)
        })
        
        return data

    def test_anomaly_detector_initialization(self):
        """Test anomaly detector initialization"""
        detector = AnomalyDetector(contamination=0.05)
        assert detector is not None
        assert detector.is_trained is False

    def test_feature_extraction(self, sample_data):
        """Test feature extraction for anomalies"""
        detector = AnomalyDetector()
        features = detector._extract_features(sample_data)
        
        assert not features.empty
        assert 'price_zscore' in features.columns
        assert 'volume_zscore' in features.columns

    def test_model_training(self, sample_data):
        """Test anomaly detector training"""
        detector = AnomalyDetector(contamination=0.05)
        detector.train(sample_data)
        
        assert detector.is_trained is True

    def test_anomaly_detection(self, sample_data):
        """Test detecting anomalies"""
        detector = AnomalyDetector(contamination=0.05)
        detector.train(sample_data)
        
        prediction = detector.detect(sample_data)
        
        assert prediction in [-1, 1]  # -1 for anomaly, 1 for normal

    def test_anomaly_score(self, sample_data):
        """Test anomaly scoring"""
        detector = AnomalyDetector(contamination=0.05)
        detector.train(sample_data)
        
        score = detector.get_anomaly_score(sample_data)
        
        assert isinstance(score, (int, float, np.number))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
