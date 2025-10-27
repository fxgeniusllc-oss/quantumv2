"""
VOLATILITY PREDICTION MODEL
Advanced volatility forecasting using ensemble methods
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging


class VolatilityModel:
    """
    Advanced volatility prediction model
    """
    
    def __init__(self, n_estimators=300, max_depth=15):
        self.logger = logging.getLogger('VolatilityModel')
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def _calculate_volatility(self, data, window=20):
        """Calculate historical volatility"""
        returns = np.log(data['close'] / data['close'].shift(1))
        volatility = returns.rolling(window=window).std() * np.sqrt(window)
        return volatility

    def _extract_features(self, data):
        """Extract volatility-related features"""
        features = pd.DataFrame()
        
        # Historical volatilities at different windows
        features['vol_5'] = self._calculate_volatility(data, 5)
        features['vol_10'] = self._calculate_volatility(data, 10)
        features['vol_20'] = self._calculate_volatility(data, 20)
        features['vol_50'] = self._calculate_volatility(data, 50)
        
        # Price range features
        features['high_low_range'] = (data['high'] - data['low']) / data['close']
        features['close_open_range'] = abs(data['close'] - data['open']) / data['open']
        
        # Volume volatility
        features['volume_std'] = data['volume'].rolling(window=20).std()
        
        return features

    def train(self, historical_data, target_horizon=1):
        """Train volatility prediction model"""
        self.logger.info("Training volatility model...")
        
        features = self._extract_features(historical_data)
        target = self._calculate_volatility(historical_data, window=20).shift(-target_horizon)
        
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        X = features[valid_idx]
        y = target[valid_idx]
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        score = self.model.score(X_scaled, y)
        self.logger.info(f"Volatility model trained with R² score: {score:.4f}")
        
        return score

    def predict(self, current_data):
        """Predict future volatility"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        features = self._extract_features(current_data)
        X = features.iloc[[-1]].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)[0]
