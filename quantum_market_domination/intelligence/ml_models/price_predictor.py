"""
ADVANCED PRICE PREDICTION MODEL
Uses Random Forest and feature engineering for price forecasting
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging


class PricePredictor:
    """
    Advanced price prediction using Random Forest
    """
    
    def __init__(self, n_estimators=500, max_depth=20):
        self.logger = logging.getLogger('PricePredictor')
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []

    def _extract_features(self, data):
        """
        Extract features from market data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Feature matrix
        """
        features = pd.DataFrame()
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        features['sma_5'] = data['close'].rolling(window=5).mean()
        features['sma_20'] = data['close'].rolling(window=20).mean()
        features['sma_50'] = data['close'].rolling(window=50).mean()
        
        # Volatility
        features['volatility_5'] = data['close'].rolling(window=5).std()
        features['volatility_20'] = data['close'].rolling(window=20).std()
        
        # Volume features
        features['volume'] = data['volume']
        features['volume_ma'] = data['volume'].rolling(window=20).mean()
        
        # Price momentum
        features['momentum_5'] = data['close'] - data['close'].shift(5)
        features['momentum_20'] = data['close'] - data['close'].shift(20)
        
        # RSI approximation
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        sma = data['close'].rolling(window=bb_window).mean()
        std = data['close'].rolling(window=bb_window).std()
        features['bb_upper'] = sma + (std * bb_std)
        features['bb_lower'] = sma - (std * bb_std)
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        
        return features

    def train(self, historical_data, target_horizon=1):
        """
        Train the price prediction model
        
        Args:
            historical_data: DataFrame with OHLCV data
            target_horizon: Number of periods ahead to predict
        """
        self.logger.info("Training price predictor...")
        
        # Extract features
        features = self._extract_features(historical_data)
        
        # Create target (future price)
        target = historical_data['close'].shift(-target_horizon)
        
        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        X = features[valid_idx]
        y = target[valid_idx]
        
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training score
        score = self.model.score(X_scaled, y)
        self.logger.info(f"Model trained with R² score: {score:.4f}")
        
        return score

    def predict(self, current_data):
        """
        Predict future price
        
        Args:
            current_data: DataFrame with recent OHLCV data
            
        Returns:
            Predicted price
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Extract features
        features = self._extract_features(current_data)
        
        # Get the most recent complete row
        X = features.iloc[[-1]]
        
        # Handle NaN values
        if X.isna().any().any():
            self.logger.warning("NaN values in features, using mean imputation")
            X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        
        return prediction

    def get_feature_importance(self):
        """
        Get feature importance scores
        
        Returns:
            Dictionary of feature names and importance scores
        """
        if not self.is_trained:
            return {}
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
