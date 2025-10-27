"""
ANOMALY DETECTION SYSTEM
Identifies unusual market behavior and potential opportunities
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging


class AnomalyDetector:
    """
    Market anomaly detection using Isolation Forest
    """
    
    def __init__(self, contamination=0.05):
        self.logger = logging.getLogger('AnomalyDetector')
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def _extract_features(self, data):
        """Extract features for anomaly detection"""
        features = pd.DataFrame()
        
        # Price anomalies
        features['price_zscore'] = (data['close'] - data['close'].rolling(20).mean()) / data['close'].rolling(20).std()
        
        # Volume anomalies
        features['volume_zscore'] = (data['volume'] - data['volume'].rolling(20).mean()) / data['volume'].rolling(20).std()
        
        # Return anomalies
        returns = data['close'].pct_change()
        features['return_zscore'] = (returns - returns.rolling(20).mean()) / returns.rolling(20).std()
        
        # Volatility spikes
        volatility = returns.rolling(20).std()
        features['volatility_zscore'] = (volatility - volatility.rolling(20).mean()) / volatility.rolling(20).std()
        
        return features

    def train(self, historical_data):
        """Train anomaly detection model"""
        self.logger.info("Training anomaly detector...")
        
        features = self._extract_features(historical_data)
        X = features.dropna()
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True
        
        self.logger.info("Anomaly detector trained")

    def detect(self, current_data):
        """
        Detect anomalies in current data
        
        Returns:
            -1 for anomaly, 1 for normal
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before detection")
        
        features = self._extract_features(current_data)
        X = features.iloc[[-1]].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        
        if prediction == -1:
            self.logger.warning("Anomaly detected in market data")
        
        return prediction

    def get_anomaly_score(self, current_data):
        """Get anomaly score (lower = more anomalous)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")
        
        features = self._extract_features(current_data)
        X = features.iloc[[-1]].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        score = self.model.score_samples(X_scaled)[0]
        return score
