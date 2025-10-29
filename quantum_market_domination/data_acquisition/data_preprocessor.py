"""
Data Preprocessor
Handles data cleaning, transformation, and feature engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging


class DataPreprocessor:
    """
    Advanced data preprocessing and feature engineering
    Cleans and transforms raw market data for ML models
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger('DataPreprocessor')
        self.config = config or {}
        
        # Feature engineering parameters
        self.sma_windows = [5, 10, 20, 50, 200]
        self.ema_windows = [12, 26]
        self.rsi_period = 14
        self.bollinger_period = 20
        self.bollinger_std = 2
        
    def clean_ohlcv_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate OHLCV data
        
        Args:
            data: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = data.copy()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers (price spikes beyond 3 standard deviations)
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df[col] = df[col].clip(mean - 3*std, mean + 3*std)
        
        # Ensure OHLC consistency
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            df['high'] = df[['high', 'open', 'close']].max(axis=1)
            df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        return df
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        data = df.copy()
        
        if 'close' not in data.columns:
            self.logger.warning("No 'close' column found in data")
            return data
            
        # Simple Moving Averages
        for window in self.sma_windows:
            data[f'sma_{window}'] = data['close'].rolling(window=window).mean()
            
        # Exponential Moving Averages
        for window in self.ema_windows:
            data[f'ema_{window}'] = data['close'].ewm(span=window, adjust=False).mean()
            
        # MACD
        if 'ema_12' in data.columns and 'ema_26' in data.columns:
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
            data['macd_histogram'] = data['macd'] - data['macd_signal']
            
        # RSI (Relative Strength Index)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        rolling_mean = data['close'].rolling(window=self.bollinger_period).mean()
        rolling_std = data['close'].rolling(window=self.bollinger_period).std()
        data['bb_upper'] = rolling_mean + (rolling_std * self.bollinger_std)
        data['bb_middle'] = rolling_mean
        data['bb_lower'] = rolling_mean - (rolling_std * self.bollinger_std)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # Momentum indicators
        data['momentum_5'] = data['close'] - data['close'].shift(5)
        data['momentum_10'] = data['close'] - data['close'].shift(10)
        data['rate_of_change'] = data['close'].pct_change(periods=10)
        
        # Volume indicators (if volume exists)
        if 'volume' in data.columns:
            data['volume_sma_20'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma_20']
            
        # Volatility
        data['volatility_10'] = data['close'].pct_change().rolling(window=10).std()
        data['volatility_20'] = data['close'].pct_change().rolling(window=20).std()
        data['volatility_50'] = data['close'].pct_change().rolling(window=50).std()
        
        # Average True Range (ATR)
        if all(col in data.columns for col in ['high', 'low', 'close']):
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            data['atr'] = true_range.rolling(window=14).mean()
            
        return data
        
    def normalize_features(self, df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize feature values
        
        Args:
            df: DataFrame with features
            method: Normalization method ('minmax', 'zscore', 'robust')
            
        Returns:
            Normalized DataFrame
        """
        data = df.copy()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            # Min-Max normalization (0-1 range)
            for col in numeric_columns:
                min_val = data[col].min()
                max_val = data[col].max()
                if max_val > min_val:
                    data[col] = (data[col] - min_val) / (max_val - min_val)
                    
        elif method == 'zscore':
            # Z-score normalization (mean=0, std=1)
            for col in numeric_columns:
                mean = data[col].mean()
                std = data[col].std()
                if std > 0:
                    data[col] = (data[col] - mean) / std
                    
        elif method == 'robust':
            # Robust normalization using median and IQR
            for col in numeric_columns:
                median = data[col].median()
                q75, q25 = data[col].quantile([0.75, 0.25])
                iqr = q75 - q25
                if iqr > 0:
                    data[col] = (data[col] - median) / iqr
                    
        return data
        
    def create_lagged_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features for time series
        
        Args:
            df: DataFrame with time series data
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        data = df.copy()
        
        for col in columns:
            if col in data.columns:
                for lag in lags:
                    data[f'{col}_lag_{lag}'] = data[col].shift(lag)
                    
        return data
        
    def create_rolling_features(self, df: pd.DataFrame, column: str, windows: List[int]) -> pd.DataFrame:
        """
        Create rolling window statistics
        
        Args:
            df: DataFrame with time series data
            column: Column to calculate rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        data = df.copy()
        
        if column in data.columns:
            for window in windows:
                data[f'{column}_rolling_mean_{window}'] = data[column].rolling(window=window).mean()
                data[f'{column}_rolling_std_{window}'] = data[column].rolling(window=window).std()
                data[f'{column}_rolling_min_{window}'] = data[column].rolling(window=window).min()
                data[f'{column}_rolling_max_{window}'] = data[column].rolling(window=window).max()
                
        return data
        
    def remove_nan_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with NaN values
        
        Args:
            df: DataFrame to clean
            
        Returns:
            DataFrame without NaN rows
        """
        initial_len = len(df)
        df_clean = df.dropna()
        removed = initial_len - len(df_clean)
        
        if removed > 0:
            self.logger.info(f"Removed {removed} rows with NaN values")
            
        return df_clean
        
    def prepare_for_ml(self, df: pd.DataFrame, target_column: Optional[str] = None) -> tuple:
        """
        Prepare data for machine learning
        
        Args:
            df: DataFrame with features
            target_column: Name of target column (if supervised learning)
            
        Returns:
            Tuple of (features_df, target_series) or just features_df
        """
        data = df.copy()
        
        # Remove non-numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Remove infinite values
        numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
        
        # Remove NaN values
        numeric_data = self.remove_nan_rows(numeric_data)
        
        if target_column and target_column in numeric_data.columns:
            target = numeric_data[target_column]
            features = numeric_data.drop(columns=[target_column])
            return features, target
        else:
            return numeric_data
