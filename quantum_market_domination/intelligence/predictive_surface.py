"""
PREDICTIVE MARKET SURFACE MAPPING
Creates multi-dimensional predictive surfaces for market analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import griddata, interp2d
import logging


class PredictiveSurface:
    """
    Multi-dimensional market surface mapping for advanced predictions
    Uses interpolation and surface modeling to predict market conditions
    """
    
    def __init__(self, horizon: int = 500):
        """
        Initialize predictive surface mapper
        
        Args:
            horizon: Predictive horizon in data points
        """
        self.logger = logging.getLogger('PredictiveSurface')
        self.horizon = horizon
        self.surfaces: Dict[str, Dict] = {}
        self.historical_data: Dict[str, pd.DataFrame] = {}
        
    def add_market_data(self, symbol: str, data: pd.DataFrame):
        """
        Add market data for surface generation
        
        Args:
            symbol: Trading pair symbol
            data: DataFrame with OHLCV and optional features
        """
        if symbol not in self.historical_data:
            self.historical_data[symbol] = pd.DataFrame()
            
        # Append new data and keep only last horizon points
        self.historical_data[symbol] = pd.concat([self.historical_data[symbol], data])
        if len(self.historical_data[symbol]) > self.horizon:
            self.historical_data[symbol] = self.historical_data[symbol].iloc[-self.horizon:]
            
        self.logger.debug(f"Updated market data for {symbol}: {len(self.historical_data[symbol])} points")
    
    def generate_price_volatility_surface(self, symbol: str) -> Dict:
        """
        Generate price-volatility surface
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict with surface data and interpolation function
        """
        if symbol not in self.historical_data or self.historical_data[symbol].empty:
            self.logger.warning(f"No data available for {symbol}")
            return {}
            
        data = self.historical_data[symbol]
        
        if len(data) < 20:
            self.logger.warning(f"Insufficient data for surface generation ({len(data)} points)")
            return {}
            
        # Calculate returns and volatility
        returns = data['close'].pct_change().dropna()
        
        # Calculate rolling volatility at different windows
        vol_windows = [5, 10, 20, 50]
        volatility_data = {}
        
        for window in vol_windows:
            if len(returns) >= window:
                volatility_data[f'vol_{window}'] = returns.rolling(window=window).std()
        
        # Create surface data
        surface_data = {
            'prices': data['close'].values,
            'returns': returns.values,
            'volatility': volatility_data,
            'timestamps': data.index.values if hasattr(data.index, 'values') else np.arange(len(data))
        }
        
        # Store surface
        self.surfaces[f"{symbol}_price_vol"] = surface_data
        
        self.logger.info(f"Generated price-volatility surface for {symbol}")
        return surface_data
    
    def generate_volume_profile_surface(self, symbol: str) -> Dict:
        """
        Generate volume-price surface (volume profile)
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict with volume profile data
        """
        if symbol not in self.historical_data or self.historical_data[symbol].empty:
            self.logger.warning(f"No data available for {symbol}")
            return {}
            
        data = self.historical_data[symbol]
        
        if 'volume' not in data.columns:
            self.logger.warning(f"No volume data for {symbol}")
            return {}
            
        # Create price bins
        price_min = data['close'].min()
        price_max = data['close'].max()
        num_bins = min(50, len(data) // 10)  # Adaptive binning
        
        if num_bins < 5:
            num_bins = 5
            
        price_bins = np.linspace(price_min, price_max, num_bins)
        
        # Calculate volume at each price level
        volume_profile = []
        for i in range(len(price_bins) - 1):
            mask = (data['close'] >= price_bins[i]) & (data['close'] < price_bins[i + 1])
            total_volume = data.loc[mask, 'volume'].sum()
            avg_price = (price_bins[i] + price_bins[i + 1]) / 2
            volume_profile.append({
                'price': avg_price,
                'volume': total_volume,
                'price_min': price_bins[i],
                'price_max': price_bins[i + 1]
            })
        
        surface_data = {
            'volume_profile': volume_profile,
            'total_volume': data['volume'].sum(),
            'avg_volume': data['volume'].mean(),
            'price_range': (price_min, price_max)
        }
        
        self.surfaces[f"{symbol}_volume"] = surface_data
        
        self.logger.info(f"Generated volume profile surface for {symbol}")
        return surface_data
    
    def generate_momentum_surface(self, symbol: str) -> Dict:
        """
        Generate momentum surface across multiple timeframes
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict with momentum surface data
        """
        if symbol not in self.historical_data or self.historical_data[symbol].empty:
            self.logger.warning(f"No data available for {symbol}")
            return {}
            
        data = self.historical_data[symbol]
        
        if len(data) < 20:
            self.logger.warning(f"Insufficient data for momentum surface ({len(data)} points)")
            return {}
            
        # Calculate momentum at different windows
        momentum_windows = [5, 10, 20, 50]
        momentum_data = {}
        
        for window in momentum_windows:
            if len(data) >= window:
                momentum = data['close'].pct_change(periods=window)
                momentum_data[f'momentum_{window}'] = momentum.values
        
        # Calculate RSI at different periods
        rsi_periods = [14, 28]
        rsi_data = {}
        
        for period in rsi_periods:
            if len(data) >= period + 1:
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi_data[f'rsi_{period}'] = rsi.values
        
        surface_data = {
            'momentum': momentum_data,
            'rsi': rsi_data,
            'timestamps': data.index.values if hasattr(data.index, 'values') else np.arange(len(data))
        }
        
        self.surfaces[f"{symbol}_momentum"] = surface_data
        
        self.logger.info(f"Generated momentum surface for {symbol}")
        return surface_data
    
    def interpolate_surface(self, symbol: str, surface_type: str, 
                           target_points: int = 100) -> Optional[np.ndarray]:
        """
        Interpolate surface for smoother visualization and prediction
        
        Args:
            symbol: Trading pair symbol
            surface_type: Type of surface (price_vol, volume, momentum)
            target_points: Number of interpolation points
            
        Returns:
            Interpolated surface array
        """
        surface_key = f"{symbol}_{surface_type}"
        
        if surface_key not in self.surfaces:
            self.logger.warning(f"Surface {surface_key} not found")
            return None
            
        surface = self.surfaces[surface_key]
        
        # Different interpolation strategies based on surface type
        if surface_type == 'price_vol':
            if 'prices' not in surface or 'returns' not in surface:
                return None
                
            # Simple linear interpolation for demo
            prices = surface['prices']
            returns = surface['returns']
            
            # Remove NaN values
            mask = ~(np.isnan(prices) | np.isnan(returns))
            if not mask.any():
                return None
                
            valid_prices = prices[mask]
            valid_returns = returns[mask]
            
            if len(valid_prices) < 3:
                return None
                
            # Create interpolation grid
            price_range = np.linspace(valid_prices.min(), valid_prices.max(), target_points)
            interpolated = np.interp(price_range, valid_prices, valid_returns)
            
            return interpolated
            
        return None
    
    def predict_future_surface(self, symbol: str, steps_ahead: int = 10) -> Dict:
        """
        Predict future market surface evolution
        
        Args:
            symbol: Trading pair symbol
            steps_ahead: Number of steps to predict
            
        Returns:
            Dict with predicted surface characteristics
        """
        if symbol not in self.historical_data or self.historical_data[symbol].empty:
            self.logger.warning(f"No data available for {symbol}")
            return {}
            
        data = self.historical_data[symbol]
        
        if len(data) < 50:
            self.logger.warning(f"Insufficient data for prediction ({len(data)} points)")
            return {}
            
        # Calculate trend
        recent_data = data['close'].iloc[-50:]
        trend = (recent_data.iloc[-1] - recent_data.iloc[0]) / recent_data.iloc[0]
        
        # Calculate recent volatility
        returns = data['close'].pct_change().dropna()
        recent_volatility = returns.iloc[-20:].std() if len(returns) >= 20 else returns.std()
        
        # Simple prediction based on trend and volatility
        current_price = data['close'].iloc[-1]
        predicted_prices = []
        
        for i in range(1, steps_ahead + 1):
            # Simple random walk with drift
            predicted_price = current_price * (1 + trend / 50) ** i
            predicted_prices.append(predicted_price)
        
        prediction = {
            'predicted_prices': predicted_prices,
            'trend': trend,
            'volatility': recent_volatility,
            'confidence': max(0.5, 1.0 - abs(trend)),  # Higher volatility = lower confidence
            'steps_ahead': steps_ahead
        }
        
        self.logger.info(f"Generated {steps_ahead}-step prediction for {symbol}")
        return prediction
    
    def get_support_resistance_levels(self, symbol: str, num_levels: int = 5) -> Dict[str, List[float]]:
        """
        Identify support and resistance levels from price surface
        
        Args:
            symbol: Trading pair symbol
            num_levels: Number of levels to identify
            
        Returns:
            Dict with support and resistance levels
        """
        if symbol not in self.historical_data or self.historical_data[symbol].empty:
            self.logger.warning(f"No data available for {symbol}")
            return {'support': [], 'resistance': []}
            
        data = self.historical_data[symbol]
        
        if len(data) < 20:
            return {'support': [], 'resistance': []}
            
        # Find local minima (support) and maxima (resistance)
        prices = data['close'].values
        
        support_levels = []
        resistance_levels = []
        
        # Simple peak/trough detection
        for i in range(2, len(prices) - 2):
            # Local minimum (support)
            if prices[i] < prices[i-1] and prices[i] < prices[i-2] and \
               prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                support_levels.append(prices[i])
                
            # Local maximum (resistance)
            if prices[i] > prices[i-1] and prices[i] > prices[i-2] and \
               prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                resistance_levels.append(prices[i])
        
        # Get most significant levels (by frequency/strength)
        support_levels = sorted(support_levels)[:num_levels] if support_levels else []
        resistance_levels = sorted(resistance_levels, reverse=True)[:num_levels] if resistance_levels else []
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    
    def get_surface_summary(self, symbol: str) -> Dict:
        """
        Get comprehensive summary of all surfaces for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict with surface summaries
        """
        summary = {
            'symbol': symbol,
            'data_points': len(self.historical_data.get(symbol, [])),
            'surfaces': []
        }
        
        # List all surfaces for this symbol
        for key in self.surfaces:
            if key.startswith(symbol):
                summary['surfaces'].append(key)
        
        # Add support/resistance
        levels = self.get_support_resistance_levels(symbol)
        summary['support_levels'] = levels['support']
        summary['resistance_levels'] = levels['resistance']
        
        # Add prediction
        prediction = self.predict_future_surface(symbol, steps_ahead=5)
        if prediction:
            summary['short_term_prediction'] = prediction
        
        return summary
