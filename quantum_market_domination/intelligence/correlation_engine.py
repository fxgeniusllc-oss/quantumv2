"""
CROSS-MARKET CORRELATION ANALYSIS
Analyzes correlations between different markets and trading pairs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging


class CorrelationEngine:
    """
    Cross-market correlation analysis for identifying relationships
    between different trading pairs and exchanges
    """
    
    def __init__(self, correlation_depth: int = 500):
        """
        Initialize correlation engine
        
        Args:
            correlation_depth: Historical data points for correlation calculation
        """
        self.logger = logging.getLogger('CorrelationEngine')
        self.correlation_depth = correlation_depth
        self.correlation_matrix = None
        self.price_history: Dict[str, pd.Series] = {}
        
    def update_price(self, symbol: str, price: float, timestamp: pd.Timestamp = None):
        """
        Update price history for a symbol
        
        Args:
            symbol: Trading pair symbol
            price: Current price
            timestamp: Price timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()
            
        if symbol not in self.price_history:
            self.price_history[symbol] = pd.Series(dtype=float)
            
        # Add new price
        self.price_history[symbol][timestamp] = price
        
        # Keep only last correlation_depth data points
        if len(self.price_history[symbol]) > self.correlation_depth:
            self.price_history[symbol] = self.price_history[symbol].iloc[-self.correlation_depth:]
    
    def calculate_correlation_matrix(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix for specified symbols
        
        Args:
            symbols: List of symbols to analyze (None = all symbols)
            
        Returns:
            Correlation matrix DataFrame
        """
        if symbols is None:
            symbols = list(self.price_history.keys())
            
        if len(symbols) < 2:
            self.logger.warning("Need at least 2 symbols for correlation analysis")
            return pd.DataFrame()
            
        # Create DataFrame with price returns
        returns_data = {}
        for symbol in symbols:
            if symbol in self.price_history and len(self.price_history[symbol]) > 1:
                returns_data[symbol] = self.price_history[symbol].pct_change().dropna()
                
        if not returns_data:
            self.logger.warning("No valid data for correlation calculation")
            return pd.DataFrame()
            
        # Align all series to common timestamps
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        # Calculate correlation matrix
        self.correlation_matrix = returns_df.corr()
        self.logger.info(f"Calculated correlation matrix for {len(symbols)} symbols")
        
        return self.correlation_matrix
    
    def get_highly_correlated_pairs(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Find highly correlated trading pairs
        
        Args:
            threshold: Minimum correlation coefficient (0-1)
            
        Returns:
            List of (symbol1, symbol2, correlation) tuples
        """
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            self.logger.warning("No correlation matrix available")
            return []
            
        correlated_pairs = []
        
        # Iterate through upper triangle of correlation matrix
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i + 1, len(self.correlation_matrix.columns)):
                symbol1 = self.correlation_matrix.columns[i]
                symbol2 = self.correlation_matrix.columns[j]
                correlation = self.correlation_matrix.iloc[i, j]
                
                if abs(correlation) >= threshold:
                    correlated_pairs.append((symbol1, symbol2, correlation))
                    
        # Sort by absolute correlation value
        correlated_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        self.logger.info(f"Found {len(correlated_pairs)} highly correlated pairs")
        return correlated_pairs
    
    def get_uncorrelated_pairs(self, threshold: float = 0.3) -> List[Tuple[str, str, float]]:
        """
        Find uncorrelated trading pairs (good for diversification)
        
        Args:
            threshold: Maximum correlation coefficient (0-1)
            
        Returns:
            List of (symbol1, symbol2, correlation) tuples
        """
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            self.logger.warning("No correlation matrix available")
            return []
            
        uncorrelated_pairs = []
        
        # Iterate through upper triangle of correlation matrix
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i + 1, len(self.correlation_matrix.columns)):
                symbol1 = self.correlation_matrix.columns[i]
                symbol2 = self.correlation_matrix.columns[j]
                correlation = self.correlation_matrix.iloc[i, j]
                
                if abs(correlation) <= threshold:
                    uncorrelated_pairs.append((symbol1, symbol2, correlation))
                    
        # Sort by absolute correlation value
        uncorrelated_pairs.sort(key=lambda x: abs(x[2]))
        
        self.logger.info(f"Found {len(uncorrelated_pairs)} uncorrelated pairs")
        return uncorrelated_pairs
    
    def get_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Get correlation between two specific symbols
        
        Args:
            symbol1: First trading pair
            symbol2: Second trading pair
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            self.logger.warning("No correlation matrix available")
            return 0.0
            
        if symbol1 not in self.correlation_matrix.columns or symbol2 not in self.correlation_matrix.columns:
            self.logger.warning(f"Symbols {symbol1} or {symbol2} not in correlation matrix")
            return 0.0
            
        return self.correlation_matrix.loc[symbol1, symbol2]
    
    def calculate_rolling_correlation(self, symbol1: str, symbol2: str, window: int = 50) -> pd.Series:
        """
        Calculate rolling correlation between two symbols
        
        Args:
            symbol1: First trading pair
            symbol2: Second trading pair
            window: Rolling window size
            
        Returns:
            Series of rolling correlations
        """
        if symbol1 not in self.price_history or symbol2 not in self.price_history:
            self.logger.warning(f"Missing price history for {symbol1} or {symbol2}")
            return pd.Series(dtype=float)
            
        # Get returns
        returns1 = self.price_history[symbol1].pct_change().dropna()
        returns2 = self.price_history[symbol2].pct_change().dropna()
        
        # Align to common timestamps
        combined = pd.DataFrame({'s1': returns1, 's2': returns2}).dropna()
        
        if len(combined) < window:
            self.logger.warning(f"Insufficient data for rolling correlation (need {window}, have {len(combined)})")
            return pd.Series(dtype=float)
            
        # Calculate rolling correlation
        rolling_corr = combined['s1'].rolling(window=window).corr(combined['s2'])
        
        return rolling_corr
    
    def get_correlation_stability(self, symbol1: str, symbol2: str, window: int = 50) -> Dict[str, float]:
        """
        Analyze stability of correlation between two symbols
        
        Args:
            symbol1: First trading pair
            symbol2: Second trading pair
            window: Rolling window size
            
        Returns:
            Dict with correlation statistics
        """
        rolling_corr = self.calculate_rolling_correlation(symbol1, symbol2, window)
        
        if rolling_corr.empty:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'stability': 0.0
            }
            
        stats = {
            'mean': rolling_corr.mean(),
            'std': rolling_corr.std(),
            'min': rolling_corr.min(),
            'max': rolling_corr.max(),
            'stability': 1.0 - rolling_corr.std()  # Higher = more stable
        }
        
        return stats
