"""
Correlation Engine
Advanced cross-market correlation analysis and detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
import logging


class CorrelationEngine:
    """
    Advanced market correlation analysis engine
    Analyzes cross-market correlations and relationships
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger('CorrelationEngine')
        self.config = config or {}
        
        # Correlation parameters
        self.correlation_depth = self.config.get('CORRELATION_DEPTH', 500)
        self.correlation_threshold = 0.7  # Strong correlation threshold
        self.min_samples = 30  # Minimum samples for reliable correlation
        
        # Data storage
        self.price_history: Dict[str, pd.Series] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.correlation_cache: Dict[Tuple[str, str], float] = {}
        
    def add_price_data(self, symbol: str, prices: pd.Series):
        """
        Add price data for correlation analysis
        
        Args:
            symbol: Trading symbol
            prices: Price series with datetime index
        """
        # Keep only recent data based on correlation depth
        if len(prices) > self.correlation_depth:
            prices = prices.iloc[-self.correlation_depth:]
            
        self.price_history[symbol] = prices
        self.logger.debug(f"Added price data for {symbol}: {len(prices)} points")
        
    def calculate_correlation_matrix(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix for all tracked symbols
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix DataFrame
        """
        if len(self.price_history) < 2:
            self.logger.warning("Need at least 2 symbols for correlation analysis")
            return pd.DataFrame()
            
        # Align all price series to common timestamps
        price_df = pd.DataFrame(self.price_history)
        
        # Remove NaN values
        price_df = price_df.dropna()
        
        if len(price_df) < self.min_samples:
            self.logger.warning(f"Insufficient samples ({len(price_df)}) for reliable correlation")
            return pd.DataFrame()
            
        # Calculate correlation matrix
        if method == 'pearson':
            self.correlation_matrix = price_df.corr(method='pearson')
        elif method == 'spearman':
            self.correlation_matrix = price_df.corr(method='spearman')
        elif method == 'kendall':
            self.correlation_matrix = price_df.corr(method='kendall')
        else:
            raise ValueError(f"Unknown correlation method: {method}")
            
        return self.correlation_matrix
        
    def get_correlation(self, symbol1: str, symbol2: str, method: str = 'pearson') -> float:
        """
        Calculate correlation between two symbols
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            method: Correlation method
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        # Check cache
        cache_key = (symbol1, symbol2) if symbol1 < symbol2 else (symbol2, symbol1)
        if cache_key in self.correlation_cache:
            return self.correlation_cache[cache_key]
            
        if symbol1 not in self.price_history or symbol2 not in self.price_history:
            self.logger.warning(f"Missing price data for {symbol1} or {symbol2}")
            return 0.0
            
        # Align price series
        prices1 = self.price_history[symbol1]
        prices2 = self.price_history[symbol2]
        
        df = pd.DataFrame({symbol1: prices1, symbol2: prices2}).dropna()
        
        if len(df) < self.min_samples:
            self.logger.warning(f"Insufficient overlapping samples for {symbol1} and {symbol2}")
            return 0.0
            
        # Calculate correlation
        if method == 'pearson':
            corr, _ = pearsonr(df[symbol1], df[symbol2])
        elif method == 'spearman':
            corr, _ = spearmanr(df[symbol1], df[symbol2])
        else:
            raise ValueError(f"Unknown correlation method: {method}")
            
        # Cache result
        self.correlation_cache[cache_key] = corr
        
        return corr
        
    def find_correlated_pairs(self, threshold: Optional[float] = None) -> List[Tuple[str, str, float]]:
        """
        Find strongly correlated symbol pairs
        
        Args:
            threshold: Correlation threshold (default: use config threshold)
            
        Returns:
            List of (symbol1, symbol2, correlation) tuples
        """
        if threshold is None:
            threshold = self.correlation_threshold
            
        if self.correlation_matrix is None:
            self.calculate_correlation_matrix()
            
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return []
            
        pairs = []
        symbols = list(self.correlation_matrix.columns)
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1 = symbols[i]
                symbol2 = symbols[j]
                corr = self.correlation_matrix.loc[symbol1, symbol2]
                
                if abs(corr) >= threshold:
                    pairs.append((symbol1, symbol2, corr))
                    
        # Sort by absolute correlation strength
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return pairs
        
    def calculate_rolling_correlation(self, symbol1: str, symbol2: str, window: int = 30) -> pd.Series:
        """
        Calculate rolling correlation between two symbols
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            window: Rolling window size
            
        Returns:
            Rolling correlation series
        """
        if symbol1 not in self.price_history or symbol2 not in self.price_history:
            return pd.Series()
            
        prices1 = self.price_history[symbol1]
        prices2 = self.price_history[symbol2]
        
        df = pd.DataFrame({symbol1: prices1, symbol2: prices2}).dropna()
        
        if len(df) < window:
            self.logger.warning(f"Insufficient data for rolling correlation (need {window}, have {len(df)})")
            return pd.Series()
            
        # Calculate rolling correlation
        rolling_corr = df[symbol1].rolling(window=window).corr(df[symbol2])
        
        return rolling_corr
        
    def detect_correlation_changes(self, symbol1: str, symbol2: str, 
                                   short_window: int = 20, 
                                   long_window: int = 100) -> Dict:
        """
        Detect changes in correlation over time
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            short_window: Short-term window
            long_window: Long-term window
            
        Returns:
            Dictionary with correlation analysis
        """
        short_corr = self.calculate_rolling_correlation(symbol1, symbol2, short_window)
        long_corr = self.calculate_rolling_correlation(symbol1, symbol2, long_window)
        
        if short_corr.empty or long_corr.empty:
            return {}
            
        # Get latest values
        current_short = short_corr.iloc[-1] if len(short_corr) > 0 else 0.0
        current_long = long_corr.iloc[-1] if len(long_corr) > 0 else 0.0
        
        # Detect regime change
        correlation_change = current_short - current_long
        
        return {
            'short_term_correlation': current_short,
            'long_term_correlation': current_long,
            'correlation_change': correlation_change,
            'regime_shift': abs(correlation_change) > 0.3,
            'strengthening': correlation_change > 0.3,
            'weakening': correlation_change < -0.3
        }
        
    def calculate_lead_lag_relationship(self, symbol1: str, symbol2: str, max_lag: int = 10) -> Dict:
        """
        Analyze lead-lag relationship between two symbols
        
        Args:
            symbol1: First symbol (potential leader)
            symbol2: Second symbol (potential follower)
            max_lag: Maximum lag to test
            
        Returns:
            Dictionary with lead-lag analysis
        """
        if symbol1 not in self.price_history or symbol2 not in self.price_history:
            return {}
            
        prices1 = self.price_history[symbol1]
        prices2 = self.price_history[symbol2]
        
        df = pd.DataFrame({symbol1: prices1, symbol2: prices2}).dropna()
        
        if len(df) < max_lag * 2:
            return {}
            
        # Calculate correlations at different lags
        lag_correlations = {}
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # symbol1 leads symbol2
                corr = df[symbol1].iloc[:lag].corr(df[symbol2].iloc[-lag:])
            elif lag > 0:
                # symbol2 leads symbol1
                corr = df[symbol1].iloc[lag:].corr(df[symbol2].iloc[:-lag])
            else:
                # No lag
                corr = df[symbol1].corr(df[symbol2])
                
            lag_correlations[lag] = corr
            
        # Find optimal lag
        optimal_lag = max(lag_correlations.items(), key=lambda x: abs(x[1]))
        
        return {
            'optimal_lag': optimal_lag[0],
            'optimal_correlation': optimal_lag[1],
            'lag_correlations': lag_correlations,
            'symbol1_leads': optimal_lag[0] < 0,
            'symbol2_leads': optimal_lag[0] > 0,
            'synchronous': optimal_lag[0] == 0
        }
        
    def get_correlation_clusters(self, n_clusters: int = 3) -> Dict[int, List[str]]:
        """
        Identify clusters of correlated symbols
        
        Args:
            n_clusters: Number of clusters to identify
            
        Returns:
            Dictionary mapping cluster ID to list of symbols
        """
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            self.calculate_correlation_matrix()
            
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return {}
            
        from sklearn.cluster import AgglomerativeClustering
        
        # Use correlation distance for clustering
        distance_matrix = 1 - abs(self.correlation_matrix)
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        
        labels = clustering.fit_predict(distance_matrix)
        
        # Group symbols by cluster
        clusters = {}
        symbols = list(self.correlation_matrix.columns)
        
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(symbols[i])
            
        return clusters
