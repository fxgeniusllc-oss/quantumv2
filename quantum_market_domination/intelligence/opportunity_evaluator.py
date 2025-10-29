"""
OPPORTUNITY DETECTION AND SCORING
Evaluates and scores trading opportunities across multiple dimensions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from enum import Enum
import logging


class OpportunityType(Enum):
    """Types of trading opportunities"""
    TREND = "trend"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    ARBITRAGE = "arbitrage"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"


class OpportunityEvaluator:
    """
    Evaluates and scores trading opportunities using multi-dimensional analysis
    """
    
    def __init__(self, min_score: float = 0.6):
        """
        Initialize opportunity evaluator
        
        Args:
            min_score: Minimum score threshold (0-1) for valid opportunities
        """
        self.logger = logging.getLogger('OpportunityEvaluator')
        self.min_score = min_score
        self.opportunities: List[Dict] = []
        self.scoring_weights = {
            'technical': 0.3,
            'volatility': 0.2,
            'volume': 0.15,
            'risk_reward': 0.25,
            'timing': 0.1
        }
        
    def evaluate_trend_opportunity(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Evaluate trend-following opportunity
        
        Args:
            symbol: Trading pair symbol
            data: Historical price data
            
        Returns:
            Dict with opportunity details and score
        """
        if len(data) < 50:
            return {'valid': False, 'reason': 'Insufficient data'}
            
        # Calculate trend indicators
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        
        current_price = data['close'].iloc[-1]
        sma_20_current = sma_20.iloc[-1]
        sma_50_current = sma_50.iloc[-1]
        
        # Trend strength
        trend_strength = 0.0
        if sma_20_current > sma_50_current:
            # Uptrend
            trend_strength = (sma_20_current - sma_50_current) / sma_50_current
            direction = "long"
        else:
            # Downtrend
            trend_strength = (sma_50_current - sma_20_current) / sma_50_current
            direction = "short"
            
        # Price position relative to SMAs
        price_position_score = 0.0
        if direction == "long" and current_price > sma_20_current > sma_50_current:
            price_position_score = 1.0
        elif direction == "short" and current_price < sma_20_current < sma_50_current:
            price_position_score = 1.0
        else:
            price_position_score = 0.5
            
        # Volume confirmation
        volume_score = self._calculate_volume_score(data)
        
        # Calculate overall score
        technical_score = min(abs(trend_strength) * 10, 1.0)
        score = (
            technical_score * self.scoring_weights['technical'] +
            price_position_score * self.scoring_weights['technical'] +
            volume_score * self.scoring_weights['volume']
        ) / (self.scoring_weights['technical'] * 2 + self.scoring_weights['volume'])
        
        opportunity = {
            'valid': score >= self.min_score,
            'type': OpportunityType.TREND.value,
            'symbol': symbol,
            'direction': direction,
            'score': score,
            'trend_strength': trend_strength,
            'price': current_price,
            'entry': current_price,
            'stop_loss': sma_50_current if direction == "long" else sma_20_current * 1.02,
            'target': current_price * (1 + abs(trend_strength) * 2) if direction == "long" 
                     else current_price * (1 - abs(trend_strength) * 2),
            'confidence': score
        }
        
        return opportunity
    
    def evaluate_breakout_opportunity(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Evaluate breakout opportunity
        
        Args:
            symbol: Trading pair symbol
            data: Historical price data
            
        Returns:
            Dict with opportunity details and score
        """
        if len(data) < 20:
            return {'valid': False, 'reason': 'Insufficient data'}
            
        # Calculate recent high/low
        lookback = min(20, len(data))
        recent_high = data['high'].iloc[-lookback:].max()
        recent_low = data['low'].iloc[-lookback:].max()
        current_price = data['close'].iloc[-1]
        
        # Check for breakout
        breakout_threshold = 0.02  # 2% above high or below low
        
        if current_price > recent_high * (1 + breakout_threshold):
            # Bullish breakout
            direction = "long"
            breakout_strength = (current_price - recent_high) / recent_high
        elif current_price < recent_low * (1 - breakout_threshold):
            # Bearish breakout
            direction = "short"
            breakout_strength = (recent_low - current_price) / recent_low
        else:
            return {'valid': False, 'reason': 'No breakout detected'}
            
        # Volume confirmation
        volume_score = self._calculate_volume_score(data)
        
        # Volatility check
        volatility_score = self._calculate_volatility_score(data)
        
        # Calculate overall score
        score = (
            min(abs(breakout_strength) * 10, 1.0) * self.scoring_weights['technical'] +
            volume_score * self.scoring_weights['volume'] +
            volatility_score * self.scoring_weights['volatility']
        ) / (self.scoring_weights['technical'] + self.scoring_weights['volume'] + 
             self.scoring_weights['volatility'])
        
        # Risk/reward calculation
        risk = abs(current_price - recent_low if direction == "long" else recent_high - current_price)
        reward = abs(breakout_strength) * current_price * 2
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        opportunity = {
            'valid': score >= self.min_score and risk_reward_ratio >= 2,
            'type': OpportunityType.BREAKOUT.value,
            'symbol': symbol,
            'direction': direction,
            'score': score,
            'breakout_strength': breakout_strength,
            'price': current_price,
            'entry': current_price,
            'stop_loss': recent_low if direction == "long" else recent_high,
            'target': current_price * (1 + breakout_strength * 3) if direction == "long"
                     else current_price * (1 - breakout_strength * 3),
            'risk_reward': risk_reward_ratio,
            'confidence': score
        }
        
        return opportunity
    
    def evaluate_reversal_opportunity(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Evaluate mean reversion opportunity
        
        Args:
            symbol: Trading pair symbol
            data: Historical price data
            
        Returns:
            Dict with opportunity details and score
        """
        if len(data) < 20:
            return {'valid': False, 'reason': 'Insufficient data'}
            
        # Calculate Bollinger Bands
        sma_20 = data['close'].rolling(20).mean()
        std_20 = data['close'].rolling(20).std()
        upper_band = sma_20 + (std_20 * 2)
        lower_band = sma_20 - (std_20 * 2)
        
        current_price = data['close'].iloc[-1]
        current_sma = sma_20.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        # Check for reversal setup
        if current_price <= current_lower:
            # Oversold - potential long
            direction = "long"
            deviation = (current_lower - current_price) / current_sma
        elif current_price >= current_upper:
            # Overbought - potential short
            direction = "short"
            deviation = (current_price - current_upper) / current_sma
        else:
            return {'valid': False, 'reason': 'No reversal setup'}
            
        # RSI confirmation
        rsi_score = self._calculate_rsi_score(data)
        
        # Volume divergence check
        volume_score = self._calculate_volume_score(data)
        
        # Calculate overall score
        score = (
            min(abs(deviation) * 10, 1.0) * self.scoring_weights['technical'] +
            rsi_score * self.scoring_weights['technical'] +
            volume_score * self.scoring_weights['volume']
        ) / (self.scoring_weights['technical'] * 2 + self.scoring_weights['volume'])
        
        opportunity = {
            'valid': score >= self.min_score,
            'type': OpportunityType.REVERSAL.value,
            'symbol': symbol,
            'direction': direction,
            'score': score,
            'deviation': deviation,
            'price': current_price,
            'entry': current_price,
            'stop_loss': current_lower * 0.98 if direction == "long" else current_upper * 1.02,
            'target': current_sma,
            'confidence': score
        }
        
        return opportunity
    
    def evaluate_volatility_opportunity(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Evaluate volatility-based opportunity
        
        Args:
            symbol: Trading pair symbol
            data: Historical price data
            
        Returns:
            Dict with opportunity details and score
        """
        if len(data) < 20:
            return {'valid': False, 'reason': 'Insufficient data'}
            
        # Calculate volatility
        returns = data['close'].pct_change().dropna()
        current_vol = returns.iloc[-5:].std() if len(returns) >= 5 else 0
        avg_vol = returns.std()
        
        # Volatility change
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        # Determine strategy
        if vol_ratio < 0.5:
            # Low volatility - expect expansion
            strategy = "volatility_expansion"
            score = (1.0 - vol_ratio) * 0.7
        elif vol_ratio > 1.5:
            # High volatility - expect contraction
            strategy = "volatility_contraction"
            score = (vol_ratio - 1.0) * 0.5
        else:
            return {'valid': False, 'reason': 'Normal volatility regime'}
            
        opportunity = {
            'valid': score >= self.min_score,
            'type': OpportunityType.VOLATILITY.value,
            'symbol': symbol,
            'strategy': strategy,
            'score': score,
            'current_volatility': current_vol,
            'average_volatility': avg_vol,
            'volatility_ratio': vol_ratio,
            'confidence': score
        }
        
        return opportunity
    
    def _calculate_volume_score(self, data: pd.DataFrame) -> float:
        """Calculate volume confirmation score"""
        if 'volume' not in data.columns or len(data) < 10:
            return 0.5  # Neutral score if no volume data
            
        recent_volume = data['volume'].iloc[-5:].mean()
        avg_volume = data['volume'].iloc[-20:].mean()
        
        if avg_volume == 0:
            return 0.5
            
        volume_ratio = recent_volume / avg_volume
        
        # Higher volume = higher score (capped at 1.0)
        return min(volume_ratio / 1.5, 1.0)
    
    def _calculate_volatility_score(self, data: pd.DataFrame) -> float:
        """Calculate volatility score"""
        if len(data) < 20:
            return 0.5
            
        returns = data['close'].pct_change().dropna()
        current_vol = returns.iloc[-5:].std() if len(returns) >= 5 else 0
        avg_vol = returns.std()
        
        if avg_vol == 0:
            return 0.5
            
        vol_ratio = current_vol / avg_vol
        
        # Moderate volatility is ideal (0.8-1.2)
        if 0.8 <= vol_ratio <= 1.2:
            return 1.0
        elif vol_ratio < 0.8:
            return vol_ratio / 0.8
        else:
            return 1.2 / vol_ratio
    
    def _calculate_rsi_score(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI-based score"""
        if len(data) < period + 1:
            return 0.5
            
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        # Oversold (good for long) or overbought (good for short)
        if current_rsi < 30:
            return (30 - current_rsi) / 30  # Higher score when more oversold
        elif current_rsi > 70:
            return (current_rsi - 70) / 30  # Higher score when more overbought
        else:
            return 0.0  # Neutral zone
    
    def evaluate_all_opportunities(self, symbol: str, data: pd.DataFrame) -> List[Dict]:
        """
        Evaluate all opportunity types for a symbol
        
        Args:
            symbol: Trading pair symbol
            data: Historical price data
            
        Returns:
            List of valid opportunities sorted by score
        """
        opportunities = []
        
        # Evaluate each opportunity type
        trend_opp = self.evaluate_trend_opportunity(symbol, data)
        if trend_opp.get('valid', False):
            opportunities.append(trend_opp)
            
        breakout_opp = self.evaluate_breakout_opportunity(symbol, data)
        if breakout_opp.get('valid', False):
            opportunities.append(breakout_opp)
            
        reversal_opp = self.evaluate_reversal_opportunity(symbol, data)
        if reversal_opp.get('valid', False):
            opportunities.append(reversal_opp)
            
        volatility_opp = self.evaluate_volatility_opportunity(symbol, data)
        if volatility_opp.get('valid', False):
            opportunities.append(volatility_opp)
        
        # Sort by score (highest first)
        opportunities.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        self.logger.info(f"Found {len(opportunities)} valid opportunities for {symbol}")
        return opportunities
    
    def get_best_opportunity(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        Get the best opportunity for a symbol
        
        Args:
            symbol: Trading pair symbol
            data: Historical price data
            
        Returns:
            Best opportunity dict or None
        """
        opportunities = self.evaluate_all_opportunities(symbol, data)
        
        if opportunities:
            return opportunities[0]
        return None
    
    def score_opportunity(self, opportunity: Dict) -> float:
        """
        Recalculate comprehensive score for an opportunity
        
        Args:
            opportunity: Opportunity dict
            
        Returns:
            Overall score (0-1)
        """
        base_score = opportunity.get('score', 0.0)
        
        # Adjust based on risk/reward if available
        risk_reward = opportunity.get('risk_reward', 0)
        if risk_reward > 0:
            risk_reward_score = min(risk_reward / 3.0, 1.0)
            base_score = (base_score * 0.7) + (risk_reward_score * 0.3)
        
        # Adjust based on confidence
        confidence = opportunity.get('confidence', 0.5)
        final_score = (base_score * 0.8) + (confidence * 0.2)
        
        return min(final_score, 1.0)
