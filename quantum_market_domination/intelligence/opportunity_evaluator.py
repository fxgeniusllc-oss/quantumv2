"""
Opportunity Evaluator
Evaluates and scores trading opportunities based on multiple factors
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging


class OpportunityType(Enum):
    """Types of trading opportunities"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"
    CORRELATION = "correlation"


@dataclass
class TradingOpportunity:
    """Trading opportunity data structure"""
    symbol: str
    opportunity_type: OpportunityType
    score: float
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    expected_return: float
    timeframe: str
    factors: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate opportunity parameters"""
        if self.score < 0 or self.score > 1:
            raise ValueError("Score must be between 0 and 1")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.risk_reward_ratio < 0:
            raise ValueError("Risk-reward ratio must be positive")


class OpportunityEvaluator:
    """
    Advanced opportunity detection and evaluation system
    Scores and ranks trading opportunities across multiple dimensions
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger('OpportunityEvaluator')
        self.config = config or {}
        
        # Evaluation parameters
        self.min_score_threshold = 0.6  # Minimum score to consider
        self.min_confidence = 0.5  # Minimum confidence level
        self.min_risk_reward = 1.5  # Minimum risk-reward ratio
        
        # Weighting factors for composite score
        self.score_weights = {
            'technical': 0.25,
            'momentum': 0.20,
            'volatility': 0.15,
            'volume': 0.15,
            'correlation': 0.10,
            'ml_prediction': 0.15
        }
        
        # Opportunity storage
        self.opportunities: List[TradingOpportunity] = []
        
    def evaluate_trend_opportunity(self, 
                                   symbol: str,
                                   price_data: pd.DataFrame,
                                   indicators: Dict) -> Optional[TradingOpportunity]:
        """
        Evaluate trend-following opportunity
        
        Args:
            symbol: Trading symbol
            price_data: Historical price data
            indicators: Technical indicators
            
        Returns:
            TradingOpportunity if valid, None otherwise
        """
        if len(price_data) < 50:
            return None
            
        current_price = price_data['close'].iloc[-1]
        
        # Check trend indicators
        sma_20 = indicators.get('sma_20', 0)
        sma_50 = indicators.get('sma_50', 0)
        sma_200 = indicators.get('sma_200', 0)
        
        # Bullish trend: price > sma_20 > sma_50 > sma_200
        trend_alignment = (
            current_price > sma_20 > sma_50 > 0 and
            sma_20 > sma_50
        )
        
        if not trend_alignment:
            return None
            
        # Calculate scores
        trend_strength = min(1.0, (current_price - sma_50) / sma_50 * 10)
        momentum_score = self._calculate_momentum_score(price_data, indicators)
        volume_score = self._calculate_volume_score(price_data, indicators)
        
        # Composite score
        score = (
            0.4 * trend_strength +
            0.3 * momentum_score +
            0.3 * volume_score
        )
        
        # Calculate entry, target, stop loss
        atr = indicators.get('atr', current_price * 0.02)
        entry_price = current_price
        target_price = entry_price + (atr * 3)  # 3 ATR target
        stop_loss = entry_price - (atr * 1.5)  # 1.5 ATR stop
        
        risk_reward = (target_price - entry_price) / (entry_price - stop_loss)
        expected_return = (target_price - entry_price) / entry_price
        
        if risk_reward < self.min_risk_reward:
            return None
            
        return TradingOpportunity(
            symbol=symbol,
            opportunity_type=OpportunityType.TREND_FOLLOWING,
            score=score,
            confidence=0.7,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward,
            expected_return=expected_return,
            timeframe='4H',
            factors={
                'trend_strength': trend_strength,
                'momentum': momentum_score,
                'volume': volume_score
            }
        )
        
    def evaluate_mean_reversion_opportunity(self,
                                           symbol: str,
                                           price_data: pd.DataFrame,
                                           indicators: Dict) -> Optional[TradingOpportunity]:
        """
        Evaluate mean reversion opportunity
        
        Args:
            symbol: Trading symbol
            price_data: Historical price data
            indicators: Technical indicators
            
        Returns:
            TradingOpportunity if valid, None otherwise
        """
        if len(price_data) < 20:
            return None
            
        current_price = price_data['close'].iloc[-1]
        
        # Check Bollinger Bands
        bb_upper = indicators.get('bb_upper', 0)
        bb_lower = indicators.get('bb_lower', 0)
        bb_middle = indicators.get('bb_middle', 0)
        
        if bb_upper == 0 or bb_lower == 0:
            return None
            
        # Check for oversold condition
        rsi = indicators.get('rsi', 50)
        
        # Mean reversion setup: price near lower band and RSI oversold
        oversold = current_price <= bb_lower * 1.02 and rsi < 30
        
        if not oversold:
            return None
            
        # Calculate reversion scores
        deviation = abs(current_price - bb_middle) / bb_middle
        reversion_score = min(1.0, deviation * 10)
        rsi_score = max(0, (30 - rsi) / 30)  # Higher score for more oversold
        
        score = 0.6 * reversion_score + 0.4 * rsi_score
        
        # Calculate targets
        entry_price = current_price
        target_price = bb_middle  # Target mean reversion to middle band
        stop_loss = bb_lower * 0.98  # Below lower band
        
        risk_reward = (target_price - entry_price) / (entry_price - stop_loss)
        expected_return = (target_price - entry_price) / entry_price
        
        if risk_reward < self.min_risk_reward:
            return None
            
        return TradingOpportunity(
            symbol=symbol,
            opportunity_type=OpportunityType.MEAN_REVERSION,
            score=score,
            confidence=0.65,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward,
            expected_return=expected_return,
            timeframe='1H',
            factors={
                'deviation': deviation,
                'rsi': rsi,
                'reversion_score': reversion_score
            }
        )
        
    def evaluate_breakout_opportunity(self,
                                     symbol: str,
                                     price_data: pd.DataFrame,
                                     indicators: Dict) -> Optional[TradingOpportunity]:
        """
        Evaluate breakout opportunity
        
        Args:
            symbol: Trading symbol
            price_data: Historical price data
            indicators: Technical indicators
            
        Returns:
            TradingOpportunity if valid, None otherwise
        """
        if len(price_data) < 30:
            return None
            
        current_price = price_data['close'].iloc[-1]
        high_20 = price_data['high'].iloc[-20:].max()
        low_20 = price_data['low'].iloc[-20:].min()
        
        # Check for consolidation
        range_width = (high_20 - low_20) / low_20
        
        # Look for breakout from consolidation
        volatility = indicators.get('volatility_20', 0)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        
        # Breakout criteria
        is_breakout = (
            current_price > high_20 * 1.005 and  # Price above recent high
            range_width < 0.1 and  # Recent consolidation
            volume_ratio > 1.5  # Volume spike
        )
        
        if not is_breakout:
            return None
            
        # Calculate breakout scores
        breakout_strength = min(1.0, (current_price - high_20) / high_20 * 20)
        volume_score = min(1.0, volume_ratio / 2.0)
        consolidation_score = max(0, 1.0 - range_width * 10)
        
        score = (
            0.4 * breakout_strength +
            0.3 * volume_score +
            0.3 * consolidation_score
        )
        
        # Calculate targets
        entry_price = current_price
        range_height = high_20 - low_20
        target_price = high_20 + range_height  # Measured move
        stop_loss = high_20 * 0.99  # Below breakout level
        
        risk_reward = (target_price - entry_price) / (entry_price - stop_loss)
        expected_return = (target_price - entry_price) / entry_price
        
        if risk_reward < self.min_risk_reward:
            return None
            
        return TradingOpportunity(
            symbol=symbol,
            opportunity_type=OpportunityType.BREAKOUT,
            score=score,
            confidence=0.6,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward,
            expected_return=expected_return,
            timeframe='15M',
            factors={
                'breakout_strength': breakout_strength,
                'volume_ratio': volume_ratio,
                'consolidation': consolidation_score
            }
        )
        
    def _calculate_momentum_score(self, price_data: pd.DataFrame, indicators: Dict) -> float:
        """Calculate momentum score"""
        momentum_5 = indicators.get('momentum_5', 0)
        momentum_10 = indicators.get('momentum_10', 0)
        
        if momentum_5 == 0 or momentum_10 == 0:
            return 0.5
            
        # Positive momentum
        score = np.tanh(momentum_5 / price_data['close'].iloc[-1] * 100) * 0.5 + 0.5
        return max(0, min(1, score))
        
    def _calculate_volume_score(self, price_data: pd.DataFrame, indicators: Dict) -> float:
        """Calculate volume score"""
        if 'volume' not in price_data.columns:
            return 0.5
            
        volume_ratio = indicators.get('volume_ratio', 1.0)
        
        # Higher volume is better (up to 3x average)
        score = min(1.0, volume_ratio / 3.0)
        return score
        
    def rank_opportunities(self, opportunities: List[TradingOpportunity]) -> List[TradingOpportunity]:
        """
        Rank opportunities by composite score
        
        Args:
            opportunities: List of trading opportunities
            
        Returns:
            Sorted list of opportunities
        """
        # Filter by minimum thresholds
        filtered = [
            opp for opp in opportunities
            if opp.score >= self.min_score_threshold and
               opp.confidence >= self.min_confidence and
               opp.risk_reward_ratio >= self.min_risk_reward
        ]
        
        # Sort by composite ranking score
        filtered.sort(
            key=lambda x: x.score * x.confidence * min(x.risk_reward_ratio / 2, 1.5),
            reverse=True
        )
        
        return filtered
        
    def add_opportunity(self, opportunity: TradingOpportunity):
        """Add opportunity to tracking list"""
        self.opportunities.append(opportunity)
        self.logger.info(f"Added {opportunity.opportunity_type.value} opportunity for {opportunity.symbol}")
        
    def get_best_opportunities(self, n: int = 5) -> List[TradingOpportunity]:
        """
        Get top N opportunities
        
        Args:
            n: Number of opportunities to return
            
        Returns:
            List of best opportunities
        """
        ranked = self.rank_opportunities(self.opportunities)
        return ranked[:n]
        
    def clear_opportunities(self):
        """Clear all tracked opportunities"""
        self.opportunities = []
