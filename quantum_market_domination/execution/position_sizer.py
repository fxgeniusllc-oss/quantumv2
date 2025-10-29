"""
Position Sizer
Dynamic position sizing based on risk, volatility, and portfolio metrics
"""

import numpy as np
from typing import Dict, List, Optional
import logging


class PositionSizer:
    """
    Advanced position sizing algorithm
    Calculates optimal position sizes based on multiple risk factors
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger('PositionSizer')
        self.config = config or {}
        
        # Risk parameters
        self.max_single_trade_risk = self.config.get('max_single_trade_risk', 0.05)  # 5%
        self.max_portfolio_risk = self.config.get('total_portfolio_risk', 0.15)  # 15%
        self.kelly_fraction = 0.25  # Conservative Kelly criterion
        
        # Position sizing methods
        self.default_method = 'kelly'  # 'fixed', 'volatility', 'kelly', 'optimal_f'
        
    def calculate_position_size(self,
                               capital: float,
                               entry_price: float,
                               stop_loss: float,
                               method: Optional[str] = None,
                               win_rate: Optional[float] = None,
                               win_loss_ratio: Optional[float] = None,
                               volatility: Optional[float] = None) -> Dict:
        """
        Calculate optimal position size
        
        Args:
            capital: Total available capital
            entry_price: Entry price for trade
            stop_loss: Stop loss price
            method: Position sizing method
            win_rate: Historical win rate (for Kelly)
            win_loss_ratio: Average win/loss ratio (for Kelly)
            volatility: Asset volatility (for volatility-based sizing)
            
        Returns:
            Dictionary with position size details
        """
        method = method or self.default_method
        
        if entry_price <= 0 or stop_loss <= 0:
            self.logger.error("Invalid entry or stop loss price")
            return {'shares': 0, 'position_value': 0, 'risk_amount': 0}
            
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            self.logger.error("Risk per share is zero")
            return {'shares': 0, 'position_value': 0, 'risk_amount': 0}
            
        # Calculate position size based on method
        if method == 'fixed':
            result = self._fixed_risk_sizing(capital, risk_per_share)
        elif method == 'volatility':
            result = self._volatility_adjusted_sizing(capital, risk_per_share, volatility)
        elif method == 'kelly':
            result = self._kelly_criterion_sizing(capital, entry_price, risk_per_share, 
                                                  win_rate, win_loss_ratio)
        elif method == 'optimal_f':
            result = self._optimal_f_sizing(capital, risk_per_share)
        else:
            self.logger.warning(f"Unknown method {method}, using fixed")
            result = self._fixed_risk_sizing(capital, risk_per_share)
            
        # Apply portfolio-level constraints
        result = self._apply_portfolio_constraints(result, capital)
        
        return result
        
    def _fixed_risk_sizing(self, capital: float, risk_per_share: float) -> Dict:
        """
        Fixed percentage risk per trade
        
        Args:
            capital: Available capital
            risk_per_share: Risk amount per share
            
        Returns:
            Position size details
        """
        risk_amount = capital * self.max_single_trade_risk
        shares = int(risk_amount / risk_per_share)
        
        return {
            'method': 'fixed_risk',
            'shares': shares,
            'position_value': 0,  # Will be set by caller
            'risk_amount': risk_amount,
            'risk_percentage': self.max_single_trade_risk * 100
        }
        
    def _volatility_adjusted_sizing(self, 
                                    capital: float,
                                    risk_per_share: float,
                                    volatility: Optional[float]) -> Dict:
        """
        Volatility-adjusted position sizing
        Reduce position size in high volatility conditions
        
        Args:
            capital: Available capital
            risk_per_share: Risk amount per share
            volatility: Asset volatility (annualized)
            
        Returns:
            Position size details
        """
        if volatility is None or volatility <= 0:
            # Fallback to fixed risk
            return self._fixed_risk_sizing(capital, risk_per_share)
            
        # Adjust risk based on volatility
        # Higher volatility = smaller position
        base_volatility = 0.3  # 30% annualized as baseline
        volatility_adjustment = min(1.0, base_volatility / volatility)
        
        adjusted_risk = self.max_single_trade_risk * volatility_adjustment
        risk_amount = capital * adjusted_risk
        shares = int(risk_amount / risk_per_share)
        
        return {
            'method': 'volatility_adjusted',
            'shares': shares,
            'position_value': 0,
            'risk_amount': risk_amount,
            'risk_percentage': adjusted_risk * 100,
            'volatility': volatility,
            'volatility_adjustment': volatility_adjustment
        }
        
    def _kelly_criterion_sizing(self,
                                capital: float,
                                entry_price: float,
                                risk_per_share: float,
                                win_rate: Optional[float],
                                win_loss_ratio: Optional[float]) -> Dict:
        """
        Kelly Criterion position sizing
        Optimal bet size based on edge
        
        Args:
            capital: Available capital
            entry_price: Entry price
            risk_per_share: Risk per share
            win_rate: Historical win rate (0-1)
            win_loss_ratio: Average win/loss ratio
            
        Returns:
            Position size details
        """
        # Default to conservative values if not provided
        win_rate = win_rate or 0.55  # 55% win rate
        win_loss_ratio = win_loss_ratio or 1.5  # 1.5:1 win/loss ratio
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        p = win_rate
        q = 1 - win_rate
        b = win_loss_ratio
        
        kelly_percentage = (p * b - q) / b
        
        # Apply fractional Kelly for safety
        kelly_percentage = max(0, kelly_percentage * self.kelly_fraction)
        
        # Cap at max single trade risk
        kelly_percentage = min(kelly_percentage, self.max_single_trade_risk)
        
        risk_amount = capital * kelly_percentage
        shares = int(risk_amount / risk_per_share)
        
        return {
            'method': 'kelly_criterion',
            'shares': shares,
            'position_value': 0,
            'risk_amount': risk_amount,
            'risk_percentage': kelly_percentage * 100,
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio,
            'kelly_fraction': self.kelly_fraction
        }
        
    def _optimal_f_sizing(self, capital: float, risk_per_share: float) -> Dict:
        """
        Optimal F position sizing
        Based on Ralph Vince's Optimal f
        
        Args:
            capital: Available capital
            risk_per_share: Risk per share
            
        Returns:
            Position size details
        """
        # Simplified optimal f (would need historical trade data for true calculation)
        # Using conservative estimate
        optimal_f = 0.2  # 20% of capital
        
        # Cap at max single trade risk
        optimal_f = min(optimal_f, self.max_single_trade_risk)
        
        risk_amount = capital * optimal_f
        shares = int(risk_amount / risk_per_share)
        
        return {
            'method': 'optimal_f',
            'shares': shares,
            'position_value': 0,
            'risk_amount': risk_amount,
            'risk_percentage': optimal_f * 100,
            'optimal_f': optimal_f
        }
        
    def _apply_portfolio_constraints(self, result: Dict, capital: float) -> Dict:
        """
        Apply portfolio-level risk constraints
        
        Args:
            result: Initial position size calculation
            capital: Total capital
            
        Returns:
            Adjusted position size
        """
        # Ensure we don't exceed maximum single trade risk
        max_risk_amount = capital * self.max_single_trade_risk
        
        if result['risk_amount'] > max_risk_amount:
            adjustment_factor = max_risk_amount / result['risk_amount']
            result['shares'] = int(result['shares'] * adjustment_factor)
            result['risk_amount'] = max_risk_amount
            result['constrained'] = True
            self.logger.info("Position size constrained by max single trade risk")
        else:
            result['constrained'] = False
            
        return result
        
    def calculate_pyramid_sizes(self,
                               initial_capital: float,
                               entry_price: float,
                               stop_loss: float,
                               n_levels: int = 3) -> List[Dict]:
        """
        Calculate position sizes for pyramiding strategy
        
        Args:
            initial_capital: Starting capital
            entry_price: Initial entry price
            stop_loss: Stop loss price
            n_levels: Number of pyramid levels
            
        Returns:
            List of position sizes for each level
        """
        pyramid_positions = []
        
        # Decrease position size at each level
        size_reduction = 0.6  # Each level is 60% of previous
        
        remaining_capital = initial_capital
        current_entry = entry_price
        
        for level in range(n_levels):
            # Calculate position for this level
            position = self.calculate_position_size(
                remaining_capital,
                current_entry,
                stop_loss,
                method='fixed'
            )
            
            if position['shares'] == 0:
                break
                
            pyramid_positions.append({
                'level': level + 1,
                'shares': position['shares'],
                'entry_price': current_entry,
                'risk_amount': position['risk_amount']
            })
            
            # Adjust for next level
            remaining_capital -= position['risk_amount']
            remaining_capital *= size_reduction
            current_entry *= 1.02  # Assume 2% higher entry for next level
            
        return pyramid_positions
        
    def scale_out_sizes(self,
                       total_shares: int,
                       n_levels: int = 3,
                       strategy: str = 'equal') -> List[Dict]:
        """
        Calculate position sizes for scaling out
        
        Args:
            total_shares: Total position size
            n_levels: Number of exit levels
            strategy: 'equal', 'decreasing', or 'increasing'
            
        Returns:
            List of shares to exit at each level
        """
        if strategy == 'equal':
            # Equal size at each level
            shares_per_level = total_shares // n_levels
            scales = [{'level': i+1, 'shares': shares_per_level} for i in range(n_levels)]
            
            # Add remainder to last level
            remainder = total_shares - (shares_per_level * n_levels)
            scales[-1]['shares'] += remainder
            
        elif strategy == 'decreasing':
            # Larger exits early
            weights = [2.0 ** (n_levels - i) for i in range(n_levels)]
            total_weight = sum(weights)
            
            scales = []
            remaining = total_shares
            
            for i, weight in enumerate(weights[:-1]):
                shares = int(total_shares * weight / total_weight)
                scales.append({'level': i+1, 'shares': shares})
                remaining -= shares
                
            # Last level gets remainder
            scales.append({'level': n_levels, 'shares': remaining})
            
        elif strategy == 'increasing':
            # Larger exits late
            weights = [2.0 ** i for i in range(n_levels)]
            total_weight = sum(weights)
            
            scales = []
            remaining = total_shares
            
            for i, weight in enumerate(weights[:-1]):
                shares = int(total_shares * weight / total_weight)
                scales.append({'level': i+1, 'shares': shares})
                remaining -= shares
                
            # Last level gets remainder
            scales.append({'level': n_levels, 'shares': remaining})
            
        else:
            raise ValueError(f"Unknown scaling strategy: {strategy}")
            
        return scales
