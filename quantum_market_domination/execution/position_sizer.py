"""
DYNAMIC POSITION SIZING
Advanced position sizing algorithms based on risk, volatility, and market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging


class PositionSizer:
    """
    Dynamic position sizing using multiple strategies:
    - Fixed fractional
    - Kelly Criterion
    - Volatility-based
    - Risk parity
    """
    
    def __init__(self, max_position_pct: float = 0.1, max_leverage: float = 3.0):
        """
        Initialize position sizer
        
        Args:
            max_position_pct: Maximum position size as % of portfolio (0-1)
            max_leverage: Maximum leverage allowed
        """
        self.logger = logging.getLogger('PositionSizer')
        self.max_position_pct = max_position_pct
        self.max_leverage = max_leverage
        
    def calculate_fixed_fractional(self, portfolio_value: float, 
                                   risk_per_trade: float = 0.02) -> float:
        """
        Fixed fractional position sizing
        
        Args:
            portfolio_value: Total portfolio value
            risk_per_trade: Risk per trade as fraction (0-1)
            
        Returns:
            Position size in portfolio currency
        """
        position_size = portfolio_value * risk_per_trade
        max_size = portfolio_value * self.max_position_pct
        
        return min(position_size, max_size)
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, 
                                  avg_loss: float, portfolio_value: float,
                                  fraction: float = 0.25) -> float:
        """
        Kelly Criterion position sizing
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive)
            portfolio_value: Total portfolio value
            fraction: Fraction of Kelly to use (0-1), default 0.25 for safety
            
        Returns:
            Position size in portfolio currency
        """
        if avg_loss <= 0 or win_rate <= 0:
            self.logger.warning("Invalid parameters for Kelly criterion")
            return self.calculate_fixed_fractional(portfolio_value)
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win_rate, q = 1-p, b = avg_win/avg_loss
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Apply safety fraction
        kelly_fraction = kelly_fraction * fraction
        
        # Ensure non-negative and within limits
        kelly_fraction = max(0, min(kelly_fraction, self.max_position_pct))
        
        position_size = portfolio_value * kelly_fraction
        
        self.logger.debug(f"Kelly position size: {position_size:.2f} ({kelly_fraction:.2%} of portfolio)")
        return position_size
    
    def calculate_volatility_based(self, portfolio_value: float, 
                                   volatility: float, target_volatility: float = 0.15,
                                   price: float = 1.0) -> float:
        """
        Volatility-based position sizing (Vol targeting)
        
        Args:
            portfolio_value: Total portfolio value
            volatility: Asset volatility (annualized std dev)
            target_volatility: Target portfolio volatility
            price: Current asset price
            
        Returns:
            Position size in units
        """
        if volatility <= 0:
            self.logger.warning("Invalid volatility, using fixed fractional")
            return self.calculate_fixed_fractional(portfolio_value) / price
        
        # Scale position inversely with volatility
        volatility_scalar = target_volatility / volatility
        position_value = portfolio_value * volatility_scalar * self.max_position_pct
        
        # Cap at max position size
        max_value = portfolio_value * self.max_position_pct
        position_value = min(position_value, max_value)
        
        # Convert to units
        position_units = position_value / price
        
        self.logger.debug(f"Volatility-based position: {position_units:.4f} units (value: {position_value:.2f})")
        return position_units
    
    def calculate_risk_based(self, portfolio_value: float, entry_price: float,
                            stop_loss: float, risk_amount: float) -> float:
        """
        Risk-based position sizing (fixed $ risk per trade)
        
        Args:
            portfolio_value: Total portfolio value
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_amount: Amount to risk in portfolio currency
            
        Returns:
            Position size in units
        """
        if entry_price <= 0 or stop_loss <= 0:
            self.logger.warning("Invalid prices for risk-based sizing")
            return 0.0
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit <= 0:
            self.logger.warning("Invalid risk per unit")
            return 0.0
        
        # Calculate position size
        position_units = risk_amount / risk_per_unit
        
        # Cap at max position value
        position_value = position_units * entry_price
        max_value = portfolio_value * self.max_position_pct
        
        if position_value > max_value:
            position_units = max_value / entry_price
            
        self.logger.debug(f"Risk-based position: {position_units:.4f} units (risk: {risk_amount:.2f})")
        return position_units
    
    def calculate_optimal_position(self, portfolio_value: float, 
                                   entry_price: float,
                                   stop_loss: Optional[float] = None,
                                   volatility: Optional[float] = None,
                                   win_rate: Optional[float] = None,
                                   avg_win: Optional[float] = None,
                                   avg_loss: Optional[float] = None,
                                   strategy: str = 'adaptive') -> Dict:
        """
        Calculate optimal position size using specified strategy
        
        Args:
            portfolio_value: Total portfolio value
            entry_price: Entry price
            stop_loss: Stop loss price (optional)
            volatility: Asset volatility (optional)
            win_rate: Historical win rate (optional)
            avg_win: Average win amount (optional)
            avg_loss: Average loss amount (optional)
            strategy: Sizing strategy ('fixed', 'kelly', 'volatility', 'risk', 'adaptive')
            
        Returns:
            Dict with position sizing details
        """
        if portfolio_value <= 0 or entry_price <= 0:
            return {
                'units': 0.0,
                'value': 0.0,
                'strategy': strategy,
                'valid': False,
                'reason': 'Invalid input parameters'
            }
        
        position_units = 0.0
        
        if strategy == 'fixed':
            position_value = self.calculate_fixed_fractional(portfolio_value)
            position_units = position_value / entry_price
            
        elif strategy == 'kelly' and win_rate and avg_win and avg_loss:
            position_value = self.calculate_kelly_criterion(
                win_rate, avg_win, avg_loss, portfolio_value
            )
            position_units = position_value / entry_price
            
        elif strategy == 'volatility' and volatility:
            position_units = self.calculate_volatility_based(
                portfolio_value, volatility, price=entry_price
            )
            
        elif strategy == 'risk' and stop_loss:
            risk_amount = portfolio_value * 0.02  # 2% risk per trade
            position_units = self.calculate_risk_based(
                portfolio_value, entry_price, stop_loss, risk_amount
            )
            
        elif strategy == 'adaptive':
            # Use best available method
            methods = []
            
            if stop_loss:
                risk_amount = portfolio_value * 0.02
                risk_units = self.calculate_risk_based(
                    portfolio_value, entry_price, stop_loss, risk_amount
                )
                methods.append(('risk', risk_units))
                
            if volatility:
                vol_units = self.calculate_volatility_based(
                    portfolio_value, volatility, price=entry_price
                )
                methods.append(('volatility', vol_units))
                
            if win_rate and avg_win and avg_loss:
                kelly_value = self.calculate_kelly_criterion(
                    win_rate, avg_win, avg_loss, portfolio_value
                )
                kelly_units = kelly_value / entry_price
                methods.append(('kelly', kelly_units))
            
            if not methods:
                # Fallback to fixed fractional
                fixed_value = self.calculate_fixed_fractional(portfolio_value)
                position_units = fixed_value / entry_price
            else:
                # Use average of available methods
                position_units = np.mean([units for _, units in methods])
                
        else:
            # Default to fixed fractional
            position_value = self.calculate_fixed_fractional(portfolio_value)
            position_units = position_value / entry_price
        
        # Calculate position value
        position_value = position_units * entry_price
        
        # Apply leverage if needed
        leveraged_value = position_value
        leverage_used = 1.0
        
        if position_value > portfolio_value:
            leverage_used = position_value / portfolio_value
            if leverage_used > self.max_leverage:
                # Scale down to max leverage
                position_units = (portfolio_value * self.max_leverage) / entry_price
                position_value = position_units * entry_price
                leverage_used = self.max_leverage
        
        # Calculate position metrics
        position_pct = (position_value / leverage_used) / portfolio_value
        
        return {
            'units': position_units,
            'value': position_value,
            'position_pct': position_pct,
            'leverage': leverage_used,
            'strategy': strategy,
            'valid': True,
            'entry_price': entry_price,
            'max_loss': abs(entry_price - stop_loss) * position_units if stop_loss else None,
            'max_loss_pct': abs(entry_price - stop_loss) / entry_price if stop_loss else None
        }
    
    def adjust_position_for_correlation(self, base_position: Dict, 
                                       correlation: float,
                                       reduction_factor: float = 0.5) -> Dict:
        """
        Adjust position size based on portfolio correlation
        
        Args:
            base_position: Base position dict from calculate_optimal_position
            correlation: Correlation with existing positions (-1 to 1)
            reduction_factor: How much to reduce for high correlation (0-1)
            
        Returns:
            Adjusted position dict
        """
        if not base_position.get('valid', False):
            return base_position
        
        # Reduce position for highly correlated assets
        correlation_penalty = abs(correlation) * reduction_factor
        adjustment = 1.0 - correlation_penalty
        
        adjusted_position = base_position.copy()
        adjusted_position['units'] *= adjustment
        adjusted_position['value'] *= adjustment
        adjusted_position['correlation_adjusted'] = True
        adjusted_position['correlation'] = correlation
        adjusted_position['adjustment_factor'] = adjustment
        
        self.logger.info(f"Adjusted position for correlation {correlation:.2f}: "
                        f"{adjustment:.2%} of original")
        
        return adjusted_position
    
    def calculate_pyramid_sizing(self, portfolio_value: float, 
                                entry_price: float,
                                num_entries: int = 3,
                                total_position_pct: float = 0.1) -> List[Dict]:
        """
        Calculate position sizes for pyramiding into a position
        
        Args:
            portfolio_value: Total portfolio value
            entry_price: Initial entry price
            num_entries: Number of pyramid entries
            total_position_pct: Total position as % of portfolio
            
        Returns:
            List of position dicts for each entry
        """
        if num_entries <= 0:
            return []
        
        total_value = portfolio_value * total_position_pct
        entries = []
        
        # Decreasing position sizes (e.g., 50%, 30%, 20%)
        weights = [1.0 / (i + 1) for i in range(num_entries)]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        for i, weight in enumerate(weights):
            entry_value = total_value * weight
            entry_units = entry_value / entry_price
            
            entries.append({
                'entry_number': i + 1,
                'units': entry_units,
                'value': entry_value,
                'weight': weight,
                'entry_price': entry_price
            })
        
        self.logger.info(f"Calculated {num_entries} pyramid entries")
        return entries
    
    def calculate_scaling_out(self, position_units: float, 
                              num_exits: int = 3) -> List[Dict]:
        """
        Calculate position sizes for scaling out of a position
        
        Args:
            position_units: Total position size in units
            num_exits: Number of exit points
            
        Returns:
            List of exit dicts
        """
        if num_exits <= 0 or position_units <= 0:
            return []
        
        exits = []
        remaining_units = position_units
        
        # Increasing exit sizes (e.g., 30%, 30%, 40%)
        for i in range(num_exits):
            if i == num_exits - 1:
                # Last exit takes all remaining
                exit_units = remaining_units
            else:
                # Proportional exits
                exit_units = position_units / num_exits
                
            exits.append({
                'exit_number': i + 1,
                'units': exit_units,
                'pct_of_position': exit_units / position_units
            })
            
            remaining_units -= exit_units
        
        self.logger.info(f"Calculated {num_exits} scaling exits")
        return exits
