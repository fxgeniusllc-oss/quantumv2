"""
ADVANCED RISK MANAGEMENT SYSTEM
Comprehensive risk controls with position sizing and dynamic stop-loss mechanisms
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_single_trade_risk: float = 0.05  # 5%
    total_portfolio_risk: float = 0.15  # 15%
    stop_loss_threshold: float = 0.10  # 10%
    max_concurrent_trades: int = 10
    max_leverage: float = 10.0


class RiskManager:
    """
    Advanced risk management system
    """
    
    def __init__(self, config, initial_capital=100000):
        self.logger = logging.getLogger('RiskManager')
        self.config = config
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.open_positions = {}
        self.closed_trades = []
        
        # Load risk parameters
        risk_params = config.get_risk_parameters()
        self.limits = RiskLimits(
            max_single_trade_risk=risk_params.get('max_single_trade_risk', 0.05),
            total_portfolio_risk=risk_params.get('total_portfolio_risk', 0.15),
            stop_loss_threshold=risk_params.get('stop_loss_threshold', 0.10)
        )
        
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss_price: float, confidence: float = 1.0) -> float:
        """
        Calculate position size based on risk parameters
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            confidence: Confidence level (0-1)
            
        Returns:
            Position size in base currency
        """
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        # Maximum risk amount
        max_risk_amount = self.current_capital * self.limits.max_single_trade_risk * confidence
        
        # Position size
        position_size = max_risk_amount / risk_per_unit
        
        # Apply leverage limits
        max_position = (self.current_capital * self.limits.max_leverage) / entry_price
        position_size = min(position_size, max_position)
        
        self.logger.info(
            f"Position size for {symbol}: {position_size:.4f} "
            f"(risk: ${max_risk_amount:.2f})"
        )
        
        return position_size

    def can_open_position(self, symbol: str, position_size: float, 
                         entry_price: float) -> bool:
        """
        Check if new position can be opened
        
        Args:
            symbol: Trading symbol
            position_size: Proposed position size
            entry_price: Entry price
            
        Returns:
            Boolean indicating if position can be opened
        """
        # Check concurrent trades limit
        if len(self.open_positions) >= self.limits.max_concurrent_trades:
            self.logger.warning("Maximum concurrent trades reached")
            return False
        
        # Check total portfolio risk
        total_risk = self._calculate_total_risk()
        new_position_risk = (position_size * entry_price) / self.current_capital
        
        if total_risk + new_position_risk > self.limits.total_portfolio_risk:
            self.logger.warning(
                f"Total portfolio risk exceeded: "
                f"{total_risk + new_position_risk:.2%} > {self.limits.total_portfolio_risk:.2%}"
            )
            return False
        
        # Check available capital
        required_capital = position_size * entry_price / self.limits.max_leverage
        if required_capital > self.current_capital * 0.9:  # Keep 10% buffer
            self.logger.warning("Insufficient capital")
            return False
        
        return True

    def _calculate_total_risk(self) -> float:
        """Calculate total portfolio risk from open positions"""
        total_risk = 0
        for position in self.open_positions.values():
            position_value = position['size'] * position['entry_price']
            total_risk += position_value / self.current_capital
        
        return total_risk

    def open_position(self, symbol: str, size: float, entry_price: float,
                     stop_loss: float, take_profit: Optional[float] = None):
        """Open a new position"""
        if symbol in self.open_positions:
            self.logger.warning(f"Position already exists for {symbol}")
            return False
        
        self.open_positions[symbol] = {
            'size': size,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'unrealized_pnl': 0
        }
        
        self.logger.info(
            f"Opened position: {symbol} - Size: {size:.4f} @ {entry_price:.2f}"
        )
        
        return True

    def update_position(self, symbol: str, current_price: float):
        """Update position with current price and check stop loss"""
        if symbol not in self.open_positions:
            return
        
        position = self.open_positions[symbol]
        
        # Calculate unrealized PnL
        position['unrealized_pnl'] = (
            (current_price - position['entry_price']) * position['size']
        )
        
        # Check stop loss
        if current_price <= position['stop_loss']:
            self.logger.warning(f"Stop loss triggered for {symbol}")
            self.close_position(symbol, current_price, reason="stop_loss")
            return
        
        # Check take profit
        if position.get('take_profit') and current_price >= position['take_profit']:
            self.logger.info(f"Take profit reached for {symbol}")
            self.close_position(symbol, current_price, reason="take_profit")
            return

    def close_position(self, symbol: str, exit_price: float, reason: str = "manual"):
        """Close a position"""
        if symbol not in self.open_positions:
            return
        
        position = self.open_positions.pop(symbol)
        
        # Calculate realized PnL
        pnl = (exit_price - position['entry_price']) * position['size']
        pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        
        # Update capital
        self.current_capital += pnl
        
        # Record trade
        self.closed_trades.append({
            'symbol': symbol,
            'size': position['size'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason
        })
        
        self.logger.info(
            f"Closed position: {symbol} - PnL: ${pnl:.2f} ({pnl_pct:.2%}) - Reason: {reason}"
        )

    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        if not self.closed_trades:
            return {}
        
        total_pnl = sum(trade['pnl'] for trade in self.closed_trades)
        winning_trades = [t for t in self.closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in self.closed_trades if t['pnl'] <= 0]
        
        return {
            'total_trades': len(self.closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.closed_trades),
            'total_pnl': total_pnl,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'current_capital': self.current_capital,
            'open_positions': len(self.open_positions)
        }
