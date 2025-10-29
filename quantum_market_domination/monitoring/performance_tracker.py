"""
PERFORMANCE TRACKING AND METRICS
Tracks and analyzes trading performance with comprehensive metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging


class PerformanceTracker:
    """
    Comprehensive performance tracking and analysis
    """
    
    def __init__(self):
        """Initialize performance tracker"""
        self.logger = logging.getLogger('PerformanceTracker')
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.daily_returns: pd.Series = pd.Series(dtype=float)
        self.starting_capital: Optional[float] = None
        
    def record_trade(self, trade: Dict):
        """
        Record a completed trade
        
        Args:
            trade: Dict with trade details (symbol, entry, exit, pnl, etc.)
        """
        required_fields = ['symbol', 'entry_price', 'exit_price', 'quantity', 'pnl']
        if not all(field in trade for field in required_fields):
            self.logger.warning(f"Trade missing required fields: {required_fields}")
            return
        
        # Add timestamp if not present
        if 'timestamp' not in trade:
            trade['timestamp'] = datetime.now()
            
        # Calculate additional metrics
        trade['return'] = trade['pnl'] / (trade['entry_price'] * trade['quantity'])
        trade['win'] = trade['pnl'] > 0
        
        self.trades.append(trade)
        self.logger.debug(f"Recorded trade: {trade['symbol']} PnL: {trade['pnl']:.2f}")
    
    def update_equity(self, equity: float, timestamp: datetime = None):
        """
        Update equity curve
        
        Args:
            equity: Current portfolio equity
            timestamp: Timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        if self.starting_capital is None:
            self.starting_capital = equity
            
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'return': (equity - self.starting_capital) / self.starting_capital
        })
        
        # Update daily returns
        date = timestamp.date()
        self.daily_returns[date] = (equity - self.starting_capital) / self.starting_capital
    
    def get_win_rate(self) -> float:
        """Calculate win rate"""
        if not self.trades:
            return 0.0
            
        wins = sum(1 for trade in self.trades if trade.get('win', False))
        return wins / len(self.trades)
    
    def get_profit_factor(self) -> float:
        """Calculate profit factor (gross profits / gross losses)"""
        if not self.trades:
            return 0.0
            
        gross_profit = sum(trade['pnl'] for trade in self.trades if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in self.trades if trade['pnl'] < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
            
        return gross_profit / gross_loss
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if len(self.daily_returns) < 2:
            return 0.0
            
        excess_returns = self.daily_returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if excess_returns.std() == 0:
            return 0.0
            
        # Annualized Sharpe ratio
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        return sharpe
    
    def get_sortino_ratio(self, risk_free_rate: float = 0.0, target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio (uses downside deviation)
        
        Args:
            risk_free_rate: Annual risk-free rate
            target_return: Target return threshold
            
        Returns:
            Sortino ratio
        """
        if len(self.daily_returns) < 2:
            return 0.0
            
        excess_returns = self.daily_returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < target_return]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
            
        # Annualized Sortino ratio
        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
        return sortino
    
    def get_max_drawdown(self) -> Dict[str, float]:
        """
        Calculate maximum drawdown
        
        Returns:
            Dict with max drawdown info
        """
        if not self.equity_curve:
            return {'max_drawdown': 0.0, 'max_drawdown_pct': 0.0}
            
        equity_values = [point['equity'] for point in self.equity_curve]
        cummax = np.maximum.accumulate(equity_values)
        drawdown = (cummax - equity_values) / cummax
        
        max_dd_idx = np.argmax(drawdown)
        max_drawdown_pct = drawdown[max_dd_idx]
        max_drawdown = cummax[max_dd_idx] - equity_values[max_dd_idx]
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'max_drawdown_date': self.equity_curve[max_dd_idx]['timestamp']
        }
    
    def get_average_trade_metrics(self) -> Dict[str, float]:
        """Get average trade metrics"""
        if not self.trades:
            return {
                'avg_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_return': 0.0
            }
            
        winning_trades = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        losing_trades = [t['pnl'] for t in self.trades if t['pnl'] < 0]
        
        return {
            'avg_pnl': np.mean([t['pnl'] for t in self.trades]),
            'avg_win': np.mean(winning_trades) if winning_trades else 0.0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0.0,
            'avg_return': np.mean([t['return'] for t in self.trades]),
            'avg_win_return': np.mean([t['return'] for t in self.trades if t['pnl'] > 0]) if winning_trades else 0.0,
            'avg_loss_return': np.mean([t['return'] for t in self.trades if t['pnl'] < 0]) if losing_trades else 0.0
        }
    
    def get_expectancy(self) -> float:
        """
        Calculate expectancy (average expected profit per trade)
        
        Returns:
            Expectancy value
        """
        if not self.trades:
            return 0.0
            
        win_rate = self.get_win_rate()
        avg_metrics = self.get_average_trade_metrics()
        
        expectancy = (win_rate * avg_metrics['avg_win']) + \
                     ((1 - win_rate) * avg_metrics['avg_loss'])
        
        return expectancy
    
    def get_calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown)
        
        Returns:
            Calmar ratio
        """
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0
            
        # Calculate annual return
        total_days = (self.equity_curve[-1]['timestamp'] - 
                     self.equity_curve[0]['timestamp']).days
        
        if total_days == 0:
            return 0.0
            
        total_return = self.equity_curve[-1]['return']
        annual_return = total_return * (365 / total_days)
        
        # Get max drawdown
        max_dd = self.get_max_drawdown()['max_drawdown_pct']
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
            
        return annual_return / max_dd
    
    def get_consecutive_wins_losses(self) -> Dict[str, int]:
        """Get maximum consecutive wins and losses"""
        if not self.trades:
            return {'max_consecutive_wins': 0, 'max_consecutive_losses': 0}
            
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self.trades:
            if trade.get('win', False):
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return {
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses,
            'current_streak': current_wins if current_wins > 0 else -current_losses
        }
    
    def get_trade_duration_stats(self) -> Dict:
        """Get statistics on trade durations"""
        durations = []
        for trade in self.trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600  # hours
                durations.append(duration)
        
        if not durations:
            return {
                'avg_duration_hours': 0.0,
                'median_duration_hours': 0.0,
                'min_duration_hours': 0.0,
                'max_duration_hours': 0.0
            }
        
        return {
            'avg_duration_hours': np.mean(durations),
            'median_duration_hours': np.median(durations),
            'min_duration_hours': np.min(durations),
            'max_duration_hours': np.max(durations)
        }
    
    def get_performance_summary(self) -> Dict:
        """
        Get comprehensive performance summary
        
        Returns:
            Dict with all performance metrics
        """
        summary = {
            'total_trades': len(self.trades),
            'win_rate': self.get_win_rate(),
            'profit_factor': self.get_profit_factor(),
            'expectancy': self.get_expectancy(),
            'sharpe_ratio': self.get_sharpe_ratio(),
            'sortino_ratio': self.get_sortino_ratio(),
            'calmar_ratio': self.get_calmar_ratio(),
        }
        
        # Add average trade metrics
        summary.update(self.get_average_trade_metrics())
        
        # Add drawdown info
        summary.update(self.get_max_drawdown())
        
        # Add consecutive stats
        summary.update(self.get_consecutive_wins_losses())
        
        # Add duration stats
        summary.update(self.get_trade_duration_stats())
        
        # Add total PnL
        if self.trades:
            summary['total_pnl'] = sum(t['pnl'] for t in self.trades)
            summary['total_return'] = sum(t['return'] for t in self.trades)
        else:
            summary['total_pnl'] = 0.0
            summary['total_return'] = 0.0
        
        # Add equity info
        if self.equity_curve:
            summary['current_equity'] = self.equity_curve[-1]['equity']
            summary['starting_equity'] = self.starting_capital
            summary['equity_return'] = self.equity_curve[-1]['return']
        
        return summary
    
    def get_performance_by_symbol(self) -> Dict[str, Dict]:
        """
        Get performance metrics broken down by symbol
        
        Returns:
            Dict mapping symbols to their performance metrics
        """
        if not self.trades:
            return {}
        
        symbols = {}
        for trade in self.trades:
            symbol = trade['symbol']
            if symbol not in symbols:
                symbols[symbol] = []
            symbols[symbol].append(trade)
        
        performance_by_symbol = {}
        for symbol, trades in symbols.items():
            total_pnl = sum(t['pnl'] for t in trades)
            wins = sum(1 for t in trades if t.get('win', False))
            
            performance_by_symbol[symbol] = {
                'trades': len(trades),
                'total_pnl': total_pnl,
                'win_rate': wins / len(trades),
                'avg_pnl': total_pnl / len(trades)
            }
        
        return performance_by_symbol
    
    def export_trades(self) -> pd.DataFrame:
        """
        Export trades as DataFrame
        
        Returns:
            DataFrame with all trades
        """
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def export_equity_curve(self) -> pd.DataFrame:
        """
        Export equity curve as DataFrame
        
        Returns:
            DataFrame with equity curve
        """
        if not self.equity_curve:
            return pd.DataFrame()
        
        return pd.DataFrame(self.equity_curve)
