"""
Performance Tracker
Tracks and analyzes trading performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging


@dataclass
class TradeRecord:
    """Individual trade record"""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    shares: int = 0
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    fees: float = 0.0
    holding_time: Optional[float] = None  # hours
    strategy: str = ""
    metadata: Dict = field(default_factory=dict)


class PerformanceTracker:
    """
    Comprehensive trading performance tracking and analysis
    Monitors wins, losses, drawdowns, and various performance metrics
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger('PerformanceTracker')
        self.config = config or {}
        
        # Trade history
        self.trades: List[TradeRecord] = []
        self.open_trades: Dict[str, TradeRecord] = {}
        
        # Performance metrics
        self.initial_capital = self.config.get('initial_capital', 100000.0)
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        
        # Equity curve
        self.equity_curve: List[Dict] = []
        
    def record_trade_entry(self,
                          trade_id: str,
                          symbol: str,
                          side: str,
                          entry_price: float,
                          shares: int,
                          strategy: str = "") -> TradeRecord:
        """
        Record trade entry
        
        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            side: 'buy' or 'sell'
            entry_price: Entry price
            shares: Number of shares
            strategy: Strategy name
            
        Returns:
            TradeRecord object
        """
        trade = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            entry_time=datetime.now(),
            entry_price=entry_price,
            shares=shares,
            strategy=strategy
        )
        
        self.open_trades[trade_id] = trade
        self.logger.info(f"Recorded entry for {trade_id}: {symbol} @ {entry_price}")
        
        return trade
        
    def record_trade_exit(self,
                         trade_id: str,
                         exit_price: float,
                         fees: float = 0.0) -> Optional[TradeRecord]:
        """
        Record trade exit and calculate P&L
        
        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            fees: Trading fees
            
        Returns:
            Completed TradeRecord
        """
        if trade_id not in self.open_trades:
            self.logger.error(f"Trade {trade_id} not found in open trades")
            return None
            
        trade = self.open_trades[trade_id]
        trade.exit_time = datetime.now()
        trade.exit_price = exit_price
        trade.fees = fees
        
        # Calculate holding time in hours
        trade.holding_time = (trade.exit_time - trade.entry_time).total_seconds() / 3600
        
        # Calculate P&L
        if trade.side == 'buy':
            trade.pnl = (exit_price - trade.entry_price) * trade.shares - fees
        else:  # sell/short
            trade.pnl = (trade.entry_price - exit_price) * trade.shares - fees
            
        trade.pnl_percentage = (trade.pnl / (trade.entry_price * trade.shares)) * 100
        
        # Update capital
        self.current_capital += trade.pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        
        # Move to completed trades
        self.trades.append(trade)
        del self.open_trades[trade_id]
        
        # Update equity curve
        self.equity_curve.append({
            'timestamp': trade.exit_time,
            'capital': self.current_capital,
            'trade_pnl': trade.pnl
        })
        
        self.logger.info(f"Closed {trade_id}: P&L = ${trade.pnl:.2f} ({trade.pnl_percentage:.2f}%)")
        
        return trade
        
    def get_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.trades:
            return {'message': 'No completed trades'}
            
        df = pd.DataFrame([vars(t) for t in self.trades])
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = df['pnl'].sum()
        total_fees = df['fees'].sum()
        net_pnl = total_pnl - total_fees
        
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        profit_factor = abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else float('inf')
        
        # Drawdown
        drawdown_metrics = self._calculate_drawdown()
        
        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(df)
        
        # Average holding time
        avg_holding_time = df['holding_time'].mean()
        
        # Return metrics
        total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate * 100,
            'total_pnl': total_pnl,
            'total_fees': total_fees,
            'net_pnl': net_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': drawdown_metrics['max_drawdown'],
            'max_drawdown_pct': drawdown_metrics['max_drawdown_pct'],
            'current_drawdown': drawdown_metrics['current_drawdown'],
            'avg_holding_time_hours': avg_holding_time,
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_return_pct': total_return
        }
        
    def _calculate_drawdown(self) -> Dict:
        """Calculate drawdown metrics"""
        if not self.equity_curve:
            return {
                'max_drawdown': 0,
                'max_drawdown_pct': 0,
                'current_drawdown': 0
            }
            
        equity = pd.Series([point['capital'] for point in self.equity_curve])
        
        # Calculate running maximum
        running_max = equity.expanding().max()
        
        # Calculate drawdown
        drawdown = equity - running_max
        drawdown_pct = (drawdown / running_max) * 100
        
        max_drawdown = drawdown.min()
        max_drawdown_pct = drawdown_pct.min()
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'current_drawdown': current_drawdown
        }
        
    def _calculate_sharpe_ratio(self, df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            df: DataFrame with trade data
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if len(df) < 2:
            return 0.0
            
        returns = df['pnl_percentage'] / 100
        
        avg_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
            
        # Annualize (assuming daily trades for simplicity)
        sharpe = (avg_return - risk_free_rate) / std_return * np.sqrt(252)
        
        return sharpe
        
    def get_trade_statistics_by_symbol(self) -> Dict[str, Dict]:
        """
        Get performance statistics grouped by symbol
        
        Returns:
            Dictionary of statistics per symbol
        """
        if not self.trades:
            return {}
            
        df = pd.DataFrame([vars(t) for t in self.trades])
        
        stats_by_symbol = {}
        
        for symbol in df['symbol'].unique():
            symbol_trades = df[df['symbol'] == symbol]
            
            winning = len(symbol_trades[symbol_trades['pnl'] > 0])
            total = len(symbol_trades)
            
            stats_by_symbol[symbol] = {
                'total_trades': total,
                'win_rate': (winning / total * 100) if total > 0 else 0,
                'total_pnl': symbol_trades['pnl'].sum(),
                'avg_pnl': symbol_trades['pnl'].mean(),
                'best_trade': symbol_trades['pnl'].max(),
                'worst_trade': symbol_trades['pnl'].min()
            }
            
        return stats_by_symbol
        
    def get_trade_statistics_by_strategy(self) -> Dict[str, Dict]:
        """
        Get performance statistics grouped by strategy
        
        Returns:
            Dictionary of statistics per strategy
        """
        if not self.trades:
            return {}
            
        df = pd.DataFrame([vars(t) for t in self.trades])
        
        stats_by_strategy = {}
        
        for strategy in df['strategy'].unique():
            if not strategy:
                continue
                
            strategy_trades = df[df['strategy'] == strategy]
            
            winning = len(strategy_trades[strategy_trades['pnl'] > 0])
            total = len(strategy_trades)
            
            stats_by_strategy[strategy] = {
                'total_trades': total,
                'win_rate': (winning / total * 100) if total > 0 else 0,
                'total_pnl': strategy_trades['pnl'].sum(),
                'avg_pnl': strategy_trades['pnl'].mean(),
                'best_trade': strategy_trades['pnl'].max(),
                'worst_trade': strategy_trades['pnl'].min()
            }
            
        return stats_by_strategy
        
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve as DataFrame
        
        Returns:
            DataFrame with timestamp and capital
        """
        if not self.equity_curve:
            return pd.DataFrame()
            
        return pd.DataFrame(self.equity_curve)
        
    def export_trades(self, filepath: str):
        """
        Export trade history to CSV
        
        Args:
            filepath: Output file path
        """
        if not self.trades:
            self.logger.warning("No trades to export")
            return
            
        df = pd.DataFrame([vars(t) for t in self.trades])
        df.to_csv(filepath, index=False)
        self.logger.info(f"Exported {len(self.trades)} trades to {filepath}")
        
    def reset(self):
        """Reset all performance data"""
        self.trades = []
        self.open_trades = {}
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.equity_curve = []
        self.logger.info("Performance tracker reset")
