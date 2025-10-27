"""
TRADE EXECUTOR
Multi-exchange trade execution with advanced order management
"""

import asyncio
import logging
from typing import Dict, Optional
import ccxt


class TradeExecutor:
    """
    Multi-exchange trade execution system
    """
    
    def __init__(self, config, risk_manager):
        self.logger = logging.getLogger('TradeExecutor')
        self.config = config
        self.risk_manager = risk_manager
        self.exchanges = {}
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

    def add_exchange(self, exchange_name: str, exchange_instance):
        """Add an exchange for trading"""
        self.exchanges[exchange_name] = exchange_instance
        self.logger.info(f"Added exchange: {exchange_name}")

    async def execute_market_order(self, exchange_name: str, symbol: str,
                                   side: str, amount: float) -> Optional[Dict]:
        """
        Execute market order
        
        Args:
            exchange_name: Name of the exchange
            symbol: Trading symbol
            side: 'buy' or 'sell'
            amount: Order amount
            
        Returns:
            Order result dictionary
        """
        if exchange_name not in self.exchanges:
            self.logger.error(f"Exchange {exchange_name} not found")
            return None
        
        exchange = self.exchanges[exchange_name]
        
        try:
            self.logger.info(
                f"Executing {side} market order: {amount} {symbol} on {exchange_name}"
            )
            
            # Execute order
            order = await asyncio.to_thread(
                exchange.create_market_order,
                symbol,
                side,
                amount
            )
            
            self.logger.info(f"Order executed: {order.get('id')}")
            return order
            
        except ccxt.InsufficientFunds as e:
            self.logger.error(f"Insufficient funds: {e}")
            return None
        except ccxt.NetworkError as e:
            self.logger.error(f"Network error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            return None

    async def execute_limit_order(self, exchange_name: str, symbol: str,
                                  side: str, amount: float, price: float) -> Optional[Dict]:
        """Execute limit order"""
        if exchange_name not in self.exchanges:
            self.logger.error(f"Exchange {exchange_name} not found")
            return None
        
        exchange = self.exchanges[exchange_name]
        
        try:
            self.logger.info(
                f"Executing {side} limit order: {amount} {symbol} @ {price} on {exchange_name}"
            )
            
            order = await asyncio.to_thread(
                exchange.create_limit_order,
                symbol,
                side,
                amount,
                price
            )
            
            self.logger.info(f"Limit order placed: {order.get('id')}")
            return order
            
        except Exception as e:
            self.logger.error(f"Error executing limit order: {e}")
            return None

    async def cancel_order(self, exchange_name: str, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        if exchange_name not in self.exchanges:
            return False
        
        exchange = self.exchanges[exchange_name]
        
        try:
            await asyncio.to_thread(exchange.cancel_order, order_id, symbol)
            self.logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False

    async def get_order_status(self, exchange_name: str, order_id: str,
                              symbol: str) -> Optional[Dict]:
        """Get order status"""
        if exchange_name not in self.exchanges:
            return None
        
        exchange = self.exchanges[exchange_name]
        
        try:
            order = await asyncio.to_thread(
                exchange.fetch_order,
                order_id,
                symbol
            )
            return order
        except Exception as e:
            self.logger.error(f"Error fetching order status: {e}")
            return None

    async def execute_strategy_signal(self, signal: Dict):
        """
        Execute a trading signal with risk management
        
        Args:
            signal: Dictionary containing trading signal information
        """
        symbol = signal.get('symbol')
        action = signal.get('action')  # 'buy' or 'sell'
        exchange = signal.get('exchange', 'binance')
        entry_price = signal.get('entry_price')
        stop_loss = signal.get('stop_loss')
        confidence = signal.get('confidence', 1.0)
        
        if not all([symbol, action, entry_price, stop_loss]):
            self.logger.error("Invalid signal: missing required fields")
            return
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            symbol,
            entry_price,
            stop_loss,
            confidence
        )
        
        # Check if position can be opened
        if not self.risk_manager.can_open_position(symbol, position_size, entry_price):
            self.logger.warning(f"Cannot open position for {symbol}")
            return
        
        # Execute order
        order = await self.execute_market_order(
            exchange,
            symbol,
            action,
            position_size
        )
        
        if order:
            # Register position with risk manager
            self.risk_manager.open_position(
                symbol,
                position_size,
                entry_price,
                stop_loss,
                signal.get('take_profit')
            )
