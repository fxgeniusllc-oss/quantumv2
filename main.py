"""
QUANTUM MARKET DOMINATION - MAIN APPLICATION
Entry point for the trading system
"""

import asyncio
import sys
import signal
import logging
from quantum_market_domination.core.config_manager import QuantumConfigManager
from quantum_market_domination.core.system_monitor import QuantumSystemMonitor
from quantum_market_domination.data_acquisition.quantum_collector import QuantumMarketDominationCollector
from quantum_market_domination.execution.risk_manager import RiskManager
from quantum_market_domination.execution.trade_executor import TradeExecutor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('QuantumMain')


class QuantumTradingSystem:
    """
    Main Quantum Trading System orchestrator
    """
    
    def __init__(self):
        logger.info("Initializing Quantum Market Domination System...")
        
        # Initialize core components
        self.config = QuantumConfigManager()
        self.system_monitor = QuantumSystemMonitor(
            alert_threshold=self.config.get_alert_threshold()
        )
        self.risk_manager = RiskManager(self.config)
        self.trade_executor = TradeExecutor(self.config, self.risk_manager)
        self.collector = QuantumMarketDominationCollector(self.config)
        
        # Add exchanges to executor
        for name, exchange in self.collector.exchanges.items():
            self.trade_executor.add_exchange(name, exchange)
        
        logger.info("System initialized successfully")

    async def start(self, symbols=None, duration=None):
        """
        Start the trading system
        
        Args:
            symbols: List of symbols to trade (default: major crypto pairs)
            duration: Optional duration in seconds (None for continuous)
        """
        if symbols is None:
            symbols = [
                'BTC/USDT', 'ETH/USDT', 'XRP/USDT',
                'DOGE/USDT', 'ADA/USDT'
            ]
        
        logger.info(f"Starting Quantum Trading System for {len(symbols)} symbols")
        logger.info(f"Environment: {self.config.environment}")
        
        # Start system monitor
        monitor_task = asyncio.create_task(
            self.system_monitor.monitor_system(interval=10)
        )
        
        # Start trading strategy
        try:
            await self.collector.execute_quantum_strategy(symbols, duration)
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        finally:
            monitor_task.cancel()
            await self.shutdown()

    async def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("Shutting down Quantum Trading System...")
        
        # Close all open positions
        for symbol in list(self.risk_manager.open_positions.keys()):
            logger.info(f"Closing position: {symbol}")
            # In production, fetch current price and close
            # self.risk_manager.close_position(symbol, current_price, "shutdown")
        
        # Display performance metrics
        metrics = self.risk_manager.get_performance_metrics()
        if metrics:
            logger.info("=== Performance Summary ===")
            logger.info(f"Total Trades: {metrics.get('total_trades', 0)}")
            logger.info(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
            logger.info(f"Total Return: {metrics.get('total_return', 0):.2%}")
            logger.info(f"Current Capital: ${metrics.get('current_capital', 0):.2f}")
        
        # Display system metrics
        sys_metrics = self.system_monitor.get_metrics_summary()
        if sys_metrics:
            logger.info("=== System Metrics ===")
            logger.info(f"Avg CPU: {sys_metrics.get('avg_cpu', 0):.1f}%")
            logger.info(f"Avg Memory: {sys_metrics.get('avg_memory', 0):.1f}%")
        
        # Display market intelligence
        intel_summary = self.collector.get_market_intelligence_summary()
        logger.info("=== Market Intelligence ===")
        logger.info(f"Correlations tracked: {intel_summary.get('correlation_count', 0)}")
        logger.info(f"Anomalies detected: {intel_summary.get('anomalies_detected', 0)}")
        logger.info(f"Predictive surfaces: {intel_summary.get('predictive_surfaces', 0)}")
        
        logger.info("Shutdown complete")


async def main():
    """Main entry point"""
    system = QuantumTradingSystem()
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Signal received, initiating shutdown...")
        raise KeyboardInterrupt
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start system
    await system.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
