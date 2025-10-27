"""
DeFi Strategy Main Application
"""

import asyncio
import sys
import logging
from ultimate-defi-domination.core.config import DominanceConfig
from ultimate-defi-domination.engines.python_engine.strategy_engine import StrategyEngine


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('DeFiMain')


async def main():
    """Main entry point for DeFi strategy"""
    logger.info("Starting DeFi Domination System...")
    
    # Initialize configuration and strategy
    config = DominanceConfig()
    strategy = StrategyEngine(config)
    
    # Execute strategy
    try:
        await strategy.execute_strategy(duration=60)  # Run for 60 seconds
    except KeyboardInterrupt:
        logger.info("Strategy interrupted by user")
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
    
    logger.info("DeFi strategy completed")


if __name__ == "__main__":
    asyncio.run(main())
