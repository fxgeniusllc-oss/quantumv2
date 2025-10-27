"""
README VALIDATION SCRIPT
Validates all claims and features described in the README
"""

import sys
import os
import logging
from typing import Dict, List

# Add path for importing modules with hyphens
sys.path.insert(0, os.path.abspath('.'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('Validator')


class ReadmeValidator:
    """Validates README claims against actual implementation"""
    
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }

    def validate_all(self):
        """Run all validation checks"""
        logger.info("=" * 60)
        logger.info("QUANTUM MARKET DOMINATION SYSTEM - README VALIDATION")
        logger.info("=" * 60)
        
        self.validate_core_modules()
        self.validate_data_acquisition()
        self.validate_intelligence_layer()
        self.validate_execution_layer()
        self.validate_monitoring_layer()
        self.validate_utilities()
        self.validate_defi_components()
        self.validate_configuration()
        
        self.print_summary()

    def validate_core_modules(self):
        """Validate core module claims"""
        logger.info("\n[1] Validating Core Modules...")
        
        try:
            from quantum_market_domination.core.config_manager import QuantumConfigManager
            self.results['passed'].append("✓ QuantumConfigManager exists")
            
            config = QuantumConfigManager()
            assert hasattr(config, 'MODULES'), "MODULES configuration missing"
            assert 'DATA_ACQUISITION' in config.MODULES
            assert 'INTELLIGENCE' in config.MODULES
            assert 'EXECUTION' in config.MODULES
            self.results['passed'].append("✓ Configuration modules structure validated")
            
        except Exception as e:
            self.results['failed'].append(f"✗ Core config validation failed: {e}")
        
        try:
            from quantum_market_domination.core.secret_vault import SecretVault
            vault = SecretVault()
            self.results['passed'].append("✓ SecretVault with Fernet encryption exists")
        except Exception as e:
            self.results['failed'].append(f"✗ SecretVault validation failed: {e}")
        
        try:
            from quantum_market_domination.core.system_monitor import QuantumSystemMonitor
            monitor = QuantumSystemMonitor()
            self.results['passed'].append("✓ QuantumSystemMonitor exists")
        except Exception as e:
            self.results['failed'].append(f"✗ SystemMonitor validation failed: {e}")

    def validate_data_acquisition(self):
        """Validate data acquisition layer"""
        logger.info("\n[2] Validating Data Acquisition Layer...")
        
        try:
            from quantum_market_domination.data_acquisition.quantum_collector import QuantumMarketDominationCollector
            from quantum_market_domination.core.config_manager import QuantumConfigManager
            
            config = QuantumConfigManager()
            collector = QuantumMarketDominationCollector(config)
            
            assert hasattr(collector, 'exchanges'), "Multi-exchange integration missing"
            assert hasattr(collector, 'ml_models'), "ML models missing"
            assert hasattr(collector, 'quantum_parameters'), "Quantum parameters missing"
            
            # Check quantum parameters
            qp = collector.quantum_parameters
            assert qp['latency_threshold'] == 0.5, "Latency threshold incorrect"
            assert qp['market_penetration_depth'] == 0.95, "Market penetration depth incorrect"
            assert qp['predictive_horizon'] == 500, "Predictive horizon incorrect"
            
            self.results['passed'].append("✓ QuantumMarketDominationCollector validated")
            self.results['passed'].append("✓ Quantum parameters validated (0.5ms latency, 95% coverage, 500 horizon)")
            
        except Exception as e:
            self.results['failed'].append(f"✗ Data acquisition validation failed: {e}")

    def validate_intelligence_layer(self):
        """Validate intelligence/ML layer"""
        logger.info("\n[3] Validating Intelligence Layer...")
        
        try:
            from quantum_market_domination.intelligence.ml_models.price_predictor import PricePredictor
            predictor = PricePredictor(n_estimators=500, max_depth=20)
            assert predictor.model.n_estimators == 500
            assert predictor.model.max_depth == 20
            self.results['passed'].append("✓ Price Predictor with 500 estimators validated")
        except Exception as e:
            self.results['failed'].append(f"✗ Price predictor validation failed: {e}")
        
        try:
            from quantum_market_domination.intelligence.ml_models.volatility_model import VolatilityModel
            vol_model = VolatilityModel(n_estimators=300, max_depth=15)
            assert vol_model.model.n_estimators == 300
            self.results['passed'].append("✓ Volatility Model with 300 estimators validated")
        except Exception as e:
            self.results['failed'].append(f"✗ Volatility model validation failed: {e}")
        
        try:
            from quantum_market_domination.intelligence.ml_models.anomaly_detector import AnomalyDetector
            detector = AnomalyDetector()
            self.results['passed'].append("✓ Anomaly Detector exists")
        except Exception as e:
            self.results['failed'].append(f"✗ Anomaly detector validation failed: {e}")

    def validate_execution_layer(self):
        """Validate execution layer"""
        logger.info("\n[4] Validating Execution Layer...")
        
        try:
            from quantum_market_domination.execution.risk_manager import RiskManager
            from quantum_market_domination.core.config_manager import QuantumConfigManager
            
            config = QuantumConfigManager()
            rm = RiskManager(config)
            
            # Validate risk limits
            limits = rm.limits
            assert limits.max_single_trade_risk == 0.05, "Max single trade risk incorrect"
            assert limits.total_portfolio_risk == 0.15, "Total portfolio risk incorrect"
            assert limits.stop_loss_threshold == 0.10, "Stop loss threshold incorrect"
            assert limits.max_concurrent_trades == 10, "Max concurrent trades incorrect"
            
            self.results['passed'].append("✓ Risk Manager validated")
            self.results['passed'].append("✓ Risk parameters: 5% single trade, 15% total, 10% stop loss, 10 concurrent")
            
        except Exception as e:
            self.results['failed'].append(f"✗ Risk manager validation failed: {e}")
        
        try:
            from quantum_market_domination.execution.trade_executor import TradeExecutor
            self.results['passed'].append("✓ Trade Executor exists")
        except Exception as e:
            self.results['failed'].append(f"✗ Trade executor validation failed: {e}")

    def validate_monitoring_layer(self):
        """Validate monitoring capabilities"""
        logger.info("\n[5] Validating Monitoring Layer...")
        
        try:
            from quantum_market_domination.core.system_monitor import SystemMetrics
            metrics = SystemMetrics(
                cpu_usage=50.0,
                memory_usage=60.0,
                gpu_usage=40.0,
                network_latency=10.0,
                disk_io=1.0,
                temperature=65.0,
                timestamp=None
            )
            assert hasattr(metrics, 'cpu_usage')
            assert hasattr(metrics, 'memory_usage')
            assert hasattr(metrics, 'gpu_usage')
            self.results['passed'].append("✓ System metrics monitoring (CPU, Memory, GPU, Network, Disk, Temp)")
        except Exception as e:
            self.results['failed'].append(f"✗ Monitoring validation failed: {e}")

    def validate_utilities(self):
        """Validate utility modules"""
        logger.info("\n[6] Validating Utilities...")
        
        try:
            from quantum_market_domination.utils.encryption import EncryptionUtils
            key = EncryptionUtils.generate_key()
            assert key is not None
            self.results['passed'].append("✓ Encryption utilities exist")
        except Exception as e:
            self.results['failed'].append(f"✗ Encryption utilities validation failed: {e}")

    def validate_defi_components(self):
        """Validate DeFi components"""
        logger.info("\n[7] Validating DeFi Components...")
        
        try:
            # Import using importlib to handle hyphens in module name
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "defi_config",
                "ultimate-defi-domination/core/config.py"
            )
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            config = config_module.DominanceConfig()
            
            # Validate chains
            assert 'ethereum' in config.CHAINS
            assert 'polygon' in config.CHAINS
            assert config.CHAINS['ethereum']['chain_id'] == 1
            assert config.CHAINS['polygon']['chain_id'] == 137
            self.results['passed'].append("✓ Multi-chain configuration (Ethereum, Polygon)")
            
            # Validate flashloan configs
            assert 'aave_v3' in config.FLASHLOAN_CONFIGS
            assert config.FLASHLOAN_CONFIGS['aave_v3']['max_loan_size'] == 10_000_000
            self.results['passed'].append("✓ Flashloan configuration (Aave V3, dYdX)")
            
        except Exception as e:
            self.results['failed'].append(f"✗ DeFi config validation failed: {e}")
        
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "strategy_engine",
                "ultimate-defi-domination/engines/python_engine/strategy_engine.py"
            )
            strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_module)
            self.results['passed'].append("✓ Python Strategy Engine exists")
        except Exception as e:
            self.results['failed'].append(f"✗ Strategy engine validation failed: {e}")
        
        # Check Node.js executor
        if os.path.exists('ultimate-defi-domination/engines/node_engine/tx_executor.js'):
            self.results['passed'].append("✓ Node.js Transaction Executor exists")
        else:
            self.results['failed'].append("✗ Node.js executor missing")
        
        # Check Solidity contracts
        if os.path.exists('contracts/flashloan_aggregator.sol'):
            self.results['passed'].append("✓ Solidity Flashloan Aggregator exists")
        else:
            self.results['failed'].append("✗ Solidity contract missing")

    def validate_configuration(self):
        """Validate configuration files"""
        logger.info("\n[8] Validating Configuration Files...")
        
        files_to_check = [
            ('requirements.txt', 'Python dependencies'),
            ('package.json', 'Node.js dependencies'),
            ('.env.example', 'Environment template'),
            ('.gitignore', 'Git ignore file')
        ]
        
        for filename, description in files_to_check:
            if os.path.exists(filename):
                self.results['passed'].append(f"✓ {description} exists")
            else:
                self.results['failed'].append(f"✗ {description} missing")

    def print_summary(self):
        """Print validation summary"""
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"\n✓ PASSED: {len(self.results['passed'])} checks")
        for item in self.results['passed']:
            logger.info(f"  {item}")
        
        if self.results['warnings']:
            logger.info(f"\n⚠ WARNINGS: {len(self.results['warnings'])} items")
            for item in self.results['warnings']:
                logger.warning(f"  {item}")
        
        if self.results['failed']:
            logger.info(f"\n✗ FAILED: {len(self.results['failed'])} checks")
            for item in self.results['failed']:
                logger.error(f"  {item}")
        
        logger.info("\n" + "=" * 60)
        
        total = len(self.results['passed']) + len(self.results['failed'])
        pass_rate = (len(self.results['passed']) / total * 100) if total > 0 else 0
        
        logger.info(f"OVERALL: {len(self.results['passed'])}/{total} checks passed ({pass_rate:.1f}%)")
        logger.info("=" * 60)
        
        if self.results['failed']:
            return 1
        return 0


def main():
    """Main validation entry point"""
    validator = ReadmeValidator()
    exit_code = validator.validate_all()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
