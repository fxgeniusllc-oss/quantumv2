"""
REGULATORY COMPLIANCE MONITORING
Monitors and enforces regulatory compliance requirements
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum


class ComplianceRule(Enum):
    """Types of compliance rules"""
    POSITION_LIMIT = "position_limit"
    DAILY_TRADE_LIMIT = "daily_trade_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    WASH_SALE = "wash_sale"
    PATTERN_DAY_TRADER = "pattern_day_trader"
    CONCENTRATION = "concentration"
    CIRCUIT_BREAKER = "circuit_breaker"
    KYC_VERIFICATION = "kyc_verification"


class ComplianceStatus(Enum):
    """Compliance status"""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"


class ComplianceChecker:
    """
    Regulatory compliance monitoring and enforcement
    """
    
    def __init__(self):
        """Initialize compliance checker"""
        self.logger = logging.getLogger('ComplianceChecker')
        self.violations: List[Dict] = []
        self.warnings: List[Dict] = []
        self.compliance_rules = {
            ComplianceRule.POSITION_LIMIT: {
                'enabled': True,
                'max_position_pct': 0.25,  # 25% of portfolio
                'max_single_position_value': 1000000  # $1M
            },
            ComplianceRule.DAILY_TRADE_LIMIT: {
                'enabled': True,
                'max_trades_per_day': 100,
                'max_volume_per_day': 10000000  # $10M
            },
            ComplianceRule.LEVERAGE_LIMIT: {
                'enabled': True,
                'max_leverage': 3.0,
                'warning_leverage': 2.5
            },
            ComplianceRule.CONCENTRATION: {
                'enabled': True,
                'max_single_asset_pct': 0.30,  # 30%
                'max_sector_pct': 0.50  # 50%
            },
            ComplianceRule.PATTERN_DAY_TRADER: {
                'enabled': True,
                'min_equity': 25000,  # $25k minimum for PDT
                'max_day_trades_per_week': 3  # If under $25k
            }
        }
        self.trade_history: List[Dict] = []
        self.daily_stats: Dict[str, Dict] = {}
        
    def check_position_limits(self, position: Dict, portfolio_value: float) -> Dict:
        """
        Check if position complies with position limits
        
        Args:
            position: Position dict with symbol, size, value
            portfolio_value: Total portfolio value
            
        Returns:
            Dict with compliance status
        """
        rule = self.compliance_rules[ComplianceRule.POSITION_LIMIT]
        
        if not rule['enabled']:
            return {'status': ComplianceStatus.COMPLIANT.value, 'rule': ComplianceRule.POSITION_LIMIT.value}
        
        position_value = position.get('value', 0)
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
        
        violations = []
        
        # Check percentage limit
        if position_pct > rule['max_position_pct']:
            violations.append({
                'type': 'percentage_limit',
                'current': position_pct,
                'limit': rule['max_position_pct'],
                'message': f"Position {position['symbol']} exceeds {rule['max_position_pct']:.1%} limit "
                          f"(current: {position_pct:.1%})"
            })
        
        # Check absolute value limit
        if position_value > rule['max_single_position_value']:
            violations.append({
                'type': 'value_limit',
                'current': position_value,
                'limit': rule['max_single_position_value'],
                'message': f"Position {position['symbol']} exceeds ${rule['max_single_position_value']:,.0f} limit "
                          f"(current: ${position_value:,.0f})"
            })
        
        if violations:
            self._record_violation(ComplianceRule.POSITION_LIMIT, violations)
            return {
                'status': ComplianceStatus.VIOLATION.value,
                'rule': ComplianceRule.POSITION_LIMIT.value,
                'violations': violations
            }
        
        return {'status': ComplianceStatus.COMPLIANT.value, 'rule': ComplianceRule.POSITION_LIMIT.value}
    
    def check_daily_trade_limits(self, proposed_trade: Dict) -> Dict:
        """
        Check if proposed trade complies with daily trade limits
        
        Args:
            proposed_trade: Proposed trade dict
            
        Returns:
            Dict with compliance status
        """
        rule = self.compliance_rules[ComplianceRule.DAILY_TRADE_LIMIT]
        
        if not rule['enabled']:
            return {'status': ComplianceStatus.COMPLIANT.value, 'rule': ComplianceRule.DAILY_TRADE_LIMIT.value}
        
        today = datetime.now().date()
        
        if today not in self.daily_stats:
            self.daily_stats[today] = {
                'trade_count': 0,
                'total_volume': 0.0
            }
        
        stats = self.daily_stats[today]
        violations = []
        
        # Check trade count
        if stats['trade_count'] >= rule['max_trades_per_day']:
            violations.append({
                'type': 'trade_count',
                'current': stats['trade_count'],
                'limit': rule['max_trades_per_day'],
                'message': f"Daily trade limit reached ({rule['max_trades_per_day']} trades)"
            })
        
        # Check volume
        trade_value = proposed_trade.get('value', 0)
        if stats['total_volume'] + trade_value > rule['max_volume_per_day']:
            violations.append({
                'type': 'volume_limit',
                'current': stats['total_volume'],
                'proposed': trade_value,
                'limit': rule['max_volume_per_day'],
                'message': f"Daily volume limit would be exceeded "
                          f"(current: ${stats['total_volume']:,.0f}, "
                          f"limit: ${rule['max_volume_per_day']:,.0f})"
            })
        
        if violations:
            self._record_violation(ComplianceRule.DAILY_TRADE_LIMIT, violations)
            return {
                'status': ComplianceStatus.VIOLATION.value,
                'rule': ComplianceRule.DAILY_TRADE_LIMIT.value,
                'violations': violations
            }
        
        return {'status': ComplianceStatus.COMPLIANT.value, 'rule': ComplianceRule.DAILY_TRADE_LIMIT.value}
    
    def check_leverage_limits(self, leverage: float) -> Dict:
        """
        Check if leverage complies with limits
        
        Args:
            leverage: Current leverage ratio
            
        Returns:
            Dict with compliance status
        """
        rule = self.compliance_rules[ComplianceRule.LEVERAGE_LIMIT]
        
        if not rule['enabled']:
            return {'status': ComplianceStatus.COMPLIANT.value, 'rule': ComplianceRule.LEVERAGE_LIMIT.value}
        
        if leverage > rule['max_leverage']:
            violation = {
                'type': 'leverage_limit',
                'current': leverage,
                'limit': rule['max_leverage'],
                'message': f"Leverage exceeds maximum ({leverage:.2f}x > {rule['max_leverage']:.2f}x)"
            }
            self._record_violation(ComplianceRule.LEVERAGE_LIMIT, [violation])
            return {
                'status': ComplianceStatus.VIOLATION.value,
                'rule': ComplianceRule.LEVERAGE_LIMIT.value,
                'violations': [violation]
            }
        
        if leverage > rule['warning_leverage']:
            warning = {
                'type': 'leverage_warning',
                'current': leverage,
                'threshold': rule['warning_leverage'],
                'message': f"Leverage approaching maximum ({leverage:.2f}x > {rule['warning_leverage']:.2f}x)"
            }
            self._record_warning(ComplianceRule.LEVERAGE_LIMIT, warning)
            return {
                'status': ComplianceStatus.WARNING.value,
                'rule': ComplianceRule.LEVERAGE_LIMIT.value,
                'warnings': [warning]
            }
        
        return {'status': ComplianceStatus.COMPLIANT.value, 'rule': ComplianceRule.LEVERAGE_LIMIT.value}
    
    def check_concentration_limits(self, portfolio: Dict) -> Dict:
        """
        Check portfolio concentration limits
        
        Args:
            portfolio: Dict with positions and total value
            
        Returns:
            Dict with compliance status
        """
        rule = self.compliance_rules[ComplianceRule.CONCENTRATION]
        
        if not rule['enabled']:
            return {'status': ComplianceStatus.COMPLIANT.value, 'rule': ComplianceRule.CONCENTRATION.value}
        
        total_value = portfolio.get('total_value', 0)
        if total_value == 0:
            return {'status': ComplianceStatus.COMPLIANT.value, 'rule': ComplianceRule.CONCENTRATION.value}
        
        positions = portfolio.get('positions', [])
        violations = []
        
        # Check single asset concentration
        for position in positions:
            position_pct = position['value'] / total_value
            if position_pct > rule['max_single_asset_pct']:
                violations.append({
                    'type': 'asset_concentration',
                    'symbol': position['symbol'],
                    'current': position_pct,
                    'limit': rule['max_single_asset_pct'],
                    'message': f"Asset {position['symbol']} exceeds concentration limit "
                              f"({position_pct:.1%} > {rule['max_single_asset_pct']:.1%})"
                })
        
        if violations:
            self._record_violation(ComplianceRule.CONCENTRATION, violations)
            return {
                'status': ComplianceStatus.VIOLATION.value,
                'rule': ComplianceRule.CONCENTRATION.value,
                'violations': violations
            }
        
        return {'status': ComplianceStatus.COMPLIANT.value, 'rule': ComplianceRule.CONCENTRATION.value}
    
    def check_pattern_day_trader(self, equity: float, day_trades_this_week: int) -> Dict:
        """
        Check Pattern Day Trader (PDT) rule compliance
        
        Args:
            equity: Account equity
            day_trades_this_week: Number of day trades in the past 5 days
            
        Returns:
            Dict with compliance status
        """
        rule = self.compliance_rules[ComplianceRule.PATTERN_DAY_TRADER]
        
        if not rule['enabled']:
            return {'status': ComplianceStatus.COMPLIANT.value, 'rule': ComplianceRule.PATTERN_DAY_TRADER.value}
        
        # If equity is below minimum, check day trade count
        if equity < rule['min_equity']:
            if day_trades_this_week >= rule['max_day_trades_per_week']:
                violation = {
                    'type': 'pdt_violation',
                    'equity': equity,
                    'min_equity': rule['min_equity'],
                    'day_trades': day_trades_this_week,
                    'limit': rule['max_day_trades_per_week'],
                    'message': f"Pattern Day Trader rule violation: "
                              f"{day_trades_this_week} day trades with ${equity:,.0f} equity "
                              f"(minimum ${rule['min_equity']:,.0f} required)"
                }
                self._record_violation(ComplianceRule.PATTERN_DAY_TRADER, [violation])
                return {
                    'status': ComplianceStatus.VIOLATION.value,
                    'rule': ComplianceRule.PATTERN_DAY_TRADER.value,
                    'violations': [violation]
                }
            
            # Warning if approaching limit
            if day_trades_this_week == rule['max_day_trades_per_week'] - 1:
                warning = {
                    'type': 'pdt_warning',
                    'day_trades': day_trades_this_week,
                    'limit': rule['max_day_trades_per_week'],
                    'message': f"Approaching PDT limit: {day_trades_this_week} of "
                              f"{rule['max_day_trades_per_week']} day trades used"
                }
                self._record_warning(ComplianceRule.PATTERN_DAY_TRADER, warning)
                return {
                    'status': ComplianceStatus.WARNING.value,
                    'rule': ComplianceRule.PATTERN_DAY_TRADER.value,
                    'warnings': [warning]
                }
        
        return {'status': ComplianceStatus.COMPLIANT.value, 'rule': ComplianceRule.PATTERN_DAY_TRADER.value}
    
    def check_wash_sale(self, proposed_trade: Dict) -> Dict:
        """
        Check for potential wash sale violations (simplified)
        
        Args:
            proposed_trade: Proposed trade dict
            
        Returns:
            Dict with compliance status
        """
        symbol = proposed_trade.get('symbol')
        side = proposed_trade.get('side')
        
        # Look for recent trades in opposite direction
        cutoff = datetime.now() - timedelta(days=30)
        recent_trades = [
            t for t in self.trade_history
            if t['symbol'] == symbol and t['timestamp'] >= cutoff
        ]
        
        # Simple wash sale detection: selling at loss then rebuying within 30 days
        if side == 'buy':
            recent_sales = [t for t in recent_trades if t['side'] == 'sell' and t.get('pnl', 0) < 0]
            if recent_sales:
                warning = {
                    'type': 'wash_sale_warning',
                    'symbol': symbol,
                    'message': f"Potential wash sale: buying {symbol} within 30 days of loss sale"
                }
                self._record_warning(ComplianceRule.WASH_SALE, warning)
                return {
                    'status': ComplianceStatus.WARNING.value,
                    'rule': ComplianceRule.WASH_SALE.value,
                    'warnings': [warning]
                }
        
        return {'status': ComplianceStatus.COMPLIANT.value, 'rule': ComplianceRule.WASH_SALE.value}
    
    def record_trade(self, trade: Dict):
        """
        Record a trade for compliance tracking
        
        Args:
            trade: Completed trade dict
        """
        trade['timestamp'] = trade.get('timestamp', datetime.now())
        self.trade_history.append(trade)
        
        # Update daily stats
        today = trade['timestamp'].date()
        if today not in self.daily_stats:
            self.daily_stats[today] = {
                'trade_count': 0,
                'total_volume': 0.0
            }
        
        self.daily_stats[today]['trade_count'] += 1
        self.daily_stats[today]['total_volume'] += trade.get('value', 0)
        
        # Clean up old daily stats (keep last 30 days)
        cutoff = datetime.now().date() - timedelta(days=30)
        self.daily_stats = {
            date: stats for date, stats in self.daily_stats.items()
            if date >= cutoff
        }
    
    def _record_violation(self, rule: ComplianceRule, violations: List[Dict]):
        """Record compliance violation"""
        self.violations.append({
            'timestamp': datetime.now(),
            'rule': rule.value,
            'violations': violations
        })
        self.logger.error(f"Compliance violation: {rule.value} - {violations}")
    
    def _record_warning(self, rule: ComplianceRule, warning: Dict):
        """Record compliance warning"""
        self.warnings.append({
            'timestamp': datetime.now(),
            'rule': rule.value,
            'warning': warning
        })
        self.logger.warning(f"Compliance warning: {rule.value} - {warning}")
    
    def check_all_rules(self, context: Dict) -> Dict:
        """
        Check all enabled compliance rules
        
        Args:
            context: Dict with all relevant data (portfolio, trades, etc.)
            
        Returns:
            Dict with overall compliance status
        """
        results = {
            'overall_status': ComplianceStatus.COMPLIANT.value,
            'rule_results': []
        }
        
        has_violations = False
        has_warnings = False
        
        # Check position limits
        if 'positions' in context and 'portfolio_value' in context:
            for position in context['positions']:
                result = self.check_position_limits(position, context['portfolio_value'])
                results['rule_results'].append(result)
                if result['status'] == ComplianceStatus.VIOLATION.value:
                    has_violations = True
                elif result['status'] == ComplianceStatus.WARNING.value:
                    has_warnings = True
        
        # Check leverage limits
        if 'leverage' in context:
            result = self.check_leverage_limits(context['leverage'])
            results['rule_results'].append(result)
            if result['status'] == ComplianceStatus.VIOLATION.value:
                has_violations = True
            elif result['status'] == ComplianceStatus.WARNING.value:
                has_warnings = True
        
        # Check concentration
        if 'portfolio' in context:
            result = self.check_concentration_limits(context['portfolio'])
            results['rule_results'].append(result)
            if result['status'] == ComplianceStatus.VIOLATION.value:
                has_violations = True
        
        # Set overall status
        if has_violations:
            results['overall_status'] = ComplianceStatus.VIOLATION.value
        elif has_warnings:
            results['overall_status'] = ComplianceStatus.WARNING.value
        
        return results
    
    def get_violations(self, hours: int = 24) -> List[Dict]:
        """Get recent violations"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [v for v in self.violations if v['timestamp'] >= cutoff]
    
    def get_warnings(self, hours: int = 24) -> List[Dict]:
        """Get recent warnings"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [w for w in self.warnings if w['timestamp'] >= cutoff]
    
    def update_rule(self, rule: ComplianceRule, settings: Dict):
        """
        Update compliance rule settings
        
        Args:
            rule: Rule to update
            settings: New settings dict
        """
        if rule in self.compliance_rules:
            self.compliance_rules[rule].update(settings)
            self.logger.info(f"Updated compliance rule: {rule.value}")
    
    def get_compliance_report(self) -> Dict:
        """
        Generate comprehensive compliance report
        
        Returns:
            Dict with compliance statistics
        """
        return {
            'total_violations': len(self.violations),
            'total_warnings': len(self.warnings),
            'recent_violations': self.get_violations(hours=24),
            'recent_warnings': self.get_warnings(hours=24),
            'enabled_rules': [
                rule.value for rule, settings in self.compliance_rules.items()
                if settings.get('enabled', False)
            ],
            'total_trades_tracked': len(self.trade_history)
        }
