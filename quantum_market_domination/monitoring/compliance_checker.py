"""
Compliance Checker
Regulatory compliance monitoring and validation
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class ComplianceRule(Enum):
    """Compliance rule types"""
    POSITION_LIMIT = "position_limit"
    TRADE_FREQUENCY = "trade_frequency"
    WASH_SALE = "wash_sale"
    MARKET_MANIPULATION = "market_manipulation"
    CIRCUIT_BREAKER = "circuit_breaker"
    CAPITAL_REQUIREMENT = "capital_requirement"
    LEVERAGE_LIMIT = "leverage_limit"


@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    violation_id: str
    timestamp: datetime
    rule: ComplianceRule
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    affected_symbol: str
    details: Dict
    resolved: bool = False
    
    def __str__(self):
        return f"[{self.severity.upper()}] {self.rule.value}: {self.description}"


class ComplianceChecker:
    """
    Regulatory compliance monitoring system
    Validates trading activities against regulatory requirements
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger('ComplianceChecker')
        self.config = config or {}
        
        # Compliance parameters
        self.position_limits = self.config.get('position_limits', {
            'max_single_position_value': 100000,  # Max position value per symbol
            'max_position_percentage': 0.25,  # Max % of portfolio per position
            'max_total_positions': 10  # Max concurrent positions
        })
        
        self.trade_frequency_limits = {
            'max_trades_per_day': 100,
            'max_trades_per_hour': 20,
            'max_trades_per_minute': 5
        }
        
        self.leverage_limits = {
            'max_leverage': 3.0,  # 3x leverage
            'margin_call_threshold': 0.25  # 25% equity
        }
        
        # Violation tracking
        self.violations: List[ComplianceViolation] = []
        self.violation_counter = 0
        
        # Trade tracking for wash sale detection
        self.recent_trades: List[Dict] = []
        self.wash_sale_window = timedelta(days=30)
        
        # Circuit breaker tracking
        self.price_movements: Dict[str, List[Dict]] = {}
        self.circuit_breaker_threshold = 0.10  # 10% move
        
    def check_position_compliance(self,
                                  symbol: str,
                                  position_value: float,
                                  total_portfolio_value: float,
                                  open_positions: int) -> Dict:
        """
        Check position size compliance
        
        Args:
            symbol: Trading symbol
            position_value: Value of position
            total_portfolio_value: Total portfolio value
            open_positions: Number of open positions
            
        Returns:
            Compliance check result
        """
        violations = []
        
        # Check absolute position limit
        if position_value > self.position_limits['max_single_position_value']:
            violations.append(self._record_violation(
                rule=ComplianceRule.POSITION_LIMIT,
                severity='high',
                description=f"Position value ${position_value:,.2f} exceeds limit",
                symbol=symbol,
                details={
                    'position_value': position_value,
                    'limit': self.position_limits['max_single_position_value']
                }
            ))
            
        # Check percentage limit
        position_pct = position_value / total_portfolio_value if total_portfolio_value > 0 else 0
        if position_pct > self.position_limits['max_position_percentage']:
            violations.append(self._record_violation(
                rule=ComplianceRule.POSITION_LIMIT,
                severity='medium',
                description=f"Position {position_pct*100:.1f}% exceeds {self.position_limits['max_position_percentage']*100}% limit",
                symbol=symbol,
                details={
                    'position_percentage': position_pct,
                    'limit': self.position_limits['max_position_percentage']
                }
            ))
            
        # Check concurrent positions limit
        if open_positions >= self.position_limits['max_total_positions']:
            violations.append(self._record_violation(
                rule=ComplianceRule.POSITION_LIMIT,
                severity='medium',
                description=f"Open positions ({open_positions}) at limit",
                symbol=symbol,
                details={
                    'open_positions': open_positions,
                    'limit': self.position_limits['max_total_positions']
                }
            ))
            
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'checked_rules': ['position_limits']
        }
        
    def check_trade_frequency_compliance(self,
                                        recent_trades: List[Dict]) -> Dict:
        """
        Check trade frequency compliance
        
        Args:
            recent_trades: List of recent trades with timestamps
            
        Returns:
            Compliance check result
        """
        violations = []
        now = datetime.now()
        
        # Count trades in different time windows
        trades_last_minute = len([t for t in recent_trades 
                                 if (now - t['timestamp']).total_seconds() < 60])
        trades_last_hour = len([t for t in recent_trades 
                               if (now - t['timestamp']).total_seconds() < 3600])
        trades_last_day = len([t for t in recent_trades 
                              if (now - t['timestamp']).total_seconds() < 86400])
        
        # Check limits
        if trades_last_minute > self.trade_frequency_limits['max_trades_per_minute']:
            violations.append(self._record_violation(
                rule=ComplianceRule.TRADE_FREQUENCY,
                severity='critical',
                description=f"Excessive trading: {trades_last_minute} trades in last minute",
                symbol='ALL',
                details={
                    'trades_last_minute': trades_last_minute,
                    'limit': self.trade_frequency_limits['max_trades_per_minute']
                }
            ))
            
        if trades_last_hour > self.trade_frequency_limits['max_trades_per_hour']:
            violations.append(self._record_violation(
                rule=ComplianceRule.TRADE_FREQUENCY,
                severity='high',
                description=f"High trading frequency: {trades_last_hour} trades in last hour",
                symbol='ALL',
                details={
                    'trades_last_hour': trades_last_hour,
                    'limit': self.trade_frequency_limits['max_trades_per_hour']
                }
            ))
            
        if trades_last_day > self.trade_frequency_limits['max_trades_per_day']:
            violations.append(self._record_violation(
                rule=ComplianceRule.TRADE_FREQUENCY,
                severity='medium',
                description=f"Daily trade limit reached: {trades_last_day} trades",
                symbol='ALL',
                details={
                    'trades_last_day': trades_last_day,
                    'limit': self.trade_frequency_limits['max_trades_per_day']
                }
            ))
            
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'trade_counts': {
                'last_minute': trades_last_minute,
                'last_hour': trades_last_hour,
                'last_day': trades_last_day
            }
        }
        
    def check_wash_sale(self,
                       symbol: str,
                       action: str,
                       trade_time: datetime) -> Dict:
        """
        Check for wash sale violations
        A wash sale occurs when selling at a loss and buying back within 30 days
        
        Args:
            symbol: Trading symbol
            action: 'buy' or 'sell'
            trade_time: Trade timestamp
            
        Returns:
            Compliance check result
        """
        violations = []
        
        # Get recent trades for this symbol
        symbol_trades = [t for t in self.recent_trades 
                        if t['symbol'] == symbol and 
                        (trade_time - t['timestamp']) <= self.wash_sale_window]
        
        # Check for wash sale pattern
        if action == 'buy':
            # Look for recent sells at a loss
            recent_losses = [t for t in symbol_trades 
                           if t['action'] == 'sell' and t.get('pnl', 0) < 0]
            
            if recent_losses:
                violations.append(self._record_violation(
                    rule=ComplianceRule.WASH_SALE,
                    severity='medium',
                    description=f"Potential wash sale: buying {symbol} after recent loss",
                    symbol=symbol,
                    details={
                        'action': action,
                        'recent_loss_count': len(recent_losses),
                        'window_days': self.wash_sale_window.days
                    }
                ))
                
        # Record this trade
        self.recent_trades.append({
            'symbol': symbol,
            'action': action,
            'timestamp': trade_time
        })
        
        # Clean up old trades
        cutoff = trade_time - self.wash_sale_window
        self.recent_trades = [t for t in self.recent_trades if t['timestamp'] > cutoff]
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }
        
    def check_leverage_compliance(self,
                                 equity: float,
                                 borrowed: float,
                                 portfolio_value: float) -> Dict:
        """
        Check leverage and margin compliance
        
        Args:
            equity: Account equity
            borrowed: Borrowed amount
            portfolio_value: Total portfolio value
            
        Returns:
            Compliance check result
        """
        violations = []
        
        # Calculate current leverage
        leverage = portfolio_value / equity if equity > 0 else 0
        
        # Check leverage limit
        if leverage > self.leverage_limits['max_leverage']:
            violations.append(self._record_violation(
                rule=ComplianceRule.LEVERAGE_LIMIT,
                severity='critical',
                description=f"Leverage {leverage:.2f}x exceeds {self.leverage_limits['max_leverage']}x limit",
                symbol='ACCOUNT',
                details={
                    'current_leverage': leverage,
                    'max_leverage': self.leverage_limits['max_leverage'],
                    'equity': equity,
                    'borrowed': borrowed
                }
            ))
            
        # Check margin call threshold
        equity_ratio = equity / portfolio_value if portfolio_value > 0 else 0
        if equity_ratio < self.leverage_limits['margin_call_threshold']:
            violations.append(self._record_violation(
                rule=ComplianceRule.CAPITAL_REQUIREMENT,
                severity='critical',
                description=f"Equity ratio {equity_ratio*100:.1f}% below margin call threshold",
                symbol='ACCOUNT',
                details={
                    'equity_ratio': equity_ratio,
                    'threshold': self.leverage_limits['margin_call_threshold']
                }
            ))
            
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'leverage': leverage,
            'equity_ratio': equity_ratio
        }
        
    def check_circuit_breaker(self,
                             symbol: str,
                             current_price: float,
                             reference_price: float) -> Dict:
        """
        Check for circuit breaker conditions (extreme price movements)
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            reference_price: Reference price (e.g., day open)
            
        Returns:
            Compliance check result
        """
        violations = []
        
        # Calculate price change
        price_change = abs(current_price - reference_price) / reference_price
        
        if price_change >= self.circuit_breaker_threshold:
            violations.append(self._record_violation(
                rule=ComplianceRule.CIRCUIT_BREAKER,
                severity='high',
                description=f"Circuit breaker: {symbol} moved {price_change*100:.1f}%",
                symbol=symbol,
                details={
                    'current_price': current_price,
                    'reference_price': reference_price,
                    'change_percentage': price_change * 100,
                    'threshold': self.circuit_breaker_threshold * 100
                }
            ))
            
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'price_change': price_change
        }
        
    def _record_violation(self,
                         rule: ComplianceRule,
                         severity: str,
                         description: str,
                         symbol: str,
                         details: Dict) -> ComplianceViolation:
        """Record a compliance violation"""
        self.violation_counter += 1
        
        violation = ComplianceViolation(
            violation_id=f"COMP-{self.violation_counter:06d}",
            timestamp=datetime.now(),
            rule=rule,
            severity=severity,
            description=description,
            affected_symbol=symbol,
            details=details
        )
        
        self.violations.append(violation)
        self.logger.warning(str(violation))
        
        return violation
        
    def get_violations(self,
                      rule: Optional[ComplianceRule] = None,
                      severity: Optional[str] = None,
                      unresolved_only: bool = False) -> List[ComplianceViolation]:
        """
        Get compliance violations
        
        Args:
            rule: Filter by rule type
            severity: Filter by severity
            unresolved_only: Only unresolved violations
            
        Returns:
            List of violations
        """
        violations = self.violations
        
        if rule:
            violations = [v for v in violations if v.rule == rule]
            
        if severity:
            violations = [v for v in violations if v.severity == severity]
            
        if unresolved_only:
            violations = [v for v in violations if not v.resolved]
            
        return violations
        
    def resolve_violation(self, violation_id: str):
        """Mark a violation as resolved"""
        for violation in self.violations:
            if violation.violation_id == violation_id:
                violation.resolved = True
                self.logger.info(f"Resolved violation {violation_id}")
                return True
                
        return False
        
    def get_compliance_report(self) -> Dict:
        """
        Generate compliance report
        
        Returns:
            Compliance summary report
        """
        total_violations = len(self.violations)
        unresolved = len([v for v in self.violations if not v.resolved])
        
        by_severity = {}
        for severity in ['low', 'medium', 'high', 'critical']:
            count = len([v for v in self.violations if v.severity == severity and not v.resolved])
            by_severity[severity] = count
            
        by_rule = {}
        for rule in ComplianceRule:
            count = len([v for v in self.violations if v.rule == rule and not v.resolved])
            by_rule[rule.value] = count
            
        return {
            'total_violations': total_violations,
            'unresolved_violations': unresolved,
            'by_severity': by_severity,
            'by_rule': by_rule,
            'compliance_score': max(0, 100 - (unresolved * 5))  # Simple scoring
        }
