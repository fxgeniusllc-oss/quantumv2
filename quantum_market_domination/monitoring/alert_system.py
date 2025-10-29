"""
REAL-TIME ALERTING SYSTEM
Multi-channel alerting for critical events and conditions
"""

import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""
    SYSTEM = "system"
    TRADE = "trade"
    RISK = "risk"
    PERFORMANCE = "performance"
    MARKET = "market"
    COMPLIANCE = "compliance"


class AlertSystem:
    """
    Real-time alerting system with multiple channels and severity levels
    """
    
    def __init__(self, alert_threshold: Dict[str, float] = None):
        """
        Initialize alert system
        
        Args:
            alert_threshold: Dict of thresholds for different metrics
        """
        self.logger = logging.getLogger('AlertSystem')
        self.alerts: List[Dict] = []
        self.alert_threshold = alert_threshold or {
            'cpu_usage': 85.0,
            'memory_usage': 85.0,
            'drawdown': 0.10,  # 10%
            'daily_loss': 0.05,  # 5%
            'position_risk': 0.15  # 15%
        }
        self.alert_handlers: Dict[AlertLevel, List[Callable]] = {
            AlertLevel.INFO: [],
            AlertLevel.WARNING: [],
            AlertLevel.ERROR: [],
            AlertLevel.CRITICAL: []
        }
        self.alert_cooldown: Dict[str, datetime] = {}
        self.cooldown_period = timedelta(minutes=5)  # Default cooldown
        
    def register_handler(self, level: AlertLevel, handler: Callable):
        """
        Register a custom alert handler
        
        Args:
            level: Alert level to handle
            handler: Callable that takes alert dict as parameter
        """
        self.alert_handlers[level].append(handler)
        self.logger.info(f"Registered handler for {level.value} alerts")
    
    def _should_send_alert(self, alert_key: str) -> bool:
        """
        Check if alert should be sent (respects cooldown)
        
        Args:
            alert_key: Unique identifier for alert type
            
        Returns:
            True if alert should be sent
        """
        if alert_key not in self.alert_cooldown:
            return True
            
        time_since_last = datetime.now() - self.alert_cooldown[alert_key]
        return time_since_last >= self.cooldown_period
    
    def send_alert(self, level: AlertLevel, alert_type: AlertType, 
                   message: str, data: Optional[Dict] = None,
                   cooldown_key: Optional[str] = None):
        """
        Send an alert
        
        Args:
            level: Severity level
            alert_type: Type of alert
            message: Alert message
            data: Additional data dict
            cooldown_key: Key for cooldown (prevents spam)
        """
        # Check cooldown
        if cooldown_key and not self._should_send_alert(cooldown_key):
            self.logger.debug(f"Alert {cooldown_key} in cooldown period")
            return
        
        alert = {
            'timestamp': datetime.now(),
            'level': level.value,
            'type': alert_type.value,
            'message': message,
            'data': data or {}
        }
        
        # Store alert
        self.alerts.append(alert)
        
        # Update cooldown
        if cooldown_key:
            self.alert_cooldown[cooldown_key] = datetime.now()
        
        # Log alert
        log_method = getattr(self.logger, level.value.lower())
        log_method(f"[{alert_type.value.upper()}] {message}")
        
        # Call registered handlers
        for handler in self.alert_handlers[level]:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
    
    def check_system_health(self, metrics: Dict):
        """
        Check system health metrics and alert if needed
        
        Args:
            metrics: Dict with system metrics (cpu, memory, etc.)
        """
        # CPU usage
        if metrics.get('cpu_usage', 0) > self.alert_threshold['cpu_usage']:
            self.send_alert(
                AlertLevel.WARNING,
                AlertType.SYSTEM,
                f"High CPU usage: {metrics['cpu_usage']:.1f}%",
                {'cpu_usage': metrics['cpu_usage']},
                cooldown_key='cpu_high'
            )
        
        # Memory usage
        if metrics.get('memory_usage', 0) > self.alert_threshold['memory_usage']:
            self.send_alert(
                AlertLevel.WARNING,
                AlertType.SYSTEM,
                f"High memory usage: {metrics['memory_usage']:.1f}%",
                {'memory_usage': metrics['memory_usage']},
                cooldown_key='memory_high'
            )
        
        # GPU usage (if available)
        if 'gpu_usage' in metrics and metrics['gpu_usage'] > self.alert_threshold['cpu_usage']:
            self.send_alert(
                AlertLevel.WARNING,
                AlertType.SYSTEM,
                f"High GPU usage: {metrics['gpu_usage']:.1f}%",
                {'gpu_usage': metrics['gpu_usage']},
                cooldown_key='gpu_high'
            )
        
        # Network latency
        if 'network_latency' in metrics and metrics['network_latency'] > 100:  # 100ms
            self.send_alert(
                AlertLevel.WARNING,
                AlertType.SYSTEM,
                f"High network latency: {metrics['network_latency']:.1f}ms",
                {'network_latency': metrics['network_latency']},
                cooldown_key='latency_high'
            )
    
    def check_risk_limits(self, risk_metrics: Dict):
        """
        Check risk limits and alert if breached
        
        Args:
            risk_metrics: Dict with risk metrics
        """
        # Drawdown check
        if risk_metrics.get('drawdown', 0) > self.alert_threshold['drawdown']:
            self.send_alert(
                AlertLevel.ERROR,
                AlertType.RISK,
                f"Drawdown limit breached: {risk_metrics['drawdown']:.2%}",
                {'drawdown': risk_metrics['drawdown']},
                cooldown_key='drawdown_breach'
            )
        
        # Daily loss check
        if risk_metrics.get('daily_loss_pct', 0) > self.alert_threshold['daily_loss']:
            self.send_alert(
                AlertLevel.CRITICAL,
                AlertType.RISK,
                f"Daily loss limit breached: {risk_metrics['daily_loss_pct']:.2%}",
                {'daily_loss': risk_metrics['daily_loss_pct']},
                cooldown_key='daily_loss_breach'
            )
        
        # Position risk check
        if risk_metrics.get('position_risk', 0) > self.alert_threshold['position_risk']:
            self.send_alert(
                AlertLevel.WARNING,
                AlertType.RISK,
                f"High position risk: {risk_metrics['position_risk']:.2%}",
                {'position_risk': risk_metrics['position_risk']},
                cooldown_key='position_risk_high'
            )
        
        # Portfolio heat check
        if risk_metrics.get('portfolio_heat', 0) > 0.90:  # 90%
            self.send_alert(
                AlertLevel.CRITICAL,
                AlertType.RISK,
                f"Portfolio heat critical: {risk_metrics['portfolio_heat']:.2%}",
                {'portfolio_heat': risk_metrics['portfolio_heat']},
                cooldown_key='portfolio_heat_critical'
            )
    
    def alert_trade_execution(self, trade: Dict, success: bool):
        """
        Alert on trade execution
        
        Args:
            trade: Trade dict
            success: Whether trade was successful
        """
        if success:
            self.send_alert(
                AlertLevel.INFO,
                AlertType.TRADE,
                f"Trade executed: {trade['symbol']} {trade.get('side', 'N/A')} "
                f"{trade.get('quantity', 0)} @ {trade.get('price', 0)}",
                trade
            )
        else:
            self.send_alert(
                AlertLevel.ERROR,
                AlertType.TRADE,
                f"Trade execution failed: {trade['symbol']} - {trade.get('error', 'Unknown error')}",
                trade
            )
    
    def alert_large_trade(self, trade: Dict, size_threshold: float = 0.1):
        """
        Alert on large trades (>threshold of portfolio)
        
        Args:
            trade: Trade dict
            size_threshold: Size threshold as fraction of portfolio
        """
        position_size = trade.get('position_size_pct', 0)
        if position_size > size_threshold:
            self.send_alert(
                AlertLevel.WARNING,
                AlertType.TRADE,
                f"Large trade detected: {trade['symbol']} ({position_size:.2%} of portfolio)",
                trade,
                cooldown_key=f"large_trade_{trade['symbol']}"
            )
    
    def alert_stop_loss_triggered(self, position: Dict):
        """
        Alert when stop loss is triggered
        
        Args:
            position: Position dict
        """
        self.send_alert(
            AlertLevel.WARNING,
            AlertType.TRADE,
            f"Stop loss triggered: {position['symbol']} at {position.get('exit_price', 0)} "
            f"(Loss: {position.get('pnl', 0):.2f})",
            position
        )
    
    def alert_performance_milestone(self, metric: str, value: float, milestone: str):
        """
        Alert on performance milestones
        
        Args:
            metric: Performance metric name
            value: Current value
            milestone: Milestone description
        """
        self.send_alert(
            AlertLevel.INFO,
            AlertType.PERFORMANCE,
            f"Performance milestone reached: {metric} = {value} ({milestone})",
            {'metric': metric, 'value': value, 'milestone': milestone}
        )
    
    def alert_market_anomaly(self, symbol: str, anomaly_type: str, score: float):
        """
        Alert on market anomalies
        
        Args:
            symbol: Trading pair
            anomaly_type: Type of anomaly detected
            score: Anomaly score
        """
        self.send_alert(
            AlertLevel.WARNING,
            AlertType.MARKET,
            f"Market anomaly detected: {symbol} - {anomaly_type} (score: {score:.2f})",
            {'symbol': symbol, 'anomaly_type': anomaly_type, 'score': score},
            cooldown_key=f"anomaly_{symbol}_{anomaly_type}"
        )
    
    def alert_compliance_issue(self, issue: str, severity: str = 'warning'):
        """
        Alert on compliance issues
        
        Args:
            issue: Description of compliance issue
            severity: Severity level (warning/error/critical)
        """
        level_map = {
            'info': AlertLevel.INFO,
            'warning': AlertLevel.WARNING,
            'error': AlertLevel.ERROR,
            'critical': AlertLevel.CRITICAL
        }
        
        self.send_alert(
            level_map.get(severity.lower(), AlertLevel.WARNING),
            AlertType.COMPLIANCE,
            f"Compliance issue: {issue}",
            {'issue': issue}
        )
    
    def get_recent_alerts(self, minutes: int = 60, 
                         level: Optional[AlertLevel] = None,
                         alert_type: Optional[AlertType] = None) -> List[Dict]:
        """
        Get recent alerts
        
        Args:
            minutes: Number of minutes to look back
            level: Filter by alert level
            alert_type: Filter by alert type
            
        Returns:
            List of recent alerts
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent = [
            alert for alert in self.alerts
            if alert['timestamp'] >= cutoff_time
        ]
        
        # Apply filters
        if level:
            recent = [a for a in recent if a['level'] == level.value]
        
        if alert_type:
            recent = [a for a in recent if a['type'] == alert_type.value]
        
        return recent
    
    def get_alert_summary(self) -> Dict:
        """
        Get summary of alerts
        
        Returns:
            Dict with alert statistics
        """
        if not self.alerts:
            return {
                'total_alerts': 0,
                'by_level': {},
                'by_type': {},
                'recent_critical': []
            }
        
        # Count by level
        by_level = {}
        for alert in self.alerts:
            level = alert['level']
            by_level[level] = by_level.get(level, 0) + 1
        
        # Count by type
        by_type = {}
        for alert in self.alerts:
            alert_type = alert['type']
            by_type[alert_type] = by_type.get(alert_type, 0) + 1
        
        # Get recent critical alerts
        recent_critical = self.get_recent_alerts(minutes=60, level=AlertLevel.CRITICAL)
        
        return {
            'total_alerts': len(self.alerts),
            'by_level': by_level,
            'by_type': by_type,
            'recent_critical': recent_critical,
            'last_alert': self.alerts[-1] if self.alerts else None
        }
    
    def clear_old_alerts(self, days: int = 7):
        """
        Clear alerts older than specified days
        
        Args:
            days: Number of days to keep
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        
        before_count = len(self.alerts)
        self.alerts = [
            alert for alert in self.alerts
            if alert['timestamp'] >= cutoff_time
        ]
        after_count = len(self.alerts)
        
        cleared = before_count - after_count
        if cleared > 0:
            self.logger.info(f"Cleared {cleared} old alerts")
    
    def set_threshold(self, metric: str, value: float):
        """
        Update alert threshold
        
        Args:
            metric: Metric name
            value: New threshold value
        """
        self.alert_threshold[metric] = value
        self.logger.info(f"Updated threshold for {metric} to {value}")
