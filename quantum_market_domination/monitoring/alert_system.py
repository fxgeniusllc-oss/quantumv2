"""
Alert System
Real-time alerting for trading events and system issues
"""

import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels"""
    LOG = "log"
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    WEBHOOK = "webhook"


@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    category: str
    message: str
    details: Dict
    channels: List[AlertChannel]
    
    def __str__(self):
        return f"[{self.severity.value.upper()}] {self.category}: {self.message}"


class AlertSystem:
    """
    Real-time multi-channel alerting system
    Monitors events and triggers notifications
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger('AlertSystem')
        self.config = config or {}
        
        # Alert storage
        self.alerts: List[Alert] = []
        self.alert_counter = 0
        
        # Alert handlers
        self.handlers: Dict[AlertChannel, Callable] = {
            AlertChannel.LOG: self._handle_log_alert
        }
        
        # Alert configuration
        self.enabled_channels = [AlertChannel.LOG]  # Default to logging only
        self.alert_threshold = {
            AlertSeverity.INFO: True,
            AlertSeverity.WARNING: True,
            AlertSeverity.ERROR: True,
            AlertSeverity.CRITICAL: True
        }
        
        # Alert categories
        self.categories = {
            'system': 'System Health',
            'trade': 'Trade Execution',
            'risk': 'Risk Management',
            'market': 'Market Conditions',
            'performance': 'Performance Metrics',
            'security': 'Security Events'
        }
        
    def send_alert(self,
                  severity: AlertSeverity,
                  category: str,
                  message: str,
                  details: Optional[Dict] = None,
                  channels: Optional[List[AlertChannel]] = None) -> Alert:
        """
        Send alert through configured channels
        
        Args:
            severity: Alert severity level
            category: Alert category
            message: Alert message
            details: Additional alert details
            channels: List of channels to use (default: all enabled)
            
        Returns:
            Created Alert object
        """
        # Check if severity is enabled
        if not self.alert_threshold.get(severity, True):
            return None
            
        # Generate alert ID
        self.alert_counter += 1
        alert_id = f"ALERT-{self.alert_counter:06d}"
        
        # Create alert
        alert = Alert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            message=message,
            details=details or {},
            channels=channels or self.enabled_channels
        )
        
        # Store alert
        self.alerts.append(alert)
        
        # Send through channels
        for channel in alert.channels:
            handler = self.handlers.get(channel)
            if handler:
                try:
                    handler(alert)
                except Exception as e:
                    self.logger.error(f"Error sending alert through {channel}: {e}")
            else:
                self.logger.warning(f"No handler for channel {channel}")
                
        return alert
        
    def _handle_log_alert(self, alert: Alert):
        """Handle log-based alerts"""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(alert.severity, logging.INFO)
        
        self.logger.log(log_level, str(alert))
        
    def register_handler(self, channel: AlertChannel, handler: Callable):
        """
        Register custom alert handler
        
        Args:
            channel: Alert channel
            handler: Handler function that takes Alert as parameter
        """
        self.handlers[channel] = handler
        self.logger.info(f"Registered handler for channel {channel}")
        
    def enable_channel(self, channel: AlertChannel):
        """Enable alert channel"""
        if channel not in self.enabled_channels:
            self.enabled_channels.append(channel)
            self.logger.info(f"Enabled alert channel: {channel}")
            
    def disable_channel(self, channel: AlertChannel):
        """Disable alert channel"""
        if channel in self.enabled_channels:
            self.enabled_channels.remove(channel)
            self.logger.info(f"Disabled alert channel: {channel}")
            
    def set_severity_threshold(self, severity: AlertSeverity, enabled: bool):
        """
        Enable or disable alerts for specific severity level
        
        Args:
            severity: Alert severity
            enabled: Enable/disable flag
        """
        self.alert_threshold[severity] = enabled
        
    # Convenience methods for common alerts
    
    def alert_system_health(self, metric: str, value: float, threshold: float):
        """Alert for system health issues"""
        self.send_alert(
            severity=AlertSeverity.WARNING if value < threshold * 1.2 else AlertSeverity.CRITICAL,
            category='system',
            message=f"System health alert: {metric}",
            details={
                'metric': metric,
                'value': value,
                'threshold': threshold
            }
        )
        
    def alert_trade_execution(self, symbol: str, action: str, success: bool, details: Dict):
        """Alert for trade execution events"""
        self.send_alert(
            severity=AlertSeverity.INFO if success else AlertSeverity.ERROR,
            category='trade',
            message=f"Trade {action} for {symbol}: {'Success' if success else 'Failed'}",
            details=details
        )
        
    def alert_risk_breach(self, risk_type: str, current: float, limit: float):
        """Alert for risk limit breaches"""
        self.send_alert(
            severity=AlertSeverity.CRITICAL,
            category='risk',
            message=f"Risk limit breach: {risk_type}",
            details={
                'risk_type': risk_type,
                'current': current,
                'limit': limit,
                'breach_percentage': ((current - limit) / limit) * 100
            }
        )
        
    def alert_market_anomaly(self, symbol: str, anomaly_type: str, confidence: float):
        """Alert for market anomalies"""
        self.send_alert(
            severity=AlertSeverity.WARNING,
            category='market',
            message=f"Market anomaly detected: {symbol}",
            details={
                'symbol': symbol,
                'anomaly_type': anomaly_type,
                'confidence': confidence
            }
        )
        
    def alert_performance_milestone(self, milestone: str, value: float):
        """Alert for performance milestones"""
        self.send_alert(
            severity=AlertSeverity.INFO,
            category='performance',
            message=f"Performance milestone: {milestone}",
            details={
                'milestone': milestone,
                'value': value
            }
        )
        
    def alert_security_event(self, event_type: str, details: Dict):
        """Alert for security events"""
        self.send_alert(
            severity=AlertSeverity.CRITICAL,
            category='security',
            message=f"Security event: {event_type}",
            details=details
        )
        
    def get_recent_alerts(self, n: int = 10, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """
        Get recent alerts
        
        Args:
            n: Number of alerts to retrieve
            severity: Filter by severity (optional)
            
        Returns:
            List of recent alerts
        """
        alerts = self.alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
            
        return alerts[-n:]
        
    def get_alerts_by_category(self, category: str) -> List[Alert]:
        """
        Get alerts by category
        
        Args:
            category: Alert category
            
        Returns:
            List of alerts in category
        """
        return [a for a in self.alerts if a.category == category]
        
    def get_critical_alerts(self) -> List[Alert]:
        """Get all critical alerts"""
        return [a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]
        
    def clear_alerts(self, before: Optional[datetime] = None):
        """
        Clear alerts
        
        Args:
            before: Clear alerts before this datetime (optional, default: all)
        """
        if before:
            self.alerts = [a for a in self.alerts if a.timestamp >= before]
        else:
            self.alerts = []
            
        self.logger.info(f"Cleared alerts{' before ' + str(before) if before else ''}")
        
    def get_alert_summary(self) -> Dict:
        """
        Get summary of alerts
        
        Returns:
            Dictionary with alert statistics
        """
        if not self.alerts:
            return {'total': 0}
            
        summary = {
            'total': len(self.alerts),
            'by_severity': {},
            'by_category': {},
            'latest': self.alerts[-1] if self.alerts else None
        }
        
        for severity in AlertSeverity:
            count = len([a for a in self.alerts if a.severity == severity])
            summary['by_severity'][severity.value] = count
            
        for category in self.categories.keys():
            count = len([a for a in self.alerts if a.category == category])
            summary['by_category'][category] = count
            
        return summary
