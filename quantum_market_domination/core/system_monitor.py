"""
QUANTUM SYSTEM HEALTH AND PERFORMANCE MONITORING
Provides real-time system resource tracking and anomaly detection
"""

import psutil
import asyncio
import logging
from dataclasses import dataclass
from typing import List
from datetime import datetime


@dataclass
class SystemMetrics:
    """
    Comprehensive system performance tracking
    """
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    network_latency: float
    disk_io: float
    temperature: float
    timestamp: datetime


class QuantumSystemMonitor:
    """
    QUANTUM SYSTEM HEALTH AND PERFORMANCE MONITORING
    Provides real-time system resource tracking and anomaly detection
    """
    
    def __init__(self, alert_threshold=0.85):
        self.logger = logging.getLogger('SystemMonitor')
        self.alert_threshold = alert_threshold
        self.metrics_history: List[SystemMetrics] = []
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

    async def monitor_system(self, interval=5):
        """
        Continuous system monitoring with async capabilities
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.logger.info(f"Starting system monitoring with {interval}s interval")
        
        while True:
            try:
                metrics = self._collect_metrics()
                self._check_system_health(metrics)
                self.metrics_history.append(metrics)
                
                # Trim history to prevent unbounded growth
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error during monitoring: {e}")
                await asyncio.sleep(interval)

    def _collect_metrics(self) -> SystemMetrics:
        """
        Collect comprehensive system metrics
        
        Returns:
            SystemMetrics object with current system state
        """
        # CPU Metrics
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # Memory Metrics
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # GPU Metrics (gracefully handle if GPU not available)
        gpu_usage = 0
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_usage = gpus[0].load * 100 if gpus else 0
        except (ImportError, IndexError):
            pass
        
        # Network Latency (placeholder - measure to exchange endpoints)
        network_latency = self._measure_network_latency()
        
        # Disk I/O
        disk_io = 0
        try:
            io_counters = psutil.disk_io_counters()
            if io_counters and io_counters.write_time > 0:
                disk_io = io_counters.read_time / io_counters.write_time
        except:
            pass
        
        # Temperature (gracefully handle if sensors not available)
        temperature = 0
        try:
            temps = psutil.sensors_temperatures()
            if temps and 'coretemp' in temps:
                temp_list = temps['coretemp']
                if temp_list:
                    temperature = temp_list[0].current
        except (AttributeError, KeyError, IndexError):
            pass

        return SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            network_latency=network_latency,
            disk_io=disk_io,
            temperature=temperature,
            timestamp=datetime.now()
        )

    def _measure_network_latency(self) -> float:
        """
        Measure network latency to key endpoints
        
        Returns:
            Average latency in milliseconds
        """
        # Placeholder implementation
        # In production, ping actual exchange endpoints
        return 10.0

    def _check_system_health(self, metrics: SystemMetrics):
        """
        Advanced system health checking with multi-dimensional analysis
        
        Args:
            metrics: Current system metrics
        """
        alerts = []
        
        if metrics.cpu_usage > self.alert_threshold * 100:
            alerts.append(f"HIGH CPU: {metrics.cpu_usage:.1f}%")
        
        if metrics.memory_usage > self.alert_threshold * 100:
            alerts.append(f"HIGH MEMORY: {metrics.memory_usage:.1f}%")
        
        if metrics.gpu_usage > self.alert_threshold * 100:
            alerts.append(f"HIGH GPU: {metrics.gpu_usage:.1f}%")
        
        if metrics.network_latency > 100:  # ms
            alerts.append(f"HIGH LATENCY: {metrics.network_latency:.1f}ms")
        
        if metrics.temperature > 85:  # Celsius
            alerts.append(f"HIGH TEMPERATURE: {metrics.temperature:.1f}°C")
        
        if alerts:
            self._trigger_system_alert(alerts)

    def _trigger_system_alert(self, alerts: List[str]):
        """
        Send critical system alerts
        Supports multiple notification channels
        
        Args:
            alerts: List of alert messages
        """
        alert_message = "\n".join(alerts)
        self.logger.critical(f"SYSTEM HEALTH ALERT:\n{alert_message}")
        
        # Additional alert mechanisms (Slack, Email, etc.)
        # self._send_slack_alert(alert_message)
        # self._send_email_alert(alert_message)

    def get_metrics_summary(self) -> dict:
        """
        Get summary statistics of recent metrics
        
        Returns:
            Dictionary with metric summaries
        """
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 readings
        
        return {
            'avg_cpu': sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            'avg_memory': sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            'avg_gpu': sum(m.gpu_usage for m in recent_metrics) / len(recent_metrics),
            'avg_latency': sum(m.network_latency for m in recent_metrics) / len(recent_metrics),
            'max_cpu': max(m.cpu_usage for m in recent_metrics),
            'max_memory': max(m.memory_usage for m in recent_metrics),
            'samples': len(recent_metrics)
        }

    def _send_slack_alert(self, message: str):
        """Send alert to Slack (placeholder)"""
        pass

    def _send_email_alert(self, message: str):
        """Send alert via email (placeholder)"""
        pass
