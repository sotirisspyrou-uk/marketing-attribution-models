"""
Performance Monitoring System

Real-time monitoring of marketing channel performance with anomaly detection,
trend analysis, and automated alerting for attribution model drift.

Author: Sotiris Spyrou
Portfolio: https://verityai.co
LinkedIn: https://www.linkedin.com/in/sspyrou/

DISCLAIMER: This is demonstration code for portfolio purposes only.
Not intended for production use without proper testing and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from scipy import stats
from collections import deque
import logging

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Advanced performance monitoring system for marketing attribution.
    
    Monitors channel performance, detects anomalies, tracks model drift,
    and generates automated alerts for performance issues.
    """
    
    def __init__(self,
                 monitoring_window: int = 7,
                 anomaly_threshold: float = 2.0,
                 drift_threshold: float = 0.15,
                 alert_cooldown: int = 3600):  # 1 hour
        """
        Initialize Performance Monitor.
        
        Args:
            monitoring_window: Days of data to use for baseline calculations
            anomaly_threshold: Standard deviations for anomaly detection
            drift_threshold: Maximum allowed drift in attribution weights
            alert_cooldown: Minimum seconds between duplicate alerts
        """
        self.monitoring_window = monitoring_window
        self.anomaly_threshold = anomaly_threshold
        self.drift_threshold = drift_threshold
        self.alert_cooldown = alert_cooldown
        
        # Performance tracking
        self.performance_history = {}
        self.baseline_metrics = {}
        self.current_metrics = {}
        self.anomalies = []
        self.drift_analysis = {}
        
        # Alert management
        self.active_alerts = []
        self.alert_history = deque(maxlen=1000)
        self.last_alert_times = {}
        
        # Model drift tracking
        self.attribution_history = deque(maxlen=100)
        self.model_stability_score = 1.0
        
        logger.info("Performance monitor initialized")
    
    def update_performance_data(self,
                               channel_metrics: Dict[str, Dict[str, float]],
                               attribution_weights: Dict[str, float],
                               timestamp: Optional[datetime] = None) -> bool:
        """
        Update performance monitoring with latest metrics.
        
        Args:
            channel_metrics: Current performance metrics by channel
            attribution_weights: Current attribution weights
            timestamp: Data timestamp (defaults to now)
            
        Returns:
            Success status
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Store current metrics
            self.current_metrics = {
                'timestamp': timestamp,
                'channel_metrics': channel_metrics,
                'attribution_weights': attribution_weights
            }
            
            # Update performance history
            self._update_performance_history(timestamp, channel_metrics)
            
            # Update attribution history
            self.attribution_history.append({
                'timestamp': timestamp,
                'weights': attribution_weights.copy()
            })
            
            # Calculate baselines if we have enough data
            if len(self.performance_history) >= self.monitoring_window:
                self._calculate_baselines()
                
                # Detect anomalies
                anomalies = self._detect_anomalies(channel_metrics, timestamp)
                
                # Analyze model drift
                drift_analysis = self._analyze_model_drift()
                
                # Generate alerts
                self._generate_performance_alerts(anomalies, drift_analysis, timestamp)
            
            logger.info(f"Performance data updated at {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update performance data: {e}")
            return False
    
    def _update_performance_history(self,
                                  timestamp: datetime,
                                  channel_metrics: Dict[str, Dict[str, float]]):
        """Update historical performance data."""
        
        # Clean old data outside monitoring window
        cutoff_date = timestamp - timedelta(days=self.monitoring_window * 2)
        
        for channel in list(self.performance_history.keys()):
            self.performance_history[channel] = [
                entry for entry in self.performance_history[channel]
                if entry['timestamp'] > cutoff_date
            ]
        
        # Add new data
        for channel, metrics in channel_metrics.items():
            if channel not in self.performance_history:
                self.performance_history[channel] = []
            
            self.performance_history[channel].append({
                'timestamp': timestamp,
                'metrics': metrics.copy()
            })
    
    def _calculate_baselines(self):
        """Calculate baseline performance metrics from historical data."""
        
        self.baseline_metrics = {}
        
        for channel, history in self.performance_history.items():
            if len(history) < 3:  # Need minimum data points
                continue
            
            # Get recent data within monitoring window
            recent_data = history[-self.monitoring_window:]
            
            channel_baselines = {}
            metric_names = set()
            
            # Collect all metric names
            for entry in recent_data:
                metric_names.update(entry['metrics'].keys())
            
            # Calculate baseline statistics for each metric
            for metric_name in metric_names:
                values = [
                    entry['metrics'].get(metric_name, 0)
                    for entry in recent_data
                ]
                
                if values and any(v != 0 for v in values):
                    channel_baselines[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'trend': self._calculate_trend(values)
                    }
            
            self.baseline_metrics[channel] = channel_baselines
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1) using linear regression."""
        
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Handle edge cases
        if np.std(y) == 0:
            return 0
        
        try:
            slope, _, r_value, _, _ = stats.linregress(x, y)
            # Normalize slope by value range
            value_range = np.max(y) - np.min(y)
            if value_range > 0:
                normalized_slope = slope / value_range * len(values)
                return np.clip(normalized_slope, -1, 1)
            return 0
        except:
            return 0
    
    def _detect_anomalies(self,
                         current_metrics: Dict[str, Dict[str, float]],
                         timestamp: datetime) -> List[Dict[str, Any]]:
        """Detect performance anomalies using statistical analysis."""
        
        anomalies = []
        
        for channel, metrics in current_metrics.items():
            if channel not in self.baseline_metrics:
                continue
            
            channel_baselines = self.baseline_metrics[channel]
            
            for metric_name, current_value in metrics.items():
                if metric_name not in channel_baselines:
                    continue
                
                baseline = channel_baselines[metric_name]
                
                # Z-score anomaly detection
                if baseline['std'] > 0:
                    z_score = (current_value - baseline['mean']) / baseline['std']
                    
                    if abs(z_score) > self.anomaly_threshold:
                        anomaly_type = 'spike' if z_score > 0 else 'drop'
                        severity = 'high' if abs(z_score) > 3 else 'medium'
                        
                        anomalies.append({
                            'timestamp': timestamp,
                            'channel': channel,
                            'metric': metric_name,
                            'current_value': current_value,
                            'baseline_mean': baseline['mean'],
                            'z_score': z_score,
                            'type': anomaly_type,
                            'severity': severity,
                            'deviation_percent': abs(current_value - baseline['mean']) / baseline['mean'] * 100
                        })
        
        # Store anomalies
        self.anomalies.extend(anomalies)
        
        # Keep only recent anomalies
        cutoff_time = timestamp - timedelta(days=7)
        self.anomalies = [
            a for a in self.anomalies
            if a['timestamp'] > cutoff_time
        ]
        
        return anomalies
    
    def _analyze_model_drift(self) -> Dict[str, Any]:
        """Analyze attribution model drift over time."""
        
        if len(self.attribution_history) < 10:
            return {'insufficient_data': True}
        
        drift_analysis = {}
        
        # Get recent attribution snapshots
        recent_attributions = list(self.attribution_history)[-10:]  # Last 10 updates
        baseline_attributions = list(self.attribution_history)[-20:-10] if len(self.attribution_history) >= 20 else recent_attributions[:5]
        
        # Calculate channel-level drift
        channel_drift = {}
        all_channels = set()
        
        for snapshot in recent_attributions + baseline_attributions:
            all_channels.update(snapshot['weights'].keys())
        
        for channel in all_channels:
            # Get recent and baseline weights
            recent_weights = [
                snapshot['weights'].get(channel, 0)
                for snapshot in recent_attributions
            ]
            baseline_weights = [
                snapshot['weights'].get(channel, 0)
                for snapshot in baseline_attributions
            ]
            
            if recent_weights and baseline_weights:
                recent_mean = np.mean(recent_weights)
                baseline_mean = np.mean(baseline_weights)
                
                # Calculate drift magnitude
                if baseline_mean > 0:
                    drift_ratio = abs(recent_mean - baseline_mean) / baseline_mean
                else:
                    drift_ratio = 1.0 if recent_mean > 0 else 0.0
                
                # Calculate volatility
                recent_volatility = np.std(recent_weights)
                baseline_volatility = np.std(baseline_weights) if len(baseline_weights) > 1 else 0
                
                channel_drift[channel] = {
                    'drift_ratio': drift_ratio,
                    'recent_mean': recent_mean,
                    'baseline_mean': baseline_mean,
                    'recent_volatility': recent_volatility,
                    'baseline_volatility': baseline_volatility,
                    'is_drifting': drift_ratio > self.drift_threshold
                }
        
        # Overall model stability
        total_drift = sum(d['drift_ratio'] for d in channel_drift.values())
        num_drifting_channels = sum(1 for d in channel_drift.values() if d['is_drifting'])
        
        self.model_stability_score = max(0, 1 - (total_drift / len(channel_drift))) if channel_drift else 1.0
        
        drift_analysis = {
            'channel_drift': channel_drift,
            'total_drift_magnitude': total_drift,
            'num_drifting_channels': num_drifting_channels,
            'model_stability_score': self.model_stability_score,
            'drift_severity': 'high' if num_drifting_channels > len(channel_drift) / 2 else
                            'medium' if num_drifting_channels > 0 else 'low'
        }
        
        self.drift_analysis = drift_analysis
        return drift_analysis
    
    def _generate_performance_alerts(self,
                                   anomalies: List[Dict[str, Any]],
                                   drift_analysis: Dict[str, Any],
                                   timestamp: datetime):
        """Generate performance and drift alerts."""
        
        new_alerts = []
        
        # Anomaly alerts
        for anomaly in anomalies:
            alert_key = f"anomaly_{anomaly['channel']}_{anomaly['metric']}"
            
            if self._should_send_alert(alert_key, timestamp):
                new_alerts.append({
                    'id': alert_key,
                    'type': 'anomaly',
                    'severity': anomaly['severity'],
                    'timestamp': timestamp,
                    'channel': anomaly['channel'],
                    'metric': anomaly['metric'],
                    'title': f'{anomaly["channel"]} {anomaly["metric"]} {anomaly["type"]}',
                    'message': f'{anomaly["channel"]} {anomaly["metric"]} shows {anomaly["type"]} ({anomaly["deviation_percent"]:.1f}% deviation from baseline)',
                    'details': anomaly
                })
        
        # Model drift alerts
        if not drift_analysis.get('insufficient_data', False):
            drift_severity = drift_analysis.get('drift_severity', 'low')
            
            if drift_severity in ['high', 'medium']:
                alert_key = f"model_drift_{drift_severity}"
                
                if self._should_send_alert(alert_key, timestamp):
                    new_alerts.append({
                        'id': alert_key,
                        'type': 'model_drift',
                        'severity': drift_severity,
                        'timestamp': timestamp,
                        'title': f'Attribution Model Drift Detected',
                        'message': f'Model showing {drift_severity} drift: {drift_analysis["num_drifting_channels"]} channels drifting (stability: {drift_analysis["model_stability_score"]:.1%})',
                        'details': drift_analysis
                    })
        
        # Update alerts
        self.active_alerts.extend(new_alerts)
        self.alert_history.extend(new_alerts)
        
        # Clean up old alerts
        cutoff_time = timestamp - timedelta(hours=24)
        self.active_alerts = [
            alert for alert in self.active_alerts
            if alert['timestamp'] > cutoff_time
        ]
    
    def _should_send_alert(self, alert_key: str, timestamp: datetime) -> bool:
        """Check if alert should be sent based on cooldown period."""
        
        last_alert_time = self.last_alert_times.get(alert_key)
        
        if last_alert_time is None:
            self.last_alert_times[alert_key] = timestamp
            return True
        
        time_since_last = (timestamp - last_alert_time).total_seconds()
        
        if time_since_last >= self.alert_cooldown:
            self.last_alert_times[alert_key] = timestamp
            return True
        
        return False
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        
        return {
            'timestamp': datetime.now(),
            'monitoring_status': {
                'channels_monitored': len(self.performance_history),
                'baseline_metrics_available': len(self.baseline_metrics),
                'model_stability_score': self.model_stability_score,
                'active_alerts': len(self.active_alerts)
            },
            'current_metrics': self.current_metrics,
            'recent_anomalies': self.anomalies[-10:],
            'drift_analysis': self.drift_analysis,
            'active_alerts': self.active_alerts,
            'performance_trends': self._get_performance_trends()
        }
    
    def _get_performance_trends(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance trends for each channel."""
        
        trends = {}
        
        for channel, baselines in self.baseline_metrics.items():
            channel_trends = {}
            
            for metric, baseline in baselines.items():
                channel_trends[metric] = {
                    'trend_direction': baseline['trend'],
                    'recent_average': baseline['mean'],
                    'volatility': baseline['std'] / baseline['mean'] if baseline['mean'] > 0 else 0
                }
            
            trends[channel] = channel_trends
        
        return trends
    
    def get_channel_health_score(self, channel: str) -> float:
        """Calculate overall health score for a channel (0-1)."""
        
        if channel not in self.baseline_metrics:
            return 0.5  # Unknown
        
        score_components = []
        
        # Anomaly penalty
        recent_anomalies = [
            a for a in self.anomalies[-50:]  # Last 50 anomalies
            if a['channel'] == channel
        ]
        
        anomaly_penalty = min(len(recent_anomalies) * 0.1, 0.5)
        score_components.append(1 - anomaly_penalty)
        
        # Drift penalty
        if self.drift_analysis and channel in self.drift_analysis.get('channel_drift', {}):
            drift_info = self.drift_analysis['channel_drift'][channel]
            drift_penalty = min(drift_info['drift_ratio'], 0.3)
            score_components.append(1 - drift_penalty)
        
        # Volatility penalty
        baselines = self.baseline_metrics[channel]
        avg_volatility = 0
        
        for metric, baseline in baselines.items():
            if baseline['mean'] > 0:
                cv = baseline['std'] / baseline['mean']
                avg_volatility += min(cv, 1.0)
        
        if baselines:
            avg_volatility /= len(baselines)
            volatility_penalty = avg_volatility * 0.2
            score_components.append(1 - volatility_penalty)
        
        # Return average score
        return np.mean(score_components) if score_components else 0.5
    
    def generate_monitoring_report(self) -> str:
        """Generate comprehensive monitoring report."""
        
        report = "# Marketing Performance Monitoring Report\n\n"
        report += "**Real-time Performance Analytics by Sotiris Spyrou**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        # Monitoring Overview
        status = self.get_monitoring_dashboard()['monitoring_status']
        report += f"## Monitoring Status\n\n"
        report += f"- **Channels Monitored**: {status['channels_monitored']}\n"
        report += f"- **Model Stability**: {status['model_stability_score']:.1%}\n"
        report += f"- **Active Alerts**: {status['active_alerts']}\n\n"
        
        # Channel Health Scores
        report += f"## Channel Health Assessment\n\n"
        report += "| Channel | Health Score | Status |\n"
        report += "|---------|-------------|--------|\n"
        
        for channel in self.baseline_metrics.keys():
            health_score = self.get_channel_health_score(channel)
            status_emoji = "🟢" if health_score > 0.8 else "🟡" if health_score > 0.6 else "🔴"
            status_text = "Healthy" if health_score > 0.8 else "Warning" if health_score > 0.6 else "Critical"
            
            report += f"| {channel} | {health_score:.1%} | {status_emoji} {status_text} |\n"
        
        # Recent Anomalies
        if self.anomalies:
            recent_anomalies = self.anomalies[-5:]
            report += f"\n## Recent Anomalies ({len(recent_anomalies)})\n\n"
            
            for anomaly in recent_anomalies:
                severity_emoji = "🔴" if anomaly['severity'] == 'high' else "🟡"
                report += f"- {severity_emoji} **{anomaly['channel']}** {anomaly['metric']}: {anomaly['deviation_percent']:.1f}% deviation\n"
        
        # Model Drift Analysis
        if self.drift_analysis and not self.drift_analysis.get('insufficient_data', False):
            report += f"\n## Model Drift Analysis\n\n"
            report += f"- **Overall Stability**: {self.drift_analysis['model_stability_score']:.1%}\n"
            report += f"- **Drifting Channels**: {self.drift_analysis['num_drifting_channels']}\n"
            report += f"- **Drift Severity**: {self.drift_analysis['drift_severity'].title()}\n\n"
        
        # Active Alerts
        if self.active_alerts:
            report += f"## Active Alerts ({len(self.active_alerts)})\n\n"
            
            for alert in self.active_alerts[-5:]:  # Show last 5 alerts
                severity_emoji = "🔴" if alert['severity'] == 'high' else "🟡" if alert['severity'] == 'medium' else "🟢"
                report += f"- {severity_emoji} **{alert['title']}**: {alert['message']}\n"
        
        report += "\n---\n*This monitoring system provides real-time performance insights and anomaly detection. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for enterprise implementations.*"
        
        return report


def demo_performance_monitoring():
    """Executive demonstration of Performance Monitoring."""
    
    print("=== Performance Monitoring: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    # Initialize monitor
    monitor = PerformanceMonitor(
        monitoring_window=7,
        anomaly_threshold=2.0,
        drift_threshold=0.15
    )
    
    print("📊 Simulating 14 days of performance data...")
    
    np.random.seed(42)
    channels = ['Search', 'Display', 'Social', 'Email', 'Direct']
    
    # Simulate performance data over time with some anomalies and drift
    base_date = datetime.now() - timedelta(days=14)
    
    for day in range(14):
        timestamp = base_date + timedelta(days=day)
        
        # Simulate channel metrics with trends and anomalies
        channel_metrics = {}
        attribution_weights = {}
        
        for i, channel in enumerate(channels):
            # Base performance with trend
            base_ctr = 0.05 + (i * 0.01)
            base_cvr = 0.15 + (i * 0.02)
            base_roas = 2.0 + (i * 0.3)
            
            # Add time-based trend
            trend_factor = 1 + (day / 100)  # Slight upward trend
            
            # Add random variation
            noise = np.random.normal(1, 0.1)
            
            # Introduce anomaly on day 10 for Search
            anomaly_factor = 1
            if day == 10 and channel == 'Search':
                anomaly_factor = 0.3  # 70% drop
            
            channel_metrics[channel] = {
                'ctr': base_ctr * trend_factor * noise * anomaly_factor,
                'cvr': base_cvr * trend_factor * noise,
                'roas': base_roas * trend_factor * noise * anomaly_factor,
                'spend': 10000 + np.random.normal(0, 1000),
                'conversions': 100 + np.random.normal(0, 10)
            }
            
            # Attribution weights with gradual drift
            base_weight = 0.2
            drift_factor = 1 + (day * 0.01) if channel == 'Search' else 1 - (day * 0.005) if channel == 'Display' else 1
            attribution_weights[channel] = base_weight * drift_factor
        
        # Normalize attribution weights
        total_weight = sum(attribution_weights.values())
        attribution_weights = {k: v/total_weight for k, v in attribution_weights.items()}
        
        # Update monitor
        monitor.update_performance_data(channel_metrics, attribution_weights, timestamp)
        
        if day % 3 == 0:
            print(f"  Day {day+1}: Processed {len(channels)} channels")
    
    # Get monitoring results
    dashboard = monitor.get_monitoring_dashboard()
    
    print("\n🎯 PERFORMANCE MONITORING RESULTS")
    print("=" * 50)
    
    # Monitoring status
    status = dashboard['monitoring_status']
    print(f"\n📈 Monitoring Overview:")
    print(f"  • Channels Monitored: {status['channels_monitored']}")
    print(f"  • Model Stability: {status['model_stability_score']:.1%}")
    print(f"  • Active Alerts: {status['active_alerts']}")
    
    # Channel health scores
    print(f"\n💊 Channel Health Assessment:")
    for channel in channels:
        health_score = monitor.get_channel_health_score(channel)
        health_emoji = "🟢" if health_score > 0.8 else "🟡" if health_score > 0.6 else "🔴"
        health_status = "Healthy" if health_score > 0.8 else "Warning" if health_score > 0.6 else "Critical"
        print(f"{health_emoji} {channel:8}: {health_score:.1%} ({health_status})")
    
    # Recent anomalies
    recent_anomalies = dashboard['recent_anomalies']
    print(f"\n🚨 Recent Anomalies ({len(recent_anomalies)}):")
    
    if recent_anomalies:
        for anomaly in recent_anomalies[-3:]:  # Show last 3
            severity_emoji = "🔴" if anomaly['severity'] == 'high' else "🟡"
            print(f"{severity_emoji} {anomaly['channel']} {anomaly['metric']}: {anomaly['deviation_percent']:.1f}% deviation ({anomaly['type']})")
    else:
        print("  ✅ No anomalies detected")
    
    # Model drift analysis
    drift = dashboard.get('drift_analysis', {})
    if drift and not drift.get('insufficient_data', False):
        print(f"\n📊 Model Drift Analysis:")
        print(f"  • Overall Stability: {drift['model_stability_score']:.1%}")
        print(f"  • Drifting Channels: {drift['num_drifting_channels']}")
        print(f"  • Drift Severity: {drift['drift_severity'].title()}")
        
        # Show top drifting channels
        if 'channel_drift' in drift:
            drifting_channels = [
                (ch, metrics['drift_ratio']) 
                for ch, metrics in drift['channel_drift'].items() 
                if metrics['is_drifting']
            ]
            
            if drifting_channels:
                drifting_channels.sort(key=lambda x: x[1], reverse=True)
                print(f"\n🔄 Top Drifting Channels:")
                for channel, drift_ratio in drifting_channels[:3]:
                    print(f"  • {channel}: {drift_ratio:.1%} drift")
    
    # Active alerts
    active_alerts = dashboard['active_alerts']
    print(f"\n🚨 Active Alerts ({len(active_alerts)}):")
    
    if active_alerts:
        for alert in active_alerts[-3:]:  # Show last 3
            severity_emoji = "🔴" if alert['severity'] == 'high' else "🟡" if alert['severity'] == 'medium' else "🟢"
            print(f"{severity_emoji} {alert['title']}: {alert['message']}")
    else:
        print("  ✅ All systems operating within normal parameters")
    
    # Performance trends
    trends = dashboard['performance_trends']
    print(f"\n📈 Performance Trends:")
    
    for channel in channels[:3]:  # Show top 3 channels
        if channel in trends:
            channel_trends = trends[channel]
            trend_summary = []
            
            for metric, trend_data in channel_trends.items():
                direction = trend_data['trend_direction']
                arrow = "↗️" if direction > 0.1 else "↘️" if direction < -0.1 else "→"
                trend_summary.append(f"{metric} {arrow}")
            
            print(f"  • {channel}: {', '.join(trend_summary[:2])}")
    
    print(f"\n📋 Generating Monitoring Report...")
    report = monitor.generate_monitoring_report()
    
    print(f"\n📄 MONITORING REPORT EXCERPT:")
    print("-" * 40)
    
    # Show excerpt of the report
    report_lines = report.split('\n')
    excerpt_lines = report_lines[6:16]  # Skip header, show key sections
    for line in excerpt_lines:
        print(line)
    
    print("\n" + "="*60)
    print("🚀 Real-time performance monitoring with anomaly detection")
    print("💼 Enterprise-grade marketing analytics surveillance")
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_performance_monitoring()