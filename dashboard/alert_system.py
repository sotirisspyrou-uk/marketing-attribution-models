"""
Marketing Attribution Alert System

Real-time alerting system for marketing attribution anomalies, performance
degradation, and optimization opportunities across channels and campaigns.

Author: Sotiris Spyrou
Portfolio: https://verityai.co
LinkedIn: https://www.linkedin.com/in/sspyrou/

DISCLAIMER: This is demonstration code for portfolio purposes only.
Not intended for production use without proper testing and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of marketing attribution alerts."""
    PERFORMANCE_DROP = "performance_drop"
    ATTRIBUTION_SHIFT = "attribution_shift"
    BUDGET_OVERSPEND = "budget_overspend"
    CONVERSION_ANOMALY = "conversion_anomaly"
    CHANNEL_FAILURE = "channel_failure"
    MODEL_DRIFT = "model_drift"
    FRAUD_DETECTION = "fraud_detection"
    OPPORTUNITY = "opportunity"


@dataclass
class Alert:
    """Marketing attribution alert data structure."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    channel: Optional[str] = None
    campaign: Optional[str] = None
    metric: Optional[str] = None
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None
    recommendation: Optional[str] = None
    is_resolved: bool = False
    resolved_timestamp: Optional[datetime] = None


class AlertSystem:
    """
    Comprehensive alert system for marketing attribution monitoring.
    
    Monitors attribution performance, detects anomalies, identifies optimization
    opportunities, and sends intelligent alerts to marketing teams.
    """
    
    def __init__(self,
                 alert_retention_days: int = 30,
                 cooldown_minutes: int = 60,
                 enable_auto_resolution: bool = True):
        """
        Initialize Alert System.
        
        Args:
            alert_retention_days: Days to retain alert history
            cooldown_minutes: Minimum time between duplicate alerts
            enable_auto_resolution: Automatically resolve alerts when conditions normalize
        """
        self.alert_retention_days = alert_retention_days
        self.cooldown_minutes = cooldown_minutes
        self.enable_auto_resolution = enable_auto_resolution
        
        # Alert storage
        self.active_alerts: List[Alert] = []
        self.alert_history: deque = deque(maxlen=10000)
        self.alert_rules: Dict[str, Dict] = {}
        self.cooldown_tracker: Dict[str, datetime] = {}
        
        # Performance tracking
        self.performance_baselines: Dict[str, Dict] = {}
        self.attribution_history: deque = deque(maxlen=1000)
        self.budget_tracking: Dict[str, Dict] = {}
        
        # Alert handlers
        self.alert_handlers: Dict[AlertType, List[Callable]] = defaultdict(list)
        
        self._initialize_default_rules()
        
        logger.info("Alert system initialized")
    
    def _initialize_default_rules(self):
        """Initialize default alerting rules."""
        
        self.alert_rules = {
            # Performance degradation rules
            'conversion_rate_drop': {
                'type': AlertType.PERFORMANCE_DROP,
                'metric': 'conversion_rate',
                'threshold': -0.15,  # 15% drop
                'severity': AlertSeverity.HIGH,
                'lookback_hours': 24
            },
            'roas_drop': {
                'type': AlertType.PERFORMANCE_DROP,
                'metric': 'roas',
                'threshold': -0.20,  # 20% drop
                'severity': AlertSeverity.HIGH,
                'lookback_hours': 24
            },
            'ctr_drop': {
                'type': AlertType.PERFORMANCE_DROP,
                'metric': 'ctr',
                'threshold': -0.25,  # 25% drop
                'severity': AlertSeverity.MEDIUM,
                'lookback_hours': 12
            },
            
            # Attribution shift rules
            'attribution_shift': {
                'type': AlertType.ATTRIBUTION_SHIFT,
                'metric': 'attribution_weight',
                'threshold': 0.10,  # 10% shift
                'severity': AlertSeverity.MEDIUM,
                'lookback_hours': 48
            },
            
            # Budget rules
            'budget_overspend': {
                'type': AlertType.BUDGET_OVERSPEND,
                'metric': 'spend_pacing',
                'threshold': 1.20,  # 20% over budget
                'severity': AlertSeverity.HIGH,
                'lookback_hours': 24
            },
            
            # Anomaly detection rules
            'conversion_anomaly': {
                'type': AlertType.CONVERSION_ANOMALY,
                'metric': 'conversions',
                'threshold': 3.0,  # 3 standard deviations
                'severity': AlertSeverity.MEDIUM,
                'lookback_hours': 6
            },
            
            # Opportunity detection rules
            'underutilized_channel': {
                'type': AlertType.OPPORTUNITY,
                'metric': 'efficiency_score',
                'threshold': 1.30,  # 30% above average efficiency
                'severity': AlertSeverity.LOW,
                'lookback_hours': 72
            }
        }
    
    def process_performance_data(self,
                               channel_data: Dict[str, Dict[str, float]],
                               attribution_data: Dict[str, float],
                               budget_data: Optional[Dict[str, Dict]] = None,
                               timestamp: Optional[datetime] = None) -> List[Alert]:
        """
        Process new performance data and generate alerts.
        
        Args:
            channel_data: Current channel performance metrics
            attribution_data: Current attribution weights
            budget_data: Budget and spend information
            timestamp: Data timestamp (defaults to now)
            
        Returns:
            List of newly generated alerts
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        new_alerts = []
        
        # Update baselines and tracking
        self._update_performance_baselines(channel_data, timestamp)
        self._update_attribution_history(attribution_data, timestamp)
        
        if budget_data:
            self._update_budget_tracking(budget_data, timestamp)
        
        # Performance anomaly detection
        performance_alerts = self._detect_performance_anomalies(channel_data, timestamp)
        new_alerts.extend(performance_alerts)
        
        # Attribution shift detection
        attribution_alerts = self._detect_attribution_shifts(attribution_data, timestamp)
        new_alerts.extend(attribution_alerts)
        
        # Budget alerts
        if budget_data:
            budget_alerts = self._detect_budget_issues(budget_data, timestamp)
            new_alerts.extend(budget_alerts)
        
        # Opportunity detection
        opportunity_alerts = self._detect_opportunities(channel_data, timestamp)
        new_alerts.extend(opportunity_alerts)
        
        # Process and store alerts
        for alert in new_alerts:
            if self._should_send_alert(alert):
                self._send_alert(alert)
        
        # Auto-resolve alerts if enabled
        if self.enable_auto_resolution:
            self._auto_resolve_alerts(channel_data, attribution_data, timestamp)
        
        # Cleanup old alerts
        self._cleanup_alerts(timestamp)
        
        logger.info(f"Processed performance data: {len(new_alerts)} new alerts generated")
        return new_alerts
    
    def _update_performance_baselines(self, channel_data: Dict[str, Dict], timestamp: datetime):
        """Update performance baselines for anomaly detection."""
        
        for channel, metrics in channel_data.items():
            if channel not in self.performance_baselines:
                self.performance_baselines[channel] = defaultdict(list)
            
            # Store metrics with timestamps
            for metric, value in metrics.items():
                baseline_data = self.performance_baselines[channel][metric]
                baseline_data.append({'value': value, 'timestamp': timestamp})
                
                # Keep only recent data (24 hours)
                cutoff_time = timestamp - timedelta(hours=24)
                self.performance_baselines[channel][metric] = [
                    d for d in baseline_data if d['timestamp'] > cutoff_time
                ]
    
    def _update_attribution_history(self, attribution_data: Dict[str, float], timestamp: datetime):
        """Update attribution history for shift detection."""
        
        self.attribution_history.append({
            'timestamp': timestamp,
            'attribution': attribution_data.copy()
        })
    
    def _update_budget_tracking(self, budget_data: Dict[str, Dict], timestamp: datetime):
        """Update budget tracking for overspend alerts."""
        
        for channel, budget_info in budget_data.items():
            if channel not in self.budget_tracking:
                self.budget_tracking[channel] = []
            
            self.budget_tracking[channel].append({
                'timestamp': timestamp,
                'spend': budget_info.get('spend', 0),
                'budget': budget_info.get('budget', 0),
                'daily_budget': budget_info.get('daily_budget', 0)
            })
            
            # Keep only recent data (7 days)
            cutoff_time = timestamp - timedelta(days=7)
            self.budget_tracking[channel] = [
                d for d in self.budget_tracking[channel] 
                if d['timestamp'] > cutoff_time
            ]
    
    def _detect_performance_anomalies(self, channel_data: Dict[str, Dict], timestamp: datetime) -> List[Alert]:
        """Detect performance anomalies using statistical analysis."""
        
        alerts = []
        
        for channel, metrics in channel_data.items():
            if channel not in self.performance_baselines:
                continue
            
            for metric, current_value in metrics.items():
                baseline_data = self.performance_baselines[channel].get(metric, [])
                
                if len(baseline_data) < 10:  # Need sufficient baseline
                    continue
                
                # Calculate baseline statistics
                baseline_values = [d['value'] for d in baseline_data]
                baseline_mean = np.mean(baseline_values)
                baseline_std = np.std(baseline_values)
                
                if baseline_std == 0:
                    continue
                
                # Z-score anomaly detection
                z_score = (current_value - baseline_mean) / baseline_std
                
                # Check for significant drops
                if metric in ['conversion_rate', 'roas', 'ctr'] and z_score < -2.0:
                    severity = AlertSeverity.CRITICAL if z_score < -3.0 else AlertSeverity.HIGH
                    
                    alert = Alert(
                        alert_id=f"perf_drop_{channel}_{metric}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                        alert_type=AlertType.PERFORMANCE_DROP,
                        severity=severity,
                        title=f"{channel} {metric.upper()} Drop Detected",
                        message=f"{channel} {metric} dropped {abs(z_score):.1f} standard deviations below baseline",
                        details={
                            'z_score': z_score,
                            'current_value': current_value,
                            'baseline_mean': baseline_mean,
                            'baseline_std': baseline_std,
                            'drop_percentage': (baseline_mean - current_value) / baseline_mean * 100
                        },
                        timestamp=timestamp,
                        channel=channel,
                        metric=metric,
                        threshold_value=baseline_mean - 2 * baseline_std,
                        actual_value=current_value,
                        recommendation=self._get_performance_recommendation(channel, metric, z_score)
                    )
                    alerts.append(alert)
                
                # Check for unusual spikes
                elif z_score > 3.0:
                    alert = Alert(
                        alert_id=f"perf_spike_{channel}_{metric}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                        alert_type=AlertType.CONVERSION_ANOMALY,
                        severity=AlertSeverity.MEDIUM,
                        title=f"{channel} {metric.upper()} Spike Detected",
                        message=f"{channel} {metric} spiked {z_score:.1f} standard deviations above baseline",
                        details={
                            'z_score': z_score,
                            'current_value': current_value,
                            'baseline_mean': baseline_mean,
                            'spike_magnitude': (current_value - baseline_mean) / baseline_mean * 100
                        },
                        timestamp=timestamp,
                        channel=channel,
                        metric=metric,
                        recommendation="Investigate potential data quality issues or exceptional performance drivers"
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _detect_attribution_shifts(self, attribution_data: Dict[str, float], timestamp: datetime) -> List[Alert]:
        """Detect significant shifts in attribution weights."""
        
        alerts = []
        
        if len(self.attribution_history) < 10:  # Need baseline data
            return alerts
        
        # Calculate baseline attribution weights
        recent_attributions = list(self.attribution_history)[-10:]
        baseline_attribution = {}
        
        for channel in attribution_data.keys():
            channel_weights = [
                attr['attribution'].get(channel, 0) 
                for attr in recent_attributions
            ]
            baseline_attribution[channel] = np.mean(channel_weights)
        
        # Detect shifts
        for channel, current_weight in attribution_data.items():
            baseline_weight = baseline_attribution.get(channel, 0)
            
            if baseline_weight == 0:
                continue
            
            # Calculate shift magnitude
            shift_ratio = abs(current_weight - baseline_weight) / baseline_weight
            
            if shift_ratio > 0.25:  # 25% shift
                severity = AlertSeverity.HIGH if shift_ratio > 0.50 else AlertSeverity.MEDIUM
                
                shift_direction = "increased" if current_weight > baseline_weight else "decreased"
                
                alert = Alert(
                    alert_id=f"attr_shift_{channel}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    alert_type=AlertType.ATTRIBUTION_SHIFT,
                    severity=severity,
                    title=f"{channel} Attribution Shift",
                    message=f"{channel} attribution weight {shift_direction} by {shift_ratio:.1%}",
                    details={
                        'current_weight': current_weight,
                        'baseline_weight': baseline_weight,
                        'shift_ratio': shift_ratio,
                        'shift_direction': shift_direction
                    },
                    timestamp=timestamp,
                    channel=channel,
                    metric='attribution_weight',
                    recommendation=self._get_attribution_recommendation(channel, shift_direction, shift_ratio)
                )
                alerts.append(alert)
        
        return alerts
    
    def _detect_budget_issues(self, budget_data: Dict[str, Dict], timestamp: datetime) -> List[Alert]:
        """Detect budget overspend and pacing issues."""
        
        alerts = []
        
        for channel, budget_info in budget_data.items():
            spend = budget_info.get('spend', 0)
            budget = budget_info.get('budget', 0)
            daily_budget = budget_info.get('daily_budget', 0)
            
            if budget > 0:
                spend_ratio = spend / budget
                
                # Overspend alert
                if spend_ratio > 1.10:  # 10% overspend
                    severity = AlertSeverity.CRITICAL if spend_ratio > 1.25 else AlertSeverity.HIGH
                    
                    alert = Alert(
                        alert_id=f"budget_overspend_{channel}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                        alert_type=AlertType.BUDGET_OVERSPEND,
                        severity=severity,
                        title=f"{channel} Budget Overspend",
                        message=f"{channel} has exceeded budget by {(spend_ratio - 1) * 100:.1f}%",
                        details={
                            'spend': spend,
                            'budget': budget,
                            'overspend_amount': spend - budget,
                            'overspend_percentage': (spend_ratio - 1) * 100
                        },
                        timestamp=timestamp,
                        channel=channel,
                        metric='budget',
                        threshold_value=budget,
                        actual_value=spend,
                        recommendation="Immediate budget adjustment or campaign pause recommended"
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _detect_opportunities(self, channel_data: Dict[str, Dict], timestamp: datetime) -> List[Alert]:
        """Detect optimization opportunities."""
        
        alerts = []
        
        # Calculate efficiency scores
        efficiency_scores = {}
        for channel, metrics in channel_data.items():
            roas = metrics.get('roas', 0)
            cvr = metrics.get('cvr', 0)
            ctr = metrics.get('ctr', 0)
            
            # Simple efficiency score
            efficiency_score = (roas * cvr * ctr) if all([roas, cvr, ctr]) else 0
            efficiency_scores[channel] = efficiency_score
        
        if not efficiency_scores:
            return alerts
        
        # Find top-performing channels
        avg_efficiency = np.mean(list(efficiency_scores.values()))
        
        for channel, score in efficiency_scores.items():
            if score > avg_efficiency * 1.5:  # 50% above average
                alert = Alert(
                    alert_id=f"opportunity_{channel}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    alert_type=AlertType.OPPORTUNITY,
                    severity=AlertSeverity.LOW,
                    title=f"{channel} Scaling Opportunity",
                    message=f"{channel} showing exceptional efficiency - consider budget increase",
                    details={
                        'efficiency_score': score,
                        'avg_efficiency': avg_efficiency,
                        'performance_ratio': score / avg_efficiency,
                        'channel_metrics': channel_data[channel]
                    },
                    timestamp=timestamp,
                    channel=channel,
                    recommendation=f"Consider increasing {channel} budget by 20-30% to capitalize on strong performance"
                )
                alerts.append(alert)
        
        return alerts
    
    def _should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent based on cooldown and deduplication."""
        
        alert_key = f"{alert.alert_type.value}_{alert.channel}_{alert.metric}"
        last_sent = self.cooldown_tracker.get(alert_key)
        
        if last_sent:
            time_since_last = (alert.timestamp - last_sent).total_seconds() / 60
            if time_since_last < self.cooldown_minutes:
                return False
        
        self.cooldown_tracker[alert_key] = alert.timestamp
        return True
    
    def _send_alert(self, alert: Alert):
        """Send alert through configured channels."""
        
        # Add to active alerts
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Execute alert handlers
        for handler in self.alert_handlers[alert.alert_type]:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.info(f"Alert sent: {alert.title} ({alert.severity.value})")
    
    def _auto_resolve_alerts(self, channel_data: Dict[str, Dict], 
                           attribution_data: Dict[str, float], 
                           timestamp: datetime):
        """Automatically resolve alerts when conditions normalize."""
        
        resolved_alerts = []
        
        for alert in self.active_alerts:
            if alert.is_resolved:
                continue
            
            should_resolve = False
            
            # Check performance recovery
            if alert.alert_type == AlertType.PERFORMANCE_DROP:
                channel = alert.channel
                metric = alert.metric
                
                if channel in channel_data and metric in channel_data[channel]:
                    current_value = channel_data[channel][metric]
                    threshold = alert.threshold_value
                    
                    if threshold and current_value >= threshold * 0.95:  # 95% of threshold
                        should_resolve = True
            
            # Check attribution stabilization
            elif alert.alert_type == AlertType.ATTRIBUTION_SHIFT:
                channel = alert.channel
                if channel in attribution_data:
                    current_weight = attribution_data[channel]
                    original_weight = alert.details.get('baseline_weight', 0)
                    
                    if original_weight > 0:
                        current_shift = abs(current_weight - original_weight) / original_weight
                        if current_shift < 0.15:  # Shift reduced to < 15%
                            should_resolve = True
            
            if should_resolve:
                alert.is_resolved = True
                alert.resolved_timestamp = timestamp
                resolved_alerts.append(alert)
        
        # Remove resolved alerts from active list
        self.active_alerts = [a for a in self.active_alerts if not a.is_resolved]
        
        if resolved_alerts:
            logger.info(f"Auto-resolved {len(resolved_alerts)} alerts")
    
    def _cleanup_alerts(self, timestamp: datetime):
        """Clean up old alerts."""
        
        cutoff_time = timestamp - timedelta(days=self.alert_retention_days)
        
        # Clean up alert history
        self.alert_history = deque([
            alert for alert in self.alert_history 
            if alert.timestamp > cutoff_time
        ], maxlen=10000)
        
        # Clean up cooldown tracker
        cutoff_cooldown = timestamp - timedelta(hours=24)
        self.cooldown_tracker = {
            key: ts for key, ts in self.cooldown_tracker.items() 
            if ts > cutoff_cooldown
        }
    
    def _get_performance_recommendation(self, channel: str, metric: str, z_score: float) -> str:
        """Get performance-specific recommendations."""
        
        recommendations = {
            'conversion_rate': f"Review {channel} targeting and landing page optimization. Consider A/B testing ad creative.",
            'roas': f"Analyze {channel} spend efficiency. Consider bid adjustments and audience refinement.",
            'ctr': f"Review {channel} ad creative performance and audience relevance. Test new creative variants.",
            'cvr': f"Optimize {channel} landing pages and conversion funnel. Review user experience."
        }
        
        base_rec = recommendations.get(metric, f"Investigate {channel} {metric} performance decline")
        
        if z_score < -3.0:
            return f"URGENT: {base_rec} Consider pausing underperforming campaigns."
        else:
            return base_rec
    
    def _get_attribution_recommendation(self, channel: str, direction: str, magnitude: float) -> str:
        """Get attribution shift recommendations."""
        
        if direction == "increased":
            if magnitude > 0.5:
                return f"Major attribution increase for {channel}. Verify tracking accuracy and consider budget reallocation."
            else:
                return f"{channel} showing stronger attribution influence. Consider optimizing budget allocation."
        else:
            if magnitude > 0.5:
                return f"Significant attribution decline for {channel}. Review channel performance and competitive landscape."
            else:
                return f"{channel} attribution decreased. Monitor performance trends and adjust strategy if needed."
    
    def register_alert_handler(self, alert_type: AlertType, handler: Callable[[Alert], None]):
        """Register custom alert handler."""
        self.alert_handlers[alert_type].append(handler)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return self.active_alerts.copy()
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert system summary."""
        
        active_by_severity = defaultdict(int)
        active_by_type = defaultdict(int)
        
        for alert in self.active_alerts:
            active_by_severity[alert.severity.value] += 1
            active_by_type[alert.alert_type.value] += 1
        
        recent_alerts = [a for a in self.alert_history if 
                        (datetime.now() - a.timestamp).days < 7]
        
        return {
            'total_active_alerts': len(self.active_alerts),
            'alerts_by_severity': dict(active_by_severity),
            'alerts_by_type': dict(active_by_type),
            'alerts_last_7_days': len(recent_alerts),
            'auto_resolved_alerts': sum(1 for a in recent_alerts if a.is_resolved),
            'alert_rules_configured': len(self.alert_rules)
        }
    
    def generate_alert_report(self) -> str:
        """Generate comprehensive alert system report."""
        
        report = "# Marketing Attribution Alert System Report\n\n"
        report += "**Real-Time Marketing Surveillance by Sotiris Spyrou**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        # Alert Summary
        summary = self.get_alert_summary()
        report += f"## Alert System Status\n\n"
        report += f"- **Active Alerts**: {summary['total_active_alerts']}\n"
        report += f"- **Alerts (Last 7 Days)**: {summary['alerts_last_7_days']}\n"
        report += f"- **Auto-Resolved**: {summary['auto_resolved_alerts']}\n"
        report += f"- **Alert Rules**: {summary['alert_rules_configured']}\n\n"
        
        # Active Alerts by Severity
        if summary['alerts_by_severity']:
            report += f"### Active Alerts by Severity\n\n"
            for severity, count in summary['alerts_by_severity'].items():
                emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(severity, "📊")
                report += f"- {emoji} **{severity.title()}**: {count} alerts\n"
            report += "\n"
        
        # Recent Critical Alerts
        critical_alerts = [a for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            report += f"### Critical Alerts Requiring Immediate Action\n\n"
            for alert in critical_alerts[:5]:
                report += f"- **{alert.title}**: {alert.message}\n"
            report += "\n"
        
        # Alert Type Distribution
        if summary['alerts_by_type']:
            report += f"### Alert Categories\n\n"
            for alert_type, count in summary['alerts_by_type'].items():
                report += f"- **{alert_type.replace('_', ' ').title()}**: {count} alerts\n"
            report += "\n"
        
        report += "---\n*This alert system provides proactive monitoring and intelligent notifications. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for enterprise implementations.*"
        
        return report


def demo_alert_system():
    """Executive demonstration of Alert System."""
    
    print("=== Marketing Attribution Alert System: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    # Initialize alert system
    alert_system = AlertSystem(
        alert_retention_days=30,
        cooldown_minutes=60,
        enable_auto_resolution=True
    )
    
    print("🚨 Simulating marketing attribution monitoring...")
    
    np.random.seed(42)
    
    # Simulate baseline performance over 5 days
    channels = ['Search', 'Display', 'Social', 'Email', 'Direct']
    base_timestamp = datetime.now() - timedelta(days=5)
    
    # Build baseline
    for day in range(5):
        timestamp = base_timestamp + timedelta(days=day)
        
        # Normal performance data
        channel_data = {}
        for channel in channels:
            channel_data[channel] = {
                'conversion_rate': 0.15 + np.random.normal(0, 0.02),
                'roas': 2.5 + np.random.normal(0, 0.3),
                'ctr': 0.05 + np.random.normal(0, 0.005),
                'cvr': 0.20 + np.random.normal(0, 0.03)
            }
        
        attribution_data = {
            'Search': 0.35, 'Display': 0.25, 'Social': 0.20, 
            'Email': 0.15, 'Direct': 0.05
        }
        
        alert_system.process_performance_data(channel_data, attribution_data, timestamp=timestamp)
    
    print("📊 Baseline established over 5 days...")
    
    # Simulate day with performance issues
    problem_timestamp = datetime.now()
    
    # Performance drop scenario
    problem_channel_data = {}
    for channel in channels:
        if channel == 'Search':
            # Simulate Search performance drop
            problem_channel_data[channel] = {
                'conversion_rate': 0.08,  # 47% drop from baseline ~0.15
                'roas': 1.2,              # 52% drop from baseline ~2.5
                'ctr': 0.035,             # 30% drop from baseline ~0.05
                'cvr': 0.18               # Minor drop
            }
        else:
            # Normal performance for other channels
            problem_channel_data[channel] = {
                'conversion_rate': 0.15 + np.random.normal(0, 0.01),
                'roas': 2.5 + np.random.normal(0, 0.2),
                'ctr': 0.05 + np.random.normal(0, 0.003),
                'cvr': 0.20 + np.random.normal(0, 0.02)
            }
    
    # Attribution shift
    shifted_attribution = {
        'Search': 0.20,    # Dropped from 0.35
        'Display': 0.35,   # Increased from 0.25
        'Social': 0.25,    # Increased from 0.20
        'Email': 0.15,     # Unchanged
        'Direct': 0.05     # Unchanged
    }
    
    # Budget overspend scenario
    budget_data = {
        'Display': {'spend': 12000, 'budget': 10000, 'daily_budget': 500},
        'Social': {'spend': 8000, 'budget': 8000, 'daily_budget': 400},
        'Search': {'spend': 15000, 'budget': 15000, 'daily_budget': 750}
    }
    
    # Process problem scenario
    alerts = alert_system.process_performance_data(
        problem_channel_data, 
        shifted_attribution, 
        budget_data, 
        timestamp=problem_timestamp
    )
    
    print(f"\n🎯 ALERT SYSTEM RESULTS")
    print("=" * 50)
    
    print(f"\n🚨 Generated {len(alerts)} alerts:")
    
    # Show alerts by severity
    severity_groups = defaultdict(list)
    for alert in alerts:
        severity_groups[alert.severity].append(alert)
    
    for severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH, AlertSeverity.MEDIUM, AlertSeverity.LOW]:
        if severity in severity_groups:
            severity_emoji = {
                AlertSeverity.CRITICAL: "🔴",
                AlertSeverity.HIGH: "🟠", 
                AlertSeverity.MEDIUM: "🟡",
                AlertSeverity.LOW: "🟢"
            }[severity]
            
            print(f"\n{severity_emoji} {severity.value.upper()} SEVERITY:")
            for alert in severity_groups[severity]:
                print(f"  • {alert.title}")
                print(f"    {alert.message}")
                if alert.recommendation:
                    print(f"    💡 {alert.recommendation}")
    
    # System summary
    summary = alert_system.get_alert_summary()
    print(f"\n📈 ALERT SYSTEM SUMMARY:")
    print(f"  • Total Active Alerts: {summary['total_active_alerts']}")
    print(f"  • Critical/High Priority: {summary['alerts_by_severity'].get('critical', 0) + summary['alerts_by_severity'].get('high', 0)}")
    print(f"  • Alert Rules Configured: {summary['alert_rules_configured']}")
    print(f"  • Auto-Resolution: {'Enabled' if alert_system.enable_auto_resolution else 'Disabled'}")
    
    # Show alert breakdown by type
    print(f"\n🏷️ Alert Categories:")
    for alert_type, count in summary['alerts_by_type'].items():
        type_emoji = {
            'performance_drop': "📉",
            'attribution_shift': "🔄", 
            'budget_overspend': "💰",
            'opportunity': "🚀"
        }.get(alert_type, "📊")
        print(f"  {type_emoji} {alert_type.replace('_', ' ').title()}: {count}")
    
    # Active alerts status
    active_alerts = alert_system.get_active_alerts()
    print(f"\n⚡ Active Alerts Requiring Attention:")
    
    critical_high = [a for a in active_alerts if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]]
    if critical_high:
        for alert in critical_high[:3]:  # Show top 3
            severity_emoji = "🔴" if alert.severity == AlertSeverity.CRITICAL else "🟠"
            print(f"{severity_emoji} {alert.channel} - {alert.title}")
    else:
        print("  ✅ No critical alerts requiring immediate action")
    
    print("\n" + "="*60)
    print("🚀 Intelligent alert system for proactive marketing optimization")
    print("💼 Enterprise-grade monitoring and anomaly detection")
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_alert_system()