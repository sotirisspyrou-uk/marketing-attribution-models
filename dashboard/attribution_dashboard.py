"""
Real-Time Attribution Dashboard

Interactive dashboard for marketing attribution analysis and visualization.
Provides real-time insights into channel performance, customer journeys,
and attribution model results.

Author: Sotiris Spyrou
Portfolio: https://verityai.co
LinkedIn: https://www.linkedin.com/in/sspyrou/

DISCLAIMER: This is demonstration code for portfolio purposes only.
Not intended for production use without proper testing and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class AttributionDashboard:
    """
    Real-time attribution dashboard for marketing analytics.
    
    Provides interactive visualization and analysis of attribution results,
    channel performance, and customer journey insights.
    """
    
    def __init__(self,
                 refresh_interval: int = 300,  # 5 minutes
                 data_retention_days: int = 90,
                 enable_real_time: bool = True):
        """
        Initialize Attribution Dashboard.
        
        Args:
            refresh_interval: Data refresh interval in seconds
            data_retention_days: How many days of data to retain
            enable_real_time: Enable real-time data updates
        """
        self.refresh_interval = refresh_interval
        self.data_retention_days = data_retention_days
        self.enable_real_time = enable_real_time
        
        # Dashboard data storage
        self.attribution_data = {}
        self.performance_metrics = {}
        self.journey_analytics = {}
        self.alerts = []
        self.dashboard_config = {}
        
        # Real-time tracking
        self.last_update = None
        self.update_history = []
        
        # Initialize dashboard
        self._initialize_dashboard()
        
    def _initialize_dashboard(self):
        """Initialize dashboard configuration and default settings."""
        
        self.dashboard_config = {
            'theme': 'professional',
            'layout': 'executive',
            'auto_refresh': self.enable_real_time,
            'default_time_range': '30d',
            'currency': 'USD',
            'timezone': 'UTC',
            'widgets': {
                'attribution_overview': {'enabled': True, 'position': 1},
                'channel_performance': {'enabled': True, 'position': 2},
                'journey_flow': {'enabled': True, 'position': 3},
                'conversion_funnel': {'enabled': True, 'position': 4},
                'trend_analysis': {'enabled': True, 'position': 5},
                'alerts_panel': {'enabled': True, 'position': 6}
            }
        }
        
        logger.info("Attribution dashboard initialized")
    
    def update_data(self, 
                   attribution_results: Dict[str, Any],
                   journey_data: pd.DataFrame,
                   performance_data: Optional[pd.DataFrame] = None) -> bool:
        """
        Update dashboard with latest attribution and performance data.
        
        Args:
            attribution_results: Latest attribution model results
            journey_data: Customer journey data
            performance_data: Channel performance metrics
            
        Returns:
            Success status of data update
        """
        try:
            logger.info("Updating dashboard data")
            
            # Store attribution results
            self.attribution_data = {
                'timestamp': datetime.now(),
                'model_results': attribution_results,
                'channel_attribution': self._extract_channel_attribution(attribution_results),
                'model_performance': self._extract_model_performance(attribution_results)
            }
            
            # Process journey analytics
            self.journey_analytics = self._process_journey_data(journey_data)
            
            # Process performance metrics
            if performance_data is not None:
                self.performance_metrics = self._process_performance_data(performance_data)
            
            # Update timestamp and history
            self.last_update = datetime.now()
            self.update_history.append({
                'timestamp': self.last_update,
                'records_processed': len(journey_data),
                'channels_analyzed': len(self.attribution_data['channel_attribution'])
            })
            
            # Cleanup old history
            cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
            self.update_history = [
                h for h in self.update_history 
                if h['timestamp'] > cutoff_date
            ]
            
            # Generate alerts
            self._generate_performance_alerts()
            
            logger.info(f"Dashboard updated successfully at {self.last_update}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update dashboard data: {e}")
            return False
    
    def _extract_channel_attribution(self, attribution_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract channel attribution weights from model results."""
        
        if 'channel_attribution' in attribution_results:
            return attribution_results['channel_attribution']
        elif 'attribution_weights' in attribution_results:
            return attribution_results['attribution_weights']
        elif hasattr(attribution_results, 'get_attribution_results'):
            # Handle attribution model objects
            results_df = attribution_results.get_attribution_results()
            return dict(zip(results_df['channel'], results_df['attribution_weight']))
        else:
            return {}
    
    def _extract_model_performance(self, attribution_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract model performance metrics."""
        
        performance = {}
        
        if 'model_statistics' in attribution_results:
            stats = attribution_results['model_statistics']
            performance.update({
                'accuracy': stats.get('accuracy', 0),
                'r_squared': stats.get('r_squared', 0),
                'concentration': stats.get('attribution_concentration', 0),
                'diversity': stats.get('channel_diversity_score', 0)
            })
        
        if hasattr(attribution_results, 'get_model_statistics'):
            stats = attribution_results.get_model_statistics()
            performance.update({
                'total_customers': stats.get('total_customers', 0),
                'conversion_rate': stats.get('overall_conversion_rate', 0),
                'num_channels': stats.get('num_channels', 0)
            })
        
        return performance
    
    def _process_journey_data(self, journey_data: pd.DataFrame) -> Dict[str, Any]:
        """Process customer journey data for dashboard visualization."""
        
        analytics = {}
        
        # Basic journey metrics
        analytics['total_journeys'] = journey_data['customer_id'].nunique()
        analytics['total_touchpoints'] = len(journey_data)
        analytics['avg_journey_length'] = len(journey_data) / analytics['total_journeys'] if analytics['total_journeys'] > 0 else 0
        
        # Conversion metrics
        converting_customers = journey_data[journey_data['converted']]['customer_id'].nunique()
        analytics['conversions'] = converting_customers
        analytics['conversion_rate'] = converting_customers / analytics['total_journeys'] if analytics['total_journeys'] > 0 else 0
        
        # Channel frequency
        channel_counts = journey_data['touchpoint'].value_counts()
        analytics['channel_frequency'] = channel_counts.to_dict()
        
        # Time-based patterns
        journey_data['hour'] = pd.to_datetime(journey_data['timestamp']).dt.hour
        journey_data['day_of_week'] = pd.to_datetime(journey_data['timestamp']).dt.day_name()
        
        analytics['hourly_patterns'] = journey_data.groupby('hour').size().to_dict()
        analytics['daily_patterns'] = journey_data.groupby('day_of_week').size().to_dict()
        
        # Top journey paths
        journey_paths = journey_data.groupby('customer_id')['touchpoint'].apply(lambda x: ' → '.join(x.tolist()))
        analytics['top_journey_paths'] = journey_paths.value_counts().head(10).to_dict()
        
        return analytics
    
    def _process_performance_data(self, performance_data: pd.DataFrame) -> Dict[str, Any]:
        """Process channel performance data."""
        
        metrics = {}
        
        if 'channel' in performance_data.columns:
            # Channel-level metrics
            for _, row in performance_data.iterrows():
                channel = row['channel']
                metrics[channel] = {
                    'spend': row.get('spend', 0),
                    'impressions': row.get('impressions', 0),
                    'clicks': row.get('clicks', 0),
                    'conversions': row.get('conversions', 0),
                    'revenue': row.get('revenue', 0),
                    'ctr': row.get('clicks', 0) / row.get('impressions', 1),
                    'cvr': row.get('conversions', 0) / row.get('clicks', 1),
                    'roas': row.get('revenue', 0) / row.get('spend', 1) if row.get('spend', 0) > 0 else 0,
                    'cpa': row.get('spend', 0) / row.get('conversions', 1) if row.get('conversions', 0) > 0 else 0
                }
        
        # Overall metrics
        total_spend = sum(m.get('spend', 0) for m in metrics.values())
        total_conversions = sum(m.get('conversions', 0) for m in metrics.values())
        total_revenue = sum(m.get('revenue', 0) for m in metrics.values())
        
        metrics['_overall'] = {
            'total_spend': total_spend,
            'total_conversions': total_conversions,
            'total_revenue': total_revenue,
            'overall_roas': total_revenue / total_spend if total_spend > 0 else 0,
            'overall_cpa': total_spend / total_conversions if total_conversions > 0 else 0
        }
        
        return metrics
    
    def _generate_performance_alerts(self):
        """Generate performance alerts based on current data."""
        
        current_alerts = []
        
        # Attribution concentration alert
        if self.attribution_data.get('model_performance', {}).get('concentration', 0) > 0.7:
            current_alerts.append({
                'type': 'warning',
                'category': 'attribution',
                'message': 'High attribution concentration detected - consider diversifying channels',
                'severity': 'medium',
                'timestamp': datetime.now()
            })
        
        # Channel performance alerts
        for channel, metrics in self.performance_metrics.items():
            if channel == '_overall':
                continue
                
            # Low ROAS alert
            if metrics.get('roas', 0) < 1.0 and metrics.get('spend', 0) > 1000:
                current_alerts.append({
                    'type': 'alert',
                    'category': 'performance',
                    'message': f'{channel} showing negative ROAS: {metrics["roas"]:.2f}',
                    'severity': 'high',
                    'timestamp': datetime.now(),
                    'channel': channel
                })
            
            # High CPA alert
            if metrics.get('cpa', 0) > 100:  # Configurable threshold
                current_alerts.append({
                    'type': 'warning',
                    'category': 'efficiency',
                    'message': f'{channel} CPA above threshold: ${metrics["cpa"]:.2f}',
                    'severity': 'medium',
                    'timestamp': datetime.now(),
                    'channel': channel
                })
        
        # Conversion rate alert
        if self.journey_analytics.get('conversion_rate', 0) < 0.02:  # Below 2%
            current_alerts.append({
                'type': 'alert',
                'category': 'conversion',
                'message': f'Overall conversion rate low: {self.journey_analytics["conversion_rate"]:.1%}',
                'severity': 'high',
                'timestamp': datetime.now()
            })
        
        # Update alerts list
        self.alerts = current_alerts
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data for visualization."""
        
        return {
            'timestamp': self.last_update,
            'attribution': self.attribution_data,
            'performance': self.performance_metrics,
            'journey_analytics': self.journey_analytics,
            'alerts': self.alerts,
            'config': self.dashboard_config,
            'update_history': self.update_history[-24:]  # Last 24 updates
        }
    
    def get_attribution_widget_data(self) -> Dict[str, Any]:
        """Get data for attribution overview widget."""
        
        if not self.attribution_data:
            return {'error': 'No attribution data available'}
        
        channel_attribution = self.attribution_data.get('channel_attribution', {})
        
        # Sort channels by attribution weight
        sorted_channels = sorted(
            channel_attribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'channel_attribution': sorted_channels,
            'top_channel': sorted_channels[0][0] if sorted_channels else 'N/A',
            'attribution_balance': 1 - self.attribution_data.get('model_performance', {}).get('concentration', 0),
            'total_channels': len(channel_attribution),
            'last_updated': self.attribution_data.get('timestamp')
        }
    
    def get_performance_widget_data(self) -> Dict[str, Any]:
        """Get data for channel performance widget."""
        
        if not self.performance_metrics:
            return {'error': 'No performance data available'}
        
        # Exclude overall metrics from channel-specific analysis
        channel_metrics = {k: v for k, v in self.performance_metrics.items() if k != '_overall'}
        overall_metrics = self.performance_metrics.get('_overall', {})
        
        # Calculate performance rankings
        roas_ranking = sorted(
            [(ch, metrics.get('roas', 0)) for ch, metrics in channel_metrics.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        efficiency_ranking = sorted(
            [(ch, metrics.get('cpa', float('inf'))) for ch, metrics in channel_metrics.items()],
            key=lambda x: x[1]
        )
        
        return {
            'channel_performance': channel_metrics,
            'overall_metrics': overall_metrics,
            'roas_ranking': roas_ranking,
            'efficiency_ranking': efficiency_ranking,
            'total_spend': overall_metrics.get('total_spend', 0),
            'total_revenue': overall_metrics.get('total_revenue', 0)
        }
    
    def get_journey_widget_data(self) -> Dict[str, Any]:
        """Get data for customer journey widget."""
        
        if not self.journey_analytics:
            return {'error': 'No journey data available'}
        
        return {
            'total_journeys': self.journey_analytics.get('total_journeys', 0),
            'avg_journey_length': self.journey_analytics.get('avg_journey_length', 0),
            'conversion_rate': self.journey_analytics.get('conversion_rate', 0),
            'top_journey_paths': self.journey_analytics.get('top_journey_paths', {}),
            'channel_frequency': self.journey_analytics.get('channel_frequency', {}),
            'temporal_patterns': {
                'hourly': self.journey_analytics.get('hourly_patterns', {}),
                'daily': self.journey_analytics.get('daily_patterns', {})
            }
        }
    
    def get_alerts_widget_data(self) -> Dict[str, Any]:
        """Get data for alerts widget."""
        
        # Categorize alerts by severity
        alerts_by_severity = {'high': [], 'medium': [], 'low': []}
        
        for alert in self.alerts:
            severity = alert.get('severity', 'low')
            alerts_by_severity[severity].append(alert)
        
        return {
            'total_alerts': len(self.alerts),
            'alerts_by_severity': alerts_by_severity,
            'recent_alerts': sorted(self.alerts, key=lambda x: x['timestamp'], reverse=True)[:10],
            'alert_categories': list(set(alert.get('category', 'general') for alert in self.alerts))
        }
    
    def export_dashboard_report(self, format: str = 'json') -> str:
        """Export complete dashboard data as report."""
        
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'attribution_dashboard_export',
                'data_period': f"Last {self.data_retention_days} days",
                'author': 'Sotiris Spyrou - https://verityai.co'
            },
            'dashboard_summary': {
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'total_updates': len(self.update_history),
                'active_alerts': len(self.alerts),
                'channels_monitored': len(self.attribution_data.get('channel_attribution', {}))
            },
            'attribution_overview': self.get_attribution_widget_data(),
            'performance_summary': self.get_performance_widget_data(),
            'journey_insights': self.get_journey_widget_data(),
            'alerts_summary': self.get_alerts_widget_data()
        }
        
        if format.lower() == 'json':
            return json.dumps(report_data, indent=2, default=str)
        elif format.lower() == 'markdown':
            return self._generate_markdown_report(report_data)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate markdown format dashboard report."""
        
        report = "# Marketing Attribution Dashboard Report\n\n"
        report += "**Executive Marketing Analytics Dashboard**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        # Dashboard Summary
        summary = report_data['dashboard_summary']
        report += f"## Dashboard Summary\n\n"
        report += f"- **Last Updated**: {summary.get('last_update', 'N/A')}\n"
        report += f"- **Total Data Updates**: {summary.get('total_updates', 0)}\n"
        report += f"- **Active Alerts**: {summary.get('active_alerts', 0)}\n"
        report += f"- **Channels Monitored**: {summary.get('channels_monitored', 0)}\n\n"
        
        # Attribution Overview
        attribution = report_data.get('attribution_overview', {})
        if 'channel_attribution' in attribution:
            report += f"## Channel Attribution\n\n"
            report += "| Rank | Channel | Attribution Weight |\n"
            report += "|------|---------|--------------------|\n"
            
            for i, (channel, weight) in enumerate(attribution['channel_attribution'][:10], 1):
                report += f"| {i} | {channel} | {weight:.1%} |\n"
        
        # Performance Summary
        performance = report_data.get('performance_summary', {})
        if 'overall_metrics' in performance:
            overall = performance['overall_metrics']
            report += f"\n## Performance Overview\n\n"
            report += f"- **Total Spend**: ${overall.get('total_spend', 0):,.2f}\n"
            report += f"- **Total Revenue**: ${overall.get('total_revenue', 0):,.2f}\n"
            report += f"- **Overall ROAS**: {overall.get('overall_roas', 0):.2f}x\n"
            report += f"- **Overall CPA**: ${overall.get('overall_cpa', 0):.2f}\n\n"
        
        # Alerts Summary
        alerts = report_data.get('alerts_summary', {})
        if alerts.get('total_alerts', 0) > 0:
            report += f"## Active Alerts ({alerts['total_alerts']})\n\n"
            
            for severity in ['high', 'medium', 'low']:
                severity_alerts = alerts.get('alerts_by_severity', {}).get(severity, [])
                if severity_alerts:
                    report += f"### {severity.title()} Priority\n\n"
                    for alert in severity_alerts:
                        report += f"- **{alert.get('category', 'General').title()}**: {alert.get('message', 'N/A')}\n"
                    report += "\n"
        
        report += "---\n*This dashboard demonstrates real-time attribution analytics capabilities. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for custom implementations.*"
        
        return report


def demo_attribution_dashboard():
    """Executive demonstration of Attribution Dashboard."""
    
    print("=== Attribution Dashboard: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    # Initialize dashboard
    dashboard = AttributionDashboard(
        refresh_interval=300,
        enable_real_time=True
    )
    
    # Generate sample data
    np.random.seed(42)
    
    # Sample attribution results
    attribution_results = {
        'channel_attribution': {
            'Search': 0.35,
            'Display': 0.25,
            'Social': 0.20,
            'Email': 0.15,
            'Direct': 0.05
        },
        'model_statistics': {
            'accuracy': 0.82,
            'attribution_concentration': 0.35,
            'channel_diversity_score': 0.65,
            'total_customers': 5000,
            'overall_conversion_rate': 0.18
        }
    }
    
    # Sample journey data
    customers = []
    channels = ['Search', 'Display', 'Social', 'Email', 'Direct']
    
    for i in range(1000):
        for j in range(np.random.randint(1, 5)):
            customers.append({
                'customer_id': i,
                'touchpoint': np.random.choice(channels),
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 30)) + 
                           timedelta(hours=np.random.randint(0, 24)),
                'converted': np.random.choice([True, False], p=[0.18, 0.82])
            })
    
    journey_data = pd.DataFrame(customers)
    
    # Sample performance data
    performance_data = pd.DataFrame([
        {'channel': 'Search', 'spend': 50000, 'impressions': 1000000, 'clicks': 50000, 
         'conversions': 900, 'revenue': 180000},
        {'channel': 'Display', 'spend': 30000, 'impressions': 2000000, 'clicks': 20000, 
         'conversions': 500, 'revenue': 100000},
        {'channel': 'Social', 'spend': 25000, 'impressions': 800000, 'clicks': 16000, 
         'conversions': 320, 'revenue': 64000},
        {'channel': 'Email', 'spend': 5000, 'impressions': 100000, 'clicks': 8000, 
         'conversions': 240, 'revenue': 48000},
        {'channel': 'Direct', 'spend': 0, 'impressions': 50000, 'clicks': 5000, 
         'conversions': 150, 'revenue': 45000}
    ])
    
    # Update dashboard
    print("📊 Updating dashboard with sample data...")
    dashboard.update_data(attribution_results, journey_data, performance_data)
    
    # Display dashboard results
    print("\n🎯 ATTRIBUTION DASHBOARD OVERVIEW")
    print("=" * 50)
    
    # Attribution widget
    attribution_widget = dashboard.get_attribution_widget_data()
    print(f"\n🏆 Channel Attribution:")
    for i, (channel, weight) in enumerate(attribution_widget['channel_attribution'][:5], 1):
        rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
        print(f"{rank_emoji} {channel:8}: {weight:.1%}")
    
    print(f"  • Top Channel: {attribution_widget['top_channel']}")
    print(f"  • Attribution Balance: {attribution_widget['attribution_balance']:.1%}")
    
    # Performance widget
    performance_widget = dashboard.get_performance_widget_data()
    overall = performance_widget['overall_metrics']
    print(f"\n💰 Performance Overview:")
    print(f"  • Total Spend: ${overall['total_spend']:,.0f}")
    print(f"  • Total Revenue: ${overall['total_revenue']:,.0f}")
    print(f"  • Overall ROAS: {overall['overall_roas']:.2f}x")
    print(f"  • Overall CPA: ${overall['overall_cpa']:.2f}")
    
    # Top performing channels
    print(f"\n📈 Top ROAS Channels:")
    for i, (channel, roas) in enumerate(performance_widget['roas_ranking'][:3], 1):
        print(f"  {i}. {channel}: {roas:.2f}x ROAS")
    
    # Journey widget
    journey_widget = dashboard.get_journey_widget_data()
    print(f"\n🧭 Journey Analytics:")
    print(f"  • Total Journeys: {journey_widget['total_journeys']:,}")
    print(f"  • Avg Journey Length: {journey_widget['avg_journey_length']:.1f}")
    print(f"  • Conversion Rate: {journey_widget['conversion_rate']:.1%}")
    
    # Top journey paths
    print(f"\n🛤️ Top Journey Patterns:")
    for i, (path, count) in enumerate(list(journey_widget['top_journey_paths'].items())[:3], 1):
        print(f"  {i}. {path[:40]}{'...' if len(path) > 40 else ''} ({count} journeys)")
    
    # Alerts widget
    alerts_widget = dashboard.get_alerts_widget_data()
    print(f"\n🚨 Active Alerts ({alerts_widget['total_alerts']}):")
    
    if alerts_widget['total_alerts'] > 0:
        for alert in alerts_widget['recent_alerts'][:3]:
            severity_emoji = "🔴" if alert['severity'] == 'high' else "🟡" if alert['severity'] == 'medium' else "🟢"
            print(f"{severity_emoji} {alert['message']}")
    else:
        print("  ✅ No active alerts - all systems performing within normal parameters")
    
    # Export report
    print(f"\n📋 Generating Executive Report...")
    markdown_report = dashboard.export_dashboard_report('markdown')
    
    print(f"\n📄 DASHBOARD REPORT EXCERPT:")
    print("-" * 40)
    
    # Show first few lines of the report
    report_lines = markdown_report.split('\n')
    excerpt_lines = report_lines[6:16]  # Skip header, show summary
    for line in excerpt_lines:
        print(line)
    
    print(f"\n📊 Dashboard Configuration:")
    config = dashboard.dashboard_config
    print(f"  • Theme: {config['theme']}")
    print(f"  • Auto-refresh: {config['auto_refresh']}")
    print(f"  • Active widgets: {len([w for w in config['widgets'].values() if w['enabled']])}")
    print(f"  • Last update: {dashboard.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*60)
    print("🚀 Real-time attribution dashboard for executive insights")
    print("💼 Enterprise-grade marketing analytics visualization")
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_attribution_dashboard()