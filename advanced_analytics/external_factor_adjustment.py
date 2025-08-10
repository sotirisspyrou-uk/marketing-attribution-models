"""
External Factor Adjustment for Marketing Attribution

Adjusts attribution models for external factors like seasonality, economic indicators,
competitor actions, and market events that influence marketing performance.

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
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import logging

logger = logging.getLogger(__name__)


class ExternalFactorAdjuster:
    """
    Adjusts marketing attribution for external environmental factors.
    
    Accounts for seasonality, economic conditions, competitive actions,
    and market events that influence marketing channel effectiveness.
    """
    
    def __init__(self, 
                 adjustment_strength: float = 0.3,
                 seasonality_method: str = 'fourier',
                 economic_indicators: Optional[List[str]] = None):
        """
        Initialize External Factor Adjuster.
        
        Args:
            adjustment_strength: How much to adjust attribution (0.0-1.0)
            seasonality_method: Method for seasonality adjustment
            economic_indicators: List of economic indicators to consider
        """
        self.adjustment_strength = adjustment_strength
        self.seasonality_method = seasonality_method
        self.economic_indicators = economic_indicators or [
            'consumer_confidence', 'unemployment_rate', 'gdp_growth'
        ]
        
        # Adjustment factors storage
        self.seasonality_factors = {}
        self.economic_factors = {}
        self.competitive_factors = {}
        self.event_factors = {}
        
    def adjust_attribution(self, 
                          attribution_results: Dict[str, float],
                          journey_data: pd.DataFrame,
                          external_data: Optional[pd.DataFrame] = None,
                          events_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Adjust attribution results for external factors.
        
        Args:
            attribution_results: Original attribution weights by channel
            journey_data: Customer journey data with timestamps
            external_data: External factor data (economic, competitive)
            events_data: Market events data
            
        Returns:
            Adjusted attribution results with factor analysis
        """
        logger.info("Adjusting attribution for external factors")
        
        # Calculate adjustment factors
        seasonality_adj = self._calculate_seasonality_adjustment(journey_data)
        
        economic_adj = {}
        if external_data is not None:
            economic_adj = self._calculate_economic_adjustment(
                journey_data, external_data
            )
        
        competitive_adj = self._calculate_competitive_adjustment(
            journey_data, external_data
        )
        
        event_adj = {}
        if events_data is not None:
            event_adj = self._calculate_event_adjustment(
                journey_data, events_data
            )
        
        # Apply adjustments to attribution
        adjusted_attribution = self._apply_adjustments(
            attribution_results,
            seasonality_adj,
            economic_adj,
            competitive_adj, 
            event_adj
        )
        
        return {
            'adjusted_attribution': adjusted_attribution,
            'original_attribution': attribution_results,
            'adjustment_factors': {
                'seasonality': seasonality_adj,
                'economic': economic_adj,
                'competitive': competitive_adj,
                'events': event_adj
            },
            'adjustment_summary': self._generate_adjustment_summary(
                attribution_results, adjusted_attribution
            )
        }
    
    def _calculate_seasonality_adjustment(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate seasonality adjustment factors."""
        
        data['date'] = pd.to_datetime(data['timestamp'])
        data['month'] = data['date'].dt.month
        data['day_of_week'] = data['date'].dt.dayofweek
        data['day_of_year'] = data['date'].dt.dayofyear
        
        seasonality_factors = {}
        
        # Calculate seasonal patterns by channel
        for channel in data['touchpoint'].unique():
            channel_data = data[data['touchpoint'] == channel]
            
            if len(channel_data) < 30:  # Need sufficient data
                seasonality_factors[channel] = 1.0
                continue
            
            if self.seasonality_method == 'fourier':
                # Fourier-based seasonality
                factor = self._fourier_seasonality(channel_data)
            else:
                # Simple monthly averages
                factor = self._simple_seasonality(channel_data)
            
            seasonality_factors[channel] = factor
        
        return seasonality_factors
    
    def _fourier_seasonality(self, data: pd.DataFrame) -> float:
        """Calculate Fourier-based seasonality factor."""
        
        # Create time series of conversion rates
        daily_performance = data.groupby('date')['converted'].mean()
        
        if len(daily_performance) < 14:
            return 1.0
        
        # Fourier analysis for seasonality
        values = daily_performance.values
        n = len(values)
        
        # Remove trend
        x = np.arange(n)
        trend = np.polyfit(x, values, 1)
        detrended = values - np.polyval(trend, x)
        
        # FFT for seasonality detection
        fft_values = np.fft.fft(detrended)
        frequencies = np.fft.fftfreq(n)
        
        # Find dominant frequencies (seasonal patterns)
        power = np.abs(fft_values) ** 2
        dominant_freq_idx = np.argsort(power)[-3:]  # Top 3 frequencies
        
        # Calculate seasonal strength
        seasonal_power = np.sum(power[dominant_freq_idx])
        total_power = np.sum(power)
        
        seasonal_strength = seasonal_power / total_power if total_power > 0 else 0
        
        # Convert to adjustment factor (higher seasonality = higher factor)
        return 1.0 + (seasonal_strength - 0.5) * 0.2
    
    def _simple_seasonality(self, data: pd.DataFrame) -> float:
        """Calculate simple monthly seasonality factor."""
        
        monthly_performance = data.groupby('month')['converted'].mean()
        
        if len(monthly_performance) < 3:
            return 1.0
        
        # Calculate coefficient of variation
        cv = monthly_performance.std() / monthly_performance.mean()
        
        # Higher variation = higher seasonality factor
        return 1.0 + cv * 0.1
    
    def _calculate_economic_adjustment(self, 
                                    journey_data: pd.DataFrame,
                                    external_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate economic environment adjustment factors."""
        
        if external_data is None or external_data.empty:
            return {}
        
        journey_data['date'] = pd.to_datetime(journey_data['timestamp']).dt.date
        external_data['date'] = pd.to_datetime(external_data['date']).dt.date
        
        # Merge journey data with economic data
        merged_data = journey_data.merge(external_data, on='date', how='left')
        
        economic_factors = {}
        
        for channel in journey_data['touchpoint'].unique():
            channel_data = merged_data[merged_data['touchpoint'] == channel]
            
            if len(channel_data) < 50:
                economic_factors[channel] = 1.0
                continue
            
            # Calculate correlation with economic indicators
            correlations = []
            
            for indicator in self.economic_indicators:
                if indicator in channel_data.columns:
                    # Correlation between channel performance and economic indicator
                    channel_performance = channel_data.groupby('date')['converted'].mean()
                    economic_values = channel_data.groupby('date')[indicator].mean()
                    
                    if len(channel_performance) > 10 and len(economic_values) > 10:
                        corr, p_value = stats.pearsonr(
                            channel_performance.values, 
                            economic_values.values
                        )
                        
                        if p_value < 0.05:  # Significant correlation
                            correlations.append(abs(corr))
            
            # Average correlation strength
            avg_correlation = np.mean(correlations) if correlations else 0
            
            # Convert to adjustment factor
            economic_factors[channel] = 1.0 + (avg_correlation - 0.3) * 0.15
        
        return economic_factors
    
    def _calculate_competitive_adjustment(self,
                                        journey_data: pd.DataFrame,
                                        external_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calculate competitive environment adjustment factors."""
        
        competitive_factors = {}
        
        # Simple competitive intensity based on volume patterns
        journey_data['date'] = pd.to_datetime(journey_data['timestamp']).dt.date
        
        for channel in journey_data['touchpoint'].unique():
            channel_data = journey_data[journey_data['touchpoint'] == channel]
            
            # Daily volume analysis
            daily_volumes = channel_data.groupby('date').size()
            
            if len(daily_volumes) < 14:
                competitive_factors[channel] = 1.0
                continue
            
            # Calculate volume volatility (proxy for competitive pressure)
            volume_cv = daily_volumes.std() / daily_volumes.mean()
            
            # Higher volatility suggests competitive market
            # Adjust attribution upward for stable performance in volatile markets
            competitive_factors[channel] = 1.0 + min(volume_cv * 0.1, 0.3)
        
        return competitive_factors
    
    def _calculate_event_adjustment(self,
                                  journey_data: pd.DataFrame,
                                  events_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market events adjustment factors."""
        
        if events_data is None or events_data.empty:
            return {}
        
        journey_data['date'] = pd.to_datetime(journey_data['timestamp']).dt.date
        events_data['date'] = pd.to_datetime(events_data['date']).dt.date
        
        event_factors = {}
        
        # Identify event periods
        event_dates = set(events_data['date'].unique())
        
        for channel in journey_data['touchpoint'].unique():
            channel_data = journey_data[journey_data['touchpoint'] == channel]
            
            # Compare performance during event vs non-event periods
            event_performance = channel_data[
                channel_data['date'].isin(event_dates)
            ]['converted'].mean()
            
            normal_performance = channel_data[
                ~channel_data['date'].isin(event_dates)
            ]['converted'].mean()
            
            if normal_performance > 0:
                event_impact = event_performance / normal_performance
                # Adjust for event impact
                event_factors[channel] = 1.0 + (event_impact - 1.0) * 0.2
            else:
                event_factors[channel] = 1.0
        
        return event_factors
    
    def _apply_adjustments(self,
                          original_attribution: Dict[str, float],
                          seasonality_adj: Dict[str, float],
                          economic_adj: Dict[str, float],
                          competitive_adj: Dict[str, float],
                          event_adj: Dict[str, float]) -> Dict[str, float]:
        """Apply all adjustment factors to attribution weights."""
        
        adjusted_attribution = {}
        
        for channel, original_weight in original_attribution.items():
            # Combine adjustment factors
            total_adjustment = 1.0
            
            if channel in seasonality_adj:
                total_adjustment *= (
                    1 + (seasonality_adj[channel] - 1) * self.adjustment_strength
                )
            
            if channel in economic_adj:
                total_adjustment *= (
                    1 + (economic_adj[channel] - 1) * self.adjustment_strength
                )
            
            if channel in competitive_adj:
                total_adjustment *= (
                    1 + (competitive_adj[channel] - 1) * self.adjustment_strength
                )
            
            if channel in event_adj:
                total_adjustment *= (
                    1 + (event_adj[channel] - 1) * self.adjustment_strength
                )
            
            adjusted_attribution[channel] = original_weight * total_adjustment
        
        # Normalize to sum to 1
        total_adjusted = sum(adjusted_attribution.values())
        if total_adjusted > 0:
            adjusted_attribution = {
                ch: weight / total_adjusted 
                for ch, weight in adjusted_attribution.items()
            }
        
        return adjusted_attribution
    
    def _generate_adjustment_summary(self,
                                   original: Dict[str, float],
                                   adjusted: Dict[str, float]) -> Dict[str, Any]:
        """Generate summary of adjustments made."""
        
        summary = {
            'total_adjustment_magnitude': 0,
            'channels_adjusted_up': [],
            'channels_adjusted_down': [],
            'largest_adjustments': []
        }
        
        adjustments = []
        
        for channel in original.keys():
            original_weight = original[channel]
            adjusted_weight = adjusted.get(channel, original_weight)
            
            adjustment_ratio = adjusted_weight / original_weight if original_weight > 0 else 1.0
            adjustment_magnitude = abs(adjustment_ratio - 1.0)
            
            adjustments.append((channel, adjustment_ratio, adjustment_magnitude))
            
            if adjustment_ratio > 1.05:  # Adjusted up by more than 5%
                summary['channels_adjusted_up'].append(channel)
            elif adjustment_ratio < 0.95:  # Adjusted down by more than 5%
                summary['channels_adjusted_down'].append(channel)
        
        # Calculate total adjustment magnitude
        summary['total_adjustment_magnitude'] = np.mean([adj[2] for adj in adjustments])
        
        # Largest adjustments
        adjustments.sort(key=lambda x: x[2], reverse=True)
        summary['largest_adjustments'] = [
            {'channel': adj[0], 'adjustment_ratio': adj[1], 'magnitude': adj[2]}
            for adj in adjustments[:3]
        ]
        
        return summary
    
    def generate_factor_report(self, adjustment_results: Dict[str, Any]) -> str:
        """Generate comprehensive external factor adjustment report."""
        
        report = "# External Factor Adjustment Report\n\n"
        report += "**Portfolio Demonstration by Sotiris Spyrou**\n"
        report += "- Portfolio: https://verityai.co\n"
        report += "- LinkedIn: https://www.linkedin.com/in/sspyrou/\n\n"
        
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        # Adjustment Summary
        summary = adjustment_results['adjustment_summary']
        report += f"## Adjustment Summary\n\n"
        report += f"- **Total Adjustment Magnitude**: {summary['total_adjustment_magnitude']:.1%}\n"
        report += f"- **Channels Adjusted Up**: {len(summary['channels_adjusted_up'])}\n"
        report += f"- **Channels Adjusted Down**: {len(summary['channels_adjusted_down'])}\n\n"
        
        # Factor Analysis
        factors = adjustment_results['adjustment_factors']
        
        if factors['seasonality']:
            report += f"### Seasonality Factors\n"
            for channel, factor in factors['seasonality'].items():
                impact = "Higher" if factor > 1.1 else "Lower" if factor < 0.9 else "Neutral"
                report += f"- **{channel}**: {factor:.2f} ({impact} seasonal impact)\n"
            report += "\n"
        
        if factors['economic']:
            report += f"### Economic Environment Impact\n"
            for channel, factor in factors['economic'].items():
                sensitivity = "High" if abs(factor - 1.0) > 0.1 else "Medium" if abs(factor - 1.0) > 0.05 else "Low"
                report += f"- **{channel}**: {factor:.2f} (Economic sensitivity: {sensitivity})\n"
            report += "\n"
        
        # Largest Adjustments
        if summary['largest_adjustments']:
            report += f"### Largest Attribution Adjustments\n"
            for adj in summary['largest_adjustments']:
                direction = "increased" if adj['adjustment_ratio'] > 1 else "decreased"
                report += f"- **{adj['channel']}**: Attribution {direction} by {adj['magnitude']:.1%}\n"
        
        # Business Recommendations
        report += f"\n## Strategic Recommendations\n\n"
        
        if summary['channels_adjusted_up']:
            report += f"**🔺 Channels with Enhanced Attribution:**\n"
            for channel in summary['channels_adjusted_up']:
                report += f"- {channel}: Consider increasing investment during favorable external conditions\n"
            report += "\n"
        
        if summary['channels_adjusted_down']:
            report += f"**🔻 Channels with Reduced Attribution:**\n" 
            for channel in summary['channels_adjusted_down']:
                report += f"- {channel}: May be overcredited without external factor adjustment\n"
            report += "\n"
        
        report += "**Key Insight**: External factors can significantly impact attribution accuracy. "
        report += "Regular adjustment ensures marketing investment decisions reflect true channel performance.\n\n"
        
        report += "---\n*This analysis demonstrates advanced marketing attribution capabilities. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for custom implementations.*"
        
        return report


def demo_external_factor_adjustment():
    """Demonstration of External Factor Adjustment for portfolio showcase."""
    
    print("=== Marketing Attribution: External Factor Adjustment Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    np.random.seed(42)
    
    # Sample journey data with seasonal patterns
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    
    # Create realistic seasonal data
    seasonal_multiplier = 1 + 0.3 * np.sin(2 * np.pi * np.arange(365) / 365)  # Annual cycle
    weekly_multiplier = 1 + 0.1 * np.sin(2 * np.pi * np.arange(365) / 7)     # Weekly cycle
    
    journey_data = []
    for i, date in enumerate(dates):
        daily_multiplier = seasonal_multiplier[i] * weekly_multiplier[i]
        daily_volume = int(50 * daily_multiplier)
        
        for _ in range(daily_volume):
            journey_data.append({
                'customer_id': len(journey_data) + 1,
                'touchpoint': np.random.choice(['Search', 'Social', 'Email', 'Display'], 
                                             p=[0.4, 0.3, 0.2, 0.1]),
                'timestamp': date + timedelta(hours=np.random.randint(0, 24)),
                'converted': np.random.random() < (0.15 * daily_multiplier)
            })
    
    journey_df = pd.DataFrame(journey_data)
    
    # Sample external data (economic indicators)
    external_data = pd.DataFrame({
        'date': dates,
        'consumer_confidence': 100 + 10 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 2, 365),
        'unemployment_rate': 5 + 2 * np.cos(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 0.5, 365),
        'gdp_growth': 2.5 + 1.5 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 0.3, 365)
    })
    
    # Sample events data
    events_data = pd.DataFrame({
        'date': ['2024-03-15', '2024-07-04', '2024-11-29', '2024-12-25'],  # Major holidays
        'event': ['Spring Sale', 'Summer Sale', 'Black Friday', 'Christmas'],
        'impact': [0.3, 0.2, 0.8, 0.4]
    })
    events_data['date'] = pd.to_datetime(events_data['date'])
    
    # Original attribution results (simplified)
    original_attribution = {
        'Search': 0.40,
        'Social': 0.30, 
        'Email': 0.20,
        'Display': 0.10
    }
    
    # Initialize adjuster
    adjuster = ExternalFactorAdjuster(adjustment_strength=0.3)
    
    # Apply adjustments
    results = adjuster.adjust_attribution(
        original_attribution,
        journey_df,
        external_data,
        events_data
    )
    
    # Display results
    print("📊 ATTRIBUTION ADJUSTMENT RESULTS")
    print("=" * 50)
    
    print("\n🎯 Original vs Adjusted Attribution:")
    for channel in original_attribution.keys():
        original = results['original_attribution'][channel]
        adjusted = results['adjusted_attribution'][channel]
        change = (adjusted - original) / original
        arrow = "📈" if change > 0.02 else "📉" if change < -0.02 else "➡️"
        print(f"{arrow} {channel:8}: {original:.1%} → {adjusted:.1%} ({change:+.1%})")
    
    # Summary insights
    summary = results['adjustment_summary']
    print(f"\n📋 ADJUSTMENT SUMMARY:")
    print(f"• Total adjustment magnitude: {summary['total_adjustment_magnitude']:.1%}")
    print(f"• Channels adjusted up: {len(summary['channels_adjusted_up'])}")
    print(f"• Channels adjusted down: {len(summary['channels_adjusted_down'])}")
    
    # Generate and display report excerpt
    report = adjuster.generate_factor_report(results)
    print(f"\n📄 EXECUTIVE SUMMARY EXCERPT:")
    print("-" * 50)
    print(report.split("## Strategic Recommendations")[1].split("---")[0])
    
    print("\n" + "="*60)
    print("🚀 This demonstrates advanced attribution modeling capabilities")
    print("💼 Ready for enterprise marketing analytics implementations")
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_external_factor_adjustment()