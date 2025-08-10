"""
Incrementality Measurement for Marketing Attribution

Measures the true incremental impact of marketing channels using causal inference,
lift testing, and controlled experimentation methodologies.

Author: Sotiris Spyrou
Portfolio: https://verityai.co
LinkedIn: https://www.linkedin.com/in/sspyrou/

DISCLAIMER: This is demonstration code for portfolio purposes only.
Not intended for production use without proper testing and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
import logging

logger = logging.getLogger(__name__)


class IncrementalityAnalyzer:
    """
    Measures true incremental impact of marketing channels.
    
    Uses causal inference methods, controlled experiments, and 
    synthetic control groups to isolate marketing channel effects.
    """
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 minimum_test_duration: int = 14,
                 control_group_ratio: float = 0.3):
        """
        Initialize Incrementality Analyzer.
        
        Args:
            confidence_level: Statistical confidence level for tests
            minimum_test_duration: Minimum days for valid incrementality test
            control_group_ratio: Proportion of traffic for control groups
        """
        self.confidence_level = confidence_level
        self.minimum_test_duration = minimum_test_duration
        self.control_group_ratio = control_group_ratio
        self.alpha = 1 - confidence_level
        
    def measure_incrementality(self, 
                             journey_data: pd.DataFrame,
                             spend_data: pd.DataFrame,
                             holdout_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Comprehensive incrementality analysis across channels.
        
        Args:
            journey_data: Customer journey and conversion data
            spend_data: Marketing spend data by channel and date
            holdout_data: Optional holdout test results
            
        Returns:
            Incrementality measurements and analysis
        """
        logger.info("Starting incrementality measurement analysis")
        
        # Prepare data
        journey_data['date'] = pd.to_datetime(journey_data['timestamp']).dt.date
        spend_data['date'] = pd.to_datetime(spend_data['date']).dt.date
        
        results = {}
        
        # Method 1: Synthetic Control Analysis
        results['synthetic_control'] = self._synthetic_control_analysis(
            journey_data, spend_data
        )
        
        # Method 2: Time-Based Testing (if sufficient data)
        results['time_based_testing'] = self._time_based_incrementality(
            journey_data, spend_data
        )
        
        # Method 3: Geographic Testing Analysis
        results['geo_testing'] = self._geographic_incrementality_analysis(
            journey_data, spend_data
        )
        
        # Method 4: Holdout Analysis (if data available)
        if holdout_data is not None:
            results['holdout_analysis'] = self._holdout_incrementality_analysis(
                journey_data, holdout_data
            )
        
        # Combine results for overall incrementality estimate
        results['combined_incrementality'] = self._combine_incrementality_results(results)
        
        # Generate business recommendations
        results['recommendations'] = self._generate_incrementality_recommendations(results)
        
        return results
    
    def _synthetic_control_analysis(self, 
                                  journey_data: pd.DataFrame,
                                  spend_data: pd.DataFrame) -> Dict[str, Any]:
        """Create synthetic control groups for incrementality measurement."""
        
        results = {}
        
        # Analyze each channel
        for channel in journey_data['touchpoint'].unique():
            try:
                # Get channel performance data
                channel_performance = journey_data[
                    journey_data['touchpoint'] == channel
                ].groupby('date').agg({
                    'converted': ['sum', 'count', 'mean']
                }).round(4)
                
                channel_performance.columns = ['conversions', 'impressions', 'conversion_rate']
                
                # Get channel spend data
                channel_spend = spend_data[
                    spend_data['channel'] == channel
                ].set_index('date')['spend']
                
                # Align data
                aligned_data = channel_performance.join(channel_spend, how='inner')
                
                if len(aligned_data) < self.minimum_test_duration:
                    results[channel] = {'error': 'Insufficient data for analysis'}
                    continue
                
                # Create synthetic control using other channels
                synthetic_control = self._build_synthetic_control(
                    channel, journey_data, aligned_data.index
                )
                
                # Calculate incrementality
                incrementality = self._calculate_synthetic_incrementality(
                    aligned_data, synthetic_control
                )
                
                results[channel] = incrementality
                
            except Exception as e:
                logger.warning(f"Synthetic control analysis failed for {channel}: {e}")
                results[channel] = {'error': str(e)}
        
        return results
    
    def _build_synthetic_control(self, 
                               target_channel: str,
                               journey_data: pd.DataFrame,
                               date_range: pd.Index) -> pd.Series:
        """Build synthetic control group using other channels."""
        
        # Get performance of other channels
        other_channels = journey_data[
            journey_data['touchpoint'] != target_channel
        ]
        
        # Create weighted average of other channels
        control_performance = other_channels.groupby('date')['converted'].mean()
        
        # Align with target date range
        control_aligned = control_performance.reindex(date_range).fillna(0)
        
        return control_aligned
    
    def _calculate_synthetic_incrementality(self,
                                          treatment_data: pd.DataFrame,
                                          control_data: pd.Series) -> Dict[str, Any]:
        """Calculate incrementality using synthetic control method."""
        
        treatment_cr = treatment_data['conversion_rate']
        control_cr = control_data
        
        # Calculate lift
        incremental_effect = treatment_cr - control_cr
        relative_lift = (incremental_effect / control_cr).fillna(0)
        
        # Statistical testing
        t_stat, p_value = stats.ttest_1samp(incremental_effect.dropna(), 0)
        
        # Effect size
        effect_size = incremental_effect.mean()
        confidence_interval = stats.t.interval(
            self.confidence_level,
            len(incremental_effect) - 1,
            loc=effect_size,
            scale=stats.sem(incremental_effect.dropna())
        )
        
        # Incrementality metrics
        total_conversions = treatment_data['conversions'].sum()
        incremental_conversions = (incremental_effect * treatment_data['impressions']).sum()
        incrementality_ratio = incremental_conversions / total_conversions if total_conversions > 0 else 0
        
        return {
            'incremental_effect': effect_size,
            'relative_lift': relative_lift.mean(),
            'p_value': p_value,
            'confidence_interval': confidence_interval,
            'incrementality_ratio': incrementality_ratio,
            'total_conversions': total_conversions,
            'incremental_conversions': incremental_conversions,
            'statistical_significance': p_value < self.alpha
        }
    
    def _time_based_incrementality(self,
                                 journey_data: pd.DataFrame,
                                 spend_data: pd.DataFrame) -> Dict[str, Any]:
        """Measure incrementality using time-based on/off testing."""
        
        results = {}
        
        for channel in journey_data['touchpoint'].unique():
            # Get channel spend data
            channel_spend = spend_data[spend_data['channel'] == channel]
            
            if len(channel_spend) < self.minimum_test_duration * 2:
                results[channel] = {'error': 'Insufficient data for time-based testing'}
                continue
            
            # Identify on/off periods based on spend patterns
            spend_threshold = channel_spend['spend'].quantile(0.1)
            
            # Off periods: low/no spend
            off_periods = channel_spend[channel_spend['spend'] <= spend_threshold]['date']
            # On periods: normal spend
            on_periods = channel_spend[channel_spend['spend'] > spend_threshold]['date']
            
            if len(off_periods) < 7 or len(on_periods) < 7:
                results[channel] = {'error': 'Insufficient on/off periods for testing'}
                continue
            
            # Calculate performance in on vs off periods
            channel_journey_data = journey_data[journey_data['touchpoint'] == channel]
            
            off_performance = channel_journey_data[
                channel_journey_data['date'].isin(off_periods)
            ]['converted'].mean()
            
            on_performance = channel_journey_data[
                channel_journey_data['date'].isin(on_periods)
            ]['converted'].mean()
            
            # Calculate incrementality
            incremental_lift = on_performance - off_performance
            relative_lift = incremental_lift / off_performance if off_performance > 0 else 0
            
            # Statistical test
            off_conversions = channel_journey_data[
                channel_journey_data['date'].isin(off_periods)
            ]['converted']
            on_conversions = channel_journey_data[
                channel_journey_data['date'].isin(on_periods)  
            ]['converted']
            
            t_stat, p_value = stats.ttest_ind(on_conversions, off_conversions)
            
            results[channel] = {
                'off_period_performance': off_performance,
                'on_period_performance': on_performance,
                'incremental_lift': incremental_lift,
                'relative_lift': relative_lift,
                'p_value': p_value,
                'statistical_significance': p_value < self.alpha,
                'on_periods_count': len(on_periods),
                'off_periods_count': len(off_periods)
            }
        
        return results
    
    def _geographic_incrementality_analysis(self,
                                          journey_data: pd.DataFrame,
                                          spend_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze incrementality using geographic variation (simplified simulation)."""
        
        # Note: This is a simplified version. Real geo testing requires actual geographic data
        results = {}
        
        # Simulate geographic regions
        np.random.seed(42)
        journey_data['region'] = np.random.choice(['North', 'South', 'East', 'West'], 
                                                len(journey_data))
        
        for channel in journey_data['touchpoint'].unique():
            channel_data = journey_data[journey_data['touchpoint'] == channel]
            
            # Analyze performance by region
            regional_performance = channel_data.groupby('region')['converted'].agg([
                'mean', 'count', 'std'
            ]).round(4)
            
            regional_performance.columns = ['conversion_rate', 'sample_size', 'std_dev']
            
            if len(regional_performance) < 2:
                results[channel] = {'error': 'Insufficient geographic variation'}
                continue
            
            # Calculate coefficient of variation across regions
            cv = regional_performance['conversion_rate'].std() / regional_performance['conversion_rate'].mean()
            
            # ANOVA test for geographic differences
            region_groups = [
                channel_data[channel_data['region'] == region]['converted'].values
                for region in regional_performance.index
            ]
            
            f_stat, p_value = stats.f_oneway(*region_groups)
            
            results[channel] = {
                'regional_variation_cv': cv,
                'geographic_significance': p_value < self.alpha,
                'anova_p_value': p_value,
                'regional_performance': regional_performance.to_dict(),
                'geographic_incrementality_score': min(cv * 2, 1.0)  # Normalized score
            }
        
        return results
    
    def _holdout_incrementality_analysis(self,
                                       journey_data: pd.DataFrame,
                                       holdout_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze incrementality using holdout test results."""
        
        results = {}
        
        # Merge journey data with holdout test data
        merged_data = journey_data.merge(
            holdout_data, 
            on=['customer_id'], 
            how='inner'
        )
        
        for channel in merged_data['touchpoint'].unique():
            channel_data = merged_data[merged_data['touchpoint'] == channel]
            
            # Compare test vs control groups
            test_group = channel_data[channel_data['test_group'] == 'test']
            control_group = channel_data[channel_data['test_group'] == 'control']
            
            if len(test_group) < 100 or len(control_group) < 100:
                results[channel] = {'error': 'Insufficient holdout test data'}
                continue
            
            # Calculate incrementality metrics
            test_conversion_rate = test_group['converted'].mean()
            control_conversion_rate = control_group['converted'].mean()
            
            incremental_lift = test_conversion_rate - control_conversion_rate
            relative_lift = incremental_lift / control_conversion_rate if control_conversion_rate > 0 else 0
            
            # Statistical significance test
            t_stat, p_value = stats.ttest_ind(
                test_group['converted'], 
                control_group['converted']
            )
            
            # Confidence interval
            pooled_std = np.sqrt(
                ((len(test_group) - 1) * test_group['converted'].std()**2 +
                 (len(control_group) - 1) * control_group['converted'].std()**2) /
                (len(test_group) + len(control_group) - 2)
            )
            
            margin_of_error = stats.t.ppf(1 - self.alpha/2, 
                                        len(test_group) + len(control_group) - 2) * \
                            pooled_std * np.sqrt(1/len(test_group) + 1/len(control_group))
            
            confidence_interval = (incremental_lift - margin_of_error, 
                                 incremental_lift + margin_of_error)
            
            results[channel] = {
                'test_conversion_rate': test_conversion_rate,
                'control_conversion_rate': control_conversion_rate,
                'incremental_lift': incremental_lift,
                'relative_lift': relative_lift,
                'p_value': p_value,
                'confidence_interval': confidence_interval,
                'statistical_significance': p_value < self.alpha,
                'test_group_size': len(test_group),
                'control_group_size': len(control_group)
            }
        
        return results
    
    def _combine_incrementality_results(self, 
                                      all_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine results from different incrementality methods."""
        
        combined_results = {}
        
        # Get all channels analyzed
        all_channels = set()
        for method_results in all_results.values():
            if isinstance(method_results, dict):
                all_channels.update(method_results.keys())
        
        for channel in all_channels:
            channel_results = []
            method_weights = []
            
            # Collect results from each method
            for method_name, method_results in all_results.items():
                if (isinstance(method_results, dict) and 
                    channel in method_results and 
                    'error' not in method_results[channel]):
                    
                    result = method_results[channel]
                    
                    # Extract lift and weight by confidence
                    if 'relative_lift' in result:
                        lift = result['relative_lift']
                        # Weight by statistical significance and sample size
                        weight = 1.0 if result.get('statistical_significance', False) else 0.5
                        
                        channel_results.append(lift)
                        method_weights.append(weight)
            
            if channel_results:
                # Calculate weighted average incrementality
                if sum(method_weights) > 0:
                    combined_lift = np.average(channel_results, weights=method_weights)
                else:
                    combined_lift = np.mean(channel_results)
                
                # Calculate confidence based on method agreement
                lift_std = np.std(channel_results)
                method_agreement = 1 - min(lift_std / abs(combined_lift) if combined_lift != 0 else 1, 1)
                
                combined_results[channel] = {
                    'combined_incrementality': combined_lift,
                    'method_agreement_score': method_agreement,
                    'individual_estimates': channel_results,
                    'methods_used': len(channel_results),
                    'confidence_score': method_agreement * min(len(channel_results) / 3, 1)
                }
            else:
                combined_results[channel] = {
                    'error': 'No valid incrementality measurements available'
                }
        
        return combined_results
    
    def _generate_incrementality_recommendations(self, 
                                               results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on incrementality analysis."""
        
        recommendations = []
        
        combined_results = results.get('combined_incrementality', {})
        
        # High incrementality channels
        high_incrementality = [
            ch for ch, metrics in combined_results.items()
            if (isinstance(metrics, dict) and 
                'combined_incrementality' in metrics and
                metrics['combined_incrementality'] > 0.2 and
                metrics.get('confidence_score', 0) > 0.6)
        ]
        
        # Low incrementality channels  
        low_incrementality = [
            ch for ch, metrics in combined_results.items()
            if (isinstance(metrics, dict) and 
                'combined_incrementality' in metrics and
                metrics['combined_incrementality'] < 0.05 and
                metrics.get('confidence_score', 0) > 0.6)
        ]
        
        if high_incrementality:
            recommendations.append(
                f"🚀 **High Incrementality Channels**: {', '.join(high_incrementality)}. "
                "These channels show strong incremental impact. Consider increasing investment."
            )
        
        if low_incrementality:
            recommendations.append(
                f"⚠️ **Low Incrementality Channels**: {', '.join(low_incrementality)}. "
                "These channels may be overcredited by attribution models. Review efficiency."
            )
        
        # Methodology recommendations
        if 'holdout_analysis' in results:
            recommendations.append(
                "✅ **Holdout Testing Available**: Use holdout results as primary incrementality measure "
                "as they provide the most reliable causal inference."
            )
        else:
            recommendations.append(
                "🔬 **Recommendation**: Implement holdout testing for more accurate incrementality measurement."
            )
        
        return recommendations
    
    def generate_incrementality_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive incrementality measurement report."""
        
        report = "# Marketing Channel Incrementality Analysis\n\n"
        report += "**Advanced Marketing Analytics Portfolio**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        # Combined Results Summary
        combined = results.get('combined_incrementality', {})
        if combined:
            report += "## Incrementality Summary\n\n"
            report += "| Channel | Incrementality | Confidence | Methods Used |\n"
            report += "|---------|---------------|------------|-------------|\n"
            
            for channel, metrics in combined.items():
                if isinstance(metrics, dict) and 'combined_incrementality' in metrics:
                    inc = metrics['combined_incrementality']
                    conf = metrics.get('confidence_score', 0)
                    methods = metrics.get('methods_used', 0)
                    
                    report += f"| {channel} | {inc:.1%} | {conf:.1f} | {methods} |\n"
            report += "\n"
        
        # Method-Specific Results
        if 'holdout_analysis' in results:
            report += "### Holdout Test Results ✅\n"
            holdout = results['holdout_analysis']
            for channel, metrics in holdout.items():
                if isinstance(metrics, dict) and 'relative_lift' in metrics:
                    lift = metrics['relative_lift']
                    significant = "✓" if metrics.get('statistical_significance') else "✗"
                    report += f"- **{channel}**: {lift:.1%} lift ({significant} significant)\n"
            report += "\n"
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            report += "## Strategic Recommendations\n\n"
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n\n"
        
        # Business Impact
        report += "## Key Business Insights\n\n"
        report += "- **True Incrementality**: Traditional attribution may overestimate channel impact\n"
        report += "- **Causal Measurement**: Incrementality testing provides causal evidence of marketing effectiveness\n" 
        report += "- **Budget Optimization**: Focus investment on channels with proven incremental impact\n\n"
        
        report += "---\n*This analysis demonstrates advanced incrementality measurement capabilities. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for enterprise implementations.*"
        
        return report


def demo_incrementality_measurement():
    """Demonstration of Incrementality Measurement for portfolio showcase."""
    
    print("=== Marketing Attribution: Incrementality Measurement Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    np.random.seed(42)
    
    # Create sample data with incremental effects
    dates = pd.date_range('2024-01-01', periods=90, freq='D')
    
    # Journey data with varying incrementality by channel
    journey_data = []
    for date in dates:
        # Search: High incrementality
        search_volume = np.random.poisson(100)
        search_cr = 0.15 + 0.08  # Base + incremental
        for _ in range(search_volume):
            journey_data.append({
                'customer_id': len(journey_data) + 1,
                'touchpoint': 'Search',
                'timestamp': date,
                'converted': np.random.random() < search_cr
            })
        
        # Display: Low incrementality  
        display_volume = np.random.poisson(80)
        display_cr = 0.12 + 0.02  # Base + small incremental
        for _ in range(display_volume):
            journey_data.append({
                'customer_id': len(journey_data) + 1,
                'touchpoint': 'Display',
                'timestamp': date,
                'converted': np.random.random() < display_cr
            })
        
        # Social: Medium incrementality
        social_volume = np.random.poisson(60)
        social_cr = 0.10 + 0.05  # Base + medium incremental
        for _ in range(social_volume):
            journey_data.append({
                'customer_id': len(journey_data) + 1,
                'touchpoint': 'Social',
                'timestamp': date,
                'converted': np.random.random() < social_cr
            })
    
    journey_df = pd.DataFrame(journey_data)
    
    # Spend data
    spend_data = pd.DataFrame({
        'date': dates.tolist() * 3,
        'channel': ['Search'] * 90 + ['Display'] * 90 + ['Social'] * 90,
        'spend': np.concatenate([
            np.random.uniform(1000, 3000, 90),  # Search spend
            np.random.uniform(800, 2500, 90),   # Display spend  
            np.random.uniform(500, 1500, 90)    # Social spend
        ])
    })
    
    # Holdout test data
    holdout_data = pd.DataFrame({
        'customer_id': journey_df['customer_id'].unique(),
        'test_group': np.random.choice(['test', 'control'], 
                                     len(journey_df['customer_id'].unique()), 
                                     p=[0.7, 0.3])
    })
    
    # Initialize analyzer
    analyzer = IncrementalityAnalyzer()
    
    # Run incrementality analysis
    results = analyzer.measure_incrementality(
        journey_df, spend_data, holdout_data
    )
    
    # Display results
    print("📊 INCREMENTALITY MEASUREMENT RESULTS")
    print("=" * 55)
    
    # Combined incrementality results
    if 'combined_incrementality' in results:
        print("\n🎯 Channel Incrementality Summary:")
        for channel, metrics in results['combined_incrementality'].items():
            if isinstance(metrics, dict) and 'combined_incrementality' in metrics:
                inc = metrics['combined_incrementality']
                conf = metrics.get('confidence_score', 0)
                methods = metrics.get('methods_used', 0)
                
                confidence_icon = "🟢" if conf > 0.7 else "🟡" if conf > 0.4 else "🔴"
                print(f"{confidence_icon} {channel:8}: {inc:+.1%} incrementality (confidence: {conf:.1f}, methods: {methods})")
    
    # Holdout test results
    if 'holdout_analysis' in results:
        print(f"\n🧪 HOLDOUT TEST RESULTS:")
        holdout = results['holdout_analysis']
        for channel, metrics in holdout.items():
            if isinstance(metrics, dict) and 'relative_lift' in metrics:
                lift = metrics['relative_lift']
                p_val = metrics.get('p_value', 1.0)
                significant = "✓" if metrics.get('statistical_significance') else "✗"
                print(f"  {channel:8}: {lift:+.1%} lift, p-value: {p_val:.3f} {significant}")
    
    # Key recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        print(f"\n💡 KEY INSIGHTS:")
        for i, rec in enumerate(recommendations[:2], 1):
            # Clean up the recommendation text for display
            clean_rec = rec.replace('**', '').replace('🚀 ', '').replace('⚠️ ', '')
            print(f"  {i}. {clean_rec}")
    
    # Business impact summary
    print(f"\n📈 BUSINESS IMPACT:")
    print(f"  • Incrementality testing reveals true marketing effectiveness")
    print(f"  • Enables data-driven budget reallocation to high-impact channels")
    print(f"  • Prevents over-investment in channels with low incremental value")
    
    print("\n" + "="*60)
    print("🚀 This demonstrates advanced causal inference for marketing")
    print("💼 Ready for enterprise-level incrementality measurement")
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_incrementality_measurement()