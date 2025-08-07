"""
Competitive Impact Analysis

Analyzes the impact of competitor marketing activities on attribution models,
market share dynamics, and defensive/offensive marketing strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)


class CompetitiveImpactAnalyzer:
    """
    Analyzes competitive marketing impact on attribution and market share.
    
    Provides insights into competitor influence on customer journeys,
    defensive marketing effectiveness, and market share attribution.
    """
    
    def __init__(self, competitive_window_days: int = 7,
                 market_share_threshold: float = 0.05,
                 significance_level: float = 0.05):
        """
        Initialize Competitive Impact Analyzer.
        
        Args:
            competitive_window_days: Days to look for competitive impact
            market_share_threshold: Minimum market share to consider significant
            significance_level: Statistical significance threshold
        """
        self.competitive_window_days = competitive_window_days
        self.market_share_threshold = market_share_threshold
        self.significance_level = significance_level
        
    def analyze_competitive_impact(self, 
                                 own_journey_data: pd.DataFrame,
                                 competitor_activity_data: pd.DataFrame,
                                 market_data: Optional[pd.DataFrame] = None) -> Dict[str, any]:
        """
        Comprehensive competitive impact analysis.
        
        Args:
            own_journey_data: Company's customer journey data
            competitor_activity_data: Competitor marketing activity data
            market_data: Overall market performance data
            
        Returns:
            Dictionary with competitive analysis results
        """
        logger.info("Starting competitive impact analysis")
        
        # Validate input data
        self._validate_input_data(own_journey_data, competitor_activity_data)
        
        results = {
            'competitive_pressure_analysis': self._analyze_competitive_pressure(
                own_journey_data, competitor_activity_data
            ),
            'market_share_attribution': self._analyze_market_share_attribution(
                own_journey_data, competitor_activity_data, market_data
            ),
            'defensive_effectiveness': self._analyze_defensive_effectiveness(
                own_journey_data, competitor_activity_data
            ),
            'competitive_response_opportunities': self._identify_response_opportunities(
                own_journey_data, competitor_activity_data
            ),
            'share_of_voice_impact': self._analyze_share_of_voice_impact(
                own_journey_data, competitor_activity_data
            )
        }
        
        logger.info("Competitive impact analysis completed")
        return results
    
    def _validate_input_data(self, own_data: pd.DataFrame, 
                           competitor_data: pd.DataFrame) -> None:
        """Validate input data format and completeness."""
        
        # Check own journey data
        required_own_cols = ['customer_id', 'touchpoint', 'timestamp', 'converted']
        missing_own = [col for col in required_own_cols if col not in own_data.columns]
        if missing_own:
            raise ValueError(f"Missing columns in own_journey_data: {missing_own}")
        
        # Check competitor data
        required_comp_cols = ['competitor', 'activity_type', 'timestamp', 'spend', 'reach']
        missing_comp = [col for col in required_comp_cols if col not in competitor_data.columns]
        if missing_comp:
            raise ValueError(f"Missing columns in competitor_activity_data: {missing_comp}")
    
    def _analyze_competitive_pressure(self, own_data: pd.DataFrame,
                                    competitor_data: pd.DataFrame) -> Dict[str, any]:
        """Analyze competitive pressure on customer journeys."""
        
        own_data['date'] = pd.to_datetime(own_data['timestamp']).dt.date
        competitor_data['date'] = pd.to_datetime(competitor_data['timestamp']).dt.date
        
        # Calculate daily competitive intensity
        daily_competitive_intensity = competitor_data.groupby(['date', 'competitor']).agg({
            'spend': 'sum',
            'reach': 'sum'
        }).reset_index()
        
        # Normalize competitive intensity
        scaler = StandardScaler()
        daily_competitive_intensity['normalized_spend'] = scaler.fit_transform(
            daily_competitive_intensity[['spend']]
        )
        daily_competitive_intensity['normalized_reach'] = scaler.fit_transform(
            daily_competitive_intensity[['reach']]
        )
        
        # Calculate competitive pressure score
        daily_competitive_intensity['pressure_score'] = (
            daily_competitive_intensity['normalized_spend'] + 
            daily_competitive_intensity['normalized_reach']
        ) / 2
        
        # Aggregate by date for overall pressure
        daily_pressure = daily_competitive_intensity.groupby('date')['pressure_score'].sum()
        
        # Analyze impact on conversion rates
        own_daily_performance = own_data.groupby('date').agg({
            'customer_id': 'nunique',
            'converted': 'mean'
        }).reset_index()
        
        # Merge with competitive pressure data
        performance_with_pressure = own_daily_performance.merge(
            daily_pressure.reset_index(), on='date', how='inner'
        )
        
        # Calculate correlation between competitive pressure and performance
        if len(performance_with_pressure) > 5:
            pressure_conversion_corr = performance_with_pressure[
                ['pressure_score', 'converted']
            ].corr().iloc[0, 1]
            
            pressure_volume_corr = performance_with_pressure[
                ['pressure_score', 'customer_id']
            ].corr().iloc[0, 1]
        else:
            pressure_conversion_corr = 0.0
            pressure_volume_corr = 0.0
        
        # Identify high-pressure periods
        pressure_threshold = daily_pressure.quantile(0.8)
        high_pressure_dates = daily_pressure[daily_pressure > pressure_threshold].index
        
        return {
            'daily_competitive_pressure': daily_pressure.to_dict(),
            'pressure_conversion_correlation': pressure_conversion_corr,
            'pressure_volume_correlation': pressure_volume_corr,
            'high_pressure_periods': high_pressure_dates.tolist(),
            'average_pressure_score': daily_pressure.mean(),
            'pressure_volatility': daily_pressure.std(),
            'competitor_pressure_breakdown': daily_competitive_intensity.groupby('competitor')[
                'pressure_score'
            ].mean().to_dict()
        }
    
    def _analyze_market_share_attribution(self, own_data: pd.DataFrame,
                                        competitor_data: pd.DataFrame,
                                        market_data: Optional[pd.DataFrame]) -> Dict[str, any]:
        """Analyze market share dynamics and attribution."""
        
        # Calculate own market performance by channel
        own_performance = own_data.groupby('touchpoint').agg({
            'customer_id': 'nunique',
            'converted': ['sum', 'mean']
        }).round(4)
        
        own_performance.columns = ['customers', 'conversions', 'conversion_rate']
        
        # Estimate market share based on competitive spend
        competitor_spend_by_channel = competitor_data.groupby('activity_type')['spend'].sum()
        total_competitor_spend = competitor_spend_by_channel.sum()
        
        if market_data is not None:
            # Use actual market data if available
            market_performance = market_data.groupby('channel').agg({
                'total_customers': 'sum',
                'total_conversions': 'sum'
            })
            
            # Calculate market share
            market_share_analysis = {}
            for channel in own_performance.index:
                if channel in market_performance.index:
                    our_customers = own_performance.loc[channel, 'customers']
                    market_customers = market_performance.loc[channel, 'total_customers']
                    market_share = our_customers / market_customers if market_customers > 0 else 0
                    
                    our_conversions = own_performance.loc[channel, 'conversions']
                    market_conversions = market_performance.loc[channel, 'total_conversions']
                    conversion_share = our_conversions / market_conversions if market_conversions > 0 else 0
                    
                    market_share_analysis[channel] = {
                        'customer_market_share': market_share,
                        'conversion_market_share': conversion_share,
                        'market_share_efficiency': conversion_share / market_share if market_share > 0 else 0
                    }
        else:
            # Estimate market share using competitive spend ratios
            market_share_analysis = {}
            for channel in own_performance.index:
                if channel in competitor_spend_by_channel.index:
                    competitor_channel_spend = competitor_spend_by_channel[channel]
                    # Assume our spend is proportional to our performance
                    estimated_our_spend = own_performance.loc[channel, 'conversions'] * 1000  # Rough estimate
                    
                    estimated_market_share = estimated_our_spend / (
                        estimated_our_spend + competitor_channel_spend
                    )
                    
                    market_share_analysis[channel] = {
                        'estimated_market_share': estimated_market_share,
                        'competitive_spend_ratio': competitor_channel_spend / total_competitor_spend,
                        'our_performance_ratio': own_performance.loc[channel, 'conversions'] / 
                                               own_performance['conversions'].sum()
                    }
        
        # Identify over/under-performing channels relative to market share
        share_performance_gaps = {}
        for channel, metrics in market_share_analysis.items():
            if 'customer_market_share' in metrics and 'conversion_market_share' in metrics:
                gap = metrics['conversion_market_share'] - metrics['customer_market_share']
                share_performance_gaps[channel] = gap
            elif 'estimated_market_share' in metrics and 'our_performance_ratio' in metrics:
                gap = metrics['our_performance_ratio'] - metrics['estimated_market_share']
                share_performance_gaps[channel] = gap
        
        return {
            'market_share_by_channel': market_share_analysis,
            'share_performance_gaps': share_performance_gaps,
            'underperforming_channels': [
                ch for ch, gap in share_performance_gaps.items() if gap < -0.05
            ],
            'outperforming_channels': [
                ch for ch, gap in share_performance_gaps.items() if gap > 0.05
            ],
            'competitive_spend_distribution': competitor_spend_by_channel.to_dict()
        }
    
    def _analyze_defensive_effectiveness(self, own_data: pd.DataFrame,
                                       competitor_data: pd.DataFrame) -> Dict[str, any]:
        """Analyze effectiveness of defensive marketing strategies."""
        
        own_data['date'] = pd.to_datetime(own_data['timestamp']).dt.date
        competitor_data['date'] = pd.to_datetime(competitor_data['timestamp']).dt.date
        
        # Identify periods of high competitive activity
        daily_competitor_spend = competitor_data.groupby('date')['spend'].sum()
        high_competition_threshold = daily_competitor_spend.quantile(0.8)
        high_competition_dates = daily_competitor_spend[
            daily_competitor_spend > high_competition_threshold
        ].index
        
        # Analyze our performance during high competition periods
        high_competition_performance = own_data[
            own_data['date'].isin(high_competition_dates)
        ].groupby('touchpoint').agg({
            'customer_id': 'nunique',
            'converted': ['sum', 'mean']
        })
        high_competition_performance.columns = ['customers_high_comp', 'conversions_high_comp', 'cr_high_comp']
        
        # Analyze performance during normal periods
        normal_performance = own_data[
            ~own_data['date'].isin(high_competition_dates)
        ].groupby('touchpoint').agg({
            'customer_id': 'nunique', 
            'converted': ['sum', 'mean']
        })
        normal_performance.columns = ['customers_normal', 'conversions_normal', 'cr_normal']
        
        # Calculate defensive effectiveness metrics
        defensive_metrics = high_competition_performance.join(
            normal_performance, how='outer'
        ).fillna(0)
        
        defensive_metrics['conversion_rate_ratio'] = (
            defensive_metrics['cr_high_comp'] / 
            defensive_metrics['cr_normal'].replace(0, np.nan)
        ).fillna(0)
        
        defensive_metrics['volume_protection_ratio'] = (
            defensive_metrics['customers_high_comp'] / 
            defensive_metrics['customers_normal'].replace(0, np.nan)
        ).fillna(0)
        
        # Calculate overall defensive effectiveness score
        defensive_scores = {}
        for channel in defensive_metrics.index:
            cr_ratio = defensive_metrics.loc[channel, 'conversion_rate_ratio']
            volume_ratio = defensive_metrics.loc[channel, 'volume_protection_ratio']
            
            # Defensive score (1.0 = perfect defense, >1.0 = gained during competition)
            defensive_score = (cr_ratio + volume_ratio) / 2
            defensive_scores[channel] = defensive_score
        
        return {
            'high_competition_periods': high_competition_dates.tolist(),
            'defensive_effectiveness_by_channel': defensive_scores,
            'performance_during_competition': defensive_metrics.to_dict(),
            'vulnerable_channels': [
                ch for ch, score in defensive_scores.items() if score < 0.8
            ],
            'resilient_channels': [
                ch for ch, score in defensive_scores.items() if score > 1.2
            ]
        }
    
    def _identify_response_opportunities(self, own_data: pd.DataFrame,
                                       competitor_data: pd.DataFrame) -> Dict[str, any]:
        """Identify opportunities for competitive response."""
        
        # Analyze competitor activity patterns
        competitor_patterns = competitor_data.groupby(['competitor', 'activity_type']).agg({
            'spend': ['mean', 'std', 'count'],
            'reach': ['mean', 'std']
        }).round(2)
        
        # Identify gaps in competitive coverage
        our_channels = set(own_data['touchpoint'].unique())
        competitor_channels = set(competitor_data['activity_type'].unique())
        
        untapped_channels = our_channels - competitor_channels
        competitive_channels = our_channels & competitor_channels
        missed_opportunities = competitor_channels - our_channels
        
        # Analyze timing opportunities
        competitor_data['hour'] = pd.to_datetime(competitor_data['timestamp']).dt.hour
        competitor_data['day_of_week'] = pd.to_datetime(competitor_data['timestamp']).dt.dayofweek
        
        competitor_timing = competitor_data.groupby(['hour', 'day_of_week'])['spend'].sum()
        low_competition_times = competitor_timing[competitor_timing < competitor_timing.quantile(0.3)]
        
        # Identify competitor spending inefficiencies
        competitor_efficiency = competitor_data.groupby(['competitor', 'activity_type']).apply(
            lambda x: x['reach'].sum() / x['spend'].sum() if x['spend'].sum() > 0 else 0
        ).reset_index()
        competitor_efficiency.columns = ['competitor', 'activity_type', 'efficiency']
        
        inefficient_competitors = competitor_efficiency[
            competitor_efficiency['efficiency'] < competitor_efficiency['efficiency'].quantile(0.3)
        ]
        
        return {
            'channel_opportunities': {
                'untapped_by_competitors': list(untapped_channels),
                'highly_competitive': list(competitive_channels),
                'missed_opportunities': list(missed_opportunities)
            },
            'timing_opportunities': {
                'low_competition_times': low_competition_times.to_dict(),
                'optimal_response_windows': self._find_optimal_response_windows(competitor_data)
            },
            'competitive_inefficiencies': {
                'inefficient_competitors': inefficient_competitors.to_dict('records'),
                'opportunity_channels': inefficient_competitors['activity_type'].unique().tolist()
            },
            'response_recommendations': self._generate_response_recommendations(
                untapped_channels, missed_opportunities, low_competition_times
            )
        }
    
    def _analyze_share_of_voice_impact(self, own_data: pd.DataFrame,
                                     competitor_data: pd.DataFrame) -> Dict[str, any]:
        """Analyze share of voice vs share of conversions."""
        
        # Calculate our estimated share of voice (assuming spend proportional to performance)
        our_performance = own_data.groupby('touchpoint')['converted'].sum()
        our_estimated_spend = our_performance * 1000  # Rough spend estimate
        
        # Calculate competitor share of voice
        competitor_spend = competitor_data.groupby('activity_type')['spend'].sum()
        
        # Calculate combined share of voice
        sov_analysis = {}
        for channel in our_performance.index:
            our_spend = our_estimated_spend.get(channel, 0)
            comp_spend = competitor_spend.get(channel, 0)
            total_spend = our_spend + comp_spend
            
            if total_spend > 0:
                our_sov = our_spend / total_spend
                comp_sov = comp_spend / total_spend
                
                # Calculate our share of conversions (assuming market data)
                our_conversions = our_performance[channel]
                total_conversions = our_conversions / our_sov if our_sov > 0 else our_conversions
                our_soc = our_conversions / total_conversions if total_conversions > 0 else 0
                
                # SOV vs SOC efficiency
                sov_efficiency = our_soc / our_sov if our_sov > 0 else 0
                
                sov_analysis[channel] = {
                    'our_share_of_voice': our_sov,
                    'competitor_share_of_voice': comp_sov,
                    'our_share_of_conversions': our_soc,
                    'sov_efficiency': sov_efficiency,
                    'sov_gap': our_soc - our_sov
                }
        
        return {
            'share_of_voice_analysis': sov_analysis,
            'underweight_channels': [
                ch for ch, metrics in sov_analysis.items() 
                if metrics['sov_gap'] < -0.05
            ],
            'overperforming_channels': [
                ch for ch, metrics in sov_analysis.items() 
                if metrics['sov_efficiency'] > 1.2
            ],
            'investment_opportunities': [
                ch for ch, metrics in sov_analysis.items() 
                if metrics['sov_efficiency'] > 1.1 and metrics['our_share_of_voice'] < 0.3
            ]
        }
    
    def _find_optimal_response_windows(self, competitor_data: pd.DataFrame) -> List[Dict]:
        """Find optimal time windows for competitive response."""
        
        competitor_data['datetime'] = pd.to_datetime(competitor_data['timestamp'])
        competitor_data['hour'] = competitor_data['datetime'].dt.hour
        competitor_data['day'] = competitor_data['datetime'].dt.dayofweek
        
        # Find low-activity periods
        hourly_activity = competitor_data.groupby('hour')['spend'].sum()
        daily_activity = competitor_data.groupby('day')['spend'].sum()
        
        low_activity_hours = hourly_activity[hourly_activity < hourly_activity.quantile(0.3)].index
        low_activity_days = daily_activity[daily_activity < daily_activity.quantile(0.3)].index
        
        optimal_windows = []
        for hour in low_activity_hours:
            for day in low_activity_days:
                optimal_windows.append({
                    'day_of_week': day,
                    'hour': hour,
                    'competitive_intensity': 'low',
                    'opportunity_score': 1.0 - (hourly_activity[hour] + daily_activity[day]) / 
                                       (hourly_activity.max() + daily_activity.max())
                })
        
        return sorted(optimal_windows, key=lambda x: x['opportunity_score'], reverse=True)[:10]
    
    def _generate_response_recommendations(self, untapped_channels: Set[str],
                                         missed_opportunities: Set[str],
                                         low_competition_times: pd.Series) -> List[str]:
        """Generate actionable competitive response recommendations."""
        
        recommendations = []
        
        if untapped_channels:
            recommendations.append(
                f"🎯 Opportunity: Channels with low competitive pressure: {', '.join(untapped_channels)}. "
                "Consider increasing investment in these channels."
            )
        
        if missed_opportunities:
            recommendations.append(
                f"⚠️ Gap: Missing presence in competitive channels: {', '.join(missed_opportunities)}. "
                "Evaluate entry barriers and potential ROI."
            )
        
        if not low_competition_times.empty:
            best_times = low_competition_times.nsmallest(3)
            recommendations.append(
                f"⏰ Timing: Low competition periods identified. "
                f"Consider scheduling campaigns during these windows."
            )
        
        recommendations.append(
            "📊 Monitor competitive spend efficiency and identify opportunities "
            "to capture market share from inefficient competitors."
        )
        
        return recommendations
    
    def generate_competitive_report(self, analysis_results: Dict[str, any]) -> str:
        """Generate a comprehensive competitive analysis report."""
        
        report = "# Competitive Impact Analysis Report\n\n"
        
        # Competitive Pressure Summary
        pressure = analysis_results['competitive_pressure_analysis']
        report += f"## Competitive Pressure Summary\n"
        report += f"- Average daily competitive pressure: {pressure['average_pressure_score']:.2f}\n"
        report += f"- Pressure volatility: {pressure['pressure_volatility']:.2f}\n"
        report += f"- Correlation with conversions: {pressure['pressure_conversion_correlation']:.3f}\n\n"
        
        # Market Share Analysis
        market_share = analysis_results['market_share_attribution']
        report += f"## Market Share Performance\n"
        if market_share['outperforming_channels']:
            report += f"- Outperforming channels: {', '.join(market_share['outperforming_channels'])}\n"
        if market_share['underperforming_channels']:
            report += f"- Underperforming channels: {', '.join(market_share['underperforming_channels'])}\n"
        report += "\n"
        
        # Defensive Effectiveness
        defensive = analysis_results['defensive_effectiveness']
        report += f"## Defensive Marketing Effectiveness\n"
        report += f"- High competition periods: {len(defensive['high_competition_periods'])} days\n"
        if defensive['vulnerable_channels']:
            report += f"- Vulnerable channels: {', '.join(defensive['vulnerable_channels'])}\n"
        if defensive['resilient_channels']:
            report += f"- Resilient channels: {', '.join(defensive['resilient_channels'])}\n"
        report += "\n"
        
        # Response Opportunities
        opportunities = analysis_results['competitive_response_opportunities']
        report += f"## Response Opportunities\n"
        for rec in opportunities['response_recommendations']:
            report += f"- {rec}\n"
        
        return report


def demo_competitive_analysis():
    """Demonstration of Competitive Impact Analysis."""
    
    np.random.seed(42)
    
    # Sample own journey data
    own_data = pd.DataFrame({
        'customer_id': range(1, 1001),
        'touchpoint': np.random.choice(['Search', 'Display', 'Email', 'Social'], 1000),
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'converted': np.random.choice([True, False], 1000, p=[0.15, 0.85])
    })
    
    # Sample competitor data
    competitor_data = pd.DataFrame({
        'competitor': np.random.choice(['Competitor_A', 'Competitor_B', 'Competitor_C'], 500),
        'activity_type': np.random.choice(['Search', 'Display', 'Social', 'Video'], 500),
        'timestamp': pd.date_range('2024-01-01', periods=500, freq='2H'),
        'spend': np.random.uniform(1000, 50000, 500),
        'reach': np.random.uniform(10000, 500000, 500)
    })
    
    # Initialize analyzer
    analyzer = CompetitiveImpactAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_competitive_impact(own_data, competitor_data)
    
    # Generate report
    report = analyzer.generate_competitive_report(results)
    print(report)


if __name__ == "__main__":
    demo_competitive_analysis()