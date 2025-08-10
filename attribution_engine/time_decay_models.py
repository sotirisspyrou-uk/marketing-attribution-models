"""
Time Decay Attribution Models

Implements various time-decay attribution models that weight touchpoints based
on their temporal proximity to conversion, with customizable decay functions
and time window analysis.

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
import warnings
import logging
from scipy import stats
import math

logger = logging.getLogger(__name__)


class TimeDecayAttribution:
    """
    Advanced time decay attribution models with multiple decay functions.
    
    Implements exponential, linear, and custom decay functions to weight
    touchpoints based on their proximity to conversion events.
    """
    
    def __init__(self,
                 decay_function: str = 'exponential',
                 half_life_days: float = 7.0,
                 decay_rate: float = 0.7,
                 min_weight: float = 0.01,
                 lookback_window: int = 90):
        """
        Initialize Time Decay Attribution model.
        
        Args:
            decay_function: Type of decay ('exponential', 'linear', 'power', 'step')
            half_life_days: Half-life for exponential decay in days
            decay_rate: Decay rate parameter (0-1)
            min_weight: Minimum weight for distant touchpoints
            lookback_window: Maximum days to look back for attributing touchpoints
        """
        self.decay_function = decay_function
        self.half_life_days = half_life_days
        self.decay_rate = decay_rate
        self.min_weight = min_weight
        self.lookback_window = lookback_window
        
        # Attribution results
        self.channel_attribution = {}
        self.time_analysis = {}
        self.decay_analysis = {}
        self.model_statistics = {}
        
        # Time patterns
        self.hourly_patterns = {}
        self.daily_patterns = {}
        self.conversion_windows = {}
        
    def fit(self, journey_data: pd.DataFrame) -> 'TimeDecayAttribution':
        """
        Fit the time decay attribution model.
        
        Args:
            journey_data: Customer journey data with timestamps and conversions
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Time Decay Attribution model")
        
        # Validate input data
        required_columns = ['customer_id', 'touchpoint', 'timestamp', 'converted']
        if not all(col in journey_data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        # Prepare data
        journey_data = journey_data.copy()
        journey_data['timestamp'] = pd.to_datetime(journey_data['timestamp'])
        journey_data = journey_data.sort_values(['customer_id', 'timestamp'])
        
        # Calculate time decay attribution
        self._calculate_time_decay_attribution(journey_data)
        
        # Analyze temporal patterns
        self._analyze_temporal_patterns(journey_data)
        
        # Analyze decay characteristics
        self._analyze_decay_performance(journey_data)
        
        # Calculate model statistics
        self._calculate_model_statistics(journey_data)
        
        logger.info(f"Attribution model fitted for {len(self.channel_attribution)} channels")
        return self
    
    def _calculate_time_decay_attribution(self, data: pd.DataFrame):
        """Calculate time decay attribution for each channel."""
        
        channel_credits = {channel: 0.0 for channel in data['touchpoint'].unique()}
        total_conversions = 0
        
        # Group by customer to analyze individual journeys
        for customer_id, customer_data in data.groupby('customer_id'):
            customer_data = customer_data.sort_values('timestamp')
            converted = customer_data['converted'].any()
            
            if not converted:
                continue  # Only attribute credit for converted customers
            
            total_conversions += 1
            
            # Find conversion timestamp (use last touchpoint timestamp as proxy)
            conversion_timestamp = customer_data['timestamp'].max()
            
            # Calculate time decay weights for this journey
            journey_credits = self._calculate_journey_time_decay(
                customer_data, conversion_timestamp
            )
            
            # Add to overall channel credits
            for channel, credit in journey_credits.items():
                channel_credits[channel] += credit
        
        # Normalize attribution weights
        if total_conversions > 0:
            self.channel_attribution = {
                channel: credit / total_conversions 
                for channel, credit in channel_credits.items()
            }
        else:
            self.channel_attribution = {}
        
        logger.info(f"Calculated attribution for {total_conversions} conversions")
    
    def _calculate_journey_time_decay(self, 
                                    customer_data: pd.DataFrame,
                                    conversion_timestamp: pd.Timestamp) -> Dict[str, float]:
        """Calculate time decay attribution for a single customer journey."""
        
        journey_credits = {}
        total_weight = 0
        touchpoint_weights = []
        
        # Calculate decay weight for each touchpoint
        for _, touchpoint in customer_data.iterrows():
            touchpoint_time = touchpoint['timestamp']
            channel = touchpoint['touchpoint']
            
            # Calculate time difference in days
            time_diff_days = (conversion_timestamp - touchpoint_time).total_seconds() / (24 * 3600)
            
            # Skip touchpoints outside lookback window
            if time_diff_days > self.lookback_window:
                continue
            
            # Calculate decay weight
            decay_weight = self._calculate_decay_weight(time_diff_days)
            
            touchpoint_weights.append({
                'channel': channel,
                'weight': decay_weight,
                'time_diff': time_diff_days
            })
            
            total_weight += decay_weight
        
        # Normalize weights to sum to 1
        if total_weight > 0:
            for tp in touchpoint_weights:
                channel = tp['channel']
                normalized_weight = tp['weight'] / total_weight
                journey_credits[channel] = journey_credits.get(channel, 0) + normalized_weight
        
        return journey_credits
    
    def _calculate_decay_weight(self, time_diff_days: float) -> float:
        """Calculate decay weight based on time difference."""
        
        if time_diff_days < 0:
            time_diff_days = 0  # Handle future timestamps
        
        if self.decay_function == 'exponential':
            # Exponential decay: w = e^(-λt) where λ = ln(2)/half_life
            decay_constant = math.log(2) / self.half_life_days
            weight = math.exp(-decay_constant * time_diff_days)
            
        elif self.decay_function == 'linear':
            # Linear decay: w = 1 - (t / max_time)
            max_time = self.lookback_window
            weight = max(0, 1 - (time_diff_days / max_time))
            
        elif self.decay_function == 'power':
            # Power decay: w = (1 + t)^(-α)
            alpha = -math.log(self.decay_rate)  # Convert decay_rate to power
            weight = math.pow(1 + time_diff_days, -alpha)
            
        elif self.decay_function == 'step':
            # Step decay: discrete time buckets
            if time_diff_days <= 1:
                weight = 1.0
            elif time_diff_days <= 7:
                weight = 0.7
            elif time_diff_days <= 30:
                weight = 0.4
            else:
                weight = 0.1
                
        else:
            raise ValueError(f"Unknown decay function: {self.decay_function}")
        
        # Apply minimum weight threshold
        return max(weight, self.min_weight)
    
    def _analyze_temporal_patterns(self, data: pd.DataFrame):
        """Analyze temporal patterns in conversions and touchpoints."""
        
        # Convert timestamps
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.day_name()
        data['day_of_month'] = data['timestamp'].dt.day
        
        # Hourly patterns
        hourly_touchpoints = data.groupby('hour').size()
        hourly_conversions = data[data['converted']].groupby('hour').size()
        
        self.hourly_patterns = {
            'touchpoint_distribution': hourly_touchpoints.to_dict(),
            'conversion_distribution': hourly_conversions.to_dict(),
            'peak_touchpoint_hour': hourly_touchpoints.idxmax(),
            'peak_conversion_hour': hourly_conversions.idxmax() if not hourly_conversions.empty else 0
        }
        
        # Daily patterns
        daily_touchpoints = data.groupby('day_of_week').size()
        daily_conversions = data[data['converted']].groupby('day_of_week').size()
        
        self.daily_patterns = {
            'touchpoint_distribution': daily_touchpoints.to_dict(),
            'conversion_distribution': daily_conversions.to_dict(),
            'peak_touchpoint_day': daily_touchpoints.idxmax(),
            'peak_conversion_day': daily_conversions.idxmax() if not daily_conversions.empty else 'Monday'
        }
        
        # Conversion window analysis
        self._analyze_conversion_windows(data)
    
    def _analyze_conversion_windows(self, data: pd.DataFrame):
        """Analyze time windows from first touch to conversion."""
        
        conversion_windows = []
        
        for customer_id, customer_data in data.groupby('customer_id'):
            customer_data = customer_data.sort_values('timestamp')
            converted = customer_data['converted'].any()
            
            if converted and len(customer_data) > 1:
                first_touch = customer_data['timestamp'].min()
                last_touch = customer_data['timestamp'].max()
                
                window_days = (last_touch - first_touch).total_seconds() / (24 * 3600)
                conversion_windows.append({
                    'customer_id': customer_id,
                    'window_days': window_days,
                    'journey_length': len(customer_data),
                    'first_channel': customer_data.iloc[0]['touchpoint'],
                    'last_channel': customer_data.iloc[-1]['touchpoint']
                })
        
        if conversion_windows:
            windows_df = pd.DataFrame(conversion_windows)
            
            self.conversion_windows = {
                'mean_window_days': windows_df['window_days'].mean(),
                'median_window_days': windows_df['window_days'].median(),
                'std_window_days': windows_df['window_days'].std(),
                'percentiles': {
                    '25th': windows_df['window_days'].quantile(0.25),
                    '75th': windows_df['window_days'].quantile(0.75),
                    '90th': windows_df['window_days'].quantile(0.90)
                },
                'window_distribution': self._calculate_window_distribution(windows_df)
            }
    
    def _calculate_window_distribution(self, windows_df: pd.DataFrame) -> Dict[str, int]:
        """Calculate distribution of conversion windows."""
        
        bins = [0, 1, 7, 30, 90, float('inf')]
        labels = ['Same Day', '1-7 Days', '1-4 Weeks', '1-3 Months', '3+ Months']
        
        windows_df['window_category'] = pd.cut(
            windows_df['window_days'], 
            bins=bins, 
            labels=labels, 
            right=False
        )
        
        return windows_df['window_category'].value_counts().to_dict()
    
    def _analyze_decay_performance(self, data: pd.DataFrame):
        """Analyze how different decay parameters affect attribution."""
        
        # Test different decay parameters
        test_parameters = [
            {'half_life_days': 1.0, 'decay_rate': 0.9},
            {'half_life_days': 3.0, 'decay_rate': 0.8},
            {'half_life_days': 7.0, 'decay_rate': 0.7},
            {'half_life_days': 14.0, 'decay_rate': 0.6},
            {'half_life_days': 30.0, 'decay_rate': 0.5}
        ]
        
        parameter_performance = {}
        
        for params in test_parameters:
            # Temporarily change parameters
            original_half_life = self.half_life_days
            original_decay_rate = self.decay_rate
            
            self.half_life_days = params['half_life_days']
            self.decay_rate = params['decay_rate']
            
            # Calculate attribution with these parameters
            test_attribution = {}
            total_conversions = 0
            
            for customer_id, customer_data in data.groupby('customer_id'):
                customer_data = customer_data.sort_values('timestamp')
                converted = customer_data['converted'].any()
                
                if not converted:
                    continue
                
                total_conversions += 1
                conversion_timestamp = customer_data['timestamp'].max()
                
                journey_credits = self._calculate_journey_time_decay(
                    customer_data, conversion_timestamp
                )
                
                for channel, credit in journey_credits.items():
                    test_attribution[channel] = test_attribution.get(channel, 0) + credit
            
            # Normalize
            if total_conversions > 0:
                test_attribution = {
                    ch: credit / total_conversions 
                    for ch, credit in test_attribution.items()
                }
            
            # Calculate attribution concentration (Herfindahl index)
            concentration = sum(w**2 for w in test_attribution.values()) if test_attribution else 0
            
            parameter_performance[f"half_life_{params['half_life_days']}d"] = {
                'attribution': test_attribution,
                'concentration': concentration,
                'diversity_score': 1 - concentration,
                'parameters': params
            }
            
            # Restore original parameters
            self.half_life_days = original_half_life
            self.decay_rate = original_decay_rate
        
        self.decay_analysis = {
            'parameter_sensitivity': parameter_performance,
            'optimal_parameters': self._find_optimal_parameters(parameter_performance)
        }
    
    def _find_optimal_parameters(self, performance_results: Dict) -> Dict[str, Any]:
        """Find optimal decay parameters based on diversity and business logic."""
        
        # Score each parameter set
        scores = {}
        
        for param_name, results in performance_results.items():
            diversity = results['diversity_score']
            concentration = results['concentration']
            
            # Prefer moderate diversity (not too concentrated, not too dispersed)
            diversity_score = 1 - abs(0.7 - diversity)  # Target 70% diversity
            
            # Penalize extreme concentration
            concentration_penalty = min(concentration, 0.3)  # Penalize if > 30% concentration
            
            final_score = diversity_score - concentration_penalty
            scores[param_name] = final_score
        
        # Find best parameter set
        best_params = max(scores, key=scores.get)
        
        return {
            'recommended_parameters': performance_results[best_params]['parameters'],
            'performance_scores': scores,
            'best_configuration': best_params
        }
    
    def _calculate_model_statistics(self, data: pd.DataFrame):
        """Calculate comprehensive model statistics."""
        
        total_customers = data['customer_id'].nunique()
        converting_customers = data[data['converted']]['customer_id'].nunique()
        total_touchpoints = len(data)
        
        # Attribution concentration
        attribution_values = list(self.channel_attribution.values())
        concentration = sum(w**2 for w in attribution_values) if attribution_values else 0
        
        # Time-based metrics
        time_span_days = (data['timestamp'].max() - data['timestamp'].min()).days
        
        self.model_statistics = {
            'total_customers': total_customers,
            'converting_customers': converting_customers,
            'overall_conversion_rate': converting_customers / total_customers if total_customers > 0 else 0,
            'total_touchpoints': total_touchpoints,
            'avg_touchpoints_per_customer': total_touchpoints / total_customers if total_customers > 0 else 0,
            'time_span_days': time_span_days,
            'num_channels': len(self.channel_attribution),
            'attribution_concentration': concentration,
            'channel_diversity_score': 1 - concentration,
            'model_parameters': {
                'decay_function': self.decay_function,
                'half_life_days': self.half_life_days,
                'decay_rate': self.decay_rate,
                'min_weight': self.min_weight,
                'lookback_window': self.lookback_window
            }
        }
    
    def predict(self, journey: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Predict attribution for a single journey with timestamps.
        
        Args:
            journey: List of {'channel': str, 'timestamp': datetime} dicts
            
        Returns:
            Attribution weights for each channel in the journey
        """
        if not self.channel_attribution:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if not journey:
            return {}
        
        # Find conversion timestamp (use last touchpoint as conversion time)
        conversion_timestamp = max(tp['timestamp'] for tp in journey)
        
        # Calculate time decay for each touchpoint
        total_weight = 0
        touchpoint_weights = []
        
        for touchpoint in journey:
            channel = touchpoint['channel']
            touchpoint_time = touchpoint['timestamp']
            
            # Calculate time difference
            if isinstance(touchpoint_time, str):
                touchpoint_time = pd.to_datetime(touchpoint_time)
            
            time_diff_days = (conversion_timestamp - touchpoint_time).total_seconds() / (24 * 3600)
            
            # Calculate decay weight
            decay_weight = self._calculate_decay_weight(time_diff_days)
            
            touchpoint_weights.append({
                'channel': channel,
                'weight': decay_weight
            })
            
            total_weight += decay_weight
        
        # Normalize weights
        journey_attribution = {}
        
        if total_weight > 0:
            for tp in touchpoint_weights:
                channel = tp['channel']
                normalized_weight = tp['weight'] / total_weight
                journey_attribution[channel] = journey_attribution.get(channel, 0) + normalized_weight
        else:
            # Equal attribution fallback
            unique_channels = set(tp['channel'] for tp in journey)
            equal_weight = 1 / len(unique_channels) if unique_channels else 0
            journey_attribution = {ch: equal_weight for ch in unique_channels}
        
        return journey_attribution
    
    def get_attribution_results(self) -> pd.DataFrame:
        """Get attribution results as DataFrame."""
        
        if not self.channel_attribution:
            raise ValueError("Model not fitted. Call fit() first.")
        
        results = []
        
        for channel, attribution_weight in self.channel_attribution.items():
            results.append({
                'channel': channel,
                'attribution_weight': attribution_weight,
                'decay_influence': self._calculate_channel_decay_influence(channel)
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('attribution_weight', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def _calculate_channel_decay_influence(self, channel: str) -> float:
        """Calculate how much decay affects this channel's attribution."""
        
        # Compare with linear attribution (no decay preference)
        linear_weight = 1 / len(self.channel_attribution) if self.channel_attribution else 0
        actual_weight = self.channel_attribution.get(channel, 0)
        
        # Decay influence: positive means channel benefits from recency weighting
        return (actual_weight - linear_weight) / linear_weight if linear_weight > 0 else 0
    
    def get_temporal_analysis(self) -> Dict[str, pd.DataFrame]:
        """Get temporal pattern analysis."""
        
        analysis = {}
        
        # Hourly patterns
        if self.hourly_patterns:
            hourly_data = []
            for hour in range(24):
                touchpoints = self.hourly_patterns['touchpoint_distribution'].get(hour, 0)
                conversions = self.hourly_patterns['conversion_distribution'].get(hour, 0)
                
                hourly_data.append({
                    'hour': hour,
                    'touchpoints': touchpoints,
                    'conversions': conversions,
                    'conversion_rate': conversions / touchpoints if touchpoints > 0 else 0
                })
            
            analysis['hourly_patterns'] = pd.DataFrame(hourly_data)
        
        # Daily patterns
        if self.daily_patterns:
            daily_data = []
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            for day in days:
                touchpoints = self.daily_patterns['touchpoint_distribution'].get(day, 0)
                conversions = self.daily_patterns['conversion_distribution'].get(day, 0)
                
                daily_data.append({
                    'day_of_week': day,
                    'touchpoints': touchpoints,
                    'conversions': conversions,
                    'conversion_rate': conversions / touchpoints if touchpoints > 0 else 0
                })
            
            analysis['daily_patterns'] = pd.DataFrame(daily_data)
        
        return analysis
    
    def get_decay_analysis(self) -> pd.DataFrame:
        """Get decay parameter sensitivity analysis."""
        
        if not self.decay_analysis:
            return pd.DataFrame()
        
        analysis_data = []
        
        for param_name, results in self.decay_analysis['parameter_sensitivity'].items():
            params = results['parameters']
            
            analysis_data.append({
                'parameter_set': param_name,
                'half_life_days': params['half_life_days'],
                'decay_rate': params['decay_rate'],
                'concentration': results['concentration'],
                'diversity_score': results['diversity_score'],
                'recommended': param_name == self.decay_analysis['optimal_parameters']['best_configuration']
            })
        
        return pd.DataFrame(analysis_data)
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        return self.model_statistics.copy()
    
    def generate_executive_report(self) -> str:
        """Generate executive-level time decay attribution report."""
        
        report = "# Time Decay Attribution Analysis\n\n"
        report += "**Temporal Marketing Attribution by Sotiris Spyrou**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        # Model Configuration
        report += f"## Model Configuration\n"
        report += f"- **Decay Function**: {self.decay_function.title()}\n"
        report += f"- **Half-Life**: {self.half_life_days} days\n"
        report += f"- **Decay Rate**: {self.decay_rate}\n"
        report += f"- **Lookback Window**: {self.lookback_window} days\n"
        report += f"- **Minimum Weight**: {self.min_weight}\n\n"
        
        # Attribution Results
        attribution_df = self.get_attribution_results()
        report += f"## Channel Attribution Results\n\n"
        report += "| Rank | Channel | Attribution | Decay Influence |\n"
        report += "|------|---------|-------------|------------------|\n"
        
        for _, row in attribution_df.head(10).iterrows():
            influence_icon = "🔥" if row['decay_influence'] > 0.2 else "📈" if row['decay_influence'] > 0 else "📉"
            report += f"| {row['rank']} | {row['channel']} | {row['attribution_weight']:.1%} | {influence_icon} {row['decay_influence']:+.1%} |\n"
        
        # Temporal Insights
        if self.conversion_windows:
            windows = self.conversion_windows
            report += f"\n## Conversion Window Analysis\n\n"
            report += f"- **Average Conversion Window**: {windows['mean_window_days']:.1f} days\n"
            report += f"- **Median Conversion Window**: {windows['median_window_days']:.1f} days\n"
            report += f"- **90th Percentile**: {windows['percentiles']['90th']:.1f} days\n\n"
            
            # Window distribution
            if 'window_distribution' in windows:
                report += f"### Conversion Window Distribution\n\n"
                for category, count in windows['window_distribution'].items():
                    report += f"- **{category}**: {count} customers\n"
        
        # Optimal Parameters
        if self.decay_analysis:
            optimal = self.decay_analysis['optimal_parameters']
            recommended = optimal['recommended_parameters']
            
            report += f"\n## Parameter Optimization\n\n"
            report += f"- **Recommended Half-Life**: {recommended['half_life_days']} days\n"
            report += f"- **Recommended Decay Rate**: {recommended['decay_rate']}\n"
            report += f"- **Current Configuration**: {'✅ Optimal' if recommended['half_life_days'] == self.half_life_days else '⚠️ Sub-optimal'}\n\n"
        
        # Key Statistics
        stats = self.model_statistics
        report += f"## Key Performance Metrics\n\n"
        report += f"- **Data Time Span**: {stats['time_span_days']} days\n"
        report += f"- **Overall Conversion Rate**: {stats['overall_conversion_rate']:.1%}\n"
        report += f"- **Channel Diversity Score**: {stats['channel_diversity_score']:.1%}\n"
        report += f"- **Attribution Balance**: {1-stats['attribution_concentration']:.1%}\n\n"
        
        # Strategic Recommendations
        top_channel = attribution_df.iloc[0]['channel'] if not attribution_df.empty else 'N/A'
        report += f"## Strategic Recommendations\n\n"
        report += f"1. **Recency-Focused Strategy**: Recent touchpoints carry more weight in {self.decay_function} model\n"
        report += f"2. **Optimize {top_channel}**: Highest time-weighted attribution suggests strong late-stage influence\n"
        report += f"3. **Timing Optimization**: Focus campaigns within {self.half_life_days:.0f}-day conversion windows\n"
        report += f"4. **Channel Sequencing**: Leverage temporal patterns for journey orchestration\n"
        report += f"5. **Budget Allocation**: Weight investment toward channels effective in final {self.half_life_days:.0f} days\n\n"
        
        report += "---\n*This analysis demonstrates advanced time-based attribution modeling. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for custom implementations.*"
        
        return report


def demo_time_decay_attribution():
    """Executive demonstration of Time Decay Attribution."""
    
    print("=== Time Decay Attribution: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    np.random.seed(42)
    
    # Generate realistic customer journey data with time patterns
    customers = []
    channels = ['Search', 'Display', 'Social', 'Email', 'Direct']
    
    # Different channels have different temporal behaviors
    early_stage_channels = ['Display', 'Social']     # Awareness, early in journey
    late_stage_channels = ['Search', 'Direct']       # Conversion, close to purchase
    nurturing_channels = ['Email']                   # Throughout journey
    
    # Generate 1000 customer journeys
    for customer_id in range(1, 1001):
        # Random journey length (1-8 touchpoints)
        journey_length = np.random.choice(range(1, 9), p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.05, 0.03, 0.02])
        
        # Journey span (1-60 days)
        journey_span_days = np.random.exponential(14)  # Average 14 days, some much longer
        journey_span_days = min(journey_span_days, 90)  # Cap at 90 days
        
        # Build journey with temporal preferences
        journey_channels = []
        timestamps = []
        start_date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 30))
        
        for i in range(journey_length):
            # Time progression through journey
            time_progress = i / max(journey_length - 1, 1)
            
            # Channel selection based on journey stage
            if time_progress < 0.3:  # Early stage
                channel = np.random.choice(early_stage_channels + ['Search'], 
                                         p=[0.4, 0.3, 0.3])
            elif time_progress > 0.7:  # Late stage
                channel = np.random.choice(late_stage_channels + ['Email'], 
                                         p=[0.5, 0.3, 0.2])
            else:  # Middle stage
                channel = np.random.choice(nurturing_channels + channels, 
                                         p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
            
            journey_channels.append(channel)
            
            # Timestamp within journey span
            days_offset = time_progress * journey_span_days
            timestamp = start_date + pd.Timedelta(days=days_offset) + pd.Timedelta(hours=np.random.randint(0, 24))
            timestamps.append(timestamp)
        
        # Determine conversion based on recency and channel quality
        conversion_prob = 0.12  # Base rate
        
        if journey_length > 1:
            # Recent touchpoints boost conversion probability
            recent_channels = [ch for ch, ts in zip(journey_channels, timestamps) 
                             if (timestamps[-1] - ts).days <= 7]
            
            # Boost for recent high-intent channels
            recent_high_intent = [ch for ch in recent_channels if ch in late_stage_channels]
            conversion_prob += len(recent_high_intent) * 0.08
            
            # Boost for diverse journey with good sequencing
            if len(set(journey_channels)) >= 3:
                conversion_prob += 0.05
            
            # Boost if journey has awareness → conversion pattern
            if (any(ch in early_stage_channels for ch in journey_channels[:2]) and
                any(ch in late_stage_channels for ch in journey_channels[-2:])):
                conversion_prob += 0.1
        
        converted = np.random.random() < min(conversion_prob, 0.6)  # Cap at 60%
        
        # Add to dataset
        for channel, timestamp in zip(journey_channels, timestamps):
            customers.append({
                'customer_id': customer_id,
                'touchpoint': channel,
                'timestamp': timestamp,
                'converted': converted
            })
    
    journey_data = pd.DataFrame(customers)
    
    print(f"📊 Generated {len(journey_data)} touchpoints across {journey_data['customer_id'].nunique()} customers")
    print(f"📈 Overall conversion rate: {journey_data.groupby('customer_id')['converted'].first().mean():.1%}")
    
    # Initialize and fit model
    model = TimeDecayAttribution(
        decay_function='exponential',
        half_life_days=7.0,
        decay_rate=0.7,
        lookback_window=90
    )
    
    print(f"\n⏰ Fitting time decay attribution model...")
    model.fit(journey_data)
    
    # Display results
    print("\n📊 TIME DECAY ATTRIBUTION RESULTS")
    print("=" * 50)
    
    attribution_results = model.get_attribution_results()
    print(f"\n🏆 Channel Attribution (Recency-Weighted):")
    for _, row in attribution_results.iterrows():
        rank_emoji = "🥇" if row['rank'] == 1 else "🥈" if row['rank'] == 2 else "🥉" if row['rank'] == 3 else "📊"
        influence_emoji = "🔥" if row['decay_influence'] > 0.2 else "📈" if row['decay_influence'] > 0 else "📉"
        print(f"{rank_emoji} {row['channel']:8}: {row['attribution_weight']:.1%} attribution {influence_emoji} {row['decay_influence']:+.1%} decay effect")
    
    # Conversion window analysis
    if model.conversion_windows:
        windows = model.conversion_windows
        print(f"\n⏱️ CONVERSION WINDOW INSIGHTS:")
        print(f"  • Average journey time: {windows['mean_window_days']:.1f} days")
        print(f"  • Median journey time: {windows['median_window_days']:.1f} days")
        print(f"  • 90th percentile: {windows['percentiles']['90th']:.1f} days")
        
        # Show distribution
        print(f"\n📊 Journey Duration Distribution:")
        for category, count in windows['window_distribution'].items():
            percentage = (count / sum(windows['window_distribution'].values())) * 100
            print(f"  • {category}: {count:3} customers ({percentage:.0f}%)")
    
    # Temporal patterns
    temporal_analysis = model.get_temporal_analysis()
    if 'hourly_patterns' in temporal_analysis:
        hourly_df = temporal_analysis['hourly_patterns']
        peak_hour = hourly_df.loc[hourly_df['conversions'].idxmax(), 'hour'] if not hourly_df.empty else 0
        
        print(f"\n🕐 TEMPORAL PATTERNS:")
        print(f"  • Peak conversion hour: {int(peak_hour)}:00")
        print(f"  • Peak touchpoint hour: {model.hourly_patterns['peak_touchpoint_hour']}:00")
    
    # Parameter sensitivity
    decay_analysis = model.get_decay_analysis()
    if not decay_analysis.empty:
        print(f"\n🔬 PARAMETER SENSITIVITY:")
        recommended = decay_analysis[decay_analysis['recommended']].iloc[0] if any(decay_analysis['recommended']) else None
        
        if recommended is not None:
            print(f"  • Optimal half-life: {recommended['half_life_days']:.1f} days")
            print(f"  • Current half-life: {model.half_life_days:.1f} days")
            print(f"  • Configuration: {'✅ Optimal' if abs(recommended['half_life_days'] - model.half_life_days) < 0.1 else '⚠️ Could be improved'}")
    
    # Example prediction
    print(f"\n🔮 Journey Attribution Example:")
    sample_journey = [
        {'channel': 'Display', 'timestamp': pd.Timestamp('2024-01-01')},
        {'channel': 'Email', 'timestamp': pd.Timestamp('2024-01-05')},
        {'channel': 'Search', 'timestamp': pd.Timestamp('2024-01-08')},
        {'channel': 'Direct', 'timestamp': pd.Timestamp('2024-01-10')}
    ]
    
    journey_attribution = model.predict(sample_journey)
    print(f"Journey: Display (Jan 1) → Email (Jan 5) → Search (Jan 8) → Direct (Jan 10)")
    for channel, weight in sorted(journey_attribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {channel}: {weight:.1%} (recency-adjusted)")
    
    print("\n" + "="*60)
    print("🚀 Advanced time decay attribution for temporal marketing insights")
    print("💼 Enterprise-grade recency modeling for conversion optimization") 
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_time_decay_attribution()