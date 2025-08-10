"""
Position-Based Attribution Model (U-Shaped Attribution)

Implements position-based attribution that gives higher credit to first and last
touchpoints in the customer journey, with remaining credit distributed among
middle touchpoints.

Author: Sotiris Spyrou
Portfolio: https://verityai.co
LinkedIn: https://www.linkedin.com/in/sspyrou/

DISCLAIMER: This is demonstration code for portfolio purposes only.
Not intended for production use without proper testing and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import warnings
import logging

logger = logging.getLogger(__name__)


class PositionBasedAttribution:
    """
    Position-based attribution model (U-shaped attribution).
    
    Attributes higher value to first and last touchpoints in customer journeys,
    while distributing remaining credit among middle touchpoints. Customizable
    weighting allows for different business contexts and journey patterns.
    """
    
    def __init__(self,
                 first_touch_weight: float = 0.4,
                 last_touch_weight: float = 0.4,
                 middle_touch_weight: float = 0.2,
                 position_decay: bool = True,
                 time_weighted: bool = False):
        """
        Initialize Position-Based Attribution model.
        
        Args:
            first_touch_weight: Weight assigned to first touchpoint (0.0-1.0)
            last_touch_weight: Weight assigned to last touchpoint (0.0-1.0)
            middle_touch_weight: Weight assigned to middle touchpoints (0.0-1.0)
            position_decay: Apply decay to middle positions
            time_weighted: Apply time-based weighting
        """
        # Validate weights
        if not np.isclose(first_touch_weight + last_touch_weight + middle_touch_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")
        
        self.first_touch_weight = first_touch_weight
        self.last_touch_weight = last_touch_weight
        self.middle_touch_weight = middle_touch_weight
        self.position_decay = position_decay
        self.time_weighted = time_weighted
        
        # Attribution results
        self.channel_attribution = {}
        self.journey_analysis = {}
        self.position_analysis = {}
        self.model_statistics = {}
        
        # Journey patterns
        self.journey_patterns = defaultdict(int)
        self.conversion_rates = {}
        
    def fit(self, journey_data: pd.DataFrame) -> 'PositionBasedAttribution':
        """
        Fit the position-based attribution model.
        
        Args:
            journey_data: Customer journey data with customer_id, touchpoint, timestamp, converted
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Position-Based Attribution model")
        
        # Validate input data
        required_columns = ['customer_id', 'touchpoint', 'timestamp', 'converted']
        if not all(col in journey_data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        # Prepare data
        journey_data = journey_data.copy()
        journey_data['timestamp'] = pd.to_datetime(journey_data['timestamp'])
        journey_data = journey_data.sort_values(['customer_id', 'timestamp'])
        
        # Calculate attribution weights
        self._calculate_position_attribution(journey_data)
        
        # Analyze journey patterns
        self._analyze_journey_patterns(journey_data)
        
        # Calculate position-specific insights
        self._analyze_position_performance(journey_data)
        
        # Generate model statistics
        self._calculate_model_statistics(journey_data)
        
        logger.info(f"Attribution model fitted for {len(self.channel_attribution)} channels")
        return self
    
    def _calculate_position_attribution(self, data: pd.DataFrame):
        """Calculate position-based attribution for each channel."""
        
        channel_credits = defaultdict(float)
        total_conversions = 0
        
        # Group by customer to analyze individual journeys
        for customer_id, customer_data in data.groupby('customer_id'):
            customer_data = customer_data.sort_values('timestamp')
            journey = customer_data['touchpoint'].tolist()
            timestamps = customer_data['timestamp'].tolist()
            converted = customer_data['converted'].any()
            
            if not converted:
                continue  # Only attribute credit for converted customers
            
            total_conversions += 1
            
            # Apply position-based attribution to this journey
            journey_credits = self._calculate_journey_attribution(
                journey, timestamps, customer_id
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
    
    def _calculate_journey_attribution(self, 
                                     journey: List[str], 
                                     timestamps: List[pd.Timestamp],
                                     customer_id: str) -> Dict[str, float]:
        """Calculate attribution for a single customer journey."""
        
        if len(journey) == 0:
            return {}
        
        journey_credits = defaultdict(float)
        
        if len(journey) == 1:
            # Single touch gets full credit
            journey_credits[journey[0]] = 1.0
            
        elif len(journey) == 2:
            # Two touches: split between first and last weights
            total_weight = self.first_touch_weight + self.last_touch_weight
            if total_weight > 0:
                first_weight = self.first_touch_weight / total_weight
                last_weight = self.last_touch_weight / total_weight
                
                journey_credits[journey[0]] += first_weight
                journey_credits[journey[1]] += last_weight
            else:
                # Fallback: equal split
                journey_credits[journey[0]] = 0.5
                journey_credits[journey[1]] = 0.5
                
        else:
            # Multi-touch journey: apply U-shaped attribution
            
            # First touch
            journey_credits[journey[0]] += self.first_touch_weight
            
            # Last touch
            journey_credits[journey[-1]] += self.last_touch_weight
            
            # Middle touches
            middle_touches = journey[1:-1]
            if middle_touches and self.middle_touch_weight > 0:
                middle_attribution = self._distribute_middle_attribution(
                    middle_touches, timestamps[1:-1], customer_id
                )
                
                for channel, credit in middle_attribution.items():
                    journey_credits[channel] += credit
        
        return dict(journey_credits)
    
    def _distribute_middle_attribution(self, 
                                     middle_touches: List[str],
                                     middle_timestamps: List[pd.Timestamp],
                                     customer_id: str) -> Dict[str, float]:
        """Distribute attribution among middle touchpoints."""
        
        if not middle_touches:
            return {}
        
        middle_credits = defaultdict(float)
        
        if not self.position_decay and not self.time_weighted:
            # Simple equal distribution
            credit_per_touch = self.middle_touch_weight / len(middle_touches)
            for touch in middle_touches:
                middle_credits[touch] += credit_per_touch
                
        else:
            # Advanced distribution with decay/time weighting
            weights = self._calculate_middle_weights(
                middle_touches, middle_timestamps
            )
            
            total_weight = sum(weights)
            if total_weight > 0:
                for i, touch in enumerate(middle_touches):
                    normalized_weight = weights[i] / total_weight
                    middle_credits[touch] += self.middle_touch_weight * normalized_weight
            else:
                # Fallback to equal distribution
                credit_per_touch = self.middle_touch_weight / len(middle_touches)
                for touch in middle_touches:
                    middle_credits[touch] += credit_per_touch
        
        return dict(middle_credits)
    
    def _calculate_middle_weights(self, 
                                middle_touches: List[str],
                                middle_timestamps: List[pd.Timestamp]) -> List[float]:
        """Calculate weights for middle touchpoints."""
        
        weights = []
        
        for i, (touch, timestamp) in enumerate(zip(middle_touches, middle_timestamps)):
            weight = 1.0
            
            # Position decay: touches closer to edges get higher weight
            if self.position_decay:
                num_middle = len(middle_touches)
                distance_from_edge = min(i + 1, num_middle - i)
                # Exponential decay from edges
                position_weight = np.exp(-0.5 * (distance_from_edge - 1))
                weight *= position_weight
            
            # Time-based weighting: more recent touches get higher weight
            if self.time_weighted and len(middle_timestamps) > 1:
                # Normalize timestamps to 0-1 range within middle touches
                min_time = min(middle_timestamps)
                max_time = max(middle_timestamps)
                
                if max_time != min_time:
                    time_ratio = (timestamp - min_time) / (max_time - min_time)
                    # More recent = higher weight
                    time_weight = 0.5 + 0.5 * time_ratio
                    weight *= time_weight
            
            weights.append(weight)
        
        return weights
    
    def _analyze_journey_patterns(self, data: pd.DataFrame):
        """Analyze common journey patterns and their performance."""
        
        journey_performance = defaultdict(lambda: {'conversions': 0, 'total': 0})
        
        for customer_id, customer_data in data.groupby('customer_id'):
            customer_data = customer_data.sort_values('timestamp')
            journey = tuple(customer_data['touchpoint'].tolist())
            converted = customer_data['converted'].any()
            
            journey_performance[journey]['total'] += 1
            if converted:
                journey_performance[journey]['conversions'] += 1
        
        # Calculate conversion rates and store patterns
        self.journey_patterns = {}
        for journey, stats in journey_performance.items():
            conversion_rate = stats['conversions'] / stats['total'] if stats['total'] > 0 else 0
            self.journey_patterns[journey] = {
                'frequency': stats['total'],
                'conversions': stats['conversions'],
                'conversion_rate': conversion_rate,
                'journey_length': len(journey)
            }
        
        # Store top patterns by frequency
        sorted_patterns = sorted(
            self.journey_patterns.items(),
            key=lambda x: x[1]['frequency'],
            reverse=True
        )
        
        self.journey_analysis = {
            'total_unique_patterns': len(self.journey_patterns),
            'top_patterns': dict(sorted_patterns[:20]),
            'avg_journey_length': np.mean([p['journey_length'] for p in self.journey_patterns.values()]),
            'journey_length_distribution': self._calculate_length_distribution()
        }
        
    def _calculate_length_distribution(self) -> Dict[int, int]:
        """Calculate distribution of journey lengths."""
        length_counts = Counter()
        
        for pattern_info in self.journey_patterns.values():
            length = pattern_info['journey_length']
            frequency = pattern_info['frequency']
            length_counts[length] += frequency
            
        return dict(length_counts)
    
    def _analyze_position_performance(self, data: pd.DataFrame):
        """Analyze performance of channels by position in journey."""
        
        position_performance = defaultdict(lambda: defaultdict(int))
        
        for customer_id, customer_data in data.groupby('customer_id'):
            customer_data = customer_data.sort_values('timestamp')
            journey = customer_data['touchpoint'].tolist()
            converted = customer_data['converted'].any()
            
            for i, channel in enumerate(journey):
                if len(journey) == 1:
                    position = 'single'
                elif i == 0:
                    position = 'first'
                elif i == len(journey) - 1:
                    position = 'last'
                else:
                    position = 'middle'
                
                position_performance[channel][f'{position}_total'] += 1
                if converted:
                    position_performance[channel][f'{position}_conversions'] += 1
        
        # Calculate position-specific metrics
        self.position_analysis = {}
        
        for channel, stats in position_performance.items():
            channel_analysis = {}
            
            for position in ['first', 'middle', 'last', 'single']:
                total = stats.get(f'{position}_total', 0)
                conversions = stats.get(f'{position}_conversions', 0)
                
                if total > 0:
                    channel_analysis[f'{position}_frequency'] = total
                    channel_analysis[f'{position}_conversion_rate'] = conversions / total
                    channel_analysis[f'{position}_share'] = conversions
                else:
                    channel_analysis[f'{position}_frequency'] = 0
                    channel_analysis[f'{position}_conversion_rate'] = 0
                    channel_analysis[f'{position}_share'] = 0
            
            # Calculate position preference score
            total_conversions = sum(stats.get(f'{pos}_conversions', 0) 
                                  for pos in ['first', 'middle', 'last', 'single'])
            
            if total_conversions > 0:
                first_share = stats.get('first_conversions', 0) / total_conversions
                last_share = stats.get('last_conversions', 0) / total_conversions
                middle_share = stats.get('middle_conversions', 0) / total_conversions
                
                channel_analysis['position_preference'] = {
                    'first_preference': first_share > 0.4,
                    'last_preference': last_share > 0.4,
                    'balanced': abs(first_share - last_share) < 0.2
                }
            
            self.position_analysis[channel] = channel_analysis
    
    def _calculate_model_statistics(self, data: pd.DataFrame):
        """Calculate comprehensive model statistics."""
        
        total_customers = data['customer_id'].nunique()
        converting_customers = data[data['converted']]['customer_id'].nunique()
        total_touchpoints = len(data)
        
        # Attribution concentration (Herfindahl index)
        attribution_values = list(self.channel_attribution.values())
        concentration = sum(w**2 for w in attribution_values) if attribution_values else 0
        
        # Channel diversity
        num_channels = len(self.channel_attribution)
        max_diversity = 1 - (1/num_channels) if num_channels > 1 else 0
        actual_diversity = 1 - concentration
        diversity_score = actual_diversity / max_diversity if max_diversity > 0 else 0
        
        self.model_statistics = {
            'total_customers': total_customers,
            'converting_customers': converting_customers,
            'overall_conversion_rate': converting_customers / total_customers if total_customers > 0 else 0,
            'total_touchpoints': total_touchpoints,
            'avg_touchpoints_per_customer': total_touchpoints / total_customers if total_customers > 0 else 0,
            'num_channels': num_channels,
            'attribution_concentration': concentration,
            'channel_diversity_score': diversity_score,
            'model_parameters': {
                'first_touch_weight': self.first_touch_weight,
                'last_touch_weight': self.last_touch_weight,
                'middle_touch_weight': self.middle_touch_weight,
                'position_decay': self.position_decay,
                'time_weighted': self.time_weighted
            }
        }
    
    def predict(self, journey: List[str]) -> Dict[str, float]:
        """
        Predict attribution for a single journey.
        
        Args:
            journey: List of touchpoints in chronological order
            
        Returns:
            Attribution weights for each channel in the journey
        """
        if not self.channel_attribution:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create dummy timestamps for the journey
        dummy_timestamps = pd.date_range('2024-01-01', periods=len(journey), freq='D')
        
        # Calculate journey attribution
        journey_attribution = self._calculate_journey_attribution(
            journey, dummy_timestamps.tolist(), 'prediction'
        )
        
        return journey_attribution
    
    def get_attribution_results(self) -> pd.DataFrame:
        """
        Get attribution results as DataFrame.
        
        Returns:
            DataFrame with channel attribution weights and position insights
        """
        if not self.channel_attribution:
            raise ValueError("Model not fitted. Call fit() first.")
        
        results = []
        
        for channel, attribution_weight in self.channel_attribution.items():
            position_info = self.position_analysis.get(channel, {})
            
            results.append({
                'channel': channel,
                'attribution_weight': attribution_weight,
                'first_touch_rate': position_info.get('first_conversion_rate', 0),
                'last_touch_rate': position_info.get('last_conversion_rate', 0),
                'middle_touch_rate': position_info.get('middle_conversion_rate', 0),
                'position_preference': self._get_position_preference(channel)
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('attribution_weight', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def _get_position_preference(self, channel: str) -> str:
        """Get position preference description for a channel."""
        
        position_info = self.position_analysis.get(channel, {})
        preference = position_info.get('position_preference', {})
        
        if preference.get('first_preference', False):
            return 'First Touch'
        elif preference.get('last_preference', False):
            return 'Last Touch'
        elif preference.get('balanced', False):
            return 'Balanced'
        else:
            return 'Middle Touch'
    
    def get_journey_insights(self) -> pd.DataFrame:
        """Get insights about journey patterns."""
        
        if not self.journey_patterns:
            return pd.DataFrame()
        
        insights = []
        
        for journey, stats in self.journey_patterns.items():
            journey_str = ' → '.join(journey)
            
            insights.append({
                'journey_pattern': journey_str,
                'frequency': stats['frequency'],
                'conversions': stats['conversions'],
                'conversion_rate': stats['conversion_rate'],
                'journey_length': stats['journey_length'],
                'efficiency_score': stats['conversion_rate'] * stats['frequency']  # Frequency × conversion rate
            })
        
        df = pd.DataFrame(insights)
        df = df.sort_values('efficiency_score', ascending=False)
        
        return df
    
    def get_position_analysis(self) -> pd.DataFrame:
        """Get detailed position analysis by channel."""
        
        if not self.position_analysis:
            return pd.DataFrame()
        
        analysis = []
        
        for channel, stats in self.position_analysis.items():
            analysis.append({
                'channel': channel,
                'first_touch_frequency': stats.get('first_frequency', 0),
                'first_touch_conversion_rate': stats.get('first_conversion_rate', 0),
                'middle_touch_frequency': stats.get('middle_frequency', 0),
                'middle_touch_conversion_rate': stats.get('middle_conversion_rate', 0),
                'last_touch_frequency': stats.get('last_frequency', 0),
                'last_touch_conversion_rate': stats.get('last_conversion_rate', 0),
                'single_touch_frequency': stats.get('single_frequency', 0),
                'single_touch_conversion_rate': stats.get('single_conversion_rate', 0),
                'position_preference': self._get_position_preference(channel)
            })
        
        return pd.DataFrame(analysis)
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        return self.model_statistics.copy()
    
    def generate_executive_report(self) -> str:
        """Generate executive-level position-based attribution report."""
        
        report = "# Position-Based Attribution Analysis\n\n"
        report += "**Strategic Marketing Attribution by Sotiris Spyrou**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        # Model Configuration
        report += f"## Model Configuration\n"
        report += f"- **First Touch Weight**: {self.first_touch_weight:.1%}\n"
        report += f"- **Last Touch Weight**: {self.last_touch_weight:.1%}\n"
        report += f"- **Middle Touch Weight**: {self.middle_touch_weight:.1%}\n"
        report += f"- **Position Decay**: {'Enabled' if self.position_decay else 'Disabled'}\n"
        report += f"- **Time Weighting**: {'Enabled' if self.time_weighted else 'Disabled'}\n\n"
        
        # Attribution Results
        attribution_df = self.get_attribution_results()
        report += f"## Channel Attribution Results\n\n"
        report += "| Rank | Channel | Attribution | Position Preference |\n"
        report += "|------|---------|-------------|---------------------|\n"
        
        for _, row in attribution_df.head(10).iterrows():
            report += f"| {row['rank']} | {row['channel']} | {row['attribution_weight']:.1%} | {row['position_preference']} |\n"
        
        # Journey Insights
        journey_df = self.get_journey_insights()
        if not journey_df.empty:
            report += f"\n## Top Journey Patterns\n\n"
            report += "| Pattern | Frequency | Conversion Rate | Efficiency Score |\n"
            report += "|---------|-----------|-----------------|------------------|\n"
            
            for _, row in journey_df.head(5).iterrows():
                report += f"| {row['journey_pattern']} | {row['frequency']} | {row['conversion_rate']:.1%} | {row['efficiency_score']:.1f} |\n"
        
        # Key Statistics
        stats = self.model_statistics
        report += f"\n## Key Performance Metrics\n\n"
        report += f"- **Total Customers Analyzed**: {stats['total_customers']:,}\n"
        report += f"- **Overall Conversion Rate**: {stats['overall_conversion_rate']:.1%}\n"
        report += f"- **Average Journey Length**: {stats['avg_touchpoints_per_customer']:.1f} touchpoints\n"
        report += f"- **Channel Diversity Score**: {stats['channel_diversity_score']:.1%}\n"
        report += f"- **Attribution Concentration**: {stats['attribution_concentration']:.3f}\n\n"
        
        # Strategic Recommendations
        top_channel = attribution_df.iloc[0]['channel'] if not attribution_df.empty else 'N/A'
        report += f"## Strategic Recommendations\n\n"
        report += f"1. **Optimize {top_channel}**: Highest attribution weight indicates strong performance across positions\n"
        report += f"2. **First-Touch Strategy**: Leverage high-performing first-touch channels for awareness\n"
        report += f"3. **Last-Touch Optimization**: Focus on channels that excel at conversion\n"
        report += f"4. **Journey Orchestration**: Design sequences based on top-performing patterns\n"
        report += f"5. **Position-Specific Investment**: Allocate budget based on position effectiveness\n\n"
        
        report += "---\n*This analysis demonstrates sophisticated position-based attribution modeling. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for custom implementations.*"
        
        return report


def demo_position_based_attribution():
    """Executive demonstration of Position-Based Attribution."""
    
    print("=== Position-Based Attribution: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    np.random.seed(42)
    
    # Generate realistic customer journey data
    customers = []
    channels = ['Search', 'Display', 'Social', 'Email', 'Direct']
    
    # Different channels have different position preferences
    first_touch_channels = ['Display', 'Social']  # Awareness channels
    last_touch_channels = ['Search', 'Direct']    # Conversion channels
    middle_touch_channels = ['Email', 'Display']  # Nurturing channels
    
    # Generate 800 customer journeys
    for customer_id in range(1, 801):
        # Random journey length (1-6 touchpoints)
        journey_length = np.random.choice(range(1, 7), p=[0.15, 0.25, 0.25, 0.2, 0.1, 0.05])
        
        # Build journey with position preferences
        journey_channels = []
        timestamps = []
        start_date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 30))
        
        for i in range(journey_length):
            if i == 0:  # First touch
                channel = np.random.choice(first_touch_channels + ['Search'], 
                                         p=[0.3, 0.3, 0.4])
            elif i == journey_length - 1:  # Last touch
                channel = np.random.choice(last_touch_channels, p=[0.6, 0.4])
            else:  # Middle touches
                channel = np.random.choice(middle_touch_channels + ['Search'], 
                                         p=[0.4, 0.3, 0.3])
            
            journey_channels.append(channel)
            timestamp = start_date + pd.Timedelta(days=i) + pd.Timedelta(hours=np.random.randint(0, 24))
            timestamps.append(timestamp)
        
        # Determine conversion based on journey quality
        # Higher probability for journeys with good first-last combination
        conversion_prob = 0.15  # Base rate
        
        if journey_length > 1:
            first_channel = journey_channels[0]
            last_channel = journey_channels[-1]
            
            # Boost for awareness → conversion pattern
            if first_channel in first_touch_channels and last_channel in last_touch_channels:
                conversion_prob += 0.15
            
            # Boost for longer, diverse journeys
            unique_channels = len(set(journey_channels))
            if unique_channels >= 3:
                conversion_prob += 0.1
        
        converted = np.random.random() < conversion_prob
        
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
    model = PositionBasedAttribution(
        first_touch_weight=0.4,
        last_touch_weight=0.4, 
        middle_touch_weight=0.2,
        position_decay=True,
        time_weighted=True
    )
    
    print(f"\n🎯 Fitting U-shaped attribution model...")
    model.fit(journey_data)
    
    # Display results
    print("\n📊 POSITION-BASED ATTRIBUTION RESULTS")
    print("=" * 55)
    
    attribution_results = model.get_attribution_results()
    print(f"\n🏆 Channel Attribution & Position Preferences:")
    for _, row in attribution_results.iterrows():
        rank_emoji = "🥇" if row['rank'] == 1 else "🥈" if row['rank'] == 2 else "🥉" if row['rank'] == 3 else "📊"
        preference_emoji = "🎯" if row['position_preference'] == 'Last Touch' else "📢" if row['position_preference'] == 'First Touch' else "⚖️"
        print(f"{rank_emoji} {row['channel']:8}: {row['attribution_weight']:.1%} attribution {preference_emoji} {row['position_preference']}")
    
    # Journey pattern analysis
    journey_insights = model.get_journey_insights()
    print(f"\n🧭 Top Converting Journey Patterns:")
    for _, row in journey_insights.head(5).iterrows():
        conversion_icon = "🔥" if row['conversion_rate'] > 0.25 else "📈" if row['conversion_rate'] > 0.15 else "📊"
        print(f"{conversion_icon} {row['journey_pattern'][:40]:40} | {row['conversion_rate']:.1%} CR | {row['frequency']:3} freq")
    
    # Position analysis
    position_analysis = model.get_position_analysis()
    print(f"\n📍 Position Performance Analysis:")
    for _, row in position_analysis.iterrows():
        first_rate = row['first_touch_conversion_rate']
        last_rate = row['last_touch_conversion_rate']
        
        # Show strongest position for each channel
        if first_rate > last_rate and first_rate > 0.15:
            strength = f"First Touch: {first_rate:.1%}"
        elif last_rate > 0.15:
            strength = f"Last Touch: {last_rate:.1%}"
        else:
            strength = "Balanced Performance"
            
        print(f"  • {row['channel']:8}: {strength}")
    
    # Model statistics
    stats = model.get_model_statistics()
    print(f"\n📈 MODEL PERFORMANCE:")
    print(f"  • Journey Length (avg): {stats['avg_touchpoints_per_customer']:.1f} touchpoints")
    print(f"  • Channel Diversity: {stats['channel_diversity_score']:.1%}")
    print(f"  • Attribution Balance: {1-stats['attribution_concentration']:.1%}")
    
    # Example prediction
    print(f"\n🔮 Journey Attribution Example:")
    sample_journey = ['Display', 'Email', 'Search', 'Direct']
    journey_attribution = model.predict(sample_journey)
    print(f"Journey: {' → '.join(sample_journey)}")
    for channel, weight in sorted(journey_attribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {channel}: {weight:.1%}")
    
    print("\n" + "="*60)
    print("🚀 Strategic U-shaped attribution for customer journey optimization")
    print("💼 Enterprise-grade position-based marketing analytics") 
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_position_based_attribution()