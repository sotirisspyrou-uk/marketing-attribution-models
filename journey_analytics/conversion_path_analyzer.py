"""
Customer Conversion Path Analyzer

Advanced conversion path analysis system for understanding customer journey
sequences, identifying optimal paths to conversion, and analyzing conversion
funnels with sophisticated pattern recognition.

Author: Sotiris Spyrou
Portfolio: https://verityai.co
LinkedIn: https://www.linkedin.com/in/sspyrou/

DISCLAIMER: This is demonstration code for portfolio purposes only.
Not intended for production use without proper testing and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
import networkx as nx
from itertools import combinations

logger = logging.getLogger(__name__)


class PathType(Enum):
    """Types of conversion paths."""
    SINGLE_TOUCH = "single_touch"
    MULTI_TOUCH = "multi_touch"
    DIRECT_CONVERSION = "direct_conversion"
    ASSISTED_CONVERSION = "assisted_conversion"


class PathMetric(Enum):
    """Path analysis metrics."""
    CONVERSION_RATE = "conversion_rate"
    PATH_LENGTH = "path_length" 
    TIME_TO_CONVERSION = "time_to_conversion"
    CHANNEL_DIVERSITY = "channel_diversity"
    FREQUENCY = "frequency"


@dataclass
class ConversionPath:
    """Conversion path data structure."""
    path_id: str
    customer_id: str
    touchpoints: List[str]
    timestamps: List[datetime]
    channels: List[str]
    converted: bool
    conversion_value: float = 0.0
    path_length: int = 0
    time_to_conversion: float = 0.0
    channel_diversity: float = 0.0
    path_type: PathType = PathType.MULTI_TOUCH


@dataclass
class PathPattern:
    """Path pattern analysis results."""
    pattern: str
    frequency: int
    conversion_rate: float
    avg_time_to_conversion: float
    avg_conversion_value: float
    path_efficiency: float
    channels_involved: Set[str]


class ConversionPathAnalyzer:
    """
    Advanced conversion path analysis engine for customer journey optimization.
    
    Analyzes customer conversion paths, identifies high-performing sequences,
    and provides insights for journey optimization and funnel improvement.
    """
    
    def __init__(self,
                 min_pattern_frequency: int = 10,
                 max_path_length: int = 20,
                 time_window_days: int = 90,
                 value_threshold: float = 0.0):
        """
        Initialize Conversion Path Analyzer.
        
        Args:
            min_pattern_frequency: Minimum occurrences for pattern analysis
            max_path_length: Maximum path length to analyze
            time_window_days: Maximum days between first touch and conversion
            value_threshold: Minimum conversion value to include
        """
        self.min_pattern_frequency = min_pattern_frequency
        self.max_path_length = max_path_length
        self.time_window_days = time_window_days
        self.value_threshold = value_threshold
        
        # Analysis results
        self.conversion_paths: List[ConversionPath] = []
        self.path_patterns: Dict[str, PathPattern] = {}
        self.channel_sequences: Dict[str, Dict] = {}
        self.funnel_analysis: Dict[str, Any] = {}
        self.path_graph: nx.DiGraph = nx.DiGraph()
        
        # Performance metrics
        self.path_performance: Dict[str, Dict] = {}
        self.channel_transitions: Dict[str, Dict] = {}
        self.conversion_probabilities: Dict[str, float] = {}
        self.optimization_opportunities: List[Dict] = []
        
        logger.info("Conversion path analyzer initialized")
    
    def analyze_paths(self, journey_data: pd.DataFrame) -> 'ConversionPathAnalyzer':
        """
        Analyze customer conversion paths from journey data.
        
        Args:
            journey_data: Customer journey data with touchpoints and conversions
            
        Returns:
            Self for method chaining
        """
        logger.info("Starting conversion path analysis")
        
        # Validate input data
        required_columns = ['customer_id', 'touchpoint', 'channel', 'timestamp', 'converted']
        if not all(col in journey_data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        # Prepare data
        journey_data = journey_data.copy()
        journey_data['timestamp'] = pd.to_datetime(journey_data['timestamp'])
        journey_data = journey_data.sort_values(['customer_id', 'timestamp'])
        
        # Extract conversion paths
        self._extract_conversion_paths(journey_data)
        
        # Analyze path patterns
        self._analyze_path_patterns()
        
        # Build path network
        self._build_path_network()
        
        # Analyze channel transitions
        self._analyze_channel_transitions()
        
        # Calculate conversion probabilities
        self._calculate_conversion_probabilities()
        
        # Perform funnel analysis
        self._perform_funnel_analysis()
        
        # Identify optimization opportunities
        self._identify_optimization_opportunities()
        
        logger.info(f"Path analysis completed: {len(self.conversion_paths)} paths analyzed")
        return self
    
    def _extract_conversion_paths(self, data: pd.DataFrame):
        """Extract individual conversion paths from journey data."""
        
        paths = []
        
        for customer_id, customer_data in data.groupby('customer_id'):
            customer_data = customer_data.sort_values('timestamp')
            
            # Extract path components
            touchpoints = customer_data['touchpoint'].tolist()
            channels = customer_data['channel'].tolist()
            timestamps = customer_data['timestamp'].tolist()
            converted = customer_data['converted'].any()
            conversion_value = customer_data['conversion_value'].sum() if 'conversion_value' in customer_data else 0.0
            
            # Skip paths outside time window or too long
            if len(touchpoints) > self.max_path_length:
                continue
            
            if timestamps:
                time_span = (timestamps[-1] - timestamps[0]).days
                if time_span > self.time_window_days:
                    continue
            
            # Skip low-value conversions
            if converted and conversion_value < self.value_threshold:
                continue
            
            # Calculate path metrics
            path_length = len(touchpoints)
            unique_channels = set(channels)
            channel_diversity = len(unique_channels) / path_length if path_length > 0 else 0
            time_to_conversion = (timestamps[-1] - timestamps[0]).total_seconds() / 3600 if len(timestamps) > 1 else 0  # Hours
            
            # Determine path type
            if path_length == 1:
                path_type = PathType.DIRECT_CONVERSION if converted else PathType.SINGLE_TOUCH
            else:
                path_type = PathType.ASSISTED_CONVERSION if converted else PathType.MULTI_TOUCH
            
            # Create path object
            path = ConversionPath(
                path_id=f"path_{customer_id}",
                customer_id=str(customer_id),
                touchpoints=touchpoints,
                timestamps=timestamps,
                channels=channels,
                converted=converted,
                conversion_value=conversion_value,
                path_length=path_length,
                time_to_conversion=time_to_conversion,
                channel_diversity=channel_diversity,
                path_type=path_type
            )
            
            paths.append(path)
        
        self.conversion_paths = paths
        logger.info(f"Extracted {len(paths)} conversion paths")
    
    def _analyze_path_patterns(self):
        """Analyze common path patterns and their performance."""
        
        pattern_stats = defaultdict(lambda: {
            'frequency': 0,
            'conversions': 0,
            'conversion_values': [],
            'times_to_conversion': [],
            'channels': set()
        })
        
        for path in self.conversion_paths:
            # Create pattern string from channels
            if len(path.channels) <= 5:  # Only analyze shorter patterns for clarity
                pattern = " → ".join(path.channels)
                
                stats = pattern_stats[pattern]
                stats['frequency'] += 1
                stats['channels'].update(path.channels)
                
                if path.converted:
                    stats['conversions'] += 1
                    stats['conversion_values'].append(path.conversion_value)
                    stats['times_to_conversion'].append(path.time_to_conversion)
        
        # Convert to PathPattern objects
        patterns = {}
        
        for pattern, stats in pattern_stats.items():
            if stats['frequency'] >= self.min_pattern_frequency:
                conversion_rate = stats['conversions'] / stats['frequency'] if stats['frequency'] > 0 else 0
                avg_time = np.mean(stats['times_to_conversion']) if stats['times_to_conversion'] else 0
                avg_value = np.mean(stats['conversion_values']) if stats['conversion_values'] else 0
                
                # Calculate efficiency (conversions per hour)
                efficiency = stats['conversions'] / avg_time if avg_time > 0 else 0
                
                patterns[pattern] = PathPattern(
                    pattern=pattern,
                    frequency=stats['frequency'],
                    conversion_rate=conversion_rate,
                    avg_time_to_conversion=avg_time,
                    avg_conversion_value=avg_value,
                    path_efficiency=efficiency,
                    channels_involved=stats['channels']
                )
        
        self.path_patterns = patterns
        logger.info(f"Identified {len(patterns)} significant path patterns")
    
    def _build_path_network(self):
        """Build network graph of channel transitions."""
        
        graph = nx.DiGraph()
        
        # Add nodes for each channel
        all_channels = set()
        for path in self.conversion_paths:
            all_channels.update(path.channels)
        
        for channel in all_channels:
            graph.add_node(channel)
        
        # Add edges for transitions
        transition_weights = defaultdict(int)
        conversion_weights = defaultdict(int)
        
        for path in self.conversion_paths:
            channels = path.channels
            converted = path.converted
            
            # Add sequential transitions
            for i in range(len(channels) - 1):
                from_channel = channels[i]
                to_channel = channels[i + 1]
                
                transition_weights[(from_channel, to_channel)] += 1
                
                if converted:
                    conversion_weights[(from_channel, to_channel)] += 1
        
        # Add weighted edges
        for (from_ch, to_ch), weight in transition_weights.items():
            conversion_weight = conversion_weights.get((from_ch, to_ch), 0)
            conversion_rate = conversion_weight / weight if weight > 0 else 0
            
            graph.add_edge(from_ch, to_ch, 
                         weight=weight,
                         conversion_weight=conversion_weight,
                         conversion_rate=conversion_rate)
        
        self.path_graph = graph
        logger.info(f"Built path network with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    def _analyze_channel_transitions(self):
        """Analyze channel transition patterns and probabilities."""
        
        transitions = {}
        
        for channel in self.path_graph.nodes():
            # Outgoing transitions
            outgoing = {}
            total_outgoing = 0
            
            for successor in self.path_graph.successors(channel):
                edge_data = self.path_graph[channel][successor]
                weight = edge_data['weight']
                outgoing[successor] = weight
                total_outgoing += weight
            
            # Calculate transition probabilities
            transition_probs = {}
            if total_outgoing > 0:
                for successor, weight in outgoing.items():
                    transition_probs[successor] = weight / total_outgoing
            
            # Incoming transitions
            incoming = {}
            total_incoming = 0
            
            for predecessor in self.path_graph.predecessors(channel):
                edge_data = self.path_graph[predecessor][channel]
                weight = edge_data['weight']
                incoming[predecessor] = weight
                total_incoming += weight
            
            transitions[channel] = {
                'outgoing_transitions': outgoing,
                'transition_probabilities': transition_probs,
                'incoming_transitions': incoming,
                'total_outgoing': total_outgoing,
                'total_incoming': total_incoming,
                'is_entry_point': total_incoming == 0,
                'is_exit_point': total_outgoing == 0
            }
        
        self.channel_transitions = transitions
    
    def _calculate_conversion_probabilities(self):
        """Calculate conversion probabilities by path position and length."""
        
        position_conversions = defaultdict(lambda: {'total': 0, 'conversions': 0})
        length_conversions = defaultdict(lambda: {'total': 0, 'conversions': 0})
        
        for path in self.conversion_paths:
            length = path.path_length
            length_conversions[length]['total'] += 1
            
            if path.converted:
                length_conversions[length]['conversions'] += 1
            
            # Position-based analysis
            for i, channel in enumerate(path.channels):
                position = i + 1  # 1-indexed position
                key = f"{channel}_pos_{position}"
                
                position_conversions[key]['total'] += 1
                if path.converted:
                    position_conversions[key]['conversions'] += 1
        
        # Calculate probabilities
        probabilities = {}
        
        # By path length
        for length, stats in length_conversions.items():
            if stats['total'] >= 5:  # Minimum sample size
                prob = stats['conversions'] / stats['total']
                probabilities[f"path_length_{length}"] = prob
        
        # By position
        for pos_key, stats in position_conversions.items():
            if stats['total'] >= 10:  # Minimum sample size
                prob = stats['conversions'] / stats['total']
                probabilities[pos_key] = prob
        
        self.conversion_probabilities = probabilities
    
    def _perform_funnel_analysis(self):
        """Perform funnel analysis to identify drop-off points."""
        
        # Analyze by path position
        position_data = defaultdict(lambda: {'customers': 0, 'conversions': 0})
        
        for path in self.conversion_paths:
            for i in range(path.path_length):
                position = i + 1
                position_data[position]['customers'] += 1
                
                if path.converted:
                    position_data[position]['conversions'] += 1
        
        # Calculate funnel metrics
        funnel_steps = []
        previous_customers = None
        
        for position in sorted(position_data.keys()):
            data = position_data[position]
            customers = data['customers']
            conversions = data['conversions']
            
            # Drop-off rate
            drop_off_rate = 0
            if previous_customers is not None and previous_customers > 0:
                drop_off_rate = (previous_customers - customers) / previous_customers
            
            # Conversion rate at this step
            conversion_rate = conversions / customers if customers > 0 else 0
            
            funnel_steps.append({
                'position': position,
                'customers': customers,
                'conversions': conversions,
                'conversion_rate': conversion_rate,
                'drop_off_rate': drop_off_rate,
                'retention_rate': 1 - drop_off_rate
            })
            
            previous_customers = customers
        
        # Analyze channel-specific funnels
        channel_funnels = {}
        
        for channel in set(ch for path in self.conversion_paths for ch in path.channels):
            channel_paths = [path for path in self.conversion_paths if channel in path.channels]
            
            if len(channel_paths) >= 20:  # Minimum sample size
                converted_paths = [path for path in channel_paths if path.converted]
                
                channel_funnels[channel] = {
                    'total_paths': len(channel_paths),
                    'converted_paths': len(converted_paths),
                    'conversion_rate': len(converted_paths) / len(channel_paths),
                    'avg_path_length': np.mean([p.path_length for p in channel_paths]),
                    'avg_time_to_conversion': np.mean([p.time_to_conversion for p in converted_paths]) if converted_paths else 0
                }
        
        self.funnel_analysis = {
            'overall_funnel': funnel_steps,
            'channel_funnels': channel_funnels,
            'funnel_efficiency': len([p for p in self.conversion_paths if p.converted]) / len(self.conversion_paths) if self.conversion_paths else 0
        }
    
    def _identify_optimization_opportunities(self):
        """Identify path optimization opportunities."""
        
        opportunities = []
        
        # High-performing patterns to scale
        top_patterns = sorted(self.path_patterns.values(), 
                            key=lambda x: x.conversion_rate * x.frequency, 
                            reverse=True)[:5]
        
        for pattern in top_patterns:
            if pattern.conversion_rate > 0.3:  # High conversion rate
                opportunities.append({
                    'type': 'scale_pattern',
                    'description': f"Scale high-performing path: {pattern.pattern}",
                    'pattern': pattern.pattern,
                    'conversion_rate': pattern.conversion_rate,
                    'frequency': pattern.frequency,
                    'priority': 'high',
                    'expected_impact': pattern.frequency * 0.2  # Potential 20% increase
                })
        
        # Channels with high drop-off rates
        if self.funnel_analysis.get('channel_funnels'):
            for channel, funnel_data in self.funnel_analysis['channel_funnels'].items():
                if funnel_data['conversion_rate'] < 0.1 and funnel_data['total_paths'] > 50:
                    opportunities.append({
                        'type': 'improve_funnel',
                        'description': f"Optimize {channel} conversion funnel",
                        'channel': channel,
                        'current_rate': funnel_data['conversion_rate'],
                        'total_paths': funnel_data['total_paths'],
                        'priority': 'medium',
                        'expected_impact': funnel_data['total_paths'] * 0.05  # 5% improvement
                    })
        
        # Transition improvements
        for from_channel, transitions in self.channel_transitions.items():
            for to_channel, prob in transitions['transition_probabilities'].items():
                if prob > 0.3 and self.path_graph.has_edge(from_channel, to_channel):
                    edge_data = self.path_graph[from_channel][to_channel]
                    if edge_data['conversion_rate'] > 0.4:
                        opportunities.append({
                            'type': 'strengthen_transition',
                            'description': f"Strengthen {from_channel} → {to_channel} transition",
                            'from_channel': from_channel,
                            'to_channel': to_channel,
                            'transition_probability': prob,
                            'conversion_rate': edge_data['conversion_rate'],
                            'priority': 'low',
                            'expected_impact': edge_data['weight'] * 0.1
                        })
        
        # Sort by expected impact
        self.optimization_opportunities = sorted(opportunities, 
                                               key=lambda x: x.get('expected_impact', 0), 
                                               reverse=True)[:10]  # Top 10
    
    def get_top_conversion_paths(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top performing conversion paths."""
        
        converted_paths = [path for path in self.conversion_paths if path.converted]
        
        # Sort by efficiency (value per hour)
        top_paths = sorted(converted_paths, 
                         key=lambda x: x.conversion_value / max(x.time_to_conversion, 1), 
                         reverse=True)[:n]
        
        results = []
        for path in top_paths:
            results.append({
                'path_id': path.path_id,
                'pattern': " → ".join(path.channels),
                'conversion_value': path.conversion_value,
                'time_to_conversion_hours': path.time_to_conversion,
                'path_length': path.path_length,
                'channel_diversity': path.channel_diversity,
                'efficiency': path.conversion_value / max(path.time_to_conversion, 1)
            })
        
        return results
    
    def get_channel_performance(self) -> pd.DataFrame:
        """Get channel performance in conversion paths."""
        
        channel_stats = defaultdict(lambda: {
            'appearances': 0,
            'conversions': 0,
            'total_value': 0,
            'first_touch': 0,
            'last_touch': 0,
            'assisted_conversions': 0
        })
        
        for path in self.conversion_paths:
            for i, channel in enumerate(path.channels):
                stats = channel_stats[channel]
                stats['appearances'] += 1
                
                if path.converted:
                    stats['conversions'] += 1
                    stats['total_value'] += path.conversion_value
                    
                    if i == 0:  # First touch
                        stats['first_touch'] += 1
                    elif i == len(path.channels) - 1:  # Last touch
                        stats['last_touch'] += 1
                    else:  # Assisted
                        stats['assisted_conversions'] += 1
        
        # Convert to DataFrame
        results = []
        for channel, stats in channel_stats.items():
            results.append({
                'channel': channel,
                'total_appearances': stats['appearances'],
                'conversions': stats['conversions'],
                'conversion_rate': stats['conversions'] / stats['appearances'] if stats['appearances'] > 0 else 0,
                'total_value': stats['total_value'],
                'avg_value_per_conversion': stats['total_value'] / stats['conversions'] if stats['conversions'] > 0 else 0,
                'first_touch_conversions': stats['first_touch'],
                'last_touch_conversions': stats['last_touch'],
                'assisted_conversions': stats['assisted_conversions'],
                'assist_rate': stats['assisted_conversions'] / stats['appearances'] if stats['appearances'] > 0 else 0
            })
        
        df = pd.DataFrame(results)
        return df.sort_values('conversion_rate', ascending=False) if not df.empty else df
    
    def get_path_patterns(self, min_frequency: int = None) -> pd.DataFrame:
        """Get path patterns analysis."""
        
        min_freq = min_frequency or self.min_pattern_frequency
        
        results = []
        for pattern_name, pattern in self.path_patterns.items():
            if pattern.frequency >= min_freq:
                results.append({
                    'pattern': pattern.pattern,
                    'frequency': pattern.frequency,
                    'conversion_rate': pattern.conversion_rate,
                    'avg_time_to_conversion': pattern.avg_time_to_conversion,
                    'avg_conversion_value': pattern.avg_conversion_value,
                    'efficiency': pattern.path_efficiency,
                    'channels_count': len(pattern.channels_involved),
                    'channels': ", ".join(sorted(pattern.channels_involved))
                })
        
        df = pd.DataFrame(results)
        return df.sort_values('conversion_rate', ascending=False) if not df.empty else df
    
    def predict_conversion_probability(self, path_sequence: List[str]) -> float:
        """Predict conversion probability for a given path sequence."""
        
        if not path_sequence:
            return 0.0
        
        # Use path length probability
        length_key = f"path_length_{len(path_sequence)}"
        base_prob = self.conversion_probabilities.get(length_key, 0.15)  # Default 15%
        
        # Adjust based on pattern matching
        pattern = " → ".join(path_sequence)
        if pattern in self.path_patterns:
            pattern_prob = self.path_patterns[pattern].conversion_rate
            # Weighted average of base and pattern probability
            return (base_prob + pattern_prob * 2) / 3
        
        # Adjust based on last channel (last-touch bias)
        last_channel = path_sequence[-1]
        last_touch_key = f"{last_channel}_pos_{len(path_sequence)}"
        if last_touch_key in self.conversion_probabilities:
            position_prob = self.conversion_probabilities[last_touch_key]
            return (base_prob + position_prob) / 2
        
        return base_prob
    
    def generate_path_recommendations(self, current_path: List[str]) -> List[Dict[str, Any]]:
        """Generate recommendations for continuing a path."""
        
        if not current_path:
            return []
        
        last_channel = current_path[-1]
        recommendations = []
        
        # Get transition probabilities from last channel
        if last_channel in self.channel_transitions:
            transitions = self.channel_transitions[last_channel]
            
            for next_channel, prob in transitions['transition_probabilities'].items():
                if prob > 0.1:  # Significant transition probability
                    # Calculate expected conversion rate
                    test_path = current_path + [next_channel]
                    expected_conv_rate = self.predict_conversion_probability(test_path)
                    
                    # Get edge conversion rate
                    edge_conv_rate = 0
                    if self.path_graph.has_edge(last_channel, next_channel):
                        edge_conv_rate = self.path_graph[last_channel][next_channel]['conversion_rate']
                    
                    recommendations.append({
                        'next_channel': next_channel,
                        'transition_probability': prob,
                        'edge_conversion_rate': edge_conv_rate,
                        'expected_path_conversion_rate': expected_conv_rate,
                        'recommendation_score': prob * edge_conv_rate * expected_conv_rate
                    })
        
        # Sort by recommendation score
        return sorted(recommendations, key=lambda x: x['recommendation_score'], reverse=True)[:5]
    
    def generate_executive_report(self) -> str:
        """Generate comprehensive conversion path analysis report."""
        
        report = "# Conversion Path Analysis Report\n\n"
        report += "**Customer Journey Optimization by Sotiris Spyrou**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        # Analysis Overview
        total_paths = len(self.conversion_paths)
        converted_paths = len([p for p in self.conversion_paths if p.converted])
        overall_conversion_rate = converted_paths / total_paths if total_paths > 0 else 0
        
        report += f"## Analysis Overview\n\n"
        report += f"- **Total Paths Analyzed**: {total_paths:,}\n"
        report += f"- **Converted Paths**: {converted_paths:,}\n"
        report += f"- **Overall Conversion Rate**: {overall_conversion_rate:.1%}\n"
        report += f"- **Unique Path Patterns**: {len(self.path_patterns)}\n"
        report += f"- **Channels Analyzed**: {len(set(ch for path in self.conversion_paths for ch in path.channels))}\n\n"
        
        # Top Performing Patterns
        if self.path_patterns:
            top_patterns = sorted(self.path_patterns.values(), key=lambda x: x.conversion_rate, reverse=True)[:5]
            
            report += f"## Top Performing Path Patterns\n\n"
            report += "| Rank | Pattern | Conv. Rate | Frequency | Avg. Time | Efficiency |\n"
            report += "|------|---------|------------|-----------|-----------|------------|\n"
            
            for i, pattern in enumerate(top_patterns, 1):
                report += f"| {i} | {pattern.pattern} | {pattern.conversion_rate:.1%} | {pattern.frequency} | {pattern.avg_time_to_conversion:.1f}h | {pattern.path_efficiency:.2f} |\n"
        
        # Channel Performance
        channel_perf = self.get_channel_performance()
        if not channel_perf.empty:
            report += f"\n## Channel Performance in Paths\n\n"
            report += "| Channel | Conv. Rate | First Touch | Last Touch | Assist Rate |\n"
            report += "|---------|------------|-------------|------------|-------------|\n"
            
            for _, row in channel_perf.head(8).iterrows():
                report += f"| {row['channel']} | {row['conversion_rate']:.1%} | {row['first_touch_conversions']} | {row['last_touch_conversions']} | {row['assist_rate']:.1%} |\n"
        
        # Funnel Analysis
        if self.funnel_analysis:
            funnel_efficiency = self.funnel_analysis.get('funnel_efficiency', 0)
            report += f"\n## Conversion Funnel Insights\n\n"
            report += f"- **Overall Funnel Efficiency**: {funnel_efficiency:.1%}\n"
            
            funnel_steps = self.funnel_analysis.get('overall_funnel', [])
            if funnel_steps and len(funnel_steps) >= 2:
                worst_dropoff = max(funnel_steps[1:], key=lambda x: x['drop_off_rate'])
                report += f"- **Highest Drop-off**: Position {worst_dropoff['position']} ({worst_dropoff['drop_off_rate']:.1%})\n"
            
            # Channel funnel performance
            channel_funnels = self.funnel_analysis.get('channel_funnels', {})
            if channel_funnels:
                best_channel = max(channel_funnels.items(), key=lambda x: x[1]['conversion_rate'])
                worst_channel = min(channel_funnels.items(), key=lambda x: x[1]['conversion_rate'])
                
                report += f"- **Best Funnel Performance**: {best_channel[0]} ({best_channel[1]['conversion_rate']:.1%})\n"
                report += f"- **Needs Improvement**: {worst_channel[0]} ({worst_channel[1]['conversion_rate']:.1%})\n"
        
        # Optimization Opportunities
        if self.optimization_opportunities:
            report += f"\n## Strategic Optimization Opportunities\n\n"
            
            high_priority = [opp for opp in self.optimization_opportunities if opp.get('priority') == 'high']
            
            for i, opp in enumerate(high_priority[:3], 1):
                priority_emoji = "🔴" if opp['priority'] == 'high' else "🟡" if opp['priority'] == 'medium' else "🟢"
                report += f"### {i}. {opp['description']} {priority_emoji}\n\n"
                
                if opp['type'] == 'scale_pattern':
                    report += f"- **Pattern**: {opp['pattern']}\n"
                    report += f"- **Current Performance**: {opp['conversion_rate']:.1%} conversion rate\n"
                    report += f"- **Current Volume**: {opp['frequency']} occurrences\n"
                    report += f"- **Potential Impact**: +{opp['expected_impact']:.0f} conversions\n\n"
                
                elif opp['type'] == 'improve_funnel':
                    report += f"- **Channel**: {opp['channel']}\n"
                    report += f"- **Current Rate**: {opp['current_rate']:.1%}\n"
                    report += f"- **Volume**: {opp['total_paths']} paths\n"
                    report += f"- **Potential Impact**: +{opp['expected_impact']:.0f} conversions\n\n"
        
        # Key Insights
        report += f"## Key Strategic Insights\n\n"
        
        if self.conversion_paths:
            avg_path_length = np.mean([p.path_length for p in self.conversion_paths])
            avg_time_to_conv = np.mean([p.time_to_conversion for p in self.conversion_paths if p.converted])
            
            report += f"1. **Journey Complexity**: Average path length is {avg_path_length:.1f} touchpoints\n"
            report += f"2. **Conversion Velocity**: Average time to conversion is {avg_time_to_conv:.1f} hours\n"
            
            # Channel diversity insights
            high_diversity_paths = [p for p in self.conversion_paths if p.channel_diversity > 0.7]
            if high_diversity_paths:
                diverse_conv_rate = len([p for p in high_diversity_paths if p.converted]) / len(high_diversity_paths)
                report += f"3. **Diversity Impact**: High-diversity paths have {diverse_conv_rate:.1%} conversion rate\n"
            
            # Single vs multi-touch
            single_touch = [p for p in self.conversion_paths if p.path_length == 1]
            multi_touch = [p for p in self.conversion_paths if p.path_length > 1]
            
            if single_touch and multi_touch:
                single_conv = len([p for p in single_touch if p.converted]) / len(single_touch)
                multi_conv = len([p for p in multi_touch if p.converted]) / len(multi_touch)
                report += f"4. **Touch Complexity**: Single-touch: {single_conv:.1%} vs Multi-touch: {multi_conv:.1%}\n"
        
        # Action Plan
        report += f"\n## Recommended Action Plan\n\n"
        report += f"1. **Scale High-Performing Patterns**: Focus marketing efforts on replicating top-converting path sequences\n"
        report += f"2. **Optimize Funnel Drop-offs**: Address highest drop-off points with targeted interventions\n"
        report += f"3. **Channel Sequencing**: Optimize channel transition probabilities for better conversion rates\n"
        report += f"4. **Journey Orchestration**: Implement path recommendations for real-time optimization\n"
        report += f"5. **Continuous Monitoring**: Track path performance changes and adjust strategies accordingly\n\n"
        
        report += "---\n*Advanced conversion path analytics for journey optimization. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for custom implementations.*"
        
        return report


def demo_conversion_path_analyzer():
    """Executive demonstration of Conversion Path Analyzer."""
    
    print("=== Conversion Path Analyzer: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    np.random.seed(42)
    
    # Generate realistic customer journey data with conversion patterns
    customers = []
    channels = ['Search', 'Display', 'Social', 'Email', 'Direct', 'Affiliate', 'Referral']
    
    # Define channel characteristics for realistic patterns
    awareness_channels = ['Display', 'Social', 'Affiliate']
    consideration_channels = ['Search', 'Email', 'Referral'] 
    conversion_channels = ['Direct', 'Search', 'Email']
    
    # High-performing path patterns (will be discovered by the analyzer)
    golden_patterns = [
        ['Display', 'Email', 'Search', 'Direct'],      # Classic awareness → conversion
        ['Social', 'Email', 'Direct'],                 # Social nurturing path
        ['Search', 'Email', 'Search', 'Direct'],       # Search-dominant with email nurturing
        ['Display', 'Search', 'Direct'],               # Simple display → search → conversion
        ['Social', 'Search', 'Email', 'Direct']        # Multi-channel orchestrated path
    ]
    
    customer_id = 1
    
    # Generate golden path customers (higher conversion rates)
    for _ in range(300):
        pattern = np.random.choice(len(golden_patterns), p=[0.3, 0.25, 0.2, 0.15, 0.1])
        path_channels = golden_patterns[pattern]
        
        # Add some variation
        if np.random.random() < 0.3:
            # Add an extra touchpoint
            extra_channel = np.random.choice(channels)
            path_channels = path_channels + [extra_channel]
        
        # Generate timestamps
        start_date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 60))
        journey_span_hours = np.random.exponential(48)  # Average 2 days
        
        timestamps = []
        for i in range(len(path_channels)):
            time_offset = (i / max(len(path_channels) - 1, 1)) * journey_span_hours
            timestamp = start_date + pd.Timedelta(hours=time_offset + np.random.normal(0, 2))
            timestamps.append(timestamp)
        
        # High conversion probability for golden patterns
        converted = np.random.random() < 0.65
        conversion_value = np.random.exponential(150) + 50 if converted else 0
        
        for channel, timestamp in zip(path_channels, timestamps):
            customers.append({
                'customer_id': customer_id,
                'touchpoint': f"{channel.lower()}_touchpoint",
                'channel': channel,
                'timestamp': timestamp,
                'converted': converted,
                'conversion_value': conversion_value
            })
        
        customer_id += 1
    
    # Generate random paths (lower conversion rates)
    for _ in range(700):
        path_length = np.random.choice(range(1, 6), p=[0.2, 0.3, 0.25, 0.15, 0.1])
        path_channels = np.random.choice(channels, size=path_length, replace=False)
        
        # Generate timestamps
        start_date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 60))
        journey_span_hours = np.random.exponential(72)  # Average 3 days
        
        timestamps = []
        for i in range(path_length):
            time_offset = (i / max(path_length - 1, 1)) * journey_span_hours
            timestamp = start_date + pd.Timedelta(hours=time_offset + np.random.normal(0, 4))
            timestamps.append(timestamp)
        
        # Lower conversion probability for random paths
        converted = np.random.random() < 0.15
        conversion_value = np.random.exponential(100) + 30 if converted else 0
        
        for channel, timestamp in zip(path_channels, timestamps):
            customers.append({
                'customer_id': customer_id,
                'touchpoint': f"{channel.lower()}_touchpoint",
                'channel': channel,
                'timestamp': timestamp,
                'converted': converted,
                'conversion_value': conversion_value
            })
        
        customer_id += 1
    
    journey_data = pd.DataFrame(customers)
    
    print(f"📊 Generated {len(journey_data)} touchpoints across {journey_data['customer_id'].nunique()} customer journeys")
    print(f"📈 Overall conversion rate: {journey_data.groupby('customer_id')['converted'].first().mean():.1%}")
    print(f"💰 Total conversion value: ${journey_data.groupby('customer_id')['conversion_value'].first().sum():,.0f}")
    
    # Initialize and run path analysis
    analyzer = ConversionPathAnalyzer(
        min_pattern_frequency=8,
        max_path_length=15,
        time_window_days=90,
        value_threshold=0.0
    )
    
    print(f"\n🔍 Analyzing conversion paths and patterns...")
    analyzer.analyze_paths(journey_data)
    
    print("\n📋 CONVERSION PATH ANALYSIS RESULTS")
    print("=" * 55)
    
    # Top conversion patterns
    patterns_df = analyzer.get_path_patterns()
    print(f"\n🏆 Top Converting Path Patterns:")
    
    for i, (_, pattern) in enumerate(patterns_df.head(5).iterrows(), 1):
        conv_icon = "🔥" if pattern['conversion_rate'] > 0.5 else "📈" if pattern['conversion_rate'] > 0.3 else "📊"
        print(f"{conv_icon} {i}. {pattern['pattern']}")
        print(f"    Conversion Rate: {pattern['conversion_rate']:.1%} | Frequency: {pattern['frequency']} | Efficiency: {pattern['efficiency']:.2f}")
    
    # Channel performance in paths
    channel_perf = analyzer.get_channel_performance()
    print(f"\n📊 Channel Performance in Conversion Paths:")
    
    for _, row in channel_perf.head(6).iterrows():
        assist_icon = "🎯" if row['assist_rate'] > 0.3 else "🔗" if row['assist_rate'] > 0.1 else "📍"
        print(f"{assist_icon} {row['channel']:10}: {row['conversion_rate']:.1%} conv rate | "
              f"{row['first_touch_conversions']:2} first-touch | "
              f"{row['last_touch_conversions']:2} last-touch | {row['assist_rate']:.1%} assist")
    
    # Top performing individual paths
    top_paths = analyzer.get_top_conversion_paths(5)
    print(f"\n💎 Most Efficient Individual Conversion Paths:")
    
    for i, path in enumerate(top_paths, 1):
        efficiency_icon = "⚡" if path['efficiency'] > 5 else "💫" if path['efficiency'] > 2 else "⭐"
        print(f"{efficiency_icon} {i}. {path['pattern']}")
        print(f"    Value: ${path['conversion_value']:.0f} | Time: {path['time_to_conversion_hours']:.1f}h | Efficiency: {path['efficiency']:.2f}")
    
    # Funnel analysis insights
    if analyzer.funnel_analysis:
        funnel_eff = analyzer.funnel_analysis.get('funnel_efficiency', 0)
        print(f"\n📈 Funnel Analysis:")
        print(f"  • Overall Funnel Efficiency: {funnel_eff:.1%}")
        
        channel_funnels = analyzer.funnel_analysis.get('channel_funnels', {})
        if channel_funnels:
            best_funnel = max(channel_funnels.items(), key=lambda x: x[1]['conversion_rate'])
            worst_funnel = min(channel_funnels.items(), key=lambda x: x[1]['conversion_rate'])
            
            print(f"  • Best Channel Funnel: {best_funnel[0]} ({best_funnel[1]['conversion_rate']:.1%})")
            print(f"  • Improvement Needed: {worst_funnel[0]} ({worst_funnel[1]['conversion_rate']:.1%})")
    
    # Path prediction example
    print(f"\n🔮 Path Conversion Prediction Examples:")
    
    test_paths = [
        ['Display', 'Email', 'Search'],
        ['Social', 'Search', 'Direct'],
        ['Display', 'Social', 'Email', 'Direct'],
        ['Search'],
        ['Email', 'Direct']
    ]
    
    for path in test_paths:
        prob = analyzer.predict_conversion_probability(path)
        prob_icon = "🔥" if prob > 0.4 else "📈" if prob > 0.2 else "📊"
        print(f"{prob_icon} {' → '.join(path):25}: {prob:.1%} conversion probability")
    
    # Path recommendations
    print(f"\n💡 Path Continuation Recommendations:")
    current_path = ['Display', 'Email']
    recommendations = analyzer.generate_path_recommendations(current_path)
    
    print(f"Current Path: {' → '.join(current_path)}")
    print("Recommended Next Steps:")
    
    for i, rec in enumerate(recommendations[:3], 1):
        score_icon = "⭐" if rec['recommendation_score'] > 0.1 else "📌"
        print(f"{score_icon} {i}. {rec['next_channel']} (score: {rec['recommendation_score']:.3f})")
        print(f"    Transition Prob: {rec['transition_probability']:.1%} | Edge Conv: {rec['edge_conversion_rate']:.1%}")
    
    # Optimization opportunities
    if analyzer.optimization_opportunities:
        print(f"\n🚀 Strategic Optimization Opportunities:")
        
        for i, opp in enumerate(analyzer.optimization_opportunities[:3], 1):
            priority_icon = "🔴" if opp['priority'] == 'high' else "🟡" if opp['priority'] == 'medium' else "🟢"
            
            print(f"{priority_icon} {i}. {opp['description']}")
            if 'pattern' in opp:
                print(f"    Pattern: {opp['pattern']} | Impact: +{opp['expected_impact']:.0f} conversions")
            elif 'channel' in opp:
                print(f"    Channel: {opp['channel']} | Current Rate: {opp['current_rate']:.1%}")
    
    # Network insights
    if analyzer.path_graph:
        print(f"\n🕸️ Path Network Insights:")
        nodes = analyzer.path_graph.number_of_nodes()
        edges = analyzer.path_graph.number_of_edges()
        
        print(f"  • Network Size: {nodes} channels, {edges} transitions")
        
        # Find most connected channels
        degree_centrality = nx.degree_centrality(analyzer.path_graph)
        top_connected = max(degree_centrality.items(), key=lambda x: x[1])
        
        print(f"  • Most Connected Channel: {top_connected[0]} (centrality: {top_connected[1]:.2f})")
        
        # Entry and exit points
        entry_points = [ch for ch, data in analyzer.channel_transitions.items() if data.get('is_entry_point')]
        exit_points = [ch for ch, data in analyzer.channel_transitions.items() if data.get('is_exit_point')]
        
        if entry_points:
            print(f"  • Primary Entry Points: {', '.join(entry_points)}")
        if exit_points:
            print(f"  • Primary Exit Points: {', '.join(exit_points)}")
    
    print("\n" + "="*60)
    print("🚀 Advanced conversion path analysis for journey optimization")
    print("💼 Actionable insights for marketing funnel improvement")
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_conversion_path_analyzer()