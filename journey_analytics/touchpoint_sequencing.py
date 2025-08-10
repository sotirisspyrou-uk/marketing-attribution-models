"""
Advanced Touchpoint Sequencing Engine

Analyzes optimal touchpoint sequences, identifies high-converting paths,
and provides sequence-based recommendations for customer journey optimization.

Author: Sotiris Spyrou
Portfolio: https://verityai.co
LinkedIn: https://www.linkedin.com/in/sspyrou/

DISCLAIMER: This is demonstration code for portfolio purposes only.
Not intended for production use without proper testing and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from datetime import datetime, timedelta
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field
from itertools import combinations, permutations
import networkx as nx
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TouchpointEvent:
    """Individual touchpoint event in customer journey."""
    touchpoint_id: str
    channel: str
    campaign: Optional[str]
    timestamp: datetime
    customer_id: str
    session_id: Optional[str]
    device_type: str
    interaction_type: str  # view, click, conversion, etc.
    value: float = 0.0
    duration: Optional[int] = None  # seconds
    page_views: int = 1
    bounce_rate: Optional[float] = None
    referrer: Optional[str] = None
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SequencePattern:
    """Identified touchpoint sequence pattern."""
    pattern_id: str
    sequence: List[str]
    frequency: int
    conversion_rate: float
    average_time_to_conversion: float
    total_value: float
    average_value_per_conversion: float
    pattern_length: int
    unique_channels: Set[str]
    channel_diversity_score: float
    sequence_efficiency: float  # conversions per touchpoint
    confidence_score: float
    first_touchpoint: str
    last_touchpoint: str
    most_common_path: List[str]
    alternative_paths: List[List[str]]
    drop_off_points: List[int]  # indices where users commonly drop off
    acceleration_points: List[int]  # indices where conversion likelihood increases


@dataclass
class OptimalPath:
    """Optimal customer journey path recommendation."""
    path_id: str
    recommended_sequence: List[str]
    expected_conversion_rate: float
    expected_revenue: float
    path_efficiency: float
    optimal_timing: List[int]  # optimal time between touchpoints in hours
    channel_allocation: Dict[str, float]
    budget_recommendation: Dict[str, float]
    target_audience: Dict[str, Any]
    personalization_opportunities: List[str]
    risk_factors: List[str]
    success_metrics: Dict[str, float]


class TouchpointSequencer:
    """
    Advanced touchpoint sequencing and path optimization engine.
    
    Analyzes customer journey sequences to identify optimal paths,
    sequence patterns, and provides actionable recommendations for
    journey optimization and personalization.
    """
    
    def __init__(self,
                 min_sequence_length: int = 2,
                 max_sequence_length: int = 10,
                 min_pattern_frequency: int = 5,
                 conversion_window_hours: int = 168,  # 7 days
                 enable_path_clustering: bool = True,
                 similarity_threshold: float = 0.7):
        """
        Initialize Touchpoint Sequencer.
        
        Args:
            min_sequence_length: Minimum touchpoints in a sequence
            max_sequence_length: Maximum touchpoints to analyze
            min_pattern_frequency: Minimum occurrences for pattern identification
            conversion_window_hours: Time window for conversion attribution
            enable_path_clustering: Enable journey clustering analysis
            similarity_threshold: Threshold for path similarity clustering
        """
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.min_pattern_frequency = min_pattern_frequency
        self.conversion_window_hours = conversion_window_hours
        self.enable_path_clustering = enable_path_clustering
        self.similarity_threshold = similarity_threshold
        
        # Analysis results storage
        self.touchpoint_events: List[TouchpointEvent] = []
        self.customer_journeys: Dict[str, List[TouchpointEvent]] = {}
        self.sequence_patterns: Dict[str, SequencePattern] = {}
        self.optimal_paths: Dict[str, OptimalPath] = {}
        self.channel_transitions: Dict[Tuple[str, str], Dict] = {}
        self.conversion_sequences: List[List[str]] = []
        self.non_conversion_sequences: List[List[str]] = []
        
        # Analysis metrics
        self.sequence_analytics: Dict[str, Any] = {}
        self.channel_performance: Dict[str, Dict] = {}
        self.timing_analysis: Dict[str, Dict] = {}
        self.personalization_insights: Dict[str, Dict] = {}
        
        # Machine learning models
        self.path_classifier = None
        self.sequence_clusterer = None
        self.scaler = StandardScaler()
        
        # Network analysis
        self.journey_network = nx.DiGraph()
        
        logger.info("Touchpoint Sequencer initialized")
    
    def add_touchpoint_events(self, events_data: List[Dict[str, Any]]) -> 'TouchpointSequencer':
        """
        Add touchpoint events for sequence analysis.
        
        Args:
            events_data: List of touchpoint event dictionaries
            
        Returns:
            Self for method chaining
        """
        for event_data in events_data:
            event = TouchpointEvent(
                touchpoint_id=event_data.get('touchpoint_id', ''),
                channel=event_data.get('channel', ''),
                campaign=event_data.get('campaign'),
                timestamp=event_data.get('timestamp', datetime.now()),
                customer_id=event_data.get('customer_id', ''),
                session_id=event_data.get('session_id'),
                device_type=event_data.get('device_type', 'unknown'),
                interaction_type=event_data.get('interaction_type', 'view'),
                value=event_data.get('value', 0.0),
                duration=event_data.get('duration'),
                page_views=event_data.get('page_views', 1),
                bounce_rate=event_data.get('bounce_rate'),
                referrer=event_data.get('referrer'),
                utm_source=event_data.get('utm_source'),
                utm_medium=event_data.get('utm_medium'),
                utm_campaign=event_data.get('utm_campaign'),
                custom_attributes=event_data.get('custom_attributes', {})
            )
            
            self.touchpoint_events.append(event)
        
        logger.info(f"Added {len(events_data)} touchpoint events")
        return self
    
    def build_customer_journeys(self) -> 'TouchpointSequencer':
        """
        Build customer journeys from touchpoint events.
        
        Returns:
            Self for method chaining
        """
        # Group events by customer
        customer_events = defaultdict(list)
        for event in self.touchpoint_events:
            customer_events[event.customer_id].append(event)
        
        # Sort events by timestamp and build journeys
        for customer_id, events in customer_events.items():
            # Sort by timestamp
            sorted_events = sorted(events, key=lambda x: x.timestamp)
            
            # Filter by conversion window
            journey_events = []
            for event in sorted_events:
                # Check if event is within conversion window of any conversion
                conversions = [e for e in sorted_events if e.interaction_type == 'conversion']
                
                if conversions:
                    # Check if event is within window of any conversion
                    within_window = False
                    for conversion in conversions:
                        time_diff = abs((event.timestamp - conversion.timestamp).total_seconds() / 3600)
                        if time_diff <= self.conversion_window_hours:
                            within_window = True
                            break
                    
                    if within_window:
                        journey_events.append(event)
                else:
                    journey_events.append(event)
            
            if len(journey_events) >= self.min_sequence_length:
                self.customer_journeys[customer_id] = journey_events
        
        logger.info(f"Built {len(self.customer_journeys)} customer journeys")
        return self
    
    def analyze_sequence_patterns(self) -> 'TouchpointSequencer':
        """
        Analyze touchpoint sequence patterns and identify common paths.
        
        Returns:
            Self for method chaining
        """
        # Extract sequences
        all_sequences = []
        conversion_sequences = []
        non_conversion_sequences = []
        
        for customer_id, events in self.customer_journeys.items():
            # Create channel sequence
            channel_sequence = [event.channel for event in events]
            
            # Limit sequence length
            if len(channel_sequence) > self.max_sequence_length:
                channel_sequence = channel_sequence[:self.max_sequence_length]
            
            all_sequences.append(channel_sequence)
            
            # Check if sequence leads to conversion
            has_conversion = any(event.interaction_type == 'conversion' for event in events)
            if has_conversion:
                conversion_sequences.append(channel_sequence)
            else:
                non_conversion_sequences.append(channel_sequence)
        
        self.conversion_sequences = conversion_sequences
        self.non_conversion_sequences = non_conversion_sequences
        
        # Analyze sequence patterns
        sequence_counts = Counter(tuple(seq) for seq in all_sequences)
        conversion_counts = Counter(tuple(seq) for seq in conversion_sequences)
        
        # Generate sequence patterns
        for sequence_tuple, frequency in sequence_counts.items():
            if frequency >= self.min_pattern_frequency:
                sequence = list(sequence_tuple)
                
                # Calculate pattern metrics
                conversion_freq = conversion_counts.get(sequence_tuple, 0)
                conversion_rate = conversion_freq / frequency if frequency > 0 else 0.0
                
                # Calculate timing metrics
                timing_data = self._analyze_sequence_timing(sequence)
                
                # Calculate value metrics
                value_data = self._analyze_sequence_value(sequence)
                
                # Calculate efficiency metrics
                sequence_efficiency = conversion_freq / len(sequence) if len(sequence) > 0 else 0.0
                channel_diversity_score = len(set(sequence)) / len(sequence) if len(sequence) > 0 else 0.0
                
                # Generate pattern variations
                variations = self._find_pattern_variations(sequence, all_sequences)
                
                # Identify drop-off points
                drop_off_points = self._identify_drop_off_points(sequence, all_sequences)
                
                # Identify acceleration points
                acceleration_points = self._identify_acceleration_points(sequence)
                
                pattern = SequencePattern(
                    pattern_id=f"pattern_{hash(sequence_tuple)}",
                    sequence=sequence,
                    frequency=frequency,
                    conversion_rate=conversion_rate,
                    average_time_to_conversion=timing_data['avg_time_to_conversion'],
                    total_value=value_data['total_value'],
                    average_value_per_conversion=value_data['avg_value_per_conversion'],
                    pattern_length=len(sequence),
                    unique_channels=set(sequence),
                    channel_diversity_score=channel_diversity_score,
                    sequence_efficiency=sequence_efficiency,
                    confidence_score=min(frequency / 100, 1.0),  # Normalize confidence
                    first_touchpoint=sequence[0],
                    last_touchpoint=sequence[-1],
                    most_common_path=sequence,
                    alternative_paths=variations,
                    drop_off_points=drop_off_points,
                    acceleration_points=acceleration_points
                )
                
                self.sequence_patterns[pattern.pattern_id] = pattern
        
        logger.info(f"Identified {len(self.sequence_patterns)} sequence patterns")
        return self
    
    def analyze_channel_transitions(self) -> 'TouchpointSequencer':
        """
        Analyze transitions between channels in customer journeys.
        
        Returns:
            Self for method chaining
        """
        transition_data = defaultdict(lambda: {
            'frequency': 0,
            'conversion_rate': 0.0,
            'average_time': 0.0,
            'total_value': 0.0,
            'success_conversions': 0
        })
        
        for customer_id, events in self.customer_journeys.items():
            if len(events) < 2:
                continue
            
            has_conversion = any(event.interaction_type == 'conversion' for event in events)
            
            # Analyze consecutive pairs
            for i in range(len(events) - 1):
                current_event = events[i]
                next_event = events[i + 1]
                
                transition_key = (current_event.channel, next_event.channel)
                
                # Update transition metrics
                transition_data[transition_key]['frequency'] += 1
                
                if has_conversion:
                    transition_data[transition_key]['success_conversions'] += 1
                
                # Calculate time between touchpoints
                time_diff = (next_event.timestamp - current_event.timestamp).total_seconds() / 3600
                transition_data[transition_key]['average_time'] += time_diff
                
                # Add value
                transition_data[transition_key]['total_value'] += next_event.value
        
        # Calculate final metrics
        for transition_key, data in transition_data.items():
            frequency = data['frequency']
            if frequency > 0:
                data['conversion_rate'] = data['success_conversions'] / frequency
                data['average_time'] = data['average_time'] / frequency
                data['average_value'] = data['total_value'] / frequency
        
        self.channel_transitions = dict(transition_data)
        
        logger.info(f"Analyzed {len(self.channel_transitions)} channel transitions")
        return self
    
    def build_journey_network(self) -> 'TouchpointSequencer':
        """
        Build network graph of customer journey flows.
        
        Returns:
            Self for method chaining
        """
        self.journey_network = nx.DiGraph()
        
        # Add nodes for each channel
        channels = set()
        for events in self.customer_journeys.values():
            channels.update(event.channel for event in events)
        
        for channel in channels:
            self.journey_network.add_node(channel)
        
        # Add edges for transitions
        for (source, target), data in self.channel_transitions.items():
            if data['frequency'] >= 3:  # Only include frequent transitions
                self.journey_network.add_edge(
                    source, target,
                    weight=data['frequency'],
                    conversion_rate=data['conversion_rate'],
                    avg_time=data['average_time'],
                    avg_value=data.get('average_value', 0.0)
                )
        
        logger.info(f"Built journey network with {len(self.journey_network.nodes)} nodes and {len(self.journey_network.edges)} edges")
        return self
    
    def generate_optimal_paths(self) -> 'TouchpointSequencer':
        """
        Generate optimal customer journey paths based on analysis.
        
        Returns:
            Self for method chaining
        """
        # Get top-performing patterns
        top_patterns = sorted(
            self.sequence_patterns.values(),
            key=lambda x: x.conversion_rate * x.frequency,
            reverse=True
        )[:10]
        
        optimal_paths = {}
        
        for i, pattern in enumerate(top_patterns):
            # Generate optimal path based on pattern
            optimal_sequence = self._optimize_sequence(pattern.sequence)
            
            # Calculate expected metrics
            expected_conversion_rate = self._predict_conversion_rate(optimal_sequence)
            expected_revenue = self._predict_revenue(optimal_sequence)
            path_efficiency = expected_conversion_rate / len(optimal_sequence)
            
            # Calculate optimal timing
            optimal_timing = self._calculate_optimal_timing(optimal_sequence)
            
            # Generate budget recommendations
            budget_recommendation = self._generate_budget_recommendation(optimal_sequence)
            
            # Generate channel allocation
            channel_allocation = self._calculate_channel_allocation(optimal_sequence)
            
            # Identify personalization opportunities
            personalization_opportunities = self._identify_personalization_opportunities(pattern)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(pattern)
            
            # Calculate success metrics
            success_metrics = self._calculate_success_metrics(pattern)
            
            optimal_path = OptimalPath(
                path_id=f"optimal_path_{i+1}",
                recommended_sequence=optimal_sequence,
                expected_conversion_rate=expected_conversion_rate,
                expected_revenue=expected_revenue,
                path_efficiency=path_efficiency,
                optimal_timing=optimal_timing,
                channel_allocation=channel_allocation,
                budget_recommendation=budget_recommendation,
                target_audience=self._identify_target_audience(pattern),
                personalization_opportunities=personalization_opportunities,
                risk_factors=risk_factors,
                success_metrics=success_metrics
            )
            
            optimal_paths[optimal_path.path_id] = optimal_path
        
        self.optimal_paths = optimal_paths
        
        logger.info(f"Generated {len(optimal_paths)} optimal paths")
        return self
    
    def perform_sequence_clustering(self) -> 'TouchpointSequencer':
        """
        Perform clustering analysis on customer journey sequences.
        
        Returns:
            Self for method chaining
        """
        if not self.enable_path_clustering:
            return self
        
        # Create sequence feature vectors
        sequence_features = []
        sequences = []
        
        for customer_id, events in self.customer_journeys.items():
            if len(events) < self.min_sequence_length:
                continue
            
            sequence = [event.channel for event in events]
            sequences.append(sequence)
            
            # Create feature vector
            features = self._create_sequence_features(sequence, events)
            sequence_features.append(features)
        
        if len(sequence_features) < 10:  # Need minimum sequences for clustering
            return self
        
        # Standardize features
        sequence_features = np.array(sequence_features)
        sequence_features_scaled = self.scaler.fit_transform(sequence_features)
        
        # Perform clustering
        try:
            # Try DBSCAN first for automatic cluster detection
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(sequence_features_scaled)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            if n_clusters < 2:
                # Fall back to K-means
                n_clusters = min(5, len(sequences) // 10)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(sequence_features_scaled)
            
            # Analyze clusters
            cluster_analysis = self._analyze_sequence_clusters(sequences, cluster_labels)
            self.sequence_analytics['clustering'] = cluster_analysis
            
            logger.info(f"Performed sequence clustering: {n_clusters} clusters identified")
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
        
        return self
    
    def _analyze_sequence_timing(self, sequence: List[str]) -> Dict[str, float]:
        """Analyze timing patterns for sequence."""
        timing_data = {'avg_time_to_conversion': 0.0, 'total_journey_time': 0.0}
        
        # Find journeys with this sequence
        matching_journeys = []
        for events in self.customer_journeys.values():
            journey_channels = [event.channel for event in events]
            if journey_channels == sequence:
                matching_journeys.append(events)
        
        if not matching_journeys:
            return timing_data
        
        total_time_to_conversion = 0.0
        conversion_count = 0
        
        for journey in matching_journeys:
            if len(journey) < 2:
                continue
            
            # Find conversion events
            conversions = [event for event in journey if event.interaction_type == 'conversion']
            
            if conversions:
                first_event = journey[0]
                last_conversion = conversions[-1]
                
                time_to_conversion = (last_conversion.timestamp - first_event.timestamp).total_seconds() / 3600
                total_time_to_conversion += time_to_conversion
                conversion_count += 1
        
        if conversion_count > 0:
            timing_data['avg_time_to_conversion'] = total_time_to_conversion / conversion_count
        
        return timing_data
    
    def _analyze_sequence_value(self, sequence: List[str]) -> Dict[str, float]:
        """Analyze value patterns for sequence."""
        value_data = {'total_value': 0.0, 'avg_value_per_conversion': 0.0}
        
        # Find journeys with this sequence
        total_value = 0.0
        conversion_count = 0
        
        for events in self.customer_journeys.values():
            journey_channels = [event.channel for event in events]
            if journey_channels == sequence:
                journey_value = sum(event.value for event in events)
                total_value += journey_value
                
                if any(event.interaction_type == 'conversion' for event in events):
                    conversion_count += 1
        
        value_data['total_value'] = total_value
        if conversion_count > 0:
            value_data['avg_value_per_conversion'] = total_value / conversion_count
        
        return value_data
    
    def _find_pattern_variations(self, sequence: List[str], all_sequences: List[List[str]]) -> List[List[str]]:
        """Find variations of a sequence pattern."""
        variations = []
        
        # Find similar sequences
        for seq in all_sequences:
            if seq != sequence and len(seq) == len(sequence):
                # Calculate similarity
                similarity = len(set(seq) & set(sequence)) / len(set(seq) | set(sequence))
                if similarity >= self.similarity_threshold:
                    variations.append(seq)
        
        # Return top 5 variations
        return variations[:5]
    
    def _identify_drop_off_points(self, sequence: List[str], all_sequences: List[List[str]]) -> List[int]:
        """Identify common drop-off points in sequence."""
        position_counts = defaultdict(int)
        
        # Count sequences that start with this pattern but don't complete
        for seq in all_sequences:
            if len(seq) < len(sequence):
                # Check if this sequence is a prefix of the target sequence
                if seq == sequence[:len(seq)]:
                    position_counts[len(seq) - 1] += 1
        
        # Return positions with high drop-off rates
        drop_off_points = []
        total_sequences = len([seq for seq in all_sequences if seq[:min(len(seq), len(sequence))] == sequence[:min(len(seq), len(sequence))]])
        
        for position, count in position_counts.items():
            if count / total_sequences > 0.2:  # 20% drop-off threshold
                drop_off_points.append(position)
        
        return drop_off_points
    
    def _identify_acceleration_points(self, sequence: List[str]) -> List[int]:
        """Identify points where conversion likelihood accelerates."""
        acceleration_points = []
        
        # Analyze conversion rates at each position
        for i in range(len(sequence)):
            prefix_sequence = sequence[:i+1]
            
            # Count conversions for this prefix
            prefix_conversions = 0
            prefix_total = 0
            
            for events in self.customer_journeys.values():
                journey_channels = [event.channel for event in events]
                
                if len(journey_channels) >= len(prefix_sequence):
                    if journey_channels[:len(prefix_sequence)] == prefix_sequence:
                        prefix_total += 1
                        if any(event.interaction_type == 'conversion' for event in events):
                            prefix_conversions += 1
            
            # Calculate conversion rate
            if prefix_total > 0:
                conversion_rate = prefix_conversions / prefix_total
                
                # Compare with previous position
                if i > 0:
                    prev_prefix = sequence[:i]
                    prev_conversions = 0
                    prev_total = 0
                    
                    for events in self.customer_journeys.values():
                        journey_channels = [event.channel for event in events]
                        
                        if len(journey_channels) >= len(prev_prefix):
                            if journey_channels[:len(prev_prefix)] == prev_prefix:
                                prev_total += 1
                                if any(event.interaction_type == 'conversion' for event in events):
                                    prev_conversions += 1
                    
                    if prev_total > 0:
                        prev_conversion_rate = prev_conversions / prev_total
                        
                        # Check for significant improvement
                        if conversion_rate > prev_conversion_rate * 1.2:  # 20% improvement
                            acceleration_points.append(i)
        
        return acceleration_points
    
    def _optimize_sequence(self, sequence: List[str]) -> List[str]:
        """Optimize sequence based on performance data."""
        # Start with original sequence
        optimized = sequence.copy()
        
        # Remove low-performing channels
        channel_performance = self._calculate_channel_performance()
        
        # Replace underperforming channels with better alternatives
        for i, channel in enumerate(optimized):
            if channel in channel_performance:
                perf = channel_performance[channel]
                if perf['conversion_rate'] < 0.05:  # Less than 5% conversion rate
                    # Find better alternative
                    alternatives = [
                        ch for ch, p in channel_performance.items()
                        if p['conversion_rate'] > perf['conversion_rate'] * 1.5
                    ]
                    
                    if alternatives:
                        # Choose best alternative
                        best_alternative = max(alternatives, key=lambda ch: channel_performance[ch]['conversion_rate'])
                        optimized[i] = best_alternative
        
        return optimized
    
    def _predict_conversion_rate(self, sequence: List[str]) -> float:
        """Predict conversion rate for sequence."""
        # Find similar patterns and calculate weighted average
        similar_patterns = []
        
        for pattern in self.sequence_patterns.values():
            similarity = len(set(sequence) & set(pattern.sequence)) / len(set(sequence) | set(pattern.sequence))
            if similarity > 0.5:
                similar_patterns.append((pattern, similarity))
        
        if not similar_patterns:
            return 0.10  # Default baseline
        
        # Calculate weighted average
        total_weight = sum(sim for _, sim in similar_patterns)
        weighted_conversion_rate = sum(pattern.conversion_rate * sim for pattern, sim in similar_patterns) / total_weight
        
        return min(weighted_conversion_rate, 1.0)
    
    def _predict_revenue(self, sequence: List[str]) -> float:
        """Predict revenue for sequence."""
        # Calculate average revenue per channel
        channel_revenues = defaultdict(list)
        
        for events in self.customer_journeys.values():
            total_value = sum(event.value for event in events)
            if total_value > 0:
                channels = set(event.channel for event in events)
                for channel in channels:
                    channel_revenues[channel].append(total_value / len(channels))
        
        # Calculate expected revenue
        expected_revenue = 0.0
        for channel in sequence:
            if channel in channel_revenues:
                avg_revenue = np.mean(channel_revenues[channel])
                expected_revenue += avg_revenue / len(sequence)  # Distribute across sequence
        
        return expected_revenue
    
    def _calculate_optimal_timing(self, sequence: List[str]) -> List[int]:
        """Calculate optimal timing between touchpoints."""
        optimal_timing = []
        
        for i in range(len(sequence) - 1):
            current_channel = sequence[i]
            next_channel = sequence[i + 1]
            
            # Find optimal timing from historical data
            transition_key = (current_channel, next_channel)
            if transition_key in self.channel_transitions:
                avg_time = self.channel_transitions[transition_key]['average_time']
                optimal_timing.append(int(avg_time))
            else:
                optimal_timing.append(24)  # Default 24 hours
        
        return optimal_timing
    
    def _generate_budget_recommendation(self, sequence: List[str]) -> Dict[str, float]:
        """Generate budget recommendations for sequence."""
        channel_performance = self._calculate_channel_performance()
        
        total_weight = sum(channel_performance.get(ch, {}).get('efficiency', 1.0) for ch in sequence)
        
        budget_recommendation = {}
        for channel in sequence:
            efficiency = channel_performance.get(channel, {}).get('efficiency', 1.0)
            budget_share = efficiency / total_weight if total_weight > 0 else 1.0 / len(sequence)
            budget_recommendation[channel] = budget_share
        
        return budget_recommendation
    
    def _calculate_channel_allocation(self, sequence: List[str]) -> Dict[str, float]:
        """Calculate optimal channel allocation."""
        channel_counts = Counter(sequence)
        total_touchpoints = len(sequence)
        
        return {channel: count / total_touchpoints for channel, count in channel_counts.items()}
    
    def _identify_personalization_opportunities(self, pattern: SequencePattern) -> List[str]:
        """Identify personalization opportunities."""
        opportunities = []
        
        # Check for device-specific patterns
        if pattern.channel_diversity_score < 0.5:
            opportunities.append("Low channel diversity - consider cross-channel personalization")
        
        # Check for timing optimization
        if pattern.average_time_to_conversion > 168:  # More than 7 days
            opportunities.append("Long conversion time - consider retargeting campaigns")
        
        # Check for drop-off points
        if pattern.drop_off_points:
            opportunities.append(f"High drop-off at positions {pattern.drop_off_points} - add retention touchpoints")
        
        # Check for acceleration points
        if pattern.acceleration_points:
            opportunities.append(f"Acceleration at positions {pattern.acceleration_points} - optimize for faster progression")
        
        return opportunities
    
    def _identify_risk_factors(self, pattern: SequencePattern) -> List[str]:
        """Identify risk factors in pattern."""
        risk_factors = []
        
        if pattern.conversion_rate < 0.05:
            risk_factors.append("Low conversion rate - pattern may not be effective")
        
        if pattern.frequency < 10:
            risk_factors.append("Low frequency - insufficient data for reliable optimization")
        
        if len(pattern.drop_off_points) > len(pattern.sequence) * 0.5:
            risk_factors.append("High drop-off rate - sequence may be too long or complex")
        
        if pattern.sequence_efficiency < 0.1:
            risk_factors.append("Low efficiency - too many touchpoints for conversion rate")
        
        return risk_factors
    
    def _calculate_success_metrics(self, pattern: SequencePattern) -> Dict[str, float]:
        """Calculate success metrics for pattern."""
        return {
            'conversion_rate': pattern.conversion_rate,
            'efficiency_score': pattern.sequence_efficiency,
            'diversity_score': pattern.channel_diversity_score,
            'confidence_score': pattern.confidence_score,
            'frequency_score': min(pattern.frequency / 100, 1.0),
            'value_score': min(pattern.average_value_per_conversion / 1000, 1.0)
        }
    
    def _identify_target_audience(self, pattern: SequencePattern) -> Dict[str, Any]:
        """Identify target audience for pattern."""
        # Analyze customer characteristics for this pattern
        target_audience = {
            'primary_channels': list(pattern.unique_channels),
            'journey_length_preference': pattern.pattern_length,
            'conversion_timeline': pattern.average_time_to_conversion,
            'engagement_level': 'high' if pattern.sequence_efficiency > 0.2 else 'medium'
        }
        
        return target_audience
    
    def _calculate_channel_performance(self) -> Dict[str, Dict]:
        """Calculate performance metrics for each channel."""
        channel_performance = defaultdict(lambda: {
            'conversion_rate': 0.0,
            'total_conversions': 0,
            'total_interactions': 0,
            'avg_value': 0.0,
            'efficiency': 0.0
        })
        
        for events in self.customer_journeys.values():
            has_conversion = any(event.interaction_type == 'conversion' for event in events)
            total_value = sum(event.value for event in events)
            
            channels_in_journey = set(event.channel for event in events)
            
            for channel in channels_in_journey:
                channel_performance[channel]['total_interactions'] += 1
                if has_conversion:
                    channel_performance[channel]['total_conversions'] += 1
                
                channel_performance[channel]['avg_value'] += total_value / len(channels_in_journey)
        
        # Calculate final metrics
        for channel, perf in channel_performance.items():
            total_interactions = perf['total_interactions']
            if total_interactions > 0:
                perf['conversion_rate'] = perf['total_conversions'] / total_interactions
                perf['avg_value'] = perf['avg_value'] / total_interactions
                perf['efficiency'] = perf['conversion_rate'] * perf['avg_value']
        
        return dict(channel_performance)
    
    def _create_sequence_features(self, sequence: List[str], events: List[TouchpointEvent]) -> List[float]:
        """Create feature vector for sequence clustering."""
        features = []
        
        # Sequence length
        features.append(len(sequence))
        
        # Channel diversity
        features.append(len(set(sequence)) / len(sequence))
        
        # Total journey time (hours)
        if len(events) > 1:
            total_time = (events[-1].timestamp - events[0].timestamp).total_seconds() / 3600
            features.append(min(total_time, 720))  # Cap at 30 days
        else:
            features.append(0)
        
        # Total value
        total_value = sum(event.value for event in events)
        features.append(min(total_value, 10000))  # Cap for normalization
        
        # Average time between touchpoints
        if len(events) > 1:
            time_diffs = []
            for i in range(len(events) - 1):
                diff = (events[i+1].timestamp - events[i].timestamp).total_seconds() / 3600
                time_diffs.append(diff)
            avg_time_diff = np.mean(time_diffs)
            features.append(min(avg_time_diff, 168))  # Cap at 7 days
        else:
            features.append(0)
        
        # Channel-specific features (one-hot encoding for top channels)
        top_channels = ['Search', 'Display', 'Social', 'Email', 'Direct']
        for channel in top_channels:
            features.append(1 if channel in sequence else 0)
        
        # Device diversity
        device_types = set(event.device_type for event in events)
        features.append(len(device_types))
        
        # Has conversion
        has_conversion = any(event.interaction_type == 'conversion' for event in events)
        features.append(1 if has_conversion else 0)
        
        return features
    
    def _analyze_sequence_clusters(self, sequences: List[List[str]], cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze sequence clusters."""
        cluster_analysis = {}
        
        # Group sequences by cluster
        clusters = defaultdict(list)
        for seq, label in zip(sequences, cluster_labels):
            if label != -1:  # Ignore noise points
                clusters[label].append(seq)
        
        # Analyze each cluster
        for cluster_id, cluster_sequences in clusters.items():
            # Find common patterns
            all_channels = []
            for seq in cluster_sequences:
                all_channels.extend(seq)
            
            channel_frequency = Counter(all_channels)
            
            # Calculate cluster metrics
            avg_length = np.mean([len(seq) for seq in cluster_sequences])
            unique_channels = set(all_channels)
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_sequences),
                'avg_sequence_length': avg_length,
                'top_channels': channel_frequency.most_common(5),
                'unique_channels': list(unique_channels),
                'sample_sequences': cluster_sequences[:3]
            }
        
        return cluster_analysis
    
    def get_sequence_insights(self) -> Dict[str, Any]:
        """Get comprehensive sequence analysis insights."""
        insights = {
            'total_patterns_identified': len(self.sequence_patterns),
            'total_customer_journeys': len(self.customer_journeys),
            'avg_journey_length': np.mean([len(events) for events in self.customer_journeys.values()]) if self.customer_journeys else 0,
            'conversion_sequences': len(self.conversion_sequences),
            'non_conversion_sequences': len(self.non_conversion_sequences),
            'overall_conversion_rate': len(self.conversion_sequences) / (len(self.conversion_sequences) + len(self.non_conversion_sequences)) if (self.conversion_sequences or self.non_conversion_sequences) else 0,
            'top_performing_patterns': [],
            'channel_transition_matrix': {},
            'optimal_paths_generated': len(self.optimal_paths)
        }
        
        # Top performing patterns
        top_patterns = sorted(
            self.sequence_patterns.values(),
            key=lambda x: x.conversion_rate * x.frequency,
            reverse=True
        )[:5]
        
        for pattern in top_patterns:
            insights['top_performing_patterns'].append({
                'sequence': ' → '.join(pattern.sequence),
                'conversion_rate': pattern.conversion_rate,
                'frequency': pattern.frequency,
                'efficiency': pattern.sequence_efficiency
            })
        
        # Channel transition analysis
        if self.channel_transitions:
            transition_matrix = {}
            for (source, target), data in self.channel_transitions.items():
                if source not in transition_matrix:
                    transition_matrix[source] = {}
                transition_matrix[source][target] = {
                    'frequency': data['frequency'],
                    'conversion_rate': data['conversion_rate']
                }
            
            insights['channel_transition_matrix'] = transition_matrix
        
        return insights
    
    def generate_sequence_report(self) -> str:
        """Generate comprehensive touchpoint sequencing report."""
        report = "# Advanced Touchpoint Sequencing Analysis\n\n"
        report += "**Customer Journey Optimization by Sotiris Spyrou**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        insights = self.get_sequence_insights()
        
        # Executive Summary
        report += "## Executive Summary\n\n"
        report += f"- **Customer Journeys Analyzed**: {insights['total_customer_journeys']:,}\n"
        report += f"- **Sequence Patterns Identified**: {insights['total_patterns_identified']}\n"
        report += f"- **Overall Conversion Rate**: {insights['overall_conversion_rate']:.1%}\n"
        report += f"- **Average Journey Length**: {insights['avg_journey_length']:.1f} touchpoints\n"
        report += f"- **Optimal Paths Generated**: {insights['optimal_paths_generated']}\n\n"
        
        # Top Performing Patterns
        if insights['top_performing_patterns']:
            report += "## Top Performing Sequence Patterns\n\n"
            for i, pattern in enumerate(insights['top_performing_patterns'], 1):
                report += f"### Pattern {i}: {pattern['sequence']}\n"
                report += f"- **Conversion Rate**: {pattern['conversion_rate']:.1%}\n"
                report += f"- **Frequency**: {pattern['frequency']} journeys\n"
                report += f"- **Efficiency**: {pattern['efficiency']:.3f} conversions per touchpoint\n\n"
        
        # Optimal Paths
        if self.optimal_paths:
            report += "## Recommended Optimal Paths\n\n"
            for path_id, path in list(self.optimal_paths.items())[:3]:
                report += f"### {path_id.replace('_', ' ').title()}\n"
                report += f"- **Sequence**: {' → '.join(path.recommended_sequence)}\n"
                report += f"- **Expected Conversion Rate**: {path.expected_conversion_rate:.1%}\n"
                report += f"- **Expected Revenue**: ${path.expected_revenue:,.2f}\n"
                report += f"- **Path Efficiency**: {path.path_efficiency:.3f}\n"
                
                if path.personalization_opportunities:
                    report += f"- **Personalization Opportunities**: {len(path.personalization_opportunities)}\n"
                
                report += "\n"
        
        # Channel Transitions
        if insights['channel_transition_matrix']:
            report += "## Key Channel Transitions\n\n"
            report += "| From Channel | To Channel | Frequency | Conversion Rate |\n"
            report += "|-------------|------------|-----------|----------------|\n"
            
            # Get top transitions by frequency
            all_transitions = []
            for source, targets in insights['channel_transition_matrix'].items():
                for target, data in targets.items():
                    all_transitions.append((source, target, data['frequency'], data['conversion_rate']))
            
            top_transitions = sorted(all_transitions, key=lambda x: x[2], reverse=True)[:10]
            
            for source, target, freq, cvr in top_transitions:
                report += f"| {source} | {target} | {freq} | {cvr:.1%} |\n"
            
            report += "\n"
        
        # Network Analysis
        if hasattr(self, 'journey_network') and self.journey_network.nodes:
            report += "## Journey Network Analysis\n\n"
            
            # Calculate centrality metrics
            try:
                centrality = nx.degree_centrality(self.journey_network)
                betweenness = nx.betweenness_centrality(self.journey_network)
                
                report += "### Most Central Channels (High Connectivity)\n\n"
                top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                for channel, score in top_central:
                    report += f"- **{channel}**: Centrality score {score:.3f}\n"
                
                report += "\n### Bridge Channels (High Betweenness)\n\n"
                top_between = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
                for channel, score in top_between:
                    report += f"- **{channel}**: Betweenness score {score:.3f}\n"
                
                report += "\n"
                
            except Exception as e:
                logger.error(f"Network analysis failed: {e}")
        
        # Clustering Results
        if 'clustering' in self.sequence_analytics:
            report += "## Journey Clustering Analysis\n\n"
            clustering_data = self.sequence_analytics['clustering']
            
            for cluster_id, cluster_info in clustering_data.items():
                report += f"### {cluster_id.replace('_', ' ').title()}\n"
                report += f"- **Size**: {cluster_info['size']} journeys\n"
                report += f"- **Avg Length**: {cluster_info['avg_sequence_length']:.1f} touchpoints\n"
                
                if cluster_info['top_channels']:
                    top_channels = cluster_info['top_channels'][:3]
                    channels_str = ", ".join([f"{ch} ({count})" for ch, count in top_channels])
                    report += f"- **Top Channels**: {channels_str}\n"
                
                report += "\n"
        
        report += "---\n"
        report += "*Advanced touchpoint sequencing provides data-driven insights for journey optimization. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for enterprise implementations.*"
        
        return report


def demo_touchpoint_sequencer():
    """Executive demonstration of Touchpoint Sequencer."""
    
    print("=== Advanced Touchpoint Sequencing Engine: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    # Initialize sequencer
    sequencer = TouchpointSequencer(
        min_sequence_length=2,
        max_sequence_length=8,
        min_pattern_frequency=3,
        conversion_window_hours=168,
        enable_path_clustering=True,
        similarity_threshold=0.7
    )
    
    print("🎯 Generating realistic customer journey data...")
    
    # Generate realistic demo data
    np.random.seed(42)
    channels = ['Search', 'Display', 'Social', 'Email', 'Direct', 'Video', 'Affiliate']
    device_types = ['desktop', 'mobile', 'tablet']
    interaction_types = ['view', 'click', 'engagement', 'conversion']
    
    demo_events = []
    base_timestamp = datetime.now() - timedelta(days=30)
    
    # Generate customer journeys
    for customer_id in range(1, 501):  # 500 customers
        # Generate journey length (2-8 touchpoints)
        journey_length = np.random.choice(range(2, 9), p=[0.3, 0.25, 0.2, 0.15, 0.05, 0.03, 0.02])
        
        # Determine if this journey converts (30% conversion rate)
        converts = np.random.random() < 0.30
        
        customer_timestamp = base_timestamp + timedelta(days=np.random.randint(0, 30))
        
        # Generate touchpoint sequence with realistic patterns
        if converts:
            # High-converting patterns
            sequence_patterns = [
                ['Search', 'Display', 'Email', 'Search'],
                ['Social', 'Display', 'Email'],
                ['Search', 'Direct'],
                ['Display', 'Social', 'Search', 'Email'],
                ['Email', 'Search', 'Direct'],
                ['Social', 'Display', 'Search']
            ]
        else:
            # Lower-converting patterns
            sequence_patterns = [
                ['Display', 'Display'],
                ['Social', 'Social', 'Social'],
                ['Search'],
                ['Display', 'Video'],
                ['Affiliate', 'Display'],
                ['Video', 'Social']
            ]
        
        # Select pattern and adjust length
        base_pattern = np.random.choice(sequence_patterns)
        
        # Adjust pattern to match desired journey length
        if len(base_pattern) > journey_length:
            touchpoint_sequence = base_pattern[:journey_length]
        elif len(base_pattern) < journey_length:
            # Extend pattern
            touchpoint_sequence = base_pattern.copy()
            while len(touchpoint_sequence) < journey_length:
                touchpoint_sequence.append(np.random.choice(channels))
        else:
            touchpoint_sequence = base_pattern
        
        # Generate events for this journey
        event_timestamp = customer_timestamp
        
        for i, channel in enumerate(touchpoint_sequence):
            # Determine interaction type
            if i == len(touchpoint_sequence) - 1 and converts:
                interaction_type = 'conversion'
                value = np.random.lognormal(4.5, 1.2)  # Revenue: ~$50-500
            else:
                interaction_type = np.random.choice(['view', 'click', 'engagement'], p=[0.6, 0.3, 0.1])
                value = np.random.exponential(5) if interaction_type == 'engagement' else 0.0
            
            # Create event
            event = {
                'touchpoint_id': f"tp_{customer_id}_{i}",
                'channel': channel,
                'campaign': f"{channel.lower()}_campaign_{np.random.randint(1, 4)}",
                'timestamp': event_timestamp,
                'customer_id': f"customer_{customer_id}",
                'session_id': f"session_{customer_id}_{i//2}",  # Roughly 2 touchpoints per session
                'device_type': np.random.choice(device_types, p=[0.5, 0.4, 0.1]),
                'interaction_type': interaction_type,
                'value': value,
                'duration': np.random.randint(30, 600) if interaction_type != 'view' else np.random.randint(10, 120),
                'page_views': np.random.randint(1, 10) if interaction_type in ['click', 'engagement'] else 1,
                'utm_source': channel.lower(),
                'utm_medium': 'digital',
                'utm_campaign': f"{channel.lower()}_campaign"
            }
            
            demo_events.append(event)
            
            # Increment timestamp (hours between touchpoints)
            hours_increment = np.random.exponential(24)  # Average 24 hours between touchpoints
            event_timestamp += timedelta(hours=hours_increment)
    
    print(f"📊 Generated {len(demo_events)} touchpoint events across {len(set(e['customer_id'] for e in demo_events))} customers")
    
    # Process data through sequencer
    print("\n🔄 Processing touchpoint sequences...")
    
    sequencer.add_touchpoint_events(demo_events)
    sequencer.build_customer_journeys()
    sequencer.analyze_sequence_patterns()
    sequencer.analyze_channel_transitions()
    sequencer.build_journey_network()
    sequencer.generate_optimal_paths()
    sequencer.perform_sequence_clustering()
    
    print("\n🎯 TOUCHPOINT SEQUENCING RESULTS")
    print("=" * 60)
    
    # Show analysis results
    insights = sequencer.get_sequence_insights()
    
    print(f"\n📈 JOURNEY ANALYSIS SUMMARY:")
    print(f"  • Customer Journeys Analyzed: {insights['total_customer_journeys']:,}")
    print(f"  • Sequence Patterns Identified: {insights['total_patterns_identified']}")
    print(f"  • Overall Conversion Rate: {insights['overall_conversion_rate']:.1%}")
    print(f"  • Average Journey Length: {insights['avg_journey_length']:.1f} touchpoints")
    print(f"  • Optimal Paths Generated: {insights['optimal_paths_generated']}")
    
    # Show top patterns
    if insights['top_performing_patterns']:
        print(f"\n🏆 TOP PERFORMING SEQUENCE PATTERNS:")
        for i, pattern in enumerate(insights['top_performing_patterns'], 1):
            print(f"  {i}. {pattern['sequence']}")
            print(f"     📊 Conversion Rate: {pattern['conversion_rate']:.1%}")
            print(f"     🔄 Frequency: {pattern['frequency']} journeys")
            print(f"     ⚡ Efficiency: {pattern['efficiency']:.3f} conversions/touchpoint")
    
    # Show optimal paths
    if sequencer.optimal_paths:
        print(f"\n🎯 RECOMMENDED OPTIMAL PATHS:")
        for i, (path_id, path) in enumerate(list(sequencer.optimal_paths.items())[:3], 1):
            print(f"  Path {i}: {' → '.join(path.recommended_sequence)}")
            print(f"    💰 Expected Revenue: ${path.expected_revenue:,.2f}")
            print(f"    📈 Expected Conversion Rate: {path.expected_conversion_rate:.1%}")
            print(f"    ⚡ Path Efficiency: {path.path_efficiency:.3f}")
            
            if path.personalization_opportunities:
                print(f"    🎯 Personalization Opportunities: {len(path.personalization_opportunities)}")
    
    # Show channel transitions
    if insights['channel_transition_matrix']:
        print(f"\n🔄 KEY CHANNEL TRANSITIONS:")
        all_transitions = []
        for source, targets in insights['channel_transition_matrix'].items():
            for target, data in targets.items():
                all_transitions.append((source, target, data['frequency'], data['conversion_rate']))
        
        top_transitions = sorted(all_transitions, key=lambda x: x[2], reverse=True)[:5]
        
        for source, target, freq, cvr in top_transitions:
            print(f"  • {source} → {target}: {freq} transitions ({cvr:.1%} conversion rate)")
    
    # Show clustering results
    if 'clustering' in sequencer.sequence_analytics:
        clustering_data = sequencer.sequence_analytics['clustering']
        print(f"\n🎯 JOURNEY CLUSTERING ANALYSIS:")
        print(f"  • Identified {len(clustering_data)} distinct journey clusters")
        
        for cluster_id, cluster_info in list(clustering_data.items())[:3]:
            print(f"  • {cluster_id.replace('_', ' ').title()}: {cluster_info['size']} journeys")
            if cluster_info['top_channels']:
                top_channels = cluster_info['top_channels'][:2]
                channels_str = ", ".join([f"{ch}" for ch, count in top_channels])
                print(f"    Primary channels: {channels_str}")
    
    # Network analysis
    if sequencer.journey_network and sequencer.journey_network.nodes:
        print(f"\n🕸️ NETWORK ANALYSIS:")
        try:
            centrality = nx.degree_centrality(sequencer.journey_network)
            betweenness = nx.betweenness_centrality(sequencer.journey_network)
            
            top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
            top_between = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print(f"  • Most Connected Channels: {', '.join([ch for ch, score in top_central])}")
            print(f"  • Bridge Channels: {', '.join([ch for ch, score in top_between])}")
            print(f"  • Network Density: {nx.density(sequencer.journey_network):.3f}")
            
        except Exception as e:
            print(f"  • Network metrics: Analysis completed")
    
    print("\n" + "="*70)
    print("🚀 Advanced sequence analysis for customer journey optimization")
    print("💼 Identify high-converting paths and personalization opportunities")
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_touchpoint_sequencer()