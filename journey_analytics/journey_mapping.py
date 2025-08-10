"""
Customer Journey Mapping and Analysis System

Advanced customer journey mapping system that visualizes complete customer paths,
identifies optimal journey stages, and provides actionable insights for journey
orchestration and experience optimization.

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
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import networkx as nx
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)


class JourneyStage(Enum):
    """Customer journey stages."""
    AWARENESS = "awareness"
    CONSIDERATION = "consideration" 
    INTENT = "intent"
    PURCHASE = "purchase"
    RETENTION = "retention"
    ADVOCACY = "advocacy"


class TouchpointType(Enum):
    """Types of marketing touchpoints."""
    PAID_SEARCH = "paid_search"
    ORGANIC_SEARCH = "organic_search"
    DISPLAY_AD = "display_ad"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    DIRECT = "direct"
    REFERRAL = "referral"
    OFFLINE = "offline"
    MOBILE_APP = "mobile_app"
    VIDEO_AD = "video_ad"


@dataclass
class CustomerTouchpoint:
    """Individual customer touchpoint data."""
    customer_id: str
    touchpoint_id: str
    touchpoint_type: TouchpointType
    timestamp: datetime
    channel: str
    campaign: Optional[str] = None
    content: Optional[str] = None
    device: Optional[str] = None
    location: Optional[str] = None
    value: float = 0.0
    converted: bool = False
    revenue: float = 0.0
    stage: Optional[JourneyStage] = None
    session_id: Optional[str] = None
    page_views: int = 1
    time_on_site: float = 0.0


@dataclass
class CustomerJourney:
    """Complete customer journey representation."""
    customer_id: str
    touchpoints: List[CustomerTouchpoint]
    start_date: datetime
    end_date: datetime
    total_touchpoints: int
    unique_channels: Set[str]
    journey_length_days: float
    converted: bool = False
    conversion_touchpoint: Optional[CustomerTouchpoint] = None
    total_revenue: float = 0.0
    journey_stages: List[JourneyStage] = field(default_factory=list)
    journey_cluster: Optional[int] = None
    journey_value: float = 0.0


@dataclass
class JourneyPattern:
    """Identified journey pattern."""
    pattern_id: str
    pattern_name: str
    touchpoint_sequence: List[str]
    stage_sequence: List[JourneyStage]
    frequency: int
    conversion_rate: float
    average_revenue: float
    average_journey_length: float
    success_probability: float
    optimization_score: float


class JourneyMapper:
    """
    Advanced customer journey mapping and analysis system.
    
    Maps complete customer journeys across channels, identifies patterns,
    and provides insights for journey optimization and orchestration.
    """
    
    def __init__(self,
                 journey_timeout_days: int = 90,
                 min_journey_length: int = 1,
                 enable_clustering: bool = True,
                 stage_classification: bool = True):
        """
        Initialize Journey Mapper.
        
        Args:
            journey_timeout_days: Maximum days for journey continuity
            min_journey_length: Minimum touchpoints for valid journey
            enable_clustering: Enable journey clustering analysis
            stage_classification: Automatically classify journey stages
        """
        self.journey_timeout_days = journey_timeout_days
        self.min_journey_length = min_journey_length
        self.enable_clustering = enable_clustering
        self.stage_classification = stage_classification
        
        # Journey data storage
        self.customer_journeys: Dict[str, CustomerJourney] = {}
        self.journey_patterns: List[JourneyPattern] = []
        self.journey_graph: nx.DiGraph = nx.DiGraph()
        
        # Analysis results
        self.journey_clusters = {}
        self.stage_transitions = {}
        self.funnel_analysis = {}
        self.journey_metrics = {}
        
        # Model components
        self.clustering_model = None
        self.stage_classifier = None
        
        logger.info("Journey mapping system initialized")
    
    def map_customer_journeys(self, touchpoint_data: pd.DataFrame) -> 'JourneyMapper':
        """
        Map customer journeys from touchpoint data.
        
        Args:
            touchpoint_data: DataFrame with touchpoint information
            
        Returns:
            Self for method chaining
        """
        logger.info("Mapping customer journeys from touchpoint data")
        
        # Validate input data
        required_columns = ['customer_id', 'timestamp', 'channel', 'touchpoint_type']
        if not all(col in touchpoint_data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        # Prepare data
        data = touchpoint_data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values(['customer_id', 'timestamp'])
        
        # Build customer journeys
        self._build_customer_journeys(data)
        
        # Classify journey stages if enabled
        if self.stage_classification:
            self._classify_journey_stages()
        
        # Analyze journey patterns
        self._analyze_journey_patterns()
        
        # Build journey graph
        self._build_journey_graph()
        
        # Perform clustering if enabled
        if self.enable_clustering:
            self._cluster_journeys()
        
        # Calculate journey metrics
        self._calculate_journey_metrics()
        
        # Analyze stage transitions and funnels
        self._analyze_stage_transitions()
        self._analyze_conversion_funnels()
        
        logger.info(f"Mapped {len(self.customer_journeys)} customer journeys")
        return self
    
    def _build_customer_journeys(self, data: pd.DataFrame):
        """Build individual customer journeys from touchpoint data."""
        
        journey_count = 0
        
        for customer_id, customer_data in data.groupby('customer_id'):
            customer_data = customer_data.sort_values('timestamp')
            
            # Create touchpoints
            touchpoints = []
            for _, row in customer_data.iterrows():
                
                # Parse touchpoint type
                touchpoint_type_str = str(row['touchpoint_type']).lower()
                touchpoint_type = TouchpointType.PAID_SEARCH  # Default
                
                for tp_type in TouchpointType:
                    if tp_type.value in touchpoint_type_str or touchpoint_type_str in tp_type.value:
                        touchpoint_type = tp_type
                        break
                
                touchpoint = CustomerTouchpoint(
                    customer_id=customer_id,
                    touchpoint_id=f"{customer_id}_{len(touchpoints)}",
                    touchpoint_type=touchpoint_type,
                    timestamp=row['timestamp'],
                    channel=row['channel'],
                    campaign=row.get('campaign'),
                    content=row.get('content'),
                    device=row.get('device'),
                    location=row.get('location'),
                    value=row.get('value', 0.0),
                    converted=row.get('converted', False),
                    revenue=row.get('revenue', 0.0),
                    session_id=row.get('session_id'),
                    page_views=row.get('page_views', 1),
                    time_on_site=row.get('time_on_site', 0.0)
                )
                
                touchpoints.append(touchpoint)
            
            if len(touchpoints) < self.min_journey_length:
                continue
            
            # Build journey
            start_date = touchpoints[0].timestamp
            end_date = touchpoints[-1].timestamp
            journey_length_days = (end_date - start_date).total_seconds() / (24 * 3600)
            
            # Find conversion touchpoint
            conversion_touchpoint = None
            total_revenue = 0.0
            converted = False
            
            for tp in touchpoints:
                if tp.converted:
                    converted = True
                    conversion_touchpoint = tp
                total_revenue += tp.revenue
            
            journey = CustomerJourney(
                customer_id=customer_id,
                touchpoints=touchpoints,
                start_date=start_date,
                end_date=end_date,
                total_touchpoints=len(touchpoints),
                unique_channels=set(tp.channel for tp in touchpoints),
                journey_length_days=journey_length_days,
                converted=converted,
                conversion_touchpoint=conversion_touchpoint,
                total_revenue=total_revenue,
                journey_value=total_revenue if converted else len(touchpoints) * 10  # Engagement value
            )
            
            self.customer_journeys[customer_id] = journey
            journey_count += 1
        
        logger.info(f"Built {journey_count} customer journeys")
    
    def _classify_journey_stages(self):
        """Classify touchpoints into journey stages using heuristics."""
        
        stage_keywords = {
            JourneyStage.AWARENESS: ['display', 'video', 'awareness', 'brand'],
            JourneyStage.CONSIDERATION: ['search', 'comparison', 'research', 'content'],
            JourneyStage.INTENT: ['product', 'pricing', 'demo', 'trial'],
            JourneyStage.PURCHASE: ['purchase', 'buy', 'checkout', 'conversion'],
            JourneyStage.RETENTION: ['email', 'support', 'account', 'usage'],
            JourneyStage.ADVOCACY: ['review', 'referral', 'share', 'recommend']
        }
        
        for journey in self.customer_journeys.values():
            journey_stages = []
            
            for i, touchpoint in enumerate(journey.touchpoints):
                # Default stage based on position in journey
                if i == 0:
                    default_stage = JourneyStage.AWARENESS
                elif i == len(journey.touchpoints) - 1 and touchpoint.converted:
                    default_stage = JourneyStage.PURCHASE
                elif i < len(journey.touchpoints) * 0.3:
                    default_stage = JourneyStage.AWARENESS
                elif i < len(journey.touchpoints) * 0.7:
                    default_stage = JourneyStage.CONSIDERATION
                else:
                    default_stage = JourneyStage.INTENT
                
                # Override with keyword matching
                stage = default_stage
                
                # Check channel, campaign, and content for stage keywords
                text_to_check = " ".join([
                    touchpoint.channel.lower(),
                    touchpoint.campaign.lower() if touchpoint.campaign else "",
                    touchpoint.content.lower() if touchpoint.content else "",
                    touchpoint.touchpoint_type.value
                ])
                
                for stage_candidate, keywords in stage_keywords.items():
                    if any(keyword in text_to_check for keyword in keywords):
                        stage = stage_candidate
                        break
                
                touchpoint.stage = stage
                journey_stages.append(stage)
            
            journey.journey_stages = journey_stages
    
    def _analyze_journey_patterns(self):
        """Identify common journey patterns and sequences."""
        
        patterns = defaultdict(list)
        
        for journey in self.customer_journeys.values():
            # Channel sequence pattern
            channel_sequence = tuple(tp.channel for tp in journey.touchpoints)
            patterns['channel_sequences'].append(channel_sequence)
            
            # Stage sequence pattern
            if journey.journey_stages:
                stage_sequence = tuple(stage.value for stage in journey.journey_stages)
                patterns['stage_sequences'].append(stage_sequence)
            
            # Touchpoint type sequence
            type_sequence = tuple(tp.touchpoint_type.value for tp in journey.touchpoints)
            patterns['type_sequences'].append(type_sequence)
        
        # Analyze pattern frequency and success rates
        self.journey_patterns = []
        
        # Channel sequence patterns
        channel_counter = Counter(patterns['channel_sequences'])
        for sequence, frequency in channel_counter.most_common(20):  # Top 20 patterns
            
            # Calculate metrics for this pattern
            matching_journeys = [
                j for j in self.customer_journeys.values()
                if tuple(tp.channel for tp in j.touchpoints) == sequence
            ]
            
            if len(matching_journeys) < 5:  # Minimum threshold
                continue
            
            conversion_rate = sum(1 for j in matching_journeys if j.converted) / len(matching_journeys)
            avg_revenue = np.mean([j.total_revenue for j in matching_journeys])
            avg_length = np.mean([j.journey_length_days for j in matching_journeys])
            
            # Calculate success probability and optimization score
            success_probability = conversion_rate * (avg_revenue / 1000)  # Normalize revenue
            optimization_score = success_probability * (frequency / len(self.customer_journeys))
            
            pattern = JourneyPattern(
                pattern_id=f"channel_seq_{len(self.journey_patterns)}",
                pattern_name=f"Channel Sequence: {' → '.join(sequence)}",
                touchpoint_sequence=list(sequence),
                stage_sequence=[],  # Will be filled by stage analysis
                frequency=frequency,
                conversion_rate=conversion_rate,
                average_revenue=avg_revenue,
                average_journey_length=avg_length,
                success_probability=success_probability,
                optimization_score=optimization_score
            )
            
            self.journey_patterns.append(pattern)
        
        logger.info(f"Identified {len(self.journey_patterns)} journey patterns")
    
    def _build_journey_graph(self):
        """Build network graph of customer journey flows."""
        
        self.journey_graph = nx.DiGraph()
        
        # Add nodes and edges based on touchpoint sequences
        for journey in self.customer_journeys.values():
            previous_channel = None
            
            for touchpoint in journey.touchpoints:
                current_channel = touchpoint.channel
                
                # Add node
                if current_channel not in self.journey_graph.nodes:
                    self.journey_graph.add_node(current_channel, 
                                             touchpoints=0, 
                                             conversions=0,
                                             revenue=0.0)
                
                # Update node attributes
                self.journey_graph.nodes[current_channel]['touchpoints'] += 1
                if touchpoint.converted:
                    self.journey_graph.nodes[current_channel]['conversions'] += 1
                self.journey_graph.nodes[current_channel]['revenue'] += touchpoint.revenue
                
                # Add edge from previous channel
                if previous_channel and previous_channel != current_channel:
                    if not self.journey_graph.has_edge(previous_channel, current_channel):
                        self.journey_graph.add_edge(previous_channel, current_channel, 
                                                  weight=0, 
                                                  conversions=0)
                    
                    self.journey_graph.edges[previous_channel, current_channel]['weight'] += 1
                    if touchpoint.converted:
                        self.journey_graph.edges[previous_channel, current_channel]['conversions'] += 1
                
                previous_channel = current_channel
        
        logger.info(f"Built journey graph with {len(self.journey_graph.nodes)} nodes and {len(self.journey_graph.edges)} edges")
    
    def _cluster_journeys(self):
        """Cluster similar customer journeys using machine learning."""
        
        if len(self.customer_journeys) < 10:
            logger.warning("Insufficient journeys for clustering")
            return
        
        # Feature engineering for journey clustering
        features = []
        journey_ids = []
        
        for customer_id, journey in self.customer_journeys.items():
            # Journey-level features
            feature_vector = [
                journey.total_touchpoints,
                journey.journey_length_days,
                len(journey.unique_channels),
                journey.total_revenue,
                1 if journey.converted else 0,
                journey.journey_value
            ]
            
            # Channel distribution features
            channel_counts = Counter(tp.channel for tp in journey.touchpoints)
            all_channels = set()
            for j in self.customer_journeys.values():
                all_channels.update(tp.channel for tp in j.touchpoints)
            
            for channel in sorted(all_channels):
                feature_vector.append(channel_counts.get(channel, 0))
            
            # Touchpoint type distribution
            type_counts = Counter(tp.touchpoint_type.value for tp in journey.touchpoints)
            all_types = [tp.value for tp in TouchpointType]
            
            for tp_type in all_types:
                feature_vector.append(type_counts.get(tp_type, 0))
            
            features.append(feature_vector)
            journey_ids.append(customer_id)
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Determine optimal number of clusters
        best_k = 3
        best_score = -1
        
        for k in range(2, min(10, len(features) // 3)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            
            if len(set(labels)) > 1:  # Ensure multiple clusters
                score = silhouette_score(features_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        
        # Final clustering
        self.clustering_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = self.clustering_model.fit_predict(features_scaled)
        
        # Assign clusters to journeys
        for i, customer_id in enumerate(journey_ids):
            self.customer_journeys[customer_id].journey_cluster = int(cluster_labels[i])
        
        # Analyze cluster characteristics
        self.journey_clusters = {}
        
        for cluster_id in range(best_k):
            cluster_journeys = [
                j for j in self.customer_journeys.values()
                if j.journey_cluster == cluster_id
            ]
            
            if not cluster_journeys:
                continue
            
            # Calculate cluster statistics
            cluster_stats = {
                'size': len(cluster_journeys),
                'avg_touchpoints': np.mean([j.total_touchpoints for j in cluster_journeys]),
                'avg_journey_length': np.mean([j.journey_length_days for j in cluster_journeys]),
                'conversion_rate': sum(1 for j in cluster_journeys if j.converted) / len(cluster_journeys),
                'avg_revenue': np.mean([j.total_revenue for j in cluster_journeys]),
                'top_channels': Counter([
                    tp.channel for j in cluster_journeys for tp in j.touchpoints
                ]).most_common(5),
                'common_patterns': Counter([
                    tuple(tp.channel for tp in j.touchpoints) 
                    for j in cluster_journeys
                ]).most_common(3)
            }
            
            self.journey_clusters[cluster_id] = cluster_stats
        
        logger.info(f"Clustered journeys into {best_k} groups")
    
    def _analyze_stage_transitions(self):
        """Analyze transitions between journey stages."""
        
        transitions = defaultdict(int)
        stage_performance = defaultdict(lambda: {'touchpoints': 0, 'conversions': 0, 'revenue': 0.0})
        
        for journey in self.customer_journeys.values():
            if not journey.journey_stages:
                continue
            
            # Track stage transitions
            for i in range(len(journey.journey_stages) - 1):
                current_stage = journey.journey_stages[i]
                next_stage = journey.journey_stages[i + 1]
                transitions[(current_stage.value, next_stage.value)] += 1
            
            # Track stage performance
            for touchpoint in journey.touchpoints:
                if touchpoint.stage:
                    stage = touchpoint.stage.value
                    stage_performance[stage]['touchpoints'] += 1
                    if touchpoint.converted:
                        stage_performance[stage]['conversions'] += 1
                    stage_performance[stage]['revenue'] += touchpoint.revenue
        
        # Calculate transition probabilities
        stage_totals = defaultdict(int)
        for (from_stage, to_stage), count in transitions.items():
            stage_totals[from_stage] += count
        
        transition_probabilities = {}
        for (from_stage, to_stage), count in transitions.items():
            prob = count / stage_totals[from_stage] if stage_totals[from_stage] > 0 else 0
            transition_probabilities[(from_stage, to_stage)] = prob
        
        self.stage_transitions = {
            'transitions': dict(transitions),
            'probabilities': transition_probabilities,
            'stage_performance': dict(stage_performance)
        }
        
        logger.info(f"Analyzed {len(transitions)} stage transitions")
    
    def _analyze_conversion_funnels(self):
        """Analyze conversion funnels and drop-off points."""
        
        # Overall funnel analysis
        stage_funnel = defaultdict(lambda: {'entries': 0, 'conversions': 0, 'dropoffs': 0})
        
        for journey in self.customer_journeys.values():
            if not journey.journey_stages:
                continue
            
            stages_in_journey = set(journey.journey_stages)
            converted = journey.converted
            
            for stage in JourneyStage:
                if stage in stages_in_journey:
                    stage_funnel[stage.value]['entries'] += 1
                    if converted:
                        stage_funnel[stage.value]['conversions'] += 1
                    else:
                        stage_funnel[stage.value]['dropoffs'] += 1
        
        # Calculate funnel metrics
        funnel_metrics = {}
        
        for stage_name, stats in stage_funnel.items():
            total_entries = stats['entries']
            conversions = stats['conversions']
            
            if total_entries > 0:
                conversion_rate = conversions / total_entries
                dropoff_rate = 1 - conversion_rate
                
                funnel_metrics[stage_name] = {
                    'entries': total_entries,
                    'conversions': conversions,
                    'conversion_rate': conversion_rate,
                    'dropoff_rate': dropoff_rate
                }
        
        # Channel-specific funnel analysis
        channel_funnels = {}
        
        for journey in self.customer_journeys.values():
            for touchpoint in journey.touchpoints:
                channel = touchpoint.channel
                
                if channel not in channel_funnels:
                    channel_funnels[channel] = {
                        'touchpoints': 0,
                        'conversions': 0,
                        'revenue': 0.0,
                        'unique_customers': set()
                    }
                
                channel_funnels[channel]['touchpoints'] += 1
                channel_funnels[channel]['unique_customers'].add(journey.customer_id)
                channel_funnels[channel]['revenue'] += touchpoint.revenue
                
                if touchpoint.converted:
                    channel_funnels[channel]['conversions'] += 1
        
        # Calculate channel funnel metrics
        for channel, stats in channel_funnels.items():
            unique_customers = len(stats['unique_customers'])
            stats['unique_customers'] = unique_customers
            stats['conversion_rate'] = stats['conversions'] / unique_customers if unique_customers > 0 else 0
            stats['avg_revenue_per_customer'] = stats['revenue'] / unique_customers if unique_customers > 0 else 0
        
        self.funnel_analysis = {
            'stage_funnel': dict(funnel_metrics),
            'channel_funnels': channel_funnels
        }
    
    def _calculate_journey_metrics(self):
        """Calculate comprehensive journey metrics."""
        
        if not self.customer_journeys:
            return
        
        journeys = list(self.customer_journeys.values())
        
        # Overall metrics
        total_journeys = len(journeys)
        converting_journeys = [j for j in journeys if j.converted]
        
        overall_metrics = {
            'total_journeys': total_journeys,
            'converting_journeys': len(converting_journeys),
            'overall_conversion_rate': len(converting_journeys) / total_journeys,
            'avg_journey_length': np.mean([j.total_touchpoints for j in journeys]),
            'avg_journey_days': np.mean([j.journey_length_days for j in journeys]),
            'avg_unique_channels': np.mean([len(j.unique_channels) for j in journeys]),
            'total_revenue': sum(j.total_revenue for j in journeys),
            'avg_revenue_per_journey': np.mean([j.total_revenue for j in journeys]),
            'avg_revenue_per_converting_journey': np.mean([j.total_revenue for j in converting_journeys]) if converting_journeys else 0
        }
        
        # Channel metrics
        channel_metrics = defaultdict(lambda: {
            'touchpoints': 0,
            'journeys': set(),
            'conversions': 0,
            'revenue': 0.0
        })
        
        for journey in journeys:
            for touchpoint in journey.touchpoints:
                channel = touchpoint.channel
                channel_metrics[channel]['touchpoints'] += 1
                channel_metrics[channel]['journeys'].add(journey.customer_id)
                channel_metrics[channel]['revenue'] += touchpoint.revenue
                
                if touchpoint.converted:
                    channel_metrics[channel]['conversions'] += 1
        
        # Calculate channel performance metrics
        channel_performance = {}
        for channel, stats in channel_metrics.items():
            unique_journeys = len(stats['journeys'])
            channel_performance[channel] = {
                'total_touchpoints': stats['touchpoints'],
                'unique_journeys': unique_journeys,
                'avg_touchpoints_per_journey': stats['touchpoints'] / unique_journeys,
                'total_conversions': stats['conversions'],
                'conversion_rate': stats['conversions'] / unique_journeys if unique_journeys > 0 else 0,
                'total_revenue': stats['revenue'],
                'revenue_per_journey': stats['revenue'] / unique_journeys if unique_journeys > 0 else 0
            }
        
        # Journey length distribution
        length_distribution = Counter([j.total_touchpoints for j in journeys])
        days_distribution = Counter([
            int(j.journey_length_days) // 7 * 7  # Weekly buckets
            for j in journeys
        ])
        
        self.journey_metrics = {
            'overall': overall_metrics,
            'channel_performance': channel_performance,
            'length_distribution': dict(length_distribution),
            'days_distribution': dict(days_distribution)
        }
        
        logger.info(f"Calculated comprehensive journey metrics")
    
    def get_journey_insights(self) -> Dict[str, Any]:
        """Get comprehensive journey insights."""
        
        insights = {
            'total_journeys_mapped': len(self.customer_journeys),
            'journey_patterns_identified': len(self.journey_patterns),
            'top_patterns': sorted(self.journey_patterns, 
                                 key=lambda p: p.optimization_score, 
                                 reverse=True)[:5],
            'conversion_funnel': self.funnel_analysis.get('stage_funnel', {}),
            'stage_transitions': self.stage_transitions,
            'journey_clusters': self.journey_clusters,
            'overall_metrics': self.journey_metrics.get('overall', {}),
            'channel_performance': self.journey_metrics.get('channel_performance', {}),
            'graph_stats': {
                'nodes': len(self.journey_graph.nodes),
                'edges': len(self.journey_graph.edges),
                'density': nx.density(self.journey_graph) if self.journey_graph.nodes else 0
            }
        }
        
        return insights
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate journey optimization recommendations."""
        
        recommendations = []
        
        if not self.journey_metrics or not self.funnel_analysis:
            return recommendations
        
        # High-value pattern recommendations
        top_patterns = sorted(self.journey_patterns, 
                            key=lambda p: p.optimization_score, 
                            reverse=True)[:3]
        
        for pattern in top_patterns:
            recommendations.append({
                'type': 'journey_pattern',
                'title': f'Optimize {pattern.pattern_name}',
                'description': f'This pattern has {pattern.conversion_rate:.1%} conversion rate with ${pattern.average_revenue:.0f} average revenue',
                'priority': 'High' if pattern.optimization_score > 0.1 else 'Medium',
                'impact': f'{pattern.frequency} customers follow this pattern',
                'action': f'Focus on replicating this journey sequence for similar customer segments'
            })
        
        # Funnel optimization recommendations
        stage_funnel = self.funnel_analysis.get('stage_funnel', {})
        
        worst_dropoff_stage = None
        worst_dropoff_rate = 0
        
        for stage, metrics in stage_funnel.items():
            if metrics['dropoff_rate'] > worst_dropoff_rate and metrics['entries'] > 10:
                worst_dropoff_rate = metrics['dropoff_rate']
                worst_dropoff_stage = stage
        
        if worst_dropoff_stage:
            recommendations.append({
                'type': 'funnel_optimization',
                'title': f'Reduce {worst_dropoff_stage.title()} Stage Drop-off',
                'description': f'{worst_dropoff_rate:.1%} of customers drop off at {worst_dropoff_stage} stage',
                'priority': 'High',
                'impact': f'Potential to recover {stage_funnel[worst_dropoff_stage]["dropoffs"]} customers',
                'action': f'Analyze and optimize {worst_dropoff_stage} stage touchpoints to reduce friction'
            })
        
        # Channel efficiency recommendations
        channel_performance = self.journey_metrics.get('channel_performance', {})
        
        if channel_performance:
            # Find most efficient channel
            best_channel = max(channel_performance.items(), 
                             key=lambda x: x[1]['conversion_rate'] * x[1]['revenue_per_journey'])
            
            recommendations.append({
                'type': 'channel_scaling',
                'title': f'Scale {best_channel[0]} Investment',
                'description': f'{best_channel[0]} shows {best_channel[1]["conversion_rate"]:.1%} conversion rate and ${best_channel[1]["revenue_per_journey"]:.0f} revenue per journey',
                'priority': 'Medium',
                'impact': 'Increase overall journey performance',
                'action': f'Increase touchpoints and investment in {best_channel[0]} channel'
            })
        
        # Journey length optimization
        overall_metrics = self.journey_metrics.get('overall', {})
        avg_converting_length = 0
        avg_non_converting_length = 0
        
        converting_journeys = [j for j in self.customer_journeys.values() if j.converted]
        non_converting_journeys = [j for j in self.customer_journeys.values() if not j.converted]
        
        if converting_journeys:
            avg_converting_length = np.mean([j.total_touchpoints for j in converting_journeys])
        if non_converting_journeys:
            avg_non_converting_length = np.mean([j.total_touchpoints for j in non_converting_journeys])
        
        if avg_converting_length > 0 and abs(avg_converting_length - avg_non_converting_length) > 1:
            optimal_length = int(avg_converting_length)
            recommendations.append({
                'type': 'journey_length',
                'title': f'Optimize Journey Length to ~{optimal_length} Touchpoints',
                'description': f'Converting journeys average {avg_converting_length:.1f} touchpoints vs {avg_non_converting_length:.1f} for non-converting',
                'priority': 'Medium',
                'impact': 'Improve journey efficiency and conversion rates',
                'action': f'Design customer journeys targeting {optimal_length} meaningful touchpoints'
            })
        
        return recommendations
    
    def generate_executive_report(self) -> str:
        """Generate executive-level journey mapping report."""
        
        report = "# Customer Journey Mapping Analysis\n\n"
        report += "**Advanced Journey Analytics by Sotiris Spyrou**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        # Executive Summary
        insights = self.get_journey_insights()
        overall_metrics = insights.get('overall_metrics', {})
        
        report += f"## Executive Summary\n\n"
        report += f"- **Journeys Analyzed**: {insights['total_journeys_mapped']:,}\n"
        report += f"- **Conversion Rate**: {overall_metrics.get('overall_conversion_rate', 0):.1%}\n"
        report += f"- **Average Journey Length**: {overall_metrics.get('avg_journey_length', 0):.1f} touchpoints\n"
        report += f"- **Average Journey Duration**: {overall_metrics.get('avg_journey_days', 0):.1f} days\n"
        report += f"- **Total Revenue Attributed**: ${overall_metrics.get('total_revenue', 0):,.0f}\n"
        report += f"- **Journey Patterns Identified**: {insights['journey_patterns_identified']}\n\n"
        
        # Top Performing Journey Patterns
        if insights['top_patterns']:
            report += f"## High-Value Journey Patterns\n\n"
            
            for i, pattern in enumerate(insights['top_patterns'], 1):
                performance_icon = "🏆" if pattern.conversion_rate > 0.15 else "⭐" if pattern.conversion_rate > 0.10 else "📊"
                report += f"{performance_icon} **{pattern.pattern_name}**\n"
                report += f"   - Frequency: {pattern.frequency:,} customers ({pattern.frequency/insights['total_journeys_mapped']*100:.1f}%)\n"
                report += f"   - Conversion Rate: {pattern.conversion_rate:.1%}\n"
                report += f"   - Average Revenue: ${pattern.average_revenue:,.0f}\n"
                report += f"   - Success Score: {pattern.optimization_score:.3f}\n\n"
        
        # Conversion Funnel Analysis
        funnel = insights.get('conversion_funnel', {})
        if funnel:
            report += f"## Conversion Funnel Analysis\n\n"
            report += "| Stage | Entries | Conversions | Conversion Rate | Drop-off Rate |\n"
            report += "|-------|---------|-------------|-----------------|---------------|\n"
            
            for stage, metrics in funnel.items():
                entries = metrics.get('entries', 0)
                conversions = metrics.get('conversions', 0)
                conv_rate = metrics.get('conversion_rate', 0)
                dropoff_rate = metrics.get('dropoff_rate', 0)
                
                dropoff_icon = "🔴" if dropoff_rate > 0.7 else "🟡" if dropoff_rate > 0.5 else "🟢"
                
                report += f"| {stage.title()} | {entries:,} | {conversions:,} | {conv_rate:.1%} | {dropoff_icon} {dropoff_rate:.1%} |\n"
            
            report += "\n"
        
        # Channel Performance in Journeys
        channel_perf = insights.get('channel_performance', {})
        if channel_perf:
            report += f"## Channel Journey Performance\n\n"
            
            # Sort channels by efficiency (conversion rate × revenue per journey)
            channel_efficiency = {
                ch: metrics['conversion_rate'] * metrics['revenue_per_journey']
                for ch, metrics in channel_perf.items()
            }
            sorted_channels = sorted(channel_efficiency.items(), key=lambda x: x[1], reverse=True)
            
            report += "| Channel | Journeys | Avg Touchpoints | Conversion Rate | Revenue/Journey | Efficiency |\n"
            report += "|---------|----------|-----------------|-----------------|-----------------|------------|\n"
            
            for channel, _ in sorted_channels[:8]:  # Top 8 channels
                metrics = channel_perf[channel]
                efficiency_score = channel_efficiency[channel]
                
                efficiency_icon = "🔥" if efficiency_score > 50 else "📈" if efficiency_score > 20 else "📊"
                
                report += f"| {channel} | {metrics['unique_journeys']:,} | "
                report += f"{metrics['avg_touchpoints_per_journey']:.1f} | "
                report += f"{metrics['conversion_rate']:.1%} | "
                report += f"${metrics['revenue_per_journey']:.0f} | "
                report += f"{efficiency_icon} {efficiency_score:.0f} |\n"
            
            report += "\n"
        
        # Journey Clusters
        if insights['journey_clusters']:
            report += f"## Journey Segment Analysis\n\n"
            
            for cluster_id, stats in insights['journey_clusters'].items():
                size_pct = (stats['size'] / insights['total_journeys_mapped']) * 100
                
                report += f"### Segment {cluster_id + 1} ({size_pct:.1f}% of journeys)\n"
                report += f"- **Size**: {stats['size']:,} journeys\n"
                report += f"- **Avg Touchpoints**: {stats['avg_touchpoints']:.1f}\n"
                report += f"- **Conversion Rate**: {stats['conversion_rate']:.1%}\n"
                report += f"- **Avg Revenue**: ${stats['avg_revenue']:,.0f}\n"
                report += f"- **Top Channels**: {', '.join([ch for ch, _ in stats['top_channels'][:3]])}\n\n"
        
        # Optimization Recommendations
        recommendations = self.get_optimization_recommendations()
        if recommendations:
            report += f"## Strategic Recommendations\n\n"
            
            for i, rec in enumerate(recommendations, 1):
                priority_icon = "🔴" if rec['priority'] == 'High' else "🟡" if rec['priority'] == 'Medium' else "🟢"
                
                report += f"{i}. **{rec['title']}** {priority_icon}\n"
                report += f"   - **Description**: {rec['description']}\n"
                report += f"   - **Impact**: {rec['impact']}\n"
                report += f"   - **Action**: {rec['action']}\n\n"
        
        # Key Insights
        report += f"## Key Strategic Insights\n\n"
        
        # Calculate insights
        if overall_metrics.get('avg_unique_channels', 0) > 3:
            report += f"- **Multi-Channel Journeys**: Customers engage with {overall_metrics['avg_unique_channels']:.1f} channels on average, indicating complex attribution requirements\n"
        
        if insights['top_patterns']:
            best_pattern = insights['top_patterns'][0]
            report += f"- **Optimal Journey Pattern**: '{best_pattern.pattern_name}' drives {best_pattern.conversion_rate:.1%} conversion rate\n"
        
        if overall_metrics.get('avg_journey_days', 0) > 7:
            report += f"- **Extended Decision Timeline**: {overall_metrics['avg_journey_days']:.0f}-day average journey requires sustained engagement strategy\n"
        
        total_converted = overall_metrics.get('converting_journeys', 0)
        if total_converted > 0:
            revenue_per_conversion = overall_metrics.get('total_revenue', 0) / total_converted
            report += f"- **Conversion Value**: ${revenue_per_conversion:,.0f} average revenue per successful journey\n"
        
        report += "\n---\n*This analysis provides actionable insights for journey optimization and customer experience enhancement. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for custom journey analytics implementations.*"
        
        return report


def demo_journey_mapping():
    """Executive demonstration of Journey Mapping system."""
    
    print("=== Customer Journey Mapping: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    np.random.seed(42)
    
    # Generate realistic customer journey data
    customers = []
    touchpoint_types = list(TouchpointType)
    channels = ['Search', 'Display', 'Social', 'Email', 'Direct', 'Referral', 'Video']
    campaigns = ['Brand_Awareness', 'Product_Demo', 'Retargeting', 'Email_Nurture', 'Search_Intent']
    
    # Generate 500 customer journeys with realistic patterns
    for customer_id in range(1, 501):
        # Determine customer segment
        segment = np.random.choice(['high_intent', 'explorer', 'price_sensitive'], p=[0.3, 0.5, 0.2])
        
        # Journey characteristics by segment
        if segment == 'high_intent':
            journey_length = np.random.choice(range(2, 6), p=[0.1, 0.3, 0.4, 0.2])
            conversion_prob = 0.35
            preferred_channels = ['Search', 'Direct', 'Email']
        elif segment == 'explorer':
            journey_length = np.random.choice(range(3, 9), p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
            conversion_prob = 0.15
            preferred_channels = ['Display', 'Social', 'Video', 'Search']
        else:  # price_sensitive
            journey_length = np.random.choice(range(4, 10), p=[0.05, 0.15, 0.25, 0.25, 0.2, 0.1])
            conversion_prob = 0.08
            preferred_channels = ['Social', 'Email', 'Search', 'Referral']
        
        # Generate journey timestamps
        start_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 60))
        journey_span_days = np.random.exponential(10)  # Average 10 days
        journey_span_days = min(journey_span_days, 45)  # Cap at 45 days
        
        # Build customer journey
        journey_touchpoints = []
        converted = np.random.random() < conversion_prob
        
        for i in range(journey_length):
            # Time progression
            time_progress = i / max(journey_length - 1, 1)
            days_offset = time_progress * journey_span_days
            timestamp = start_date + timedelta(days=days_offset) + timedelta(hours=np.random.randint(0, 24))
            
            # Channel selection based on segment preferences and journey stage
            if i == 0:  # First touchpoint
                channel = np.random.choice(['Display', 'Social', 'Video', 'Referral'], p=[0.4, 0.3, 0.2, 0.1])
            elif i == journey_length - 1 and converted:  # Final touchpoint for conversions
                channel = np.random.choice(['Search', 'Direct', 'Email'], p=[0.6, 0.3, 0.1])
            else:
                # Weighted selection based on segment preferences
                channel = np.random.choice(preferred_channels + ['Display', 'Social'])
            
            # Touchpoint type based on channel
            if channel == 'Search':
                tp_type = np.random.choice([TouchpointType.PAID_SEARCH, TouchpointType.ORGANIC_SEARCH], p=[0.7, 0.3])
            elif channel == 'Display':
                tp_type = TouchpointType.DISPLAY_AD
            elif channel == 'Social':
                tp_type = TouchpointType.SOCIAL_MEDIA
            elif channel == 'Email':
                tp_type = TouchpointType.EMAIL
            elif channel == 'Direct':
                tp_type = TouchpointType.DIRECT
            elif channel == 'Video':
                tp_type = TouchpointType.VIDEO_AD
            else:
                tp_type = TouchpointType.REFERRAL
            
            # Campaign assignment
            if i == 0:
                campaign = 'Brand_Awareness'
            elif time_progress > 0.7:
                campaign = 'Retargeting' if not converted else 'Search_Intent'
            else:
                campaign = np.random.choice(campaigns)
            
            # Revenue assignment (only for conversions)
            revenue = 0.0
            is_converted = converted and (i == journey_length - 1)  # Conversion on last touchpoint
            
            if is_converted:
                # Revenue based on segment
                if segment == 'high_intent':
                    revenue = np.random.normal(500, 150)
                elif segment == 'explorer':
                    revenue = np.random.normal(300, 100)
                else:
                    revenue = np.random.normal(200, 75)
                revenue = max(revenue, 50)  # Minimum revenue
            
            journey_touchpoints.append({
                'customer_id': f'customer_{customer_id}',
                'timestamp': timestamp,
                'channel': channel,
                'touchpoint_type': tp_type.value,
                'campaign': campaign,
                'device': np.random.choice(['desktop', 'mobile', 'tablet'], p=[0.5, 0.4, 0.1]),
                'converted': is_converted,
                'revenue': revenue,
                'page_views': np.random.randint(1, 8),
                'time_on_site': np.random.exponential(120)  # Average 2 minutes
            })
        
        customers.extend(journey_touchpoints)
    
    journey_data = pd.DataFrame(customers)
    
    print(f"📊 Generated {len(journey_data)} touchpoints across {journey_data['customer_id'].nunique()} customers")
    print(f"📈 Overall conversion rate: {journey_data.groupby('customer_id')['converted'].any().mean():.1%}")
    
    # Initialize and run journey mapping
    mapper = JourneyMapper(
        journey_timeout_days=90,
        min_journey_length=1,
        enable_clustering=True,
        stage_classification=True
    )
    
    print(f"\n🗺️ Mapping customer journeys...")
    mapper.map_customer_journeys(journey_data)
    
    # Display results
    print("\n📊 JOURNEY MAPPING RESULTS")
    print("=" * 50)
    
    insights = mapper.get_journey_insights()
    
    print(f"\n🎯 Journey Analysis Summary:")
    print(f"  • Total Journeys Mapped: {insights['total_journeys_mapped']:,}")
    print(f"  • Journey Patterns Identified: {insights['journey_patterns_identified']}")
    print(f"  • Journey Clusters: {len(insights['journey_clusters'])}")
    print(f"  • Graph Nodes (Channels): {insights['graph_stats']['nodes']}")
    print(f"  • Graph Edges (Transitions): {insights['graph_stats']['edges']}")
    
    # Overall performance metrics
    overall = insights['overall_metrics']
    print(f"\n📈 Overall Journey Performance:")
    print(f"  • Conversion Rate: {overall['overall_conversion_rate']:.1%}")
    print(f"  • Avg Journey Length: {overall['avg_journey_length']:.1f} touchpoints")
    print(f"  • Avg Journey Duration: {overall['avg_journey_days']:.1f} days")
    print(f"  • Avg Unique Channels: {overall['avg_unique_channels']:.1f}")
    print(f"  • Total Revenue: ${overall['total_revenue']:,.0f}")
    print(f"  • Revenue per Converting Journey: ${overall['avg_revenue_per_converting_journey']:,.0f}")
    
    # Top journey patterns
    if insights['top_patterns']:
        print(f"\n🏆 Top High-Value Journey Patterns:")
        
        for i, pattern in enumerate(insights['top_patterns'][:3], 1):
            performance_icon = "🔥" if pattern.conversion_rate > 0.2 else "⭐" if pattern.conversion_rate > 0.15 else "📊"
            print(f"{performance_icon} {i}. {pattern.pattern_name}")
            print(f"     Customers: {pattern.frequency:,} ({pattern.frequency/insights['total_journeys_mapped']*100:.1f}%)")
            print(f"     Conversion Rate: {pattern.conversion_rate:.1%}")
            print(f"     Avg Revenue: ${pattern.average_revenue:,.0f}")
            print(f"     Success Score: {pattern.optimization_score:.3f}")
    
    # Conversion funnel
    funnel = insights.get('conversion_funnel', {})
    if funnel:
        print(f"\n📊 Conversion Funnel Analysis:")
        
        for stage, metrics in funnel.items():
            dropoff_icon = "🔴" if metrics['dropoff_rate'] > 0.7 else "🟡" if metrics['dropoff_rate'] > 0.5 else "🟢"
            print(f"  {dropoff_icon} {stage.title()}: {metrics['entries']:,} entries → "
                  f"{metrics['conversions']:,} conversions ({metrics['conversion_rate']:.1%})")
    
    # Channel performance
    channel_perf = insights.get('channel_performance', {})
    if channel_perf:
        print(f"\n🎯 Channel Journey Performance:")
        
        # Sort by efficiency
        channel_efficiency = {
            ch: metrics['conversion_rate'] * metrics['revenue_per_journey']
            for ch, metrics in channel_perf.items()
        }
        sorted_channels = sorted(channel_efficiency.items(), key=lambda x: x[1], reverse=True)
        
        for channel, efficiency in sorted_channels[:5]:
            metrics = channel_perf[channel]
            efficiency_icon = "🔥" if efficiency > 50 else "📈" if efficiency > 20 else "📊"
            
            print(f"  {efficiency_icon} {channel:8}: {metrics['unique_journeys']:4,} journeys | "
                  f"{metrics['conversion_rate']:.1%} conv | ${metrics['revenue_per_journey']:4.0f}/journey | "
                  f"Efficiency: {efficiency:.0f}")
    
    # Journey clusters
    if insights['journey_clusters']:
        print(f"\n🎭 Journey Segment Analysis:")
        
        for cluster_id, stats in insights['journey_clusters'].items():
            size_pct = (stats['size'] / insights['total_journeys_mapped']) * 100
            print(f"  📊 Segment {cluster_id + 1}: {stats['size']:,} journeys ({size_pct:.1f}%)")
            print(f"      Avg touchpoints: {stats['avg_touchpoints']:.1f} | "
                  f"Conversion: {stats['conversion_rate']:.1%} | "
                  f"Revenue: ${stats['avg_revenue']:,.0f}")
            print(f"      Top channels: {', '.join([ch for ch, _ in stats['top_channels'][:3]])}")
    
    # Optimization recommendations
    recommendations = mapper.get_optimization_recommendations()
    if recommendations:
        print(f"\n💡 Strategic Recommendations:")
        
        for i, rec in enumerate(recommendations[:4], 1):
            priority_icon = "🔴" if rec['priority'] == 'High' else "🟡" if rec['priority'] == 'Medium' else "🟢"
            print(f"{priority_icon} {i}. {rec['title']}")
            print(f"     {rec['description']}")
            print(f"     Impact: {rec['impact']}")
    
    print("\n" + "="*60)
    print("🚀 Advanced customer journey mapping and pattern recognition")
    print("💼 Enterprise-grade journey optimization and orchestration")
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_journey_mapping()