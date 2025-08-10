"""
Cross-Device Attribution Tracking

Advanced cross-device tracking and identity resolution for unified customer
journey attribution across multiple devices, platforms, and touchpoints.

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
import hashlib
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class MatchingMethod(Enum):
    """Device matching methods."""
    DETERMINISTIC = "deterministic"
    PROBABILISTIC = "probabilistic"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    HYBRID = "hybrid"


class DeviceType(Enum):
    """Device types for tracking."""
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"
    TV = "tv"
    WEARABLE = "wearable"
    UNKNOWN = "unknown"


class MatchConfidence(Enum):
    """Confidence levels for device matching."""
    VERY_HIGH = "very_high"  # 95%+
    HIGH = "high"           # 85-94%
    MEDIUM = "medium"       # 70-84%
    LOW = "low"            # 50-69%
    VERY_LOW = "very_low"  # <50%


@dataclass
class DeviceProfile:
    """Device profile data structure."""
    device_id: str
    device_type: DeviceType
    user_agent: str
    ip_address: str
    timestamps: List[datetime]
    touchpoints: List[str]
    channels: List[str]
    behavioral_features: Dict[str, Any]
    geo_data: Dict[str, str]
    session_duration: float = 0.0
    conversion_events: int = 0


@dataclass
class DeviceMatch:
    """Device matching result."""
    primary_device: str
    secondary_device: str
    match_method: MatchingMethod
    confidence_score: float
    confidence_level: MatchConfidence
    matching_features: List[str]
    match_timestamp: datetime
    is_validated: bool = False


@dataclass
class UnifiedProfile:
    """Unified cross-device user profile."""
    unified_id: str
    device_ids: Set[str]
    primary_device: str
    device_profiles: Dict[str, DeviceProfile]
    unified_journey: List[Dict[str, Any]]
    total_touchpoints: int
    total_conversions: int
    cross_device_conversions: int
    journey_span_days: float
    device_contribution: Dict[str, float]


class CrossDeviceTracker:
    """
    Advanced cross-device tracking and identity resolution system.
    
    Unifies customer journeys across devices using deterministic and 
    probabilistic matching techniques for comprehensive attribution analysis.
    """
    
    def __init__(self,
                 min_confidence_threshold: float = 0.7,
                 time_window_hours: int = 24,
                 behavioral_weight: float = 0.4,
                 temporal_weight: float = 0.3,
                 geo_weight: float = 0.3):
        """
        Initialize Cross-Device Tracker.
        
        Args:
            min_confidence_threshold: Minimum confidence for device matching
            time_window_hours: Time window for considering device matches
            behavioral_weight: Weight for behavioral matching features
            temporal_weight: Weight for temporal proximity
            geo_weight: Weight for geographic proximity
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.time_window_hours = time_window_hours
        self.behavioral_weight = behavioral_weight
        self.temporal_weight = temporal_weight
        self.geo_weight = geo_weight
        
        # Device data
        self.device_profiles: Dict[str, DeviceProfile] = {}
        self.device_matches: List[DeviceMatch] = []
        self.unified_profiles: Dict[str, UnifiedProfile] = {}
        
        # Matching algorithms
        self.feature_extractors = {}
        self.matching_rules = {}
        self.validation_metrics = {}
        
        # Analysis results
        self.cross_device_stats = {}
        self.attribution_lift = {}
        self.device_journey_analysis = {}
        
        self._initialize_matching_algorithms()
        
        logger.info("Cross-device tracker initialized")
    
    def _initialize_matching_algorithms(self):
        """Initialize device matching algorithms and rules."""
        
        # Deterministic matching rules (high confidence)
        self.matching_rules['deterministic'] = {
            'email_hash': {'weight': 1.0, 'threshold': 1.0},
            'login_id': {'weight': 1.0, 'threshold': 1.0},
            'phone_hash': {'weight': 1.0, 'threshold': 1.0},
            'customer_id': {'weight': 1.0, 'threshold': 1.0}
        }
        
        # Probabilistic matching rules (medium-high confidence)
        self.matching_rules['probabilistic'] = {
            'ip_address': {'weight': 0.6, 'threshold': 0.8},
            'user_agent_similarity': {'weight': 0.4, 'threshold': 0.7},
            'geo_proximity': {'weight': 0.7, 'threshold': 0.8},
            'time_proximity': {'weight': 0.5, 'threshold': 0.6}
        }
        
        # Behavioral matching rules (medium confidence)
        self.matching_rules['behavioral'] = {
            'channel_overlap': {'weight': 0.6, 'threshold': 0.5},
            'journey_similarity': {'weight': 0.7, 'threshold': 0.6},
            'conversion_pattern': {'weight': 0.8, 'threshold': 0.7},
            'session_pattern': {'weight': 0.4, 'threshold': 0.5}
        }
    
    def track_device_activity(self, activity_data: pd.DataFrame) -> 'CrossDeviceTracker':
        """
        Track device activity and build device profiles.
        
        Args:
            activity_data: Device activity data with touchpoints and metadata
            
        Returns:
            Self for method chaining
        """
        logger.info("Processing device activity data")
        
        # Validate input data
        required_columns = ['device_id', 'timestamp', 'touchpoint', 'channel']
        if not all(col in activity_data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        # Prepare data
        activity_data = activity_data.copy()
        activity_data['timestamp'] = pd.to_datetime(activity_data['timestamp'])
        activity_data = activity_data.sort_values(['device_id', 'timestamp'])
        
        # Build device profiles
        self._build_device_profiles(activity_data)
        
        # Extract behavioral features
        self._extract_behavioral_features()
        
        logger.info(f"Built profiles for {len(self.device_profiles)} devices")
        return self
    
    def _build_device_profiles(self, data: pd.DataFrame):
        """Build individual device profiles from activity data."""
        
        for device_id, device_data in data.groupby('device_id'):
            device_data = device_data.sort_values('timestamp')
            
            # Extract basic device info
            device_type = self._determine_device_type(
                device_data.get('user_agent', '').iloc[0] if 'user_agent' in device_data else '',
                device_data.get('screen_width', 0).iloc[0] if 'screen_width' in device_data else 0
            )
            
            # Collect touchpoints and channels
            timestamps = device_data['timestamp'].tolist()
            touchpoints = device_data['touchpoint'].tolist()
            channels = device_data['channel'].tolist()
            
            # Calculate session metrics
            session_duration = self._calculate_session_duration(timestamps)
            conversion_events = len(device_data[device_data.get('converted', False)]) if 'converted' in device_data else 0
            
            # Extract geo data
            geo_data = {
                'country': device_data.get('country', '').iloc[0] if 'country' in device_data else '',
                'region': device_data.get('region', '').iloc[0] if 'region' in device_data else '',
                'city': device_data.get('city', '').iloc[0] if 'city' in device_data else '',
                'timezone': device_data.get('timezone', '').iloc[0] if 'timezone' in device_data else ''
            }
            
            # Create device profile
            profile = DeviceProfile(
                device_id=str(device_id),
                device_type=device_type,
                user_agent=device_data.get('user_agent', '').iloc[0] if 'user_agent' in device_data else '',
                ip_address=device_data.get('ip_address', '').iloc[0] if 'ip_address' in device_data else '',
                timestamps=timestamps,
                touchpoints=touchpoints,
                channels=channels,
                behavioral_features={},
                geo_data=geo_data,
                session_duration=session_duration,
                conversion_events=conversion_events
            )
            
            self.device_profiles[str(device_id)] = profile
    
    def _determine_device_type(self, user_agent: str, screen_width: int) -> DeviceType:
        """Determine device type from user agent and screen dimensions."""
        
        user_agent_lower = user_agent.lower()
        
        if 'mobile' in user_agent_lower or 'android' in user_agent_lower or 'iphone' in user_agent_lower:
            return DeviceType.MOBILE
        elif 'tablet' in user_agent_lower or 'ipad' in user_agent_lower:
            return DeviceType.TABLET
        elif 'tv' in user_agent_lower or 'smart-tv' in user_agent_lower:
            return DeviceType.TV
        elif screen_width > 0:
            if screen_width < 768:
                return DeviceType.MOBILE
            elif screen_width < 1024:
                return DeviceType.TABLET
            else:
                return DeviceType.DESKTOP
        else:
            return DeviceType.DESKTOP  # Default assumption
    
    def _calculate_session_duration(self, timestamps: List[datetime]) -> float:
        """Calculate average session duration in minutes."""
        
        if len(timestamps) < 2:
            return 0.0
        
        # Group timestamps into sessions (30 min gaps indicate new sessions)
        sessions = []
        current_session_start = timestamps[0]
        
        for i in range(1, len(timestamps)):
            time_gap = (timestamps[i] - timestamps[i-1]).total_seconds() / 60  # Minutes
            
            if time_gap > 30:  # New session after 30 min gap
                sessions.append((current_session_start, timestamps[i-1]))
                current_session_start = timestamps[i]
        
        # Add final session
        sessions.append((current_session_start, timestamps[-1]))
        
        # Calculate average session duration
        if sessions:
            total_duration = sum((end - start).total_seconds() / 60 for start, end in sessions)
            return total_duration / len(sessions)
        
        return 0.0
    
    def _extract_behavioral_features(self):
        """Extract behavioral features for each device profile."""
        
        for device_id, profile in self.device_profiles.items():
            features = {}
            
            # Channel preferences
            channel_counts = Counter(profile.channels)
            total_touches = len(profile.channels)
            features['channel_distribution'] = {
                ch: count / total_touches for ch, count in channel_counts.items()
            }
            features['primary_channel'] = channel_counts.most_common(1)[0][0] if channel_counts else 'unknown'
            features['channel_diversity'] = len(set(profile.channels)) / total_touches if total_touches > 0 else 0
            
            # Temporal patterns
            if profile.timestamps:
                hours = [ts.hour for ts in profile.timestamps]
                days = [ts.weekday() for ts in profile.timestamps]
                
                features['active_hours'] = Counter(hours)
                features['active_days'] = Counter(days)
                features['peak_hour'] = Counter(hours).most_common(1)[0][0]
                features['peak_day'] = Counter(days).most_common(1)[0][0]
                
                # Journey span
                features['journey_span_hours'] = (profile.timestamps[-1] - profile.timestamps[0]).total_seconds() / 3600
            
            # Conversion patterns
            features['conversion_rate'] = profile.conversion_events / total_touches if total_touches > 0 else 0
            features['avg_session_duration'] = profile.session_duration
            
            # Touchpoint sequence patterns
            if len(profile.touchpoints) > 1:
                sequences = [f"{profile.touchpoints[i]}→{profile.touchpoints[i+1]}" 
                           for i in range(len(profile.touchpoints)-1)]
                features['common_sequences'] = Counter(sequences).most_common(3)
            
            profile.behavioral_features = features
    
    def perform_device_matching(self) -> 'CrossDeviceTracker':
        """
        Perform cross-device matching using multiple algorithms.
        
        Returns:
            Self for method chaining
        """
        logger.info("Starting cross-device matching")
        
        # Get all device pairs for comparison
        device_ids = list(self.device_profiles.keys())
        
        for i in range(len(device_ids)):
            for j in range(i + 1, len(device_ids)):
                device1_id = device_ids[i]
                device2_id = device_ids[j]
                
                # Skip if devices are the same type and recent (likely same user)
                device1 = self.device_profiles[device1_id]
                device2 = self.device_profiles[device2_id]
                
                # Check if matching is feasible (time window, different devices, etc.)
                if self._is_matching_feasible(device1, device2):
                    # Try different matching methods
                    best_match = self._find_best_match(device1, device2)
                    
                    if best_match and best_match.confidence_score >= self.min_confidence_threshold:
                        self.device_matches.append(best_match)
        
        logger.info(f"Found {len(self.device_matches)} device matches")
        
        # Create unified profiles
        self._create_unified_profiles()
        
        # Analyze cross-device attribution
        self._analyze_cross_device_attribution()
        
        return self
    
    def _is_matching_feasible(self, device1: DeviceProfile, device2: DeviceProfile) -> bool:
        """Check if two devices can potentially be matched."""
        
        # Don't match devices of the same type if they have overlapping activity
        if device1.device_type == device2.device_type:
            # Check for temporal overlap
            time_overlap = self._calculate_time_overlap(device1.timestamps, device2.timestamps)
            if time_overlap > 0.5:  # 50% overlap suggests different users
                return False
        
        # Check if devices have activity within reasonable time window
        if device1.timestamps and device2.timestamps:
            time_gap = abs((device1.timestamps[-1] - device2.timestamps[0]).total_seconds() / 3600)
            if time_gap > self.time_window_hours * 7:  # Within a week
                return False
        
        return True
    
    def _calculate_time_overlap(self, timestamps1: List[datetime], timestamps2: List[datetime]) -> float:
        """Calculate temporal overlap between two device activities."""
        
        if not timestamps1 or not timestamps2:
            return 0.0
        
        # Create time buckets (hourly)
        all_hours = set()
        
        for ts in timestamps1:
            all_hours.add(ts.replace(minute=0, second=0, microsecond=0))
        
        hours2 = set()
        for ts in timestamps2:
            hours2.add(ts.replace(minute=0, second=0, microsecond=0))
        
        if not all_hours:
            return 0.0
        
        overlap = len(all_hours.intersection(hours2))
        total_unique = len(all_hours.union(hours2))
        
        return overlap / total_unique if total_unique > 0 else 0.0
    
    def _find_best_match(self, device1: DeviceProfile, device2: DeviceProfile) -> Optional[DeviceMatch]:
        """Find the best match between two devices using multiple methods."""
        
        matches = []
        
        # Try deterministic matching
        deterministic_match = self._deterministic_matching(device1, device2)
        if deterministic_match:
            matches.append(deterministic_match)
        
        # Try probabilistic matching
        probabilistic_match = self._probabilistic_matching(device1, device2)
        if probabilistic_match:
            matches.append(probabilistic_match)
        
        # Try behavioral matching
        behavioral_match = self._behavioral_matching(device1, device2)
        if behavioral_match:
            matches.append(behavioral_match)
        
        # Return best match
        if matches:
            best_match = max(matches, key=lambda m: m.confidence_score)
            return best_match
        
        return None
    
    def _deterministic_matching(self, device1: DeviceProfile, device2: DeviceProfile) -> Optional[DeviceMatch]:
        """Perform deterministic matching based on exact identifiers."""
        
        matching_features = []
        confidence_score = 0.0
        
        # Check for exact matches in identifying information
        # Note: In production, these would be hashed for privacy
        if device1.ip_address and device2.ip_address:
            if device1.ip_address == device2.ip_address:
                matching_features.append('ip_address')
                confidence_score += 0.6
        
        # Geographic matching
        if (device1.geo_data.get('city') and device2.geo_data.get('city') and
            device1.geo_data['city'] == device2.geo_data['city']):
            matching_features.append('same_city')
            confidence_score += 0.3
        
        # User agent similarity (for browser fingerprinting)
        if device1.user_agent and device2.user_agent:
            ua_similarity = self._calculate_user_agent_similarity(device1.user_agent, device2.user_agent)
            if ua_similarity > 0.8:
                matching_features.append('user_agent_similarity')
                confidence_score += 0.4 * ua_similarity
        
        if confidence_score >= 0.8:  # High threshold for deterministic
            confidence_level = MatchConfidence.VERY_HIGH if confidence_score >= 0.95 else MatchConfidence.HIGH
            
            return DeviceMatch(
                primary_device=device1.device_id,
                secondary_device=device2.device_id,
                match_method=MatchingMethod.DETERMINISTIC,
                confidence_score=min(confidence_score, 1.0),
                confidence_level=confidence_level,
                matching_features=matching_features,
                match_timestamp=datetime.now()
            )
        
        return None
    
    def _probabilistic_matching(self, device1: DeviceProfile, device2: DeviceProfile) -> Optional[DeviceMatch]:
        """Perform probabilistic matching based on multiple signals."""
        
        matching_features = []
        confidence_score = 0.0
        
        # Temporal proximity
        temporal_score = self._calculate_temporal_proximity(device1.timestamps, device2.timestamps)
        if temporal_score > 0.3:
            matching_features.append('temporal_proximity')
            confidence_score += self.temporal_weight * temporal_score
        
        # Geographic proximity  
        geo_score = self._calculate_geo_proximity(device1.geo_data, device2.geo_data)
        if geo_score > 0.5:
            matching_features.append('geo_proximity')
            confidence_score += self.geo_weight * geo_score
        
        # Behavioral similarity
        behavioral_score = self._calculate_behavioral_similarity(
            device1.behavioral_features, device2.behavioral_features
        )
        if behavioral_score > 0.4:
            matching_features.append('behavioral_similarity')
            confidence_score += self.behavioral_weight * behavioral_score
        
        if confidence_score >= 0.6:  # Lower threshold for probabilistic
            if confidence_score >= 0.85:
                confidence_level = MatchConfidence.HIGH
            elif confidence_score >= 0.70:
                confidence_level = MatchConfidence.MEDIUM
            else:
                confidence_level = MatchConfidence.LOW
            
            return DeviceMatch(
                primary_device=device1.device_id,
                secondary_device=device2.device_id,
                match_method=MatchingMethod.PROBABILISTIC,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                matching_features=matching_features,
                match_timestamp=datetime.now()
            )
        
        return None
    
    def _behavioral_matching(self, device1: DeviceProfile, device2: DeviceProfile) -> Optional[DeviceMatch]:
        """Perform behavioral matching based on usage patterns."""
        
        matching_features = []
        confidence_score = 0.0
        
        # Channel preference similarity
        channel_similarity = self._calculate_channel_similarity(
            device1.behavioral_features.get('channel_distribution', {}),
            device2.behavioral_features.get('channel_distribution', {})
        )
        
        if channel_similarity > 0.5:
            matching_features.append('channel_similarity')
            confidence_score += 0.3 * channel_similarity
        
        # Journey pattern similarity
        journey_similarity = self._calculate_journey_similarity(device1, device2)
        if journey_similarity > 0.4:
            matching_features.append('journey_similarity')
            confidence_score += 0.4 * journey_similarity
        
        # Temporal pattern similarity
        temporal_pattern_score = self._calculate_temporal_pattern_similarity(device1, device2)
        if temporal_pattern_score > 0.3:
            matching_features.append('temporal_patterns')
            confidence_score += 0.3 * temporal_pattern_score
        
        if confidence_score >= 0.5:  # Lower threshold for behavioral
            if confidence_score >= 0.75:
                confidence_level = MatchConfidence.MEDIUM
            else:
                confidence_level = MatchConfidence.LOW
            
            return DeviceMatch(
                primary_device=device1.device_id,
                secondary_device=device2.device_id,
                match_method=MatchingMethod.BEHAVIORAL,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                matching_features=matching_features,
                match_timestamp=datetime.now()
            )
        
        return None
    
    def _calculate_user_agent_similarity(self, ua1: str, ua2: str) -> float:
        """Calculate similarity between user agent strings."""
        
        if not ua1 or not ua2:
            return 0.0
        
        # Simple token-based similarity
        tokens1 = set(ua1.lower().split())
        tokens2 = set(ua2.lower().split())
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_temporal_proximity(self, timestamps1: List[datetime], timestamps2: List[datetime]) -> float:
        """Calculate temporal proximity between device activities."""
        
        if not timestamps1 or not timestamps2:
            return 0.0
        
        # Find closest timestamps
        min_gap = float('inf')
        
        for ts1 in timestamps1:
            for ts2 in timestamps2:
                gap_hours = abs((ts1 - ts2).total_seconds()) / 3600
                min_gap = min(min_gap, gap_hours)
        
        # Convert to proximity score (closer = higher score)
        if min_gap <= 1:  # Within 1 hour
            return 1.0
        elif min_gap <= 6:  # Within 6 hours
            return 0.8
        elif min_gap <= 24:  # Within 1 day
            return 0.6
        elif min_gap <= 72:  # Within 3 days
            return 0.3
        else:
            return 0.1
    
    def _calculate_geo_proximity(self, geo1: Dict[str, str], geo2: Dict[str, str]) -> float:
        """Calculate geographic proximity between devices."""
        
        score = 0.0
        
        if geo1.get('country') and geo2.get('country'):
            if geo1['country'] == geo2['country']:
                score += 0.3
                
                if geo1.get('region') and geo2.get('region'):
                    if geo1['region'] == geo2['region']:
                        score += 0.3
                        
                        if geo1.get('city') and geo2.get('city'):
                            if geo1['city'] == geo2['city']:
                                score += 0.4
        
        return min(score, 1.0)
    
    def _calculate_behavioral_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate behavioral similarity between device profiles."""
        
        if not features1 or not features2:
            return 0.0
        
        similarity_scores = []
        
        # Channel distribution similarity
        if 'channel_distribution' in features1 and 'channel_distribution' in features2:
            channel_sim = self._calculate_channel_similarity(
                features1['channel_distribution'], 
                features2['channel_distribution']
            )
            similarity_scores.append(channel_sim)
        
        # Session pattern similarity
        if ('avg_session_duration' in features1 and 'avg_session_duration' in features2):
            duration1 = features1['avg_session_duration']
            duration2 = features2['avg_session_duration']
            
            if duration1 > 0 and duration2 > 0:
                duration_sim = 1 - abs(duration1 - duration2) / max(duration1, duration2)
                similarity_scores.append(duration_sim)
        
        # Peak activity time similarity
        if 'peak_hour' in features1 and 'peak_hour' in features2:
            hour_diff = abs(features1['peak_hour'] - features2['peak_hour'])
            hour_sim = 1 - (hour_diff / 12)  # Normalize by 12 hours
            similarity_scores.append(hour_sim)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0
    
    def _calculate_channel_similarity(self, channels1: Dict[str, float], channels2: Dict[str, float]) -> float:
        """Calculate channel preference similarity."""
        
        if not channels1 or not channels2:
            return 0.0
        
        # Get all channels
        all_channels = set(channels1.keys()).union(set(channels2.keys()))
        
        if not all_channels:
            return 0.0
        
        # Create vectors
        vec1 = np.array([channels1.get(ch, 0) for ch in all_channels])
        vec2 = np.array([channels2.get(ch, 0) for ch in all_channels])
        
        # Calculate cosine similarity
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        return max(similarity, 0.0)  # Ensure non-negative
    
    def _calculate_journey_similarity(self, device1: DeviceProfile, device2: DeviceProfile) -> float:
        """Calculate journey pattern similarity between devices."""
        
        if not device1.touchpoints or not device2.touchpoints:
            return 0.0
        
        # Convert touchpoints to sequences
        seq1 = " → ".join(device1.touchpoints)
        seq2 = " → ".join(device2.touchpoints)
        
        # Simple string similarity (can be enhanced with sequence alignment)
        tokens1 = set(seq1.split(" → "))
        tokens2 = set(seq2.split(" → "))
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_temporal_pattern_similarity(self, device1: DeviceProfile, device2: DeviceProfile) -> float:
        """Calculate temporal usage pattern similarity."""
        
        features1 = device1.behavioral_features
        features2 = device2.behavioral_features
        
        if not features1 or not features2:
            return 0.0
        
        # Compare peak usage times
        if 'peak_hour' in features1 and 'peak_hour' in features2:
            hour_diff = abs(features1['peak_hour'] - features2['peak_hour'])
            hour_similarity = 1 - (hour_diff / 12)  # Normalize by half day
            
            return max(hour_similarity, 0.0)
        
        return 0.0
    
    def _create_unified_profiles(self):
        """Create unified user profiles from device matches."""
        
        # Build device clusters based on matches
        device_clusters = self._build_device_clusters()
        
        for cluster_id, device_ids in device_clusters.items():
            if len(device_ids) > 1:  # Only create unified profile for multi-device users
                unified_profile = self._build_unified_profile(cluster_id, device_ids)
                self.unified_profiles[cluster_id] = unified_profile
        
        logger.info(f"Created {len(self.unified_profiles)} unified cross-device profiles")
    
    def _build_device_clusters(self) -> Dict[str, Set[str]]:
        """Build clusters of connected devices."""
        
        # Create graph of device connections
        device_connections = defaultdict(set)
        
        for match in self.device_matches:
            device_connections[match.primary_device].add(match.secondary_device)
            device_connections[match.secondary_device].add(match.primary_device)
        
        # Find connected components (clusters)
        visited = set()
        clusters = {}
        cluster_id = 0
        
        for device_id in self.device_profiles.keys():
            if device_id not in visited:
                cluster = self._dfs_cluster(device_id, device_connections, visited)
                if cluster:
                    clusters[f"unified_{cluster_id}"] = cluster
                    cluster_id += 1
        
        return clusters
    
    def _dfs_cluster(self, device_id: str, connections: Dict, visited: Set) -> Set[str]:
        """Depth-first search to find device cluster."""
        
        if device_id in visited:
            return set()
        
        visited.add(device_id)
        cluster = {device_id}
        
        for connected_device in connections.get(device_id, set()):
            if connected_device not in visited:
                cluster.update(self._dfs_cluster(connected_device, connections, visited))
        
        return cluster
    
    def _build_unified_profile(self, unified_id: str, device_ids: Set[str]) -> UnifiedProfile:
        """Build unified profile from multiple device profiles."""
        
        device_profiles = {device_id: self.device_profiles[device_id] for device_id in device_ids}
        
        # Determine primary device (most active or recent)
        primary_device = max(device_ids, 
                           key=lambda d: len(self.device_profiles[d].touchpoints))
        
        # Merge journey data
        unified_journey = []
        all_touchpoints = []
        total_conversions = 0
        cross_device_conversions = 0
        
        for device_id in device_ids:
            profile = self.device_profiles[device_id]
            
            for i, (timestamp, touchpoint, channel) in enumerate(
                zip(profile.timestamps, profile.touchpoints, profile.channels)):
                
                journey_point = {
                    'timestamp': timestamp,
                    'touchpoint': touchpoint,
                    'channel': channel,
                    'device_id': device_id,
                    'device_type': profile.device_type.value
                }
                unified_journey.append(journey_point)
                all_touchpoints.append(touchpoint)
            
            total_conversions += profile.conversion_events
        
        # Sort unified journey by timestamp
        unified_journey.sort(key=lambda x: x['timestamp'])
        
        # Calculate cross-device conversions (simplified)
        if len(device_ids) > 1:
            cross_device_conversions = total_conversions  # Assume all conversions are cross-device
        
        # Calculate journey span
        if unified_journey:
            journey_span = (unified_journey[-1]['timestamp'] - unified_journey[0]['timestamp']).days
        else:
            journey_span = 0
        
        # Calculate device contribution
        device_contribution = {}
        for device_id in device_ids:
            device_touchpoints = len(self.device_profiles[device_id].touchpoints)
            total_touchpoints = len(all_touchpoints)
            device_contribution[device_id] = device_touchpoints / total_touchpoints if total_touchpoints > 0 else 0
        
        return UnifiedProfile(
            unified_id=unified_id,
            device_ids=device_ids,
            primary_device=primary_device,
            device_profiles=device_profiles,
            unified_journey=unified_journey,
            total_touchpoints=len(all_touchpoints),
            total_conversions=total_conversions,
            cross_device_conversions=cross_device_conversions,
            journey_span_days=journey_span,
            device_contribution=device_contribution
        )
    
    def _analyze_cross_device_attribution(self):
        """Analyze cross-device attribution impact."""
        
        # Calculate cross-device statistics
        total_devices = len(self.device_profiles)
        cross_device_users = len(self.unified_profiles)
        single_device_users = total_devices - sum(len(profile.device_ids) for profile in self.unified_profiles.values())
        
        # Attribution lift analysis
        single_device_conversions = sum(
            profile.conversion_events for profile in self.device_profiles.values()
            if profile.device_id not in {device_id for unified in self.unified_profiles.values() for device_id in unified.device_ids}
        )
        
        cross_device_conversions = sum(profile.total_conversions for profile in self.unified_profiles.values())
        
        total_conversions_before_unification = sum(profile.conversion_events for profile in self.device_profiles.values())
        total_conversions_after_unification = single_device_conversions + cross_device_conversions
        
        attribution_lift = (total_conversions_after_unification - total_conversions_before_unification) / total_conversions_before_unification if total_conversions_before_unification > 0 else 0
        
        self.cross_device_stats = {
            'total_devices': total_devices,
            'cross_device_users': cross_device_users,
            'single_device_users': single_device_users,
            'cross_device_rate': cross_device_users / total_devices if total_devices > 0 else 0,
            'avg_devices_per_user': np.mean([len(profile.device_ids) for profile in self.unified_profiles.values()]) if self.unified_profiles else 1,
            'attribution_lift': attribution_lift,
            'total_conversions_unified': total_conversions_after_unification
        }
        
        # Device journey analysis
        self._analyze_device_journeys()
    
    def _analyze_device_journeys(self):
        """Analyze cross-device journey patterns."""
        
        journey_patterns = {
            'device_transitions': defaultdict(int),
            'channel_device_combinations': defaultdict(int),
            'conversion_device_types': defaultdict(int),
            'journey_length_by_devices': defaultdict(list)
        }
        
        for profile in self.unified_profiles.values():
            journey = profile.unified_journey
            
            if len(journey) > 1:
                # Analyze device transitions
                for i in range(len(journey) - 1):
                    from_device_type = journey[i]['device_type']
                    to_device_type = journey[i + 1]['device_type']
                    
                    if from_device_type != to_device_type:
                        transition = f"{from_device_type} → {to_device_type}"
                        journey_patterns['device_transitions'][transition] += 1
                
                # Channel-device combinations
                for point in journey:
                    combo = f"{point['channel']} ({point['device_type']})"
                    journey_patterns['channel_device_combinations'][combo] += 1
            
            # Conversion device analysis
            if profile.total_conversions > 0:
                # Assume last touchpoint leads to conversion
                if journey:
                    conversion_device = journey[-1]['device_type']
                    journey_patterns['conversion_device_types'][conversion_device] += profile.total_conversions
            
            # Journey length by device count
            device_count = len(profile.device_ids)
            journey_length = len(profile.unified_journey)
            journey_patterns['journey_length_by_devices'][device_count].append(journey_length)
        
        self.device_journey_analysis = {
            'top_device_transitions': dict(Counter(journey_patterns['device_transitions']).most_common(10)),
            'top_channel_device_combos': dict(Counter(journey_patterns['channel_device_combinations']).most_common(10)),
            'conversion_by_device_type': dict(journey_patterns['conversion_device_types']),
            'avg_journey_length_by_device_count': {
                device_count: np.mean(lengths) if lengths else 0
                for device_count, lengths in journey_patterns['journey_length_by_devices'].items()
            }
        }
    
    def get_cross_device_insights(self) -> Dict[str, Any]:
        """Get comprehensive cross-device insights."""
        
        return {
            'cross_device_statistics': self.cross_device_stats,
            'device_matches_summary': {
                'total_matches': len(self.device_matches),
                'matches_by_method': Counter(match.match_method.value for match in self.device_matches),
                'matches_by_confidence': Counter(match.confidence_level.value for match in self.device_matches),
                'avg_confidence_score': np.mean([match.confidence_score for match in self.device_matches]) if self.device_matches else 0
            },
            'unified_profiles_summary': {
                'total_unified_profiles': len(self.unified_profiles),
                'avg_devices_per_profile': np.mean([len(profile.device_ids) for profile in self.unified_profiles.values()]) if self.unified_profiles else 0,
                'avg_journey_length': np.mean([profile.total_touchpoints for profile in self.unified_profiles.values()]) if self.unified_profiles else 0,
                'cross_device_conversion_rate': np.mean([profile.cross_device_conversions / profile.total_touchpoints for profile in self.unified_profiles.values() if profile.total_touchpoints > 0]) if self.unified_profiles else 0
            },
            'journey_analysis': self.device_journey_analysis
        }
    
    def get_unified_attribution(self, include_single_device: bool = True) -> pd.DataFrame:
        """Get unified attribution results across devices."""
        
        results = []
        
        # Process unified profiles
        for profile in self.unified_profiles.values():
            channel_attribution = defaultdict(float)
            device_channel_attribution = defaultdict(lambda: defaultdict(float))
            
            # Simple last-touch attribution for demo (can be replaced with sophisticated models)
            total_touches = len(profile.unified_journey)
            
            for point in profile.unified_journey:
                channel = point['channel']
                device_type = point['device_type']
                
                # Equal weight attribution for simplicity
                weight = 1 / total_touches if total_touches > 0 else 0
                channel_attribution[channel] += weight
                device_channel_attribution[device_type][channel] += weight
            
            for channel, attribution in channel_attribution.items():
                results.append({
                    'user_type': 'cross_device',
                    'unified_id': profile.unified_id,
                    'channel': channel,
                    'attribution_weight': attribution,
                    'device_count': len(profile.device_ids),
                    'total_touchpoints': profile.total_touchpoints,
                    'conversions': profile.total_conversions,
                    'journey_span_days': profile.journey_span_days,
                    'primary_device_type': self.device_profiles[profile.primary_device].device_type.value
                })
        
        # Include single-device users if requested
        if include_single_device:
            single_device_profiles = {
                device_id: profile for device_id, profile in self.device_profiles.items()
                if device_id not in {device_id for unified in self.unified_profiles.values() for device_id in unified.device_ids}
            }
            
            for profile in single_device_profiles.values():
                channel_attribution = defaultdict(float)
                total_touches = len(profile.channels)
                
                for channel in profile.channels:
                    weight = 1 / total_touches if total_touches > 0 else 0
                    channel_attribution[channel] += weight
                
                for channel, attribution in channel_attribution.items():
                    results.append({
                        'user_type': 'single_device',
                        'unified_id': profile.device_id,
                        'channel': channel,
                        'attribution_weight': attribution,
                        'device_count': 1,
                        'total_touchpoints': len(profile.touchpoints),
                        'conversions': profile.conversion_events,
                        'journey_span_days': (profile.timestamps[-1] - profile.timestamps[0]).days if len(profile.timestamps) > 1 else 0,
                        'primary_device_type': profile.device_type.value
                    })
        
        return pd.DataFrame(results)
    
    def generate_executive_report(self) -> str:
        """Generate comprehensive cross-device attribution report."""
        
        report = "# Cross-Device Attribution Analysis\n\n"
        report += "**Unified Customer Journey Tracking by Sotiris Spyrou**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        insights = self.get_cross_device_insights()
        
        # Cross-Device Overview
        stats = insights['cross_device_statistics']
        report += f"## Cross-Device Tracking Overview\n\n"
        report += f"- **Total Devices Tracked**: {stats['total_devices']:,}\n"
        report += f"- **Cross-Device Users**: {stats['cross_device_users']:,}\n"
        report += f"- **Single-Device Users**: {stats['single_device_users']:,}\n"
        report += f"- **Cross-Device Rate**: {stats['cross_device_rate']:.1%}\n"
        report += f"- **Avg Devices per User**: {stats['avg_devices_per_user']:.1f}\n"
        report += f"- **Attribution Lift**: {stats['attribution_lift']:+.1%}\n\n"
        
        # Device Matching Performance
        matches = insights['device_matches_summary']
        report += f"## Device Matching Performance\n\n"
        report += f"- **Total Device Matches**: {matches['total_matches']:,}\n"
        report += f"- **Average Confidence**: {matches['avg_confidence_score']:.1%}\n\n"
        
        if matches['matches_by_method']:
            report += f"### Matching Methods Used\n\n"
            for method, count in matches['matches_by_method'].items():
                percentage = count / matches['total_matches'] * 100
                report += f"- **{method.replace('_', ' ').title()}**: {count} matches ({percentage:.1f}%)\n"
            report += "\n"
        
        if matches['matches_by_confidence']:
            report += f"### Match Confidence Distribution\n\n"
            for confidence, count in matches['matches_by_confidence'].items():
                percentage = count / matches['total_matches'] * 100
                confidence_emoji = {"very_high": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢", "very_low": "⚪"}.get(confidence, "📊")
                report += f"- {confidence_emoji} **{confidence.replace('_', ' ').title()}**: {count} matches ({percentage:.1f}%)\n"
            report += "\n"
        
        # Unified Profiles Analysis
        unified = insights['unified_profiles_summary']
        report += f"## Unified Customer Profiles\n\n"
        report += f"- **Total Unified Profiles**: {unified['total_unified_profiles']:,}\n"
        report += f"- **Avg Devices per Profile**: {unified['avg_devices_per_profile']:.1f}\n"
        report += f"- **Avg Journey Length**: {unified['avg_journey_length']:.1f} touchpoints\n"
        report += f"- **Cross-Device Conversion Rate**: {unified['cross_device_conversion_rate']:.1%}\n\n"
        
        # Journey Analysis
        journey = insights['journey_analysis']
        if journey.get('top_device_transitions'):
            report += f"## Cross-Device Journey Patterns\n\n"
            report += f"### Top Device Transitions\n\n"
            
            for transition, count in list(journey['top_device_transitions'].items())[:5]:
                report += f"- **{transition}**: {count} transitions\n"
            report += "\n"
        
        if journey.get('conversion_by_device_type'):
            report += f"### Conversions by Device Type\n\n"
            total_conversions = sum(journey['conversion_by_device_type'].values())
            
            for device_type, conversions in journey['conversion_by_device_type'].items():
                percentage = conversions / total_conversions * 100 if total_conversions > 0 else 0
                device_emoji = {"mobile": "📱", "desktop": "💻", "tablet": "📟", "tv": "📺"}.get(device_type, "📊")
                report += f"- {device_emoji} **{device_type.title()}**: {conversions} conversions ({percentage:.1f}%)\n"
            report += "\n"
        
        if journey.get('avg_journey_length_by_device_count'):
            report += f"### Journey Complexity by Device Usage\n\n"
            for device_count, avg_length in journey['avg_journey_length_by_device_count'].items():
                report += f"- **{device_count} Device{'s' if device_count > 1 else ''}**: {avg_length:.1f} avg touchpoints\n"
            report += "\n"
        
        # Attribution Impact
        attribution_df = self.get_unified_attribution()
        if not attribution_df.empty:
            # Compare cross-device vs single-device attribution
            cross_device_attr = attribution_df[attribution_df['user_type'] == 'cross_device']
            single_device_attr = attribution_df[attribution_df['user_type'] == 'single_device']
            
            if not cross_device_attr.empty and not single_device_attr.empty:
                report += f"## Attribution Impact Analysis\n\n"
                
                cross_device_avg_journey = cross_device_attr['total_touchpoints'].mean()
                single_device_avg_journey = single_device_attr['total_touchpoints'].mean()
                
                cross_device_conversion_rate = cross_device_attr.groupby('unified_id')['conversions'].sum().sum() / cross_device_attr.groupby('unified_id')['total_touchpoints'].sum().sum()
                single_device_conversion_rate = single_device_attr.groupby('unified_id')['conversions'].sum().sum() / single_device_attr.groupby('unified_id')['total_touchpoints'].sum().sum()
                
                report += f"- **Cross-Device Avg Journey**: {cross_device_avg_journey:.1f} touchpoints\n"
                report += f"- **Single-Device Avg Journey**: {single_device_avg_journey:.1f} touchpoints\n"
                report += f"- **Cross-Device Conversion Rate**: {cross_device_conversion_rate:.1%}\n"
                report += f"- **Single-Device Conversion Rate**: {single_device_conversion_rate:.1%}\n"
                report += f"- **Journey Length Lift**: {((cross_device_avg_journey - single_device_avg_journey) / single_device_avg_journey * 100):+.1f}%\n\n"
        
        # Strategic Recommendations
        report += f"## Strategic Recommendations\n\n"
        
        cross_device_rate = stats['cross_device_rate']
        if cross_device_rate > 0.3:
            report += f"1. **High Cross-Device Usage**: {cross_device_rate:.0%} of users are cross-device - prioritize unified tracking\n"
        else:
            report += f"1. **Growth Opportunity**: Only {cross_device_rate:.0%} cross-device usage - opportunity to encourage multi-device engagement\n"
        
        if stats['attribution_lift'] > 0.1:
            report += f"2. **Attribution Accuracy**: Cross-device tracking reveals {stats['attribution_lift']:+.0%} more conversions - critical for ROI measurement\n"
        
        if journey.get('top_device_transitions'):
            top_transition = list(journey['top_device_transitions'].keys())[0]
            report += f"3. **Journey Optimization**: Most common transition is {top_transition} - optimize this handoff experience\n"
        
        report += f"4. **Measurement Enhancement**: Implement consistent tracking across all devices for complete attribution\n"
        report += f"5. **Privacy Compliance**: Ensure cross-device tracking meets privacy regulations and user consent requirements\n\n"
        
        # Key Insights
        report += f"## Key Strategic Insights\n\n"
        
        if stats['avg_devices_per_user'] > 2:
            report += f"1. **Multi-Device Reality**: Users average {stats['avg_devices_per_user']:.1f} devices - traditional single-device attribution severely underestimates impact\n"
        
        if unified['avg_journey_length'] > attribution_df[attribution_df['user_type'] == 'single_device']['total_touchpoints'].mean() * 1.5:
            report += f"2. **Complex Journeys**: Cross-device users have significantly longer customer journeys requiring sophisticated attribution\n"
        
        if journey.get('conversion_by_device_type'):
            mobile_conversions = journey['conversion_by_device_type'].get('mobile', 0)
            desktop_conversions = journey['conversion_by_device_type'].get('desktop', 0)
            total_conv = mobile_conversions + desktop_conversions
            
            if total_conv > 0 and mobile_conversions / total_conv > 0.6:
                report += f"3. **Mobile-Centric Conversions**: {mobile_conversions / total_conv:.0%} of conversions happen on mobile - mobile experience is critical\n"
            elif total_conv > 0 and desktop_conversions / total_conv > 0.6:
                report += f"3. **Desktop Conversion Preference**: {desktop_conversions / total_conv:.0%} of conversions on desktop - maintain strong desktop experience\n"
        
        report += f"\n---\n*Advanced cross-device attribution for unified customer journey analysis. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for enterprise implementations.*"
        
        return report


def demo_cross_device_tracking():
    """Executive demonstration of Cross-Device Tracking."""
    
    print("=== Cross-Device Attribution Tracking: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    np.random.seed(42)
    
    # Generate realistic cross-device activity data
    devices = []
    device_id = 1
    
    # Define device characteristics
    device_types = ['mobile', 'desktop', 'tablet']
    user_agents = {
        'mobile': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15',
        'desktop': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124',
        'tablet': 'Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15'
    }
    
    channels = ['Search', 'Display', 'Social', 'Email', 'Direct', 'Video', 'Shopping']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    
    # Generate cross-device users (users with multiple devices)
    cross_device_users = 150
    
    for user_id in range(cross_device_users):
        # Each user has 2-4 devices
        num_devices = np.random.choice([2, 3, 4], p=[0.6, 0.3, 0.1])
        user_devices = np.random.choice(device_types, num_devices, replace=False)
        
        # Shared user characteristics
        user_city = np.random.choice(cities)
        user_ip_base = f"192.168.{np.random.randint(1, 255)}"
        
        # Generate realistic cross-device journey
        total_touchpoints = np.random.randint(5, 20)
        journey_span_days = np.random.exponential(7)  # Average week-long journey
        start_date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 30))
        
        # Create journey across devices
        journey_points = []
        for i in range(total_touchpoints):
            # Device selection - mobile for early discovery, desktop for research/conversion
            if i < total_touchpoints * 0.3:  # Early journey - mobile heavy
                device_type = np.random.choice(['mobile', 'tablet'], p=[0.8, 0.2])
            elif i > total_touchpoints * 0.7:  # Late journey - desktop heavy for conversion
                device_type = np.random.choice(['desktop', 'mobile'], p=[0.7, 0.3])
            else:  # Middle - mixed
                device_type = np.random.choice(user_devices)
            
            # Channel selection based on device
            if device_type == 'mobile':
                channel = np.random.choice(['Social', 'Search', 'Display', 'Video'], p=[0.4, 0.3, 0.2, 0.1])
            elif device_type == 'tablet':
                channel = np.random.choice(['Display', 'Social', 'Search', 'Video'], p=[0.3, 0.3, 0.2, 0.2])
            else:  # desktop
                channel = np.random.choice(['Search', 'Direct', 'Email', 'Display'], p=[0.4, 0.2, 0.2, 0.2])
            
            # Timestamp
            progress = i / max(total_touchpoints - 1, 1)
            days_offset = progress * journey_span_days
            
            # Add device-specific usage patterns
            if device_type == 'mobile':
                # Mobile used more during commute hours and evenings
                hour = np.random.choice([8, 9, 17, 18, 19, 20, 21], p=[0.1, 0.15, 0.15, 0.2, 0.15, 0.15, 0.1])
            elif device_type == 'desktop':
                # Desktop used during work hours and late evening
                hour = np.random.choice([10, 11, 14, 15, 22, 23], p=[0.2, 0.2, 0.15, 0.15, 0.15, 0.15])
            else:  # tablet
                # Tablet used for leisure browsing
                hour = np.random.choice([19, 20, 21, 22], p=[0.25, 0.25, 0.25, 0.25])
            
            timestamp = start_date + pd.Timedelta(days=days_offset, hours=hour, minutes=np.random.randint(0, 60))
            
            journey_points.append({
                'device_type': device_type,
                'channel': channel,
                'timestamp': timestamp,
                'touchpoint': f"{channel.lower()}_{device_type}_interaction"
            })
        
        # Sort journey by timestamp
        journey_points.sort(key=lambda x: x['timestamp'])
        
        # Create device records
        device_journey_map = defaultdict(list)
        for point in journey_points:
            device_journey_map[point['device_type']].append(point)
        
        # Conversion probability - higher for cross-device users
        converted = np.random.random() < 0.35
        
        for device_type, device_journey in device_journey_map.items():
            # Create unique device ID
            current_device_id = f"device_{device_id}"
            device_id += 1
            
            # Add some device linking signals for matching
            device_ip = f"{user_ip_base}.{np.random.randint(1, 254)}"  # Same network
            
            for point in device_journey:
                devices.append({
                    'device_id': current_device_id,
                    'user_agent': user_agents.get(device_type, ''),
                    'ip_address': device_ip,
                    'timestamp': point['timestamp'],
                    'touchpoint': point['touchpoint'],
                    'channel': point['channel'],
                    'converted': converted and point == device_journey[-1],  # Last touchpoint converts
                    'country': 'US',
                    'region': 'NY' if user_city in ['New York'] else 'CA',
                    'city': user_city,
                    'screen_width': 390 if device_type == 'mobile' else 768 if device_type == 'tablet' else 1920
                })
    
    # Generate single-device users
    single_device_users = 100
    
    for user_id in range(single_device_users):
        device_type = np.random.choice(device_types, p=[0.5, 0.4, 0.1])
        
        touchpoints = np.random.randint(1, 8)
        journey_span_days = np.random.exponential(3)  # Shorter journeys
        start_date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 30))
        
        current_device_id = f"device_{device_id}"
        device_id += 1
        
        converted = np.random.random() < 0.15  # Lower conversion rate
        
        for i in range(touchpoints):
            if device_type == 'mobile':
                channel = np.random.choice(['Social', 'Search', 'Display'], p=[0.5, 0.3, 0.2])
                hour = np.random.choice([12, 18, 19, 20], p=[0.2, 0.3, 0.25, 0.25])
            else:
                channel = np.random.choice(['Search', 'Direct', 'Display'], p=[0.5, 0.3, 0.2])
                hour = np.random.choice([10, 14, 15, 21], p=[0.3, 0.2, 0.2, 0.3])
            
            progress = i / max(touchpoints - 1, 1)
            timestamp = start_date + pd.Timedelta(
                days=progress * journey_span_days, 
                hours=hour, 
                minutes=np.random.randint(0, 60)
            )
            
            devices.append({
                'device_id': current_device_id,
                'user_agent': user_agents.get(device_type, ''),
                'ip_address': f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                'timestamp': timestamp,
                'touchpoint': f"{channel.lower()}_{device_type}_interaction",
                'channel': channel,
                'converted': converted and i == touchpoints - 1,
                'country': 'US',
                'region': np.random.choice(['CA', 'TX', 'FL']),
                'city': np.random.choice(cities),
                'screen_width': 390 if device_type == 'mobile' else 768 if device_type == 'tablet' else 1920
            })
    
    activity_data = pd.DataFrame(devices)
    
    print(f"📊 Generated activity for {activity_data['device_id'].nunique()} devices")
    print(f"📱 Device Types: {activity_data['user_agent'].apply(lambda x: 'mobile' if 'iPhone' in x else 'desktop' if 'Windows' in x else 'tablet').value_counts().to_dict()}")
    print(f"📈 Total touchpoints: {len(activity_data):,}")
    print(f"💰 Total conversions: {activity_data['converted'].sum()}")
    
    # Initialize and run cross-device tracking
    tracker = CrossDeviceTracker(
        min_confidence_threshold=0.6,
        time_window_hours=48,
        behavioral_weight=0.4,
        temporal_weight=0.3,
        geo_weight=0.3
    )
    
    print(f"\n🔍 Tracking device activity and building profiles...")
    tracker.track_device_activity(activity_data)
    
    print(f"🔗 Performing cross-device matching...")
    tracker.perform_device_matching()
    
    print("\n📋 CROSS-DEVICE ATTRIBUTION RESULTS")
    print("=" * 50)
    
    # Get comprehensive insights
    insights = tracker.get_cross_device_insights()
    
    # Display key statistics
    stats = insights['cross_device_statistics']
    print(f"\n📊 Cross-Device Overview:")
    print(f"  • Total Devices: {stats['total_devices']:,}")
    print(f"  • Cross-Device Users: {stats['cross_device_users']:,}")
    print(f"  • Single-Device Users: {stats['single_device_users']:,}")
    print(f"  • Cross-Device Rate: {stats['cross_device_rate']:.1%}")
    print(f"  • Avg Devices per User: {stats['avg_devices_per_user']:.1f}")
    print(f"  • Attribution Lift: {stats['attribution_lift']:+.1%}")
    
    # Device matching performance
    matches = insights['device_matches_summary']
    print(f"\n🔗 Device Matching Performance:")
    print(f"  • Total Matches: {matches['total_matches']}")
    print(f"  • Avg Confidence: {matches['avg_confidence_score']:.1%}")
    
    if matches['matches_by_method']:
        print(f"\n  Methods Used:")
        for method, count in matches['matches_by_method'].items():
            method_icon = {"deterministic": "🎯", "probabilistic": "📊", "behavioral": "🧠", "hybrid": "🔀"}.get(method, "📈")
            print(f"  {method_icon} {method.replace('_', ' ').title()}: {count}")
    
    if matches['matches_by_confidence']:
        print(f"\n  Confidence Distribution:")
        for confidence, count in matches['matches_by_confidence'].items():
            conf_icon = {"very_high": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢", "very_low": "⚪"}.get(confidence, "📊")
            print(f"  {conf_icon} {confidence.replace('_', ' ').title()}: {count}")
    
    # Cross-device journey analysis
    journey = insights['journey_analysis']
    
    if journey.get('top_device_transitions'):
        print(f"\n🔄 Top Device Transitions:")
        for i, (transition, count) in enumerate(list(journey['top_device_transitions'].items())[:5], 1):
            print(f"  {i}. {transition}: {count} transitions")
    
    if journey.get('conversion_by_device_type'):
        print(f"\n💰 Conversions by Device Type:")
        total_conv = sum(journey['conversion_by_device_type'].values())
        for device_type, conversions in journey['conversion_by_device_type'].items():
            device_icon = {"mobile": "📱", "desktop": "💻", "tablet": "📟"}.get(device_type, "📊")
            percentage = conversions / total_conv * 100 if total_conv > 0 else 0
            print(f"  {device_icon} {device_type.title()}: {conversions} ({percentage:.1f}%)")
    
    if journey.get('avg_journey_length_by_device_count'):
        print(f"\n📏 Journey Length by Device Usage:")
        for device_count, avg_length in journey['avg_journey_length_by_device_count'].items():
            devices_icon = "📱" if device_count == 1 else "📱💻" if device_count == 2 else "📱💻📟"
            print(f"  {devices_icon} {device_count} Device{'s' if device_count > 1 else ''}: {avg_length:.1f} avg touchpoints")
    
    # Attribution comparison
    attribution_df = tracker.get_unified_attribution()
    
    if not attribution_df.empty:
        cross_device = attribution_df[attribution_df['user_type'] == 'cross_device']
        single_device = attribution_df[attribution_df['user_type'] == 'single_device']
        
        print(f"\n📈 Attribution Impact Analysis:")
        if not cross_device.empty:
            print(f"  • Cross-Device Users: {cross_device['unified_id'].nunique()}")
            print(f"  • Avg Cross-Device Journey: {cross_device['total_touchpoints'].mean():.1f} touchpoints")
            print(f"  • Cross-Device Conversions: {cross_device['conversions'].sum()}")
        
        if not single_device.empty:
            print(f"  • Single-Device Users: {single_device['unified_id'].nunique()}")
            print(f"  • Avg Single-Device Journey: {single_device['total_touchpoints'].mean():.1f} touchpoints") 
            print(f"  • Single-Device Conversions: {single_device['conversions'].sum()}")
        
        # Channel attribution comparison
        print(f"\n🏆 Top Channels by Attribution:")
        channel_attribution = attribution_df.groupby('channel')['attribution_weight'].sum().sort_values(ascending=False)
        
        for i, (channel, attribution) in enumerate(channel_attribution.head(5).items(), 1):
            channel_icon = {"Search": "🔍", "Social": "📱", "Display": "🖥️", "Email": "📧", "Direct": "🎯", "Video": "🎬", "Shopping": "🛒"}.get(channel, "📊")
            print(f"  {channel_icon} {i}. {channel}: {attribution:.1%} attribution")
        
        # Cross-device vs single-device channel preferences
        if not cross_device.empty and not single_device.empty:
            print(f"\n🔍 Channel Preferences Comparison:")
            
            cross_device_channels = cross_device.groupby('channel')['attribution_weight'].sum().sort_values(ascending=False)
            single_device_channels = single_device.groupby('channel')['attribution_weight'].sum().sort_values(ascending=False)
            
            print("  Cross-Device Top Channels:")
            for channel, weight in cross_device_channels.head(3).items():
                print(f"    • {channel}: {weight:.1%}")
            
            print("  Single-Device Top Channels:")
            for channel, weight in single_device_channels.head(3).items():
                print(f"    • {channel}: {weight:.1%}")
    
    # Sample unified profiles
    print(f"\n👥 Sample Unified Profiles:")
    for i, (profile_id, profile) in enumerate(list(tracker.unified_profiles.items())[:3], 1):
        device_types = [tracker.device_profiles[device_id].device_type.value for device_id in profile.device_ids]
        print(f"  {i}. {profile_id}:")
        print(f"     Devices: {', '.join(set(device_types))} ({len(profile.device_ids)} total)")
        print(f"     Journey: {profile.total_touchpoints} touchpoints over {profile.journey_span_days:.1f} days")
        print(f"     Conversions: {profile.total_conversions}")
    
    print("\n" + "="*60)
    print("🚀 Advanced cross-device attribution for unified customer tracking")
    print("💼 Complete journey visibility across all touchpoints and devices") 
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_cross_device_tracking()