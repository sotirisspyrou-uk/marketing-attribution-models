"""
Offline-Online Integration System

Advanced system for integrating offline marketing activities with online customer
journeys, enabling comprehensive cross-channel attribution and unified customer
experience measurement across all touchpoints.

Author: Sotiris Spyrou
Portfolio: https://verityai.co
LinkedIn: https://www.linkedin.com/in/sspyrou/

DISCLAIMER: This is demonstration code for portfolio purposes only.
Not intended for production use without proper testing and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import uuid
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
import math

logger = logging.getLogger(__name__)


class OfflineChannel(Enum):
    """Types of offline marketing channels."""
    RETAIL_STORE = "retail_store"
    PRINT_AD = "print_ad"
    RADIO = "radio"
    TV_COMMERCIAL = "tv_commercial"
    BILLBOARD = "billboard"
    EVENT = "event"
    DIRECT_MAIL = "direct_mail"
    PHONE_CALL = "phone_call"
    SALES_MEETING = "sales_meeting"
    TRADE_SHOW = "trade_show"


class IntegrationMethod(Enum):
    """Methods for offline-online integration."""
    PROMO_CODE = "promo_code"
    PHONE_NUMBER = "phone_number"
    QR_CODE = "qr_code"
    CUSTOM_URL = "custom_url"
    GEOFENCING = "geofencing"
    TIME_WINDOW = "time_window"
    STORE_VISIT = "store_visit"
    COUPON_CODE = "coupon_code"


@dataclass
class OfflineTouchpoint:
    """Offline marketing touchpoint."""
    touchpoint_id: str
    customer_id: Optional[str]
    channel: OfflineChannel
    timestamp: datetime
    location: Optional[Tuple[float, float]] = None  # (latitude, longitude)
    campaign: Optional[str] = None
    spend: float = 0.0
    impressions: int = 0
    reach: int = 0
    integration_method: Optional[IntegrationMethod] = None
    integration_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OnlineTouchpoint:
    """Online marketing touchpoint."""
    touchpoint_id: str
    customer_id: str
    channel: str
    timestamp: datetime
    session_id: Optional[str] = None
    url: Optional[str] = None
    referrer: Optional[str] = None
    campaign: Optional[str] = None
    medium: Optional[str] = None
    source: Optional[str] = None
    converted: bool = False
    revenue: float = 0.0
    integration_signals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegratedJourney:
    """Unified offline-online customer journey."""
    customer_id: str
    journey_id: str
    offline_touchpoints: List[OfflineTouchpoint]
    online_touchpoints: List[OnlineTouchpoint]
    integration_points: List[Dict[str, Any]]
    start_timestamp: datetime
    end_timestamp: datetime
    total_offline_spend: float = 0.0
    total_online_spend: float = 0.0
    converted: bool = False
    revenue: float = 0.0
    confidence_score: float = 0.0


class OfflineOnlineIntegrator:
    """
    Advanced offline-online integration system.
    
    Combines offline marketing activities with online customer journeys
    to create unified attribution models and comprehensive customer insights.
    """
    
    def __init__(self,
                 geofence_radius_km: float = 1.0,
                 time_window_hours: int = 24,
                 confidence_threshold: float = 0.7,
                 enable_ml_matching: bool = True):
        """
        Initialize Offline-Online Integrator.
        
        Args:
            geofence_radius_km: Radius for geofencing integration in kilometers
            time_window_hours: Time window for temporal integration
            confidence_threshold: Minimum confidence for integration
            enable_ml_matching: Enable machine learning-based matching
        """
        self.geofence_radius_km = geofence_radius_km
        self.time_window_hours = time_window_hours
        self.confidence_threshold = confidence_threshold
        self.enable_ml_matching = enable_ml_matching
        
        # Data storage
        self.offline_touchpoints: List[OfflineTouchpoint] = []
        self.online_touchpoints: List[OnlineTouchpoint] = []
        self.integrated_journeys: List[IntegratedJourney] = []
        
        # Integration tracking
        self.integration_rules = {}
        self.integration_stats = {}
        self.attribution_results = {}
        
        # ML components
        self.scaler = StandardScaler()
        self.clustering_model = None
        
        self._initialize_integration_rules()
        
        logger.info("Offline-online integration system initialized")
    
    def _initialize_integration_rules(self):
        """Initialize integration rules and patterns."""
        
        self.integration_rules = {
            # Direct integration methods
            'promo_code_match': {
                'method': IntegrationMethod.PROMO_CODE,
                'confidence': 0.95,
                'time_window_hours': 168,  # 1 week
                'description': 'Direct promo code matching'
            },
            'phone_number_match': {
                'method': IntegrationMethod.PHONE_NUMBER,
                'confidence': 0.90,
                'time_window_hours': 72,  # 3 days
                'description': 'Phone number tracking'
            },
            'qr_code_scan': {
                'method': IntegrationMethod.QR_CODE,
                'confidence': 0.98,
                'time_window_hours': 24,  # 1 day
                'description': 'QR code scan tracking'
            },
            'custom_url_visit': {
                'method': IntegrationMethod.CUSTOM_URL,
                'confidence': 0.85,
                'time_window_hours': 48,  # 2 days
                'description': 'Custom URL campaign tracking'
            },
            
            # Proximity-based integration
            'geofence_match': {
                'method': IntegrationMethod.GEOFENCING,
                'confidence': 0.70,
                'time_window_hours': 6,  # 6 hours
                'description': 'Location-based geofencing'
            },
            'store_visit_attribution': {
                'method': IntegrationMethod.STORE_VISIT,
                'confidence': 0.75,
                'time_window_hours': 12,  # 12 hours
                'description': 'Physical store visit tracking'
            },
            
            # Temporal integration
            'time_window_correlation': {
                'method': IntegrationMethod.TIME_WINDOW,
                'confidence': 0.60,
                'time_window_hours': 24,  # 1 day
                'description': 'Temporal correlation analysis'
            },
            
            # Coupon-based integration
            'coupon_redemption': {
                'method': IntegrationMethod.COUPON_CODE,
                'confidence': 0.92,
                'time_window_hours': 336,  # 2 weeks
                'description': 'Coupon code redemption tracking'
            }
        }
    
    def add_offline_touchpoints(self, offline_data: pd.DataFrame) -> 'OfflineOnlineIntegrator':
        """
        Add offline marketing touchpoints.
        
        Args:
            offline_data: DataFrame with offline touchpoint data
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Adding {len(offline_data)} offline touchpoints")
        
        # Validate required columns
        required_columns = ['timestamp', 'channel', 'campaign']
        if not all(col in offline_data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        # Process each offline touchpoint
        for _, row in offline_data.iterrows():
            # Parse location if available
            location = None
            if 'latitude' in row and 'longitude' in row:
                location = (float(row['latitude']), float(row['longitude']))
            
            # Parse integration method
            integration_method = None
            integration_data = {}
            
            if 'integration_method' in row:
                method_str = str(row['integration_method']).lower()
                for method in IntegrationMethod:
                    if method.value in method_str:
                        integration_method = method
                        break
            
            # Extract integration data
            for col in row.index:
                if col.startswith('integration_'):
                    integration_data[col.replace('integration_', '')] = row[col]
            
            # Create offline touchpoint
            touchpoint = OfflineTouchpoint(
                touchpoint_id=str(uuid.uuid4()),
                customer_id=row.get('customer_id'),
                channel=OfflineChannel(row['channel'].lower().replace(' ', '_')),
                timestamp=pd.to_datetime(row['timestamp']),
                location=location,
                campaign=row.get('campaign'),
                spend=row.get('spend', 0.0),
                impressions=row.get('impressions', 0),
                reach=row.get('reach', 0),
                integration_method=integration_method,
                integration_data=integration_data,
                metadata={k: v for k, v in row.items() if not k.startswith(('integration_', 'latitude', 'longitude'))}
            )
            
            self.offline_touchpoints.append(touchpoint)
        
        logger.info(f"Added {len(offline_data)} offline touchpoints")
        return self
    
    def add_online_touchpoints(self, online_data: pd.DataFrame) -> 'OfflineOnlineIntegrator':
        """
        Add online marketing touchpoints.
        
        Args:
            online_data: DataFrame with online touchpoint data
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Adding {len(online_data)} online touchpoints")
        
        # Validate required columns
        required_columns = ['customer_id', 'timestamp', 'channel']
        if not all(col in online_data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        # Process each online touchpoint
        for _, row in online_data.iterrows():
            # Extract integration signals
            integration_signals = {}
            for col in row.index:
                if col.startswith('utm_') or col in ['promo_code', 'referrer_campaign']:
                    integration_signals[col] = row[col]
            
            # Create online touchpoint
            touchpoint = OnlineTouchpoint(
                touchpoint_id=str(uuid.uuid4()),
                customer_id=row['customer_id'],
                channel=row['channel'],
                timestamp=pd.to_datetime(row['timestamp']),
                session_id=row.get('session_id'),
                url=row.get('url'),
                referrer=row.get('referrer'),
                campaign=row.get('campaign'),
                medium=row.get('medium'),
                source=row.get('source'),
                converted=row.get('converted', False),
                revenue=row.get('revenue', 0.0),
                integration_signals=integration_signals
            )
            
            self.online_touchpoints.append(touchpoint)
        
        logger.info(f"Added {len(online_data)} online touchpoints")
        return self
    
    def integrate_journeys(self) -> 'OfflineOnlineIntegrator':
        """
        Integrate offline and online touchpoints into unified journeys.
        
        Returns:
            Self for method chaining
        """
        logger.info("Starting offline-online journey integration")
        
        # Clear previous results
        self.integrated_journeys = []
        
        # Group online touchpoints by customer
        online_by_customer = defaultdict(list)
        for tp in self.online_touchpoints:
            online_by_customer[tp.customer_id].append(tp)
        
        # Process each customer's online journey
        for customer_id, online_tps in online_by_customer.items():
            # Find matching offline touchpoints
            offline_matches = self._find_offline_matches(customer_id, online_tps)
            
            if offline_matches or online_tps:  # Create journey if we have any touchpoints
                # Create integrated journey
                journey = self._create_integrated_journey(
                    customer_id, offline_matches, online_tps
                )
                
                if journey.confidence_score >= self.confidence_threshold:
                    self.integrated_journeys.append(journey)
        
        # Calculate integration statistics
        self._calculate_integration_stats()
        
        # Calculate attribution
        self._calculate_integrated_attribution()
        
        logger.info(f"Created {len(self.integrated_journeys)} integrated journeys")
        return self
    
    def _find_offline_matches(self, customer_id: str, online_touchpoints: List[OnlineTouchpoint]) -> List[OfflineTouchpoint]:
        """Find offline touchpoints that match with online journey."""
        
        matched_offline = []
        
        for offline_tp in self.offline_touchpoints:
            # Skip if offline touchpoint already assigned to a specific customer
            if offline_tp.customer_id and offline_tp.customer_id != customer_id:
                continue
            
            # Check various integration methods
            match_confidence = self._calculate_match_confidence(offline_tp, online_touchpoints)
            
            if match_confidence >= self.confidence_threshold:
                # Mark the match confidence
                offline_tp.metadata['match_confidence'] = match_confidence
                offline_tp.metadata['matched_customer'] = customer_id
                matched_offline.append(offline_tp)
        
        return matched_offline
    
    def _calculate_match_confidence(self, offline_tp: OfflineTouchpoint, online_tps: List[OnlineTouchpoint]) -> float:
        """Calculate confidence score for offline-online matching."""
        
        max_confidence = 0.0
        
        for online_tp in online_tps:
            confidence_scores = []
            
            # Direct integration method matching
            if offline_tp.integration_method:
                direct_confidence = self._check_direct_integration(offline_tp, online_tp)
                if direct_confidence > 0:
                    confidence_scores.append(direct_confidence)
            
            # Geofencing matching
            if offline_tp.location:
                geo_confidence = self._check_geofencing_match(offline_tp, online_tp)
                if geo_confidence > 0:
                    confidence_scores.append(geo_confidence)
            
            # Temporal correlation
            temporal_confidence = self._check_temporal_correlation(offline_tp, online_tp)
            if temporal_confidence > 0:
                confidence_scores.append(temporal_confidence)
            
            # Campaign matching
            campaign_confidence = self._check_campaign_correlation(offline_tp, online_tp)
            if campaign_confidence > 0:
                confidence_scores.append(campaign_confidence)
            
            # Take maximum confidence for this online touchpoint
            if confidence_scores:
                max_confidence = max(max_confidence, max(confidence_scores))
        
        return max_confidence
    
    def _check_direct_integration(self, offline_tp: OfflineTouchpoint, online_tp: OnlineTouchpoint) -> float:
        """Check direct integration methods (promo codes, URLs, etc.)."""
        
        method = offline_tp.integration_method
        confidence = 0.0
        
        if method == IntegrationMethod.PROMO_CODE:
            # Check for promo code in online signals
            offline_promo = offline_tp.integration_data.get('promo_code', '').lower()
            online_promo = online_tp.integration_signals.get('promo_code', '').lower()
            
            if offline_promo and offline_promo == online_promo:
                confidence = 0.95
        
        elif method == IntegrationMethod.CUSTOM_URL:
            # Check for custom URL pattern
            offline_url = offline_tp.integration_data.get('tracking_url', '').lower()
            online_url = online_tp.url.lower() if online_tp.url else ''
            
            if offline_url and offline_url in online_url:
                confidence = 0.85
        
        elif method == IntegrationMethod.QR_CODE:
            # Check for QR code tracking parameter
            qr_param = online_tp.integration_signals.get('utm_source', '').lower()
            offline_qr = offline_tp.integration_data.get('qr_code_id', '').lower()
            
            if 'qr' in qr_param or (offline_qr and offline_qr in qr_param):
                confidence = 0.98
        
        elif method == IntegrationMethod.PHONE_NUMBER:
            # Check for phone call tracking
            phone_utm = online_tp.integration_signals.get('utm_medium', '').lower()
            
            if 'phone' in phone_utm or 'call' in phone_utm:
                confidence = 0.90
        
        elif method == IntegrationMethod.COUPON_CODE:
            # Check for coupon redemption
            coupon_code = online_tp.integration_signals.get('coupon', '').lower()
            offline_coupon = offline_tp.integration_data.get('coupon_code', '').lower()
            
            if offline_coupon and offline_coupon == coupon_code:
                confidence = 0.92
        
        # Apply time window constraint
        if confidence > 0:
            rule = self.integration_rules.get(f'{method.value}_match', {})
            time_window = rule.get('time_window_hours', 24)
            time_diff = abs((online_tp.timestamp - offline_tp.timestamp).total_seconds() / 3600)
            
            if time_diff <= time_window:
                return confidence
        
        return 0.0
    
    def _check_geofencing_match(self, offline_tp: OfflineTouchpoint, online_tp: OnlineTouchpoint) -> float:
        """Check geofencing-based matching."""
        
        if not offline_tp.location:
            return 0.0
        
        # Check if online touchpoint has location data
        online_location = None
        
        # Look for location in integration signals or metadata
        if 'latitude' in online_tp.integration_signals and 'longitude' in online_tp.integration_signals:
            try:
                lat = float(online_tp.integration_signals['latitude'])
                lon = float(online_tp.integration_signals['longitude'])
                online_location = (lat, lon)
            except (ValueError, TypeError):
                pass
        
        if not online_location:
            return 0.0
        
        # Calculate distance using haversine formula
        offline_coords = np.array([[math.radians(offline_tp.location[0]), math.radians(offline_tp.location[1])]])
        online_coords = np.array([[math.radians(online_location[0]), math.radians(online_location[1])]])
        
        distance_matrix = haversine_distances(offline_coords, online_coords)
        distance_km = distance_matrix[0, 0] * 6371  # Earth's radius in km
        
        # Return confidence based on distance
        if distance_km <= self.geofence_radius_km:
            # Higher confidence for closer proximity
            confidence = max(0.7, 1.0 - (distance_km / self.geofence_radius_km) * 0.3)
            
            # Apply time window
            time_diff = abs((online_tp.timestamp - offline_tp.timestamp).total_seconds() / 3600)
            if time_diff <= 6:  # 6-hour window for geofencing
                return confidence
        
        return 0.0
    
    def _check_temporal_correlation(self, offline_tp: OfflineTouchpoint, online_tp: OnlineTouchpoint) -> float:
        """Check temporal correlation between offline and online touchpoints."""
        
        time_diff_hours = abs((online_tp.timestamp - offline_tp.timestamp).total_seconds() / 3600)
        
        # Strong correlation for touchpoints within time window
        if time_diff_hours <= self.time_window_hours:
            # Exponential decay based on time difference
            confidence = 0.6 * math.exp(-time_diff_hours / (self.time_window_hours / 3))
            return max(0.3, confidence)  # Minimum 30% confidence within window
        
        return 0.0
    
    def _check_campaign_correlation(self, offline_tp: OfflineTouchpoint, online_tp: OnlineTouchpoint) -> float:
        """Check campaign name correlation."""
        
        if not offline_tp.campaign or not online_tp.campaign:
            return 0.0
        
        offline_campaign = offline_tp.campaign.lower()
        online_campaign = online_tp.campaign.lower()
        
        # Exact match
        if offline_campaign == online_campaign:
            return 0.8
        
        # Partial match (campaign contains similar keywords)
        offline_words = set(offline_campaign.split())
        online_words = set(online_campaign.split())
        
        if offline_words.intersection(online_words):
            overlap_ratio = len(offline_words.intersection(online_words)) / len(offline_words.union(online_words))
            return 0.5 + (overlap_ratio * 0.3)  # 0.5 to 0.8 based on overlap
        
        return 0.0
    
    def _create_integrated_journey(self, customer_id: str, 
                                 offline_tps: List[OfflineTouchpoint], 
                                 online_tps: List[OnlineTouchpoint]) -> IntegratedJourney:
        """Create integrated journey from offline and online touchpoints."""
        
        # Sort touchpoints by timestamp
        all_timestamps = []
        if offline_tps:
            all_timestamps.extend([tp.timestamp for tp in offline_tps])
        if online_tps:
            all_timestamps.extend([tp.timestamp for tp in online_tps])
        
        all_timestamps.sort()
        start_timestamp = all_timestamps[0] if all_timestamps else datetime.now()
        end_timestamp = all_timestamps[-1] if all_timestamps else datetime.now()
        
        # Calculate totals
        total_offline_spend = sum(tp.spend for tp in offline_tps)
        total_online_spend = 0.0  # Would need spend data in online touchpoints
        
        # Check conversion
        converted = any(tp.converted for tp in online_tps)
        revenue = sum(tp.revenue for tp in online_tps)
        
        # Calculate confidence score
        confidence_scores = []
        for offline_tp in offline_tps:
            if 'match_confidence' in offline_tp.metadata:
                confidence_scores.append(offline_tp.metadata['match_confidence'])
        
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 1.0
        
        # Identify integration points
        integration_points = []
        for offline_tp in offline_tps:
            if offline_tp.integration_method:
                integration_points.append({
                    'method': offline_tp.integration_method.value,
                    'timestamp': offline_tp.timestamp,
                    'confidence': offline_tp.metadata.get('match_confidence', 0.0),
                    'offline_touchpoint_id': offline_tp.touchpoint_id
                })
        
        journey = IntegratedJourney(
            customer_id=customer_id,
            journey_id=str(uuid.uuid4()),
            offline_touchpoints=offline_tps,
            online_touchpoints=online_tps,
            integration_points=integration_points,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            total_offline_spend=total_offline_spend,
            total_online_spend=total_online_spend,
            converted=converted,
            revenue=revenue,
            confidence_score=overall_confidence
        )
        
        return journey
    
    def _calculate_integration_stats(self):
        """Calculate integration statistics."""
        
        total_offline = len(self.offline_touchpoints)
        total_online = len(self.online_touchpoints)
        total_integrated = len(self.integrated_journeys)
        
        # Integration method usage
        method_usage = Counter()
        confidence_scores = []
        
        for journey in self.integrated_journeys:
            confidence_scores.append(journey.confidence_score)
            for integration_point in journey.integration_points:
                method_usage[integration_point['method']] += 1
        
        # Calculate metrics
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        integration_rate = total_integrated / total_online if total_online > 0 else 0.0
        
        self.integration_stats = {
            'total_offline_touchpoints': total_offline,
            'total_online_touchpoints': total_online,
            'integrated_journeys': total_integrated,
            'integration_rate': integration_rate,
            'average_confidence': avg_confidence,
            'integration_methods_used': dict(method_usage),
            'confidence_distribution': {
                'high_confidence': sum(1 for score in confidence_scores if score >= 0.8),
                'medium_confidence': sum(1 for score in confidence_scores if 0.6 <= score < 0.8),
                'low_confidence': sum(1 for score in confidence_scores if score < 0.6)
            }
        }
        
        logger.info(f"Integration stats calculated: {integration_rate:.1%} integration rate")
    
    def _calculate_integrated_attribution(self):
        """Calculate attribution across integrated offline-online journeys."""
        
        # Channel attribution
        channel_attribution = defaultdict(lambda: {
            'touchpoints': 0,
            'customers': set(),
            'offline_spend': 0.0,
            'online_conversions': 0,
            'revenue': 0.0
        })
        
        for journey in self.integrated_journeys:
            # Offline channels
            for offline_tp in journey.offline_touchpoints:
                channel = f"offline_{offline_tp.channel.value}"
                channel_attribution[channel]['touchpoints'] += 1
                channel_attribution[channel]['customers'].add(journey.customer_id)
                channel_attribution[channel]['offline_spend'] += offline_tp.spend
                
                if journey.converted:
                    channel_attribution[channel]['online_conversions'] += 1
                    channel_attribution[channel]['revenue'] += journey.revenue
            
            # Online channels
            for online_tp in journey.online_touchpoints:
                channel = f"online_{online_tp.channel}"
                channel_attribution[channel]['touchpoints'] += 1
                channel_attribution[channel]['customers'].add(journey.customer_id)
                channel_attribution[channel]['revenue'] += online_tp.revenue
                
                if online_tp.converted:
                    channel_attribution[channel]['online_conversions'] += 1
        
        # Calculate final attribution metrics
        attribution_results = {}
        
        for channel, metrics in channel_attribution.items():
            unique_customers = len(metrics['customers'])
            
            attribution_results[channel] = {
                'touchpoints': metrics['touchpoints'],
                'unique_customers': unique_customers,
                'offline_spend': metrics['offline_spend'],
                'online_conversions': metrics['online_conversions'],
                'revenue': metrics['revenue'],
                'conversion_rate': metrics['online_conversions'] / unique_customers if unique_customers > 0 else 0.0,
                'roas': metrics['revenue'] / metrics['offline_spend'] if metrics['offline_spend'] > 0 else 0.0,
                'revenue_per_customer': metrics['revenue'] / unique_customers if unique_customers > 0 else 0.0
            }
        
        self.attribution_results = attribution_results
        
        logger.info(f"Calculated attribution for {len(attribution_results)} integrated channels")
    
    def get_integration_insights(self) -> Dict[str, Any]:
        """Get comprehensive integration insights."""
        
        insights = {
            'integration_statistics': self.integration_stats,
            'attribution_results': self.attribution_results,
            'journey_analysis': self._analyze_integrated_journeys(),
            'offline_impact': self._calculate_offline_impact(),
            'integration_opportunities': self._identify_integration_opportunities()
        }
        
        return insights
    
    def _analyze_integrated_journeys(self) -> Dict[str, Any]:
        """Analyze integrated journey patterns."""
        
        if not self.integrated_journeys:
            return {}
        
        # Journey characteristics
        journey_lengths = []
        offline_ratios = []
        time_spans = []
        
        for journey in self.integrated_journeys:
            total_touchpoints = len(journey.offline_touchpoints) + len(journey.online_touchpoints)
            offline_ratio = len(journey.offline_touchpoints) / total_touchpoints if total_touchpoints > 0 else 0
            time_span = (journey.end_timestamp - journey.start_timestamp).days
            
            journey_lengths.append(total_touchpoints)
            offline_ratios.append(offline_ratio)
            time_spans.append(time_span)
        
        # Common integration patterns
        integration_patterns = Counter()
        for journey in self.integrated_journeys:
            methods = sorted([ip['method'] for ip in journey.integration_points])
            if methods:
                integration_patterns[','.join(methods)] += 1
        
        return {
            'total_integrated_journeys': len(self.integrated_journeys),
            'average_journey_length': np.mean(journey_lengths),
            'average_offline_ratio': np.mean(offline_ratios),
            'average_time_span_days': np.mean(time_spans),
            'common_integration_patterns': dict(integration_patterns.most_common(5)),
            'conversion_rate': sum(1 for j in self.integrated_journeys if j.converted) / len(self.integrated_journeys),
            'total_revenue': sum(j.revenue for j in self.integrated_journeys)
        }
    
    def _calculate_offline_impact(self) -> Dict[str, Any]:
        """Calculate the impact of offline marketing on online conversions."""
        
        # Compare journeys with vs without offline touchpoints
        with_offline = [j for j in self.integrated_journeys if j.offline_touchpoints]
        without_offline = [j for j in self.integrated_journeys if not j.offline_touchpoints]
        
        if not with_offline:
            return {'message': 'No offline touchpoints found in integrated journeys'}
        
        # Calculate metrics
        with_offline_conversion = sum(1 for j in with_offline if j.converted) / len(with_offline)
        without_offline_conversion = sum(1 for j in without_offline if j.converted) / len(without_offline) if without_offline else 0
        
        with_offline_revenue = sum(j.revenue for j in with_offline) / len(with_offline)
        without_offline_revenue = sum(j.revenue for j in without_offline) / len(without_offline) if without_offline else 0
        
        # Offline spend efficiency
        total_offline_spend = sum(j.total_offline_spend for j in with_offline)
        total_online_revenue = sum(j.revenue for j in with_offline)
        offline_roas = total_online_revenue / total_offline_spend if total_offline_spend > 0 else 0
        
        return {
            'journeys_with_offline': len(with_offline),
            'journeys_without_offline': len(without_offline),
            'conversion_rate_with_offline': with_offline_conversion,
            'conversion_rate_without_offline': without_offline_conversion,
            'conversion_lift': with_offline_conversion - without_offline_conversion,
            'avg_revenue_with_offline': with_offline_revenue,
            'avg_revenue_without_offline': without_offline_revenue,
            'revenue_lift': with_offline_revenue - without_offline_revenue,
            'offline_to_online_roas': offline_roas,
            'total_offline_investment': total_offline_spend
        }
    
    def _identify_integration_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities to improve offline-online integration."""
        
        opportunities = []
        
        # Unmatched offline touchpoints
        unmatched_offline = []
        for offline_tp in self.offline_touchpoints:
            if 'matched_customer' not in offline_tp.metadata:
                unmatched_offline.append(offline_tp)
        
        if unmatched_offline:
            opportunities.append({
                'type': 'unmatched_offline_touchpoints',
                'count': len(unmatched_offline),
                'description': 'Offline touchpoints that could not be matched to online journeys',
                'recommendation': 'Implement better tracking mechanisms (promo codes, custom URLs, QR codes)',
                'potential_impact': 'Medium'
            })
        
        # Low confidence integrations
        low_confidence_journeys = [j for j in self.integrated_journeys if j.confidence_score < 0.7]
        
        if low_confidence_journeys:
            opportunities.append({
                'type': 'low_confidence_integrations',
                'count': len(low_confidence_journeys),
                'description': 'Integrated journeys with low confidence scores',
                'recommendation': 'Improve integration methods and data collection',
                'potential_impact': 'High'
            })
        
        # Missing integration methods
        method_usage = self.integration_stats.get('integration_methods_used', {})
        all_methods = [method.value for method in IntegrationMethod]
        unused_methods = [method for method in all_methods if method not in method_usage]
        
        if unused_methods:
            opportunities.append({
                'type': 'unused_integration_methods',
                'methods': unused_methods,
                'description': 'Integration methods that are not being utilized',
                'recommendation': f'Consider implementing: {", ".join(unused_methods)}',
                'potential_impact': 'Medium'
            })
        
        # Geofencing opportunities
        location_offline = [tp for tp in self.offline_touchpoints if tp.location]
        if location_offline and 'geofencing' not in method_usage:
            opportunities.append({
                'type': 'geofencing_opportunity',
                'count': len(location_offline),
                'description': 'Offline touchpoints with location data that could use geofencing',
                'recommendation': 'Implement geofencing integration for location-based matching',
                'potential_impact': 'High'
            })
        
        return opportunities
    
    def generate_integration_report(self) -> str:
        """Generate comprehensive offline-online integration report."""
        
        report = "# Offline-Online Integration Analysis\n\n"
        report += "**Advanced Cross-Channel Integration by Sotiris Spyrou**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        # Integration Overview
        insights = self.get_integration_insights()
        stats = insights.get('integration_statistics', {})
        
        report += f"## Integration Overview\n\n"
        report += f"- **Total Offline Touchpoints**: {stats.get('total_offline_touchpoints', 0):,}\n"
        report += f"- **Total Online Touchpoints**: {stats.get('total_online_touchpoints', 0):,}\n"
        report += f"- **Integrated Journeys**: {stats.get('integrated_journeys', 0):,}\n"
        report += f"- **Integration Rate**: {stats.get('integration_rate', 0):.1%}\n"
        report += f"- **Average Confidence Score**: {stats.get('average_confidence', 0):.1%}\n\n"
        
        # Integration Methods
        methods = stats.get('integration_methods_used', {})
        if methods:
            report += f"## Integration Methods Usage\n\n"
            for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
                method_name = method.replace('_', ' ').title()
                report += f"- **{method_name}**: {count} integrations\n"
            report += "\n"
        
        # Attribution Results
        attribution = insights.get('attribution_results', {})
        if attribution:
            report += f"## Cross-Channel Attribution Results\n\n"
            report += "| Channel | Customers | Conversions | Revenue | Conversion Rate | ROAS |\n"
            report += "|---------|-----------|-------------|---------|-----------------|------|\n"
            
            # Sort by revenue
            sorted_channels = sorted(attribution.items(), key=lambda x: x[1]['revenue'], reverse=True)
            
            for channel, metrics in sorted_channels[:10]:
                channel_name = channel.replace('_', ' ').title()
                report += f"| {channel_name} | {metrics['unique_customers']:,} | "
                report += f"{metrics['online_conversions']} | ${metrics['revenue']:,.0f} | "
                report += f"{metrics['conversion_rate']:.1%} | {metrics['roas']:.1f}x |\n"
            
            report += "\n"
        
        # Journey Analysis
        journey_analysis = insights.get('journey_analysis', {})
        if journey_analysis:
            report += f"## Integrated Journey Analysis\n\n"
            report += f"- **Average Journey Length**: {journey_analysis.get('average_journey_length', 0):.1f} touchpoints\n"
            report += f"- **Average Offline Ratio**: {journey_analysis.get('average_offline_ratio', 0):.1%}\n"
            report += f"- **Average Journey Duration**: {journey_analysis.get('average_time_span_days', 0):.1f} days\n"
            report += f"- **Integrated Conversion Rate**: {journey_analysis.get('conversion_rate', 0):.1%}\n"
            report += f"- **Total Revenue**: ${journey_analysis.get('total_revenue', 0):,.0f}\n\n"
            
            # Common patterns
            patterns = journey_analysis.get('common_integration_patterns', {})
            if patterns:
                report += f"### Common Integration Patterns\n\n"
                for pattern, count in patterns.items():
                    if pattern:
                        pattern_name = pattern.replace(',', ' + ').replace('_', ' ').title()
                        report += f"- **{pattern_name}**: {count} journeys\n"
                report += "\n"
        
        # Offline Impact Assessment
        offline_impact = insights.get('offline_impact', {})
        if offline_impact and 'message' not in offline_impact:
            report += f"## Offline Marketing Impact\n\n"
            
            conversion_lift = offline_impact.get('conversion_lift', 0)
            revenue_lift = offline_impact.get('revenue_lift', 0)
            
            report += f"- **Journeys with Offline Touchpoints**: {offline_impact.get('journeys_with_offline', 0):,}\n"
            report += f"- **Conversion Rate with Offline**: {offline_impact.get('conversion_rate_with_offline', 0):.1%}\n"
            report += f"- **Conversion Rate without Offline**: {offline_impact.get('conversion_rate_without_offline', 0):.1%}\n"
            report += f"- **Conversion Lift**: {conversion_lift:+.1%}\n"
            report += f"- **Revenue Lift**: ${revenue_lift:+,.0f}\n"
            report += f"- **Offline-to-Online ROAS**: {offline_impact.get('offline_to_online_roas', 0):.1f}x\n"
            report += f"- **Total Offline Investment**: ${offline_impact.get('total_offline_investment', 0):,.0f}\n\n"
        
        # Integration Opportunities
        opportunities = insights.get('integration_opportunities', [])
        if opportunities:
            report += f"## Integration Opportunities\n\n"
            
            for i, opportunity in enumerate(opportunities, 1):
                impact_emoji = "🔴" if opportunity['potential_impact'] == 'High' else "🟡" if opportunity['potential_impact'] == 'Medium' else "🟢"
                
                report += f"{i}. **{opportunity['type'].replace('_', ' ').title()}** {impact_emoji}\n"
                report += f"   - **Description**: {opportunity['description']}\n"
                report += f"   - **Recommendation**: {opportunity['recommendation']}\n"
                
                if 'count' in opportunity:
                    report += f"   - **Impact**: {opportunity['count']} items affected\n"
                
                report += "\n"
        
        # Key Strategic Insights
        report += f"## Strategic Insights\n\n"
        
        if stats.get('integration_rate', 0) > 0.5:
            report += f"- **Strong Integration**: {stats['integration_rate']:.1%} of online journeys successfully integrated with offline touchpoints\n"
        else:
            report += f"- **Integration Gap**: Only {stats.get('integration_rate', 0):.1%} of online journeys integrated - significant opportunity for improvement\n"
        
        if offline_impact.get('conversion_lift', 0) > 0:
            report += f"- **Offline Marketing Value**: Offline touchpoints increase conversion rates by {offline_impact['conversion_lift']:+.1%}\n"
        
        offline_roas = offline_impact.get('offline_to_online_roas', 0)
        if offline_roas > 2.0:
            report += f"- **Strong Offline ROI**: {offline_roas:.1f}x ROAS demonstrates effective offline-to-online funnel\n"
        elif offline_roas > 0:
            report += f"- **Offline ROI Optimization Needed**: {offline_roas:.1f}x ROAS suggests room for improvement\n"
        
        high_confidence = stats.get('confidence_distribution', {}).get('high_confidence', 0)
        total_journeys = stats.get('integrated_journeys', 0)
        if total_journeys > 0 and high_confidence / total_journeys > 0.7:
            report += f"- **High Data Quality**: {high_confidence/total_journeys:.1%} of integrations have high confidence scores\n"
        
        report += "\n---\n*This analysis demonstrates sophisticated offline-online integration capabilities. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for custom integration implementations.*"
        
        return report


def demo_offline_online_integration():
    """Executive demonstration of Offline-Online Integration system."""
    
    print("=== Offline-Online Integration: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    np.random.seed(42)
    
    # Initialize integrator
    integrator = OfflineOnlineIntegrator(
        geofence_radius_km=1.5,
        time_window_hours=24,
        confidence_threshold=0.6,
        enable_ml_matching=True
    )
    
    # Generate offline marketing data
    print("📡 Generating offline marketing touchpoints...")
    
    offline_data = []
    campaigns = ['Spring_Sale', 'Brand_Awareness', 'Product_Launch', 'Holiday_Special']
    cities = [
        {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
        {'name': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437},
        {'name': 'Chicago', 'lat': 41.8781, 'lon': -87.6298},
        {'name': 'Houston', 'lat': 29.7604, 'lon': -95.3698}
    ]
    
    # Generate 200 offline touchpoints
    for i in range(200):
        city = np.random.choice(cities)
        campaign = np.random.choice(campaigns)
        
        # Add location noise
        lat_noise = np.random.normal(0, 0.01)
        lon_noise = np.random.normal(0, 0.01)
        
        timestamp = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 90))
        
        # Random integration methods
        integration_methods = ['promo_code', 'custom_url', 'qr_code', 'phone_number', 'geofencing']
        integration_method = np.random.choice(integration_methods) if np.random.random() > 0.3 else None
        
        offline_touchpoint = {
            'timestamp': timestamp,
            'channel': np.random.choice(['retail_store', 'print_ad', 'radio', 'tv_commercial', 'billboard']),
            'campaign': campaign,
            'latitude': city['lat'] + lat_noise,
            'longitude': city['lon'] + lon_noise,
            'spend': np.random.uniform(500, 5000),
            'impressions': np.random.randint(1000, 50000),
            'reach': np.random.randint(500, 25000),
            'integration_method': integration_method,
            'integration_promo_code': f"{campaign}_{i}" if integration_method == 'promo_code' else None,
            'integration_tracking_url': f"example.com/{campaign.lower()}_{i}" if integration_method == 'custom_url' else None,
            'integration_qr_code_id': f"QR_{campaign}_{i}" if integration_method == 'qr_code' else None
        }
        
        offline_data.append(offline_touchpoint)
    
    offline_df = pd.DataFrame(offline_data)
    integrator.add_offline_touchpoints(offline_df)
    
    print(f"📊 Added {len(offline_df)} offline touchpoints")
    
    # Generate online customer journey data
    print("💻 Generating online customer journeys...")
    
    online_data = []
    customers = [f"customer_{i}" for i in range(1, 101)]  # 100 customers
    
    for customer_id in customers:
        # Generate customer journey
        journey_length = np.random.choice([2, 3, 4, 5], p=[0.3, 0.4, 0.2, 0.1])
        
        customer_start = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 85))
        converted = np.random.random() < 0.15  # 15% conversion rate
        
        for j in range(journey_length):
            days_offset = j * np.random.exponential(2)  # Average 2 days between touchpoints
            timestamp = customer_start + timedelta(days=days_offset)
            
            # Channel selection
            channel = np.random.choice(['Search', 'Display', 'Social', 'Email', 'Direct'])
            
            # Integration signals based on offline campaigns
            integration_signals = {}
            
            # Occasionally add integration signals that match offline touchpoints
            if np.random.random() < 0.3:  # 30% chance of integration signal
                matching_campaign = np.random.choice(campaigns)
                
                if np.random.random() < 0.4:  # Promo code
                    offline_match = np.random.randint(0, 200)
                    integration_signals['promo_code'] = f"{matching_campaign}_{offline_match}"
                
                elif np.random.random() < 0.5:  # UTM source from QR
                    offline_match = np.random.randint(0, 200)
                    integration_signals['utm_source'] = f"qr_QR_{matching_campaign}_{offline_match}"
                
                elif np.random.random() < 0.6:  # Custom URL
                    offline_match = np.random.randint(0, 200)
                    integration_signals['url'] = f"https://example.com/{matching_campaign.lower()}_{offline_match}/landing"
            
            # Location data for some touchpoints (mobile users)
            if np.random.random() < 0.4:  # 40% have location
                city = np.random.choice(cities)
                integration_signals['latitude'] = city['lat'] + np.random.normal(0, 0.008)
                integration_signals['longitude'] = city['lon'] + np.random.normal(0, 0.008)
            
            # Conversion and revenue
            is_converted = converted and (j == journey_length - 1)
            revenue = np.random.normal(350, 100) if is_converted else 0.0
            revenue = max(revenue, 0)
            
            online_touchpoint = {
                'customer_id': customer_id,
                'timestamp': timestamp,
                'channel': channel,
                'campaign': np.random.choice(campaigns) if np.random.random() > 0.3 else f"online_{channel.lower()}",
                'session_id': f"sess_{customer_id}_{j}",
                'converted': is_converted,
                'revenue': revenue,
                **{f'integration_{k}' if not k.startswith(('utm_', 'promo_', 'url')) else k: v 
                   for k, v in integration_signals.items()}
            }
            
            online_data.append(online_touchpoint)
    
    online_df = pd.DataFrame(online_data)
    integrator.add_online_touchpoints(online_df)
    
    print(f"💡 Added {len(online_df)} online touchpoints for {len(customers)} customers")
    
    # Perform integration
    print(f"\n🔗 Integrating offline and online touchpoints...")
    integrator.integrate_journeys()
    
    # Display results
    print("\n📊 OFFLINE-ONLINE INTEGRATION RESULTS")
    print("=" * 50)
    
    insights = integrator.get_integration_insights()
    stats = insights['integration_statistics']
    
    print(f"\n🎯 Integration Summary:")
    print(f"  • Offline Touchpoints: {stats['total_offline_touchpoints']:,}")
    print(f"  • Online Touchpoints: {stats['total_online_touchpoints']:,}")
    print(f"  • Integrated Journeys: {stats['integrated_journeys']:,}")
    print(f"  • Integration Rate: {stats['integration_rate']:.1%}")
    print(f"  • Average Confidence: {stats['average_confidence']:.1%}")
    
    # Integration methods used
    methods_used = stats.get('integration_methods_used', {})
    if methods_used:
        print(f"\n🔧 Integration Methods Used:")
        for method, count in sorted(methods_used.items(), key=lambda x: x[1], reverse=True):
            method_name = method.replace('_', ' ').title()
            print(f"  • {method_name}: {count} integrations")
    
    # Confidence distribution
    confidence_dist = stats.get('confidence_distribution', {})
    print(f"\n📈 Confidence Distribution:")
    print(f"  • High Confidence (≥80%): {confidence_dist.get('high_confidence', 0)} journeys")
    print(f"  • Medium Confidence (60-80%): {confidence_dist.get('medium_confidence', 0)} journeys")
    print(f"  • Low Confidence (<60%): {confidence_dist.get('low_confidence', 0)} journeys")
    
    # Attribution results
    attribution = insights.get('attribution_results', {})
    if attribution:
        print(f"\n🏆 Cross-Channel Attribution:")
        
        # Sort by revenue
        sorted_channels = sorted(attribution.items(), key=lambda x: x[1]['revenue'], reverse=True)
        
        for channel, metrics in sorted_channels[:8]:  # Top 8 channels
            channel_name = channel.replace('_', ' ').title()
            roas_icon = "🔥" if metrics['roas'] > 3.0 else "📈" if metrics['roas'] > 1.5 else "📊"
            
            print(f"  {roas_icon} {channel_name}: {metrics['unique_customers']:,} customers | "
                  f"{metrics['conversion_rate']:.1%} conv | ${metrics['revenue']:,.0f} revenue | "
                  f"{metrics['roas']:.1f}x ROAS")
    
    # Journey analysis
    journey_analysis = insights.get('journey_analysis', {})
    if journey_analysis:
        print(f"\n🛤️ Journey Analysis:")
        print(f"  • Average Journey Length: {journey_analysis.get('average_journey_length', 0):.1f} touchpoints")
        print(f"  • Average Offline Ratio: {journey_analysis.get('average_offline_ratio', 0):.1%}")
        print(f"  • Average Duration: {journey_analysis.get('average_time_span_days', 0):.1f} days")
        print(f"  • Conversion Rate: {journey_analysis.get('conversion_rate', 0):.1%}")
        print(f"  • Total Revenue: ${journey_analysis.get('total_revenue', 0):,.0f}")
        
        # Common integration patterns
        patterns = journey_analysis.get('common_integration_patterns', {})
        if patterns:
            print(f"\n📊 Common Integration Patterns:")
            for pattern, count in list(patterns.items())[:3]:
                if pattern:
                    pattern_name = pattern.replace(',', ' + ').replace('_', ' ').title()
                    print(f"  • {pattern_name}: {count} journeys")
    
    # Offline impact assessment
    offline_impact = insights.get('offline_impact', {})
    if offline_impact and 'message' not in offline_impact:
        print(f"\n🎯 Offline Marketing Impact:")
        
        conversion_lift = offline_impact.get('conversion_lift', 0)
        revenue_lift = offline_impact.get('revenue_lift', 0)
        
        print(f"  • Journeys with Offline: {offline_impact.get('journeys_with_offline', 0):,}")
        print(f"  • Conversion Lift: {conversion_lift:+.1%}")
        print(f"  • Revenue Lift: ${revenue_lift:+,.0f}")
        print(f"  • Offline-to-Online ROAS: {offline_impact.get('offline_to_online_roas', 0):.1f}x")
        print(f"  • Total Offline Investment: ${offline_impact.get('total_offline_investment', 0):,.0f}")
    
    # Integration opportunities
    opportunities = insights.get('integration_opportunities', [])
    if opportunities:
        print(f"\n💡 Integration Opportunities:")
        
        for opportunity in opportunities[:3]:  # Top 3 opportunities
            impact_icon = "🔴" if opportunity['potential_impact'] == 'High' else "🟡" if opportunity['potential_impact'] == 'Medium' else "🟢"
            opportunity_name = opportunity['type'].replace('_', ' ').title()
            
            print(f"  {impact_icon} {opportunity_name}")
            print(f"     {opportunity['description']}")
            print(f"     Recommendation: {opportunity['recommendation']}")
            
            if 'count' in opportunity:
                print(f"     Impact: {opportunity['count']} items")
    
    print("\n" + "="*60)
    print("🚀 Advanced offline-online marketing integration")
    print("💼 Enterprise-grade cross-channel attribution and insights")
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_offline_online_integration()