"""
Attribution Confidence Scoring System

Provides statistical confidence measures for attribution model results,
including uncertainty quantification, model reliability assessment, and
validation metrics for marketing attribution decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import bootstrap_resample
import warnings
import logging

logger = logging.getLogger(__name__)


class AttributionConfidenceScorer:
    """
    Statistical confidence assessment for attribution model results.
    
    Provides uncertainty quantification, confidence intervals, and
    reliability metrics for attribution decisions.
    """
    
    def __init__(self, confidence_level: float = 0.95, 
                 bootstrap_samples: int = 1000,
                 min_sample_size: int = 100):
        """
        Initialize Attribution Confidence Scorer.
        
        Args:
            confidence_level: Statistical confidence level (0.95 = 95%)
            bootstrap_samples: Number of bootstrap iterations
            min_sample_size: Minimum sample size for valid confidence estimation
        """
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.min_sample_size = min_sample_size
        self.alpha = 1 - confidence_level
        
    def calculate_attribution_confidence(self, 
                                       attribution_results: Dict[str, float],
                                       journey_data: pd.DataFrame,
                                       model_type: str = "general") -> Dict[str, Dict[str, float]]:
        """
        Calculate confidence scores for attribution results.
        
        Args:
            attribution_results: Dict mapping channels to attribution weights
            journey_data: Customer journey data used for attribution
            model_type: Type of attribution model used
            
        Returns:
            Dict with confidence metrics for each channel
        """
        logger.info(f"Calculating confidence scores for {len(attribution_results)} channels")
        
        confidence_scores = {}
        
        for channel, attribution_weight in attribution_results.items():
            # Calculate channel-specific confidence metrics
            channel_confidence = self._calculate_channel_confidence(
                channel, attribution_weight, journey_data, model_type
            )
            confidence_scores[channel] = channel_confidence
        
        # Add overall model confidence
        confidence_scores['_overall'] = self._calculate_overall_confidence(
            attribution_results, journey_data, model_type
        )
        
        return confidence_scores
    
    def _calculate_channel_confidence(self, 
                                    channel: str, 
                                    attribution_weight: float,
                                    journey_data: pd.DataFrame,
                                    model_type: str) -> Dict[str, float]:
        """Calculate confidence metrics for a specific channel."""
        
        # Filter data for this channel
        channel_data = journey_data[journey_data['touchpoint'] == channel]
        
        if len(channel_data) < self.min_sample_size:
            logger.warning(f"Insufficient data for channel {channel}: {len(channel_data)} samples")
            return self._low_confidence_result()
        
        # Bootstrap confidence interval for attribution weight
        bootstrap_weights = self._bootstrap_attribution_weight(
            channel, journey_data, model_type
        )
        
        ci_lower, ci_upper = np.percentile(
            bootstrap_weights, 
            [100 * self.alpha/2, 100 * (1 - self.alpha/2)]
        )
        
        # Statistical significance test
        p_value = self._test_attribution_significance(channel, journey_data)
        
        # Sample size adequacy
        sample_adequacy = self._assess_sample_adequacy(channel_data)
        
        # Stability score (consistency across time periods)
        stability_score = self._calculate_temporal_stability(channel, journey_data)
        
        # Overall confidence score
        confidence_score = self._compute_composite_confidence(
            attribution_weight, ci_lower, ci_upper, p_value, 
            sample_adequacy, stability_score
        )
        
        return {
            'attribution_weight': attribution_weight,
            'confidence_score': confidence_score,
            'confidence_interval_lower': ci_lower,
            'confidence_interval_upper': ci_upper,
            'p_value': p_value,
            'sample_size': len(channel_data),
            'sample_adequacy': sample_adequacy,
            'stability_score': stability_score,
            'margin_of_error': (ci_upper - ci_lower) / 2,
            'relative_error': ((ci_upper - ci_lower) / 2) / max(attribution_weight, 1e-6)
        }
    
    def _bootstrap_attribution_weight(self, 
                                    channel: str, 
                                    journey_data: pd.DataFrame,
                                    model_type: str) -> np.ndarray:
        """Bootstrap sampling to estimate attribution weight distribution."""
        bootstrap_weights = []
        
        for _ in range(self.bootstrap_samples):
            # Resample journey data
            n_samples = len(journey_data)
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_data = journey_data.iloc[bootstrap_indices]
            
            # Calculate attribution weight for bootstrap sample
            try:
                weight = self._calculate_single_attribution_weight(
                    channel, bootstrap_data, model_type
                )
                bootstrap_weights.append(weight)
            except Exception as e:
                logger.warning(f"Bootstrap iteration failed for {channel}: {e}")
                continue
        
        return np.array(bootstrap_weights)
    
    def _calculate_single_attribution_weight(self, 
                                           channel: str,
                                           data: pd.DataFrame,
                                           model_type: str) -> float:
        """Calculate attribution weight for a single data sample."""
        
        if model_type == "last_touch":
            # Last touch attribution
            last_touches = data.groupby('customer_id')['touchpoint'].last()
            channel_conversions = (last_touches == channel).sum()
            total_conversions = len(last_touches)
            
        elif model_type == "first_touch":
            # First touch attribution  
            first_touches = data.groupby('customer_id')['touchpoint'].first()
            channel_conversions = (first_touches == channel).sum()
            total_conversions = len(first_touches)
            
        elif model_type == "linear":
            # Linear attribution
            customer_journeys = data.groupby('customer_id')['touchpoint'].apply(list)
            total_attributions = 0
            channel_attributions = 0
            
            for journey in customer_journeys:
                if len(journey) > 0:
                    weight_per_touch = 1 / len(journey)
                    total_attributions += 1
                    channel_attributions += journey.count(channel) * weight_per_touch
            
            return channel_attributions / max(total_attributions, 1)
            
        else:
            # Default: position-based attribution
            customer_journeys = data.groupby('customer_id')['touchpoint'].apply(list)
            total_attributions = 0
            channel_attributions = 0
            
            for journey in customer_journeys:
                if len(journey) == 1:
                    weight = 1.0 if journey[0] == channel else 0.0
                elif len(journey) == 2:
                    weight = 0.5 if channel in journey else 0.0
                else:
                    first_weight = 0.4 if journey[0] == channel else 0.0
                    last_weight = 0.4 if journey[-1] == channel else 0.0
                    middle_count = journey[1:-1].count(channel)
                    middle_weight = (0.2 / (len(journey) - 2)) * middle_count
                    weight = first_weight + last_weight + middle_weight
                
                if len(journey) > 0:
                    total_attributions += 1
                    channel_attributions += weight
            
            return channel_attributions / max(total_attributions, 1)
        
        if total_conversions == 0:
            return 0.0
        return channel_conversions / total_conversions
    
    def _test_attribution_significance(self, channel: str, 
                                     journey_data: pd.DataFrame) -> float:
        """Test statistical significance of channel attribution."""
        
        # Create binary indicator for channel presence
        customer_channels = journey_data.groupby('customer_id').agg({
            'touchpoint': lambda x: channel in x.values,
            'converted': 'first'
        })
        
        has_channel = customer_channels['touchpoint']
        converted = customer_channels['converted']
        
        if len(has_channel.unique()) < 2:
            return 1.0  # Cannot test significance
        
        # Chi-square test for association
        contingency = pd.crosstab(has_channel, converted)
        
        if contingency.shape == (2, 2):
            chi2, p_value, _, _ = stats.chi2_contingency(contingency)
            return p_value
        
        return 1.0
    
    def _assess_sample_adequacy(self, channel_data: pd.DataFrame) -> float:
        """Assess adequacy of sample size for channel."""
        n = len(channel_data)
        
        # Rule of thumb: need at least 30 observations per parameter
        min_adequate = 30
        well_adequate = 100
        
        if n < min_adequate:
            return 0.0
        elif n < well_adequate:
            return (n - min_adequate) / (well_adequate - min_adequate)
        else:
            return min(1.0, n / (2 * well_adequate))
    
    def _calculate_temporal_stability(self, channel: str, 
                                    journey_data: pd.DataFrame) -> float:
        """Calculate temporal stability of channel attribution."""
        
        journey_data['date'] = pd.to_datetime(journey_data['timestamp'])
        
        # Split into time periods
        date_range = journey_data['date'].max() - journey_data['date'].min()
        if date_range.days < 14:  # Less than 2 weeks
            return 1.0  # Assume stable for short periods
        
        # Create weekly periods
        journey_data['week'] = journey_data['date'].dt.to_period('W')
        weekly_data = []
        
        for week in journey_data['week'].unique():
            week_data = journey_data[journey_data['week'] == week]
            if len(week_data) >= 10:  # Minimum data for calculation
                try:
                    weight = self._calculate_single_attribution_weight(
                        channel, week_data, "linear"
                    )
                    weekly_data.append(weight)
                except:
                    continue
        
        if len(weekly_data) < 2:
            return 0.5  # Insufficient data for stability assessment
        
        # Calculate coefficient of variation (lower = more stable)
        mean_weight = np.mean(weekly_data)
        if mean_weight == 0:
            return 1.0
        
        cv = np.std(weekly_data) / mean_weight
        stability = max(0.0, 1.0 - cv)  # Higher stability = lower variation
        
        return min(1.0, stability)
    
    def _compute_composite_confidence(self, 
                                    attribution_weight: float,
                                    ci_lower: float, 
                                    ci_upper: float,
                                    p_value: float,
                                    sample_adequacy: float,
                                    stability_score: float) -> float:
        """Compute overall confidence score from individual metrics."""
        
        # Precision component (tighter confidence intervals = higher confidence)
        if attribution_weight > 0:
            precision_score = 1.0 - min(1.0, (ci_upper - ci_lower) / (2 * attribution_weight))
        else:
            precision_score = 0.5
        
        # Statistical significance component
        significance_score = 1.0 - p_value
        
        # Weighted combination of components
        weights = {
            'precision': 0.3,
            'significance': 0.25, 
            'sample_adequacy': 0.25,
            'stability': 0.2
        }
        
        composite_score = (
            weights['precision'] * max(0, precision_score) +
            weights['significance'] * significance_score +
            weights['sample_adequacy'] * sample_adequacy +
            weights['stability'] * stability_score
        )
        
        return min(1.0, max(0.0, composite_score))
    
    def _calculate_overall_confidence(self, 
                                    attribution_results: Dict[str, float],
                                    journey_data: pd.DataFrame,
                                    model_type: str) -> Dict[str, float]:
        """Calculate overall model confidence metrics."""
        
        # Model fit metrics
        total_customers = journey_data['customer_id'].nunique()
        total_touchpoints = len(journey_data)
        
        # Attribution completeness (should sum to ~1.0)
        total_attribution = sum(attribution_results.values())
        completeness_score = 1.0 - abs(1.0 - total_attribution)
        
        # Data coverage
        coverage_score = min(1.0, total_customers / self.min_sample_size)
        
        # Channel balance (not too concentrated)
        if len(attribution_results) > 1:
            weights = list(attribution_results.values())
            hhi = sum(w**2 for w in weights)  # Herfindahl index
            balance_score = max(0.0, 1.0 - (hhi - 1/len(weights)) / (1 - 1/len(weights)))
        else:
            balance_score = 0.5
        
        overall_confidence = np.mean([completeness_score, coverage_score, balance_score])
        
        return {
            'overall_confidence': overall_confidence,
            'completeness_score': completeness_score,
            'coverage_score': coverage_score, 
            'balance_score': balance_score,
            'total_customers': total_customers,
            'total_touchpoints': total_touchpoints,
            'total_attribution': total_attribution
        }
    
    def _low_confidence_result(self) -> Dict[str, float]:
        """Return low confidence result for insufficient data."""
        return {
            'attribution_weight': 0.0,
            'confidence_score': 0.0,
            'confidence_interval_lower': 0.0,
            'confidence_interval_upper': 0.0,
            'p_value': 1.0,
            'sample_size': 0,
            'sample_adequacy': 0.0,
            'stability_score': 0.0,
            'margin_of_error': 0.0,
            'relative_error': float('inf')
        }
    
    def generate_confidence_report(self, 
                                 confidence_scores: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Generate a comprehensive confidence report."""
        
        report_data = []
        
        for channel, metrics in confidence_scores.items():
            if channel == '_overall':
                continue
                
            report_data.append({
                'channel': channel,
                'attribution_weight': metrics['attribution_weight'],
                'confidence_score': metrics['confidence_score'],
                'confidence_level': 'High' if metrics['confidence_score'] > 0.8 else 
                                  'Medium' if metrics['confidence_score'] > 0.5 else 'Low',
                'margin_of_error': metrics['margin_of_error'],
                'sample_size': metrics['sample_size'],
                'p_value': metrics['p_value'],
                'stability_score': metrics['stability_score']
            })
        
        report_df = pd.DataFrame(report_data)
        return report_df.sort_values('confidence_score', ascending=False)
    
    def recommend_actions(self, 
                         confidence_scores: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate actionable recommendations based on confidence analysis."""
        
        recommendations = []
        
        for channel, metrics in confidence_scores.items():
            if channel == '_overall':
                continue
            
            conf_score = metrics['confidence_score']
            sample_size = metrics['sample_size']
            p_value = metrics['p_value']
            
            if conf_score < 0.3:
                recommendations.append(
                    f"❌ {channel}: Very low confidence ({conf_score:.2f}). "
                    f"Consider collecting more data (current: {sample_size} samples)"
                )
            elif conf_score < 0.6:
                recommendations.append(
                    f"⚠️ {channel}: Medium confidence ({conf_score:.2f}). "
                    f"Results may be unreliable for critical decisions"
                )
            elif p_value > 0.05:
                recommendations.append(
                    f"📊 {channel}: Not statistically significant (p={p_value:.3f}). "
                    f"Attribution may be due to random variation"
                )
            else:
                recommendations.append(
                    f"✅ {channel}: High confidence ({conf_score:.2f}). "
                    f"Attribution results are reliable for decision making"
                )
        
        # Overall recommendations
        overall_metrics = confidence_scores.get('_overall', {})
        overall_conf = overall_metrics.get('overall_confidence', 0)
        
        if overall_conf < 0.5:
            recommendations.append(
                "🔍 Overall model confidence is low. Consider:\n"
                "   • Collecting more journey data\n"
                "   • Improving data quality\n"
                "   • Using ensemble attribution methods"
            )
        
        return recommendations


def demo_confidence_scoring():
    """Demonstration of Attribution Confidence Scoring."""
    
    # Sample data
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'customer_id': np.repeat(range(1, 501), 3),
        'touchpoint': np.random.choice(['Search', 'Display', 'Email', 'Social', 'Direct'], 1500),
        'timestamp': pd.date_range('2024-01-01', periods=1500, freq='H'),
        'converted': np.random.choice([True, False], 1500, p=[0.2, 0.8])
    })
    
    # Sample attribution results
    attribution_results = {
        'Search': 0.35,
        'Display': 0.25, 
        'Email': 0.20,
        'Social': 0.15,
        'Direct': 0.05
    }
    
    # Calculate confidence scores
    scorer = AttributionConfidenceScorer()
    confidence_scores = scorer.calculate_attribution_confidence(
        attribution_results, sample_data, "linear"
    )
    
    # Generate report
    report = scorer.generate_confidence_report(confidence_scores)
    print("Confidence Report:")
    print(report)
    
    # Get recommendations
    recommendations = scorer.recommend_actions(confidence_scores)
    print("\nRecommendations:")
    for rec in recommendations:
        print(rec)


if __name__ == "__main__":
    demo_confidence_scoring()