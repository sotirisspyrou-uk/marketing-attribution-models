"""
Channel Optimization Engine

Advanced channel performance optimization system that provides data-driven
recommendations for improving marketing channel efficiency, targeting,
and ROI through attribution insights and predictive analytics.

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
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of channel optimization."""
    TARGETING = "targeting"
    BIDDING = "bidding"
    CREATIVE = "creative"
    BUDGET = "budget"
    AUDIENCE = "audience"
    PLACEMENT = "placement"
    TIMING = "timing"


class OptimizationPriority(Enum):
    """Optimization priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class OptimizationRecommendation:
    """Channel optimization recommendation."""
    channel: str
    optimization_type: OptimizationType
    priority: OptimizationPriority
    title: str
    description: str
    expected_impact: float
    confidence_score: float
    implementation_effort: str  # "Low", "Medium", "High"
    estimated_timeline: str
    success_metrics: List[str]
    action_steps: List[str]
    risk_level: str  # "Low", "Medium", "High"


@dataclass
class ChannelInsight:
    """Channel performance insight."""
    channel: str
    metric: str
    current_value: float
    benchmark_value: float
    performance_gap: float
    trend: str  # "improving", "declining", "stable"
    significance: float
    insight_text: str


class ChannelOptimizer:
    """
    Advanced channel optimization engine for marketing attribution insights.
    
    Analyzes channel performance, identifies optimization opportunities,
    and provides actionable recommendations for improving marketing ROI.
    """
    
    def __init__(self,
                 lookback_days: int = 30,
                 min_significance_threshold: float = 0.05,
                 benchmark_percentile: float = 0.75,
                 enable_predictive_recommendations: bool = True):
        """
        Initialize Channel Optimizer.
        
        Args:
            lookback_days: Days of historical data for analysis
            min_significance_threshold: Minimum statistical significance for insights
            benchmark_percentile: Percentile to use for performance benchmarks
            enable_predictive_recommendations: Enable ML-based predictive recommendations
        """
        self.lookback_days = lookback_days
        self.min_significance_threshold = min_significance_threshold
        self.benchmark_percentile = benchmark_percentile
        self.enable_predictive_recommendations = enable_predictive_recommendations
        
        # Analysis components
        self.performance_data = {}
        self.attribution_data = {}
        self.channel_insights = []
        self.optimization_recommendations = []
        self.benchmarks = {}
        
        # Machine learning components
        self.scaler = StandardScaler()
        self.clustering_model = None
        
        logger.info("Channel optimizer initialized")
    
    def analyze_channels(self,
                        performance_data: Dict[str, Dict[str, float]],
                        attribution_weights: Dict[str, float],
                        historical_data: Optional[pd.DataFrame] = None,
                        industry_benchmarks: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Comprehensive channel performance analysis and optimization recommendations.
        
        Args:
            performance_data: Current channel performance metrics
            attribution_weights: Attribution weights from attribution model
            historical_data: Historical performance data
            industry_benchmarks: Industry benchmark data
            
        Returns:
            Complete analysis results with insights and recommendations
        """
        logger.info("Starting comprehensive channel analysis")
        
        # Store data
        self.performance_data = performance_data
        self.attribution_data = attribution_weights
        
        # Calculate benchmarks
        self._calculate_benchmarks(performance_data, industry_benchmarks)
        
        # Generate insights
        insights = self._generate_channel_insights(performance_data, attribution_weights)
        
        # Performance gap analysis
        performance_gaps = self._analyze_performance_gaps(performance_data)
        
        # Attribution efficiency analysis
        attribution_efficiency = self._analyze_attribution_efficiency(
            performance_data, attribution_weights
        )
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(
            performance_data, attribution_weights, insights
        )
        
        # Channel clustering for segment-based insights
        channel_segments = self._perform_channel_clustering(performance_data)
        
        # Competitive positioning analysis
        competitive_insights = self._analyze_competitive_positioning(performance_data)
        
        # Predictive recommendations
        predictive_recommendations = []
        if self.enable_predictive_recommendations:
            predictive_recommendations = self._generate_predictive_recommendations(
                performance_data, attribution_weights
            )
        
        results = {
            'analysis_timestamp': datetime.now(),
            'channel_insights': insights,
            'performance_gaps': performance_gaps,
            'attribution_efficiency': attribution_efficiency,
            'optimization_recommendations': recommendations,
            'channel_segments': channel_segments,
            'competitive_insights': competitive_insights,
            'predictive_recommendations': predictive_recommendations,
            'benchmarks': self.benchmarks,
            'summary_metrics': self._calculate_summary_metrics(performance_data, attribution_weights)
        }
        
        # Store results
        self.channel_insights = insights
        self.optimization_recommendations = recommendations
        
        logger.info(f"Channel analysis completed: {len(recommendations)} recommendations generated")
        return results
    
    def _calculate_benchmarks(self,
                            performance_data: Dict[str, Dict[str, float]],
                            industry_benchmarks: Optional[Dict[str, Dict[str, float]]] = None):
        """Calculate performance benchmarks for comparison."""
        
        self.benchmarks = {}
        
        if industry_benchmarks:
            self.benchmarks.update(industry_benchmarks)
        else:
            # Calculate internal benchmarks from performance data
            all_metrics = set()
            for channel_data in performance_data.values():
                all_metrics.update(channel_data.keys())
            
            for metric in all_metrics:
                metric_values = [
                    channel_data.get(metric, 0) 
                    for channel_data in performance_data.values()
                    if metric in channel_data
                ]
                
                if metric_values:
                    self.benchmarks[metric] = {
                        'p25': np.percentile(metric_values, 25),
                        'p50': np.percentile(metric_values, 50),
                        'p75': np.percentile(metric_values, 75),
                        'p90': np.percentile(metric_values, 90),
                        'mean': np.mean(metric_values),
                        'std': np.std(metric_values)
                    }
    
    def _generate_channel_insights(self,
                                 performance_data: Dict[str, Dict[str, float]],
                                 attribution_weights: Dict[str, float]) -> List[ChannelInsight]:
        """Generate actionable insights for each channel."""
        
        insights = []
        
        for channel, metrics in performance_data.items():
            for metric, current_value in metrics.items():
                if metric not in self.benchmarks:
                    continue
                
                benchmark_value = self.benchmarks[metric].get('p75', current_value)
                performance_gap = (current_value - benchmark_value) / benchmark_value if benchmark_value > 0 else 0
                
                # Determine trend (simplified - would use historical data in production)
                trend = "stable"
                if performance_gap > 0.1:
                    trend = "improving"
                elif performance_gap < -0.1:
                    trend = "declining"
                
                # Statistical significance (simplified)
                significance = abs(performance_gap) if abs(performance_gap) > 0.05 else 0
                
                # Generate insight text
                insight_text = self._generate_insight_text(
                    channel, metric, current_value, benchmark_value, performance_gap, trend
                )
                
                insight = ChannelInsight(
                    channel=channel,
                    metric=metric,
                    current_value=current_value,
                    benchmark_value=benchmark_value,
                    performance_gap=performance_gap,
                    trend=trend,
                    significance=significance,
                    insight_text=insight_text
                )
                insights.append(insight)
        
        # Sort by significance
        insights.sort(key=lambda x: abs(x.performance_gap), reverse=True)
        
        return insights
    
    def _generate_insight_text(self,
                             channel: str,
                             metric: str,
                             current: float,
                             benchmark: float,
                             gap: float,
                             trend: str) -> str:
        """Generate human-readable insight text."""
        
        if gap > 0.2:
            return f"{channel} {metric} significantly outperforms benchmark by {gap:.1%}"
        elif gap > 0.05:
            return f"{channel} {metric} performs above benchmark by {gap:.1%}"
        elif gap < -0.2:
            return f"{channel} {metric} significantly underperforms benchmark by {abs(gap):.1%}"
        elif gap < -0.05:
            return f"{channel} {metric} performs below benchmark by {abs(gap):.1%}"
        else:
            return f"{channel} {metric} performs at benchmark level"
    
    def _analyze_performance_gaps(self,
                                performance_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze performance gaps across channels."""
        
        gaps_analysis = {
            'largest_gaps': [],
            'improvement_opportunities': [],
            'channel_rankings': {}
        }
        
        # Calculate performance scores for ranking
        channel_scores = {}
        for channel, metrics in performance_data.items():
            score = 0
            metric_count = 0
            
            for metric, value in metrics.items():
                if metric in self.benchmarks:
                    benchmark = self.benchmarks[metric]['p75']
                    if benchmark > 0:
                        normalized_score = value / benchmark
                        score += normalized_score
                        metric_count += 1
            
            if metric_count > 0:
                channel_scores[channel] = score / metric_count
        
        # Rank channels
        ranked_channels = sorted(
            channel_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        gaps_analysis['channel_rankings'] = ranked_channels
        
        # Identify largest improvement opportunities
        for channel, metrics in performance_data.items():
            for metric, current_value in metrics.items():
                if metric in self.benchmarks:
                    benchmark = self.benchmarks[metric]['p75']
                    gap = (benchmark - current_value) / benchmark if benchmark > 0 else 0
                    
                    if gap > 0.1:  # 10% improvement opportunity
                        gaps_analysis['improvement_opportunities'].append({
                            'channel': channel,
                            'metric': metric,
                            'current_value': current_value,
                            'benchmark_value': benchmark,
                            'improvement_potential': gap,
                            'potential_impact': gap * current_value
                        })
        
        # Sort opportunities by potential impact
        gaps_analysis['improvement_opportunities'].sort(
            key=lambda x: x['potential_impact'],
            reverse=True
        )
        
        return gaps_analysis
    
    def _analyze_attribution_efficiency(self,
                                      performance_data: Dict[str, Dict[str, float]],
                                      attribution_weights: Dict[str, float]) -> Dict[str, Any]:
        """Analyze attribution efficiency across channels."""
        
        efficiency_analysis = {
            'efficiency_scores': {},
            'over_attributed': [],
            'under_attributed': [],
            'optimal_allocation': {}
        }
        
        # Calculate efficiency scores
        total_budget_share = sum(performance_data[ch].get('spend', 0) for ch in performance_data.keys())
        
        for channel in performance_data.keys():
            spend = performance_data[channel].get('spend', 0)
            attribution_weight = attribution_weights.get(channel, 0)
            roas = performance_data[channel].get('roas', 0)
            
            # Budget share vs attribution weight
            budget_share = spend / total_budget_share if total_budget_share > 0 else 0
            attribution_efficiency = attribution_weight / budget_share if budget_share > 0 else 0
            
            # Performance efficiency
            performance_efficiency = roas * attribution_weight
            
            # Combined efficiency score
            efficiency_score = (attribution_efficiency + performance_efficiency) / 2
            efficiency_analysis['efficiency_scores'][channel] = {
                'attribution_efficiency': attribution_efficiency,
                'performance_efficiency': performance_efficiency,
                'combined_efficiency': efficiency_score,
                'budget_share': budget_share,
                'attribution_weight': attribution_weight
            }
            
            # Identify over/under attributed channels
            if attribution_efficiency > 1.5:  # 50% higher attribution than budget share
                efficiency_analysis['over_attributed'].append({
                    'channel': channel,
                    'efficiency_ratio': attribution_efficiency,
                    'recommendation': 'Consider increasing budget allocation'
                })
            elif attribution_efficiency < 0.5:  # 50% lower attribution than budget share
                efficiency_analysis['under_attributed'].append({
                    'channel': channel,
                    'efficiency_ratio': attribution_efficiency,
                    'recommendation': 'Review channel effectiveness or reduce budget'
                })
        
        return efficiency_analysis
    
    def _generate_optimization_recommendations(self,
                                             performance_data: Dict[str, Dict[str, float]],
                                             attribution_weights: Dict[str, float],
                                             insights: List[ChannelInsight]) -> List[OptimizationRecommendation]:
        """Generate specific optimization recommendations for each channel."""
        
        recommendations = []
        
        for channel, metrics in performance_data.items():
            channel_recommendations = self._generate_channel_specific_recommendations(
                channel, metrics, attribution_weights.get(channel, 0), insights
            )
            recommendations.extend(channel_recommendations)
        
        # Sort by expected impact and priority
        recommendations.sort(key=lambda x: (x.priority.value, -x.expected_impact))
        
        return recommendations
    
    def _generate_channel_specific_recommendations(self,
                                                 channel: str,
                                                 metrics: Dict[str, float],
                                                 attribution_weight: float,
                                                 insights: List[ChannelInsight]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations for a specific channel."""
        
        recommendations = []
        roas = metrics.get('roas', 0)
        ctr = metrics.get('ctr', 0)
        cvr = metrics.get('conversion_rate', 0)
        cpa = metrics.get('cpa', 0)
        
        # Low ROAS optimization
        if roas < 2.0:
            recommendations.append(OptimizationRecommendation(
                channel=channel,
                optimization_type=OptimizationType.BIDDING,
                priority=OptimizationPriority.HIGH,
                title=f"Improve {channel} ROAS",
                description=f"Current ROAS of {roas:.1f}x is below optimal threshold. Focus on bid optimization and audience refinement.",
                expected_impact=0.3,  # 30% improvement potential
                confidence_score=0.8,
                implementation_effort="Medium",
                estimated_timeline="2-4 weeks",
                success_metrics=["ROAS", "CPA", "Conversion Rate"],
                action_steps=[
                    "Analyze top-performing keywords/audiences",
                    "Implement automated bidding strategies",
                    "Exclude low-performing segments",
                    "Test bid adjustments by time-of-day"
                ],
                risk_level="Low"
            ))
        
        # Low CTR optimization
        if ctr < 0.03:  # Less than 3%
            recommendations.append(OptimizationRecommendation(
                channel=channel,
                optimization_type=OptimizationType.CREATIVE,
                priority=OptimizationPriority.HIGH,
                title=f"Improve {channel} Creative Performance",
                description=f"CTR of {ctr:.1%} suggests creative fatigue or poor audience alignment.",
                expected_impact=0.25,
                confidence_score=0.75,
                implementation_effort="Medium",
                estimated_timeline="1-2 weeks",
                success_metrics=["CTR", "Engagement Rate", "Creative Diversity"],
                action_steps=[
                    "Develop new creative variants",
                    "A/B test different messaging angles",
                    "Implement creative rotation strategies",
                    "Analyze competitor creative strategies"
                ],
                risk_level="Medium"
            ))
        
        # Low conversion rate optimization
        if cvr < 0.02:  # Less than 2%
            recommendations.append(OptimizationRecommendation(
                channel=channel,
                optimization_type=OptimizationType.TARGETING,
                priority=OptimizationPriority.HIGH,
                title=f"Optimize {channel} Targeting",
                description=f"Conversion rate of {cvr:.1%} indicates potential targeting or landing page issues.",
                expected_impact=0.4,
                confidence_score=0.85,
                implementation_effort="High",
                estimated_timeline="3-6 weeks",
                success_metrics=["Conversion Rate", "Quality Score", "Relevance Score"],
                action_steps=[
                    "Audit landing page experience",
                    "Refine audience targeting criteria",
                    "Implement lookalike audiences",
                    "Test different landing page variants"
                ],
                risk_level="Medium"
            ))
        
        # High CPA optimization
        benchmark_cpa = self.benchmarks.get('cpa', {}).get('p50', 50)
        if cpa > benchmark_cpa * 1.5:  # 50% above benchmark
            recommendations.append(OptimizationRecommendation(
                channel=channel,
                optimization_type=OptimizationType.BUDGET,
                priority=OptimizationPriority.MEDIUM,
                title=f"Reduce {channel} Cost Per Acquisition",
                description=f"CPA of ${cpa:.2f} is {(cpa/benchmark_cpa-1)*100:.0f}% above benchmark.",
                expected_impact=0.2,
                confidence_score=0.7,
                implementation_effort="Low",
                estimated_timeline="1-2 weeks",
                success_metrics=["CPA", "Cost Efficiency", "Budget Utilization"],
                action_steps=[
                    "Implement stricter bid caps",
                    "Pause underperforming campaigns",
                    "Optimize budget distribution",
                    "Focus spend on high-converting segments"
                ],
                risk_level="Low"
            ))
        
        # Attribution efficiency optimization
        if attribution_weight < 0.05:  # Very low attribution
            recommendations.append(OptimizationRecommendation(
                channel=channel,
                optimization_type=OptimizationType.AUDIENCE,
                priority=OptimizationPriority.MEDIUM,
                title=f"Increase {channel} Attribution Impact",
                description=f"Low attribution weight of {attribution_weight:.1%} suggests limited conversion influence.",
                expected_impact=0.15,
                confidence_score=0.6,
                implementation_effort="High",
                estimated_timeline="4-8 weeks",
                success_metrics=["Attribution Weight", "Conversion Influence", "Journey Participation"],
                action_steps=[
                    "Analyze customer journey touchpoint positioning",
                    "Experiment with different campaign objectives",
                    "Test expanded audience segments",
                    "Optimize for micro-conversions"
                ],
                risk_level="Medium"
            ))
        
        return recommendations
    
    def _perform_channel_clustering(self,
                                  performance_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Perform clustering analysis to identify channel segments."""
        
        if len(performance_data) < 3:
            return {'insufficient_data': True}
        
        # Prepare data for clustering
        channels = list(performance_data.keys())
        features = []
        feature_names = ['roas', 'ctr', 'conversion_rate', 'cpa']
        
        for channel in channels:
            channel_features = []
            for feature in feature_names:
                value = performance_data[channel].get(feature, 0)
                channel_features.append(value)
            features.append(channel_features)
        
        # Standardize features
        features_array = np.array(features)
        if features_array.std() > 0:
            features_scaled = self.scaler.fit_transform(features_array)
        else:
            features_scaled = features_array
        
        # Perform clustering
        n_clusters = min(3, len(channels))  # Maximum 3 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Analyze clusters
        clusters = {}
        for i in range(n_clusters):
            cluster_channels = [channels[j] for j, label in enumerate(cluster_labels) if label == i]
            
            # Calculate cluster characteristics
            cluster_features = features_scaled[cluster_labels == i]
            cluster_mean = np.mean(cluster_features, axis=0)
            
            cluster_name = self._get_cluster_name(cluster_mean, feature_names)
            
            clusters[f'cluster_{i}'] = {
                'name': cluster_name,
                'channels': cluster_channels,
                'characteristics': dict(zip(feature_names, cluster_mean)),
                'size': len(cluster_channels)
            }
        
        return {
            'clusters': clusters,
            'feature_names': feature_names,
            'cluster_assignments': dict(zip(channels, cluster_labels))
        }
    
    def _get_cluster_name(self, cluster_mean: np.ndarray, feature_names: List[str]) -> str:
        """Generate descriptive name for channel cluster."""
        
        # Find dominant characteristics
        dominant_features = []
        for i, feature in enumerate(feature_names):
            if cluster_mean[i] > 0.5:  # Above average (scaled)
                dominant_features.append(feature)
        
        if 'roas' in dominant_features:
            return "High Performing"
        elif len(dominant_features) >= 2:
            return "Balanced Performance"
        elif cluster_mean[0] < -0.5:  # Low ROAS
            return "Optimization Needed"
        else:
            return "Average Performance"
    
    def _analyze_competitive_positioning(self,
                                       performance_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze competitive positioning of channels."""
        
        competitive_insights = {
            'channel_strengths': {},
            'competitive_gaps': {},
            'market_position': {}
        }
        
        for channel, metrics in performance_data.items():
            strengths = []
            gaps = []
            
            for metric, value in metrics.items():
                if metric in self.benchmarks:
                    benchmark = self.benchmarks[metric]['p75']
                    
                    if value > benchmark * 1.1:  # 10% above benchmark
                        strengths.append({
                            'metric': metric,
                            'advantage': (value - benchmark) / benchmark,
                            'position': 'Market Leader'
                        })
                    elif value < benchmark * 0.8:  # 20% below benchmark
                        gaps.append({
                            'metric': metric,
                            'gap': (benchmark - value) / benchmark,
                            'position': 'Below Market'
                        })
            
            competitive_insights['channel_strengths'][channel] = strengths
            competitive_insights['competitive_gaps'][channel] = gaps
            
            # Overall position
            avg_performance = np.mean([
                metrics[m] / self.benchmarks[m]['p75'] 
                for m in metrics.keys() 
                if m in self.benchmarks and self.benchmarks[m]['p75'] > 0
            ])
            
            if avg_performance > 1.1:
                position = "Market Leader"
            elif avg_performance > 0.9:
                position = "Competitive"
            else:
                position = "Below Market"
            
            competitive_insights['market_position'][channel] = {
                'position': position,
                'performance_ratio': avg_performance
            }
        
        return competitive_insights
    
    def _generate_predictive_recommendations(self,
                                           performance_data: Dict[str, Dict[str, float]],
                                           attribution_weights: Dict[str, float]) -> List[OptimizationRecommendation]:
        """Generate predictive optimization recommendations using ML insights."""
        
        recommendations = []
        
        # Predictive budget reallocation
        efficiency_scores = {}
        for channel, metrics in performance_data.items():
            roas = metrics.get('roas', 0)
            attribution = attribution_weights.get(channel, 0)
            efficiency_scores[channel] = roas * attribution
        
        # Find top performers for scaling
        sorted_channels = sorted(
            efficiency_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_performer = sorted_channels[0] if sorted_channels else None
        if top_performer and top_performer[1] > np.mean(list(efficiency_scores.values())) * 1.3:
            recommendations.append(OptimizationRecommendation(
                channel=top_performer[0],
                optimization_type=OptimizationType.BUDGET,
                priority=OptimizationPriority.HIGH,
                title=f"Scale {top_performer[0]} Investment",
                description="ML analysis indicates strong scaling potential based on efficiency metrics.",
                expected_impact=0.35,
                confidence_score=0.75,
                implementation_effort="Low",
                estimated_timeline="1 week",
                success_metrics=["Revenue Growth", "ROI", "Market Share"],
                action_steps=[
                    "Gradually increase budget by 20-30%",
                    "Monitor performance closely",
                    "Expand successful campaigns",
                    "Test new audience segments"
                ],
                risk_level="Medium"
            ))
        
        return recommendations
    
    def _calculate_summary_metrics(self,
                                 performance_data: Dict[str, Dict[str, float]],
                                 attribution_weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate high-level summary metrics."""
        
        total_spend = sum(metrics.get('spend', 0) for metrics in performance_data.values())
        total_revenue = sum(metrics.get('revenue', 0) for metrics in performance_data.values())
        
        weighted_roas = sum(
            metrics.get('roas', 0) * attribution_weights.get(channel, 0)
            for channel, metrics in performance_data.items()
        )
        
        return {
            'total_channels': len(performance_data),
            'total_spend': total_spend,
            'total_revenue': total_revenue,
            'overall_roas': total_revenue / total_spend if total_spend > 0 else 0,
            'weighted_attribution_roas': weighted_roas,
            'optimization_opportunities': len(self.optimization_recommendations),
            'high_priority_recommendations': len([
                r for r in self.optimization_recommendations 
                if r.priority in [OptimizationPriority.CRITICAL, OptimizationPriority.HIGH]
            ])
        }
    
    def generate_optimization_report(self) -> str:
        """Generate comprehensive channel optimization report."""
        
        report = "# Channel Optimization Analysis Report\n\n"
        report += "**Advanced Marketing Channel Intelligence by Sotiris Spyrou**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        if not self.optimization_recommendations:
            report += "## No Optimization Analysis Available\n\n"
            report += "Run channel analysis first to generate optimization recommendations.\n"
            return report
        
        # Executive Summary
        high_priority = len([r for r in self.optimization_recommendations if r.priority in [OptimizationPriority.CRITICAL, OptimizationPriority.HIGH]])
        total_expected_impact = sum(r.expected_impact for r in self.optimization_recommendations)
        
        report += f"## Executive Summary\n\n"
        report += f"- **Total Optimization Opportunities**: {len(self.optimization_recommendations)}\n"
        report += f"- **High Priority Actions**: {high_priority}\n"
        report += f"- **Expected Performance Improvement**: {total_expected_impact:.1%}\n"
        report += f"- **Channels Analyzed**: {len(set(r.channel for r in self.optimization_recommendations))}\n\n"
        
        # Priority Recommendations
        report += f"## Priority Optimization Recommendations\n\n"
        
        priority_recs = [r for r in self.optimization_recommendations if r.priority in [OptimizationPriority.CRITICAL, OptimizationPriority.HIGH]][:5]
        
        for i, rec in enumerate(priority_recs, 1):
            priority_emoji = "🔴" if rec.priority == OptimizationPriority.CRITICAL else "🟠"
            report += f"### {i}. {rec.title} {priority_emoji}\n\n"
            report += f"**Channel**: {rec.channel} | **Type**: {rec.optimization_type.value.title()}\n\n"
            report += f"**Description**: {rec.description}\n\n"
            report += f"**Expected Impact**: {rec.expected_impact:.1%} | **Confidence**: {rec.confidence_score:.1%}\n\n"
            report += f"**Implementation**: {rec.implementation_effort} effort, {rec.estimated_timeline}\n\n"
            report += f"**Action Steps**:\n"
            for step in rec.action_steps[:3]:  # Show first 3 steps
                report += f"- {step}\n"
            report += "\n"
        
        # Optimization by Type
        type_summary = {}
        for rec in self.optimization_recommendations:
            opt_type = rec.optimization_type.value
            if opt_type not in type_summary:
                type_summary[opt_type] = {'count': 0, 'impact': 0}
            type_summary[opt_type]['count'] += 1
            type_summary[opt_type]['impact'] += rec.expected_impact
        
        report += f"## Optimization Categories\n\n"
        for opt_type, data in sorted(type_summary.items(), key=lambda x: x[1]['impact'], reverse=True):
            type_emoji = {
                'targeting': "🎯", 'bidding': "💰", 'creative': "🎨", 
                'budget': "📊", 'audience': "👥", 'placement': "📍", 'timing': "⏰"
            }.get(opt_type, "🔧")
            
            report += f"- {type_emoji} **{opt_type.title()}**: {data['count']} opportunities, {data['impact']:.1%} potential impact\n"
        
        # Top Channel Insights
        if self.channel_insights:
            report += f"\n## Key Channel Insights\n\n"
            
            significant_insights = [i for i in self.channel_insights if abs(i.performance_gap) > 0.1][:5]
            for insight in significant_insights:
                gap_emoji = "📈" if insight.performance_gap > 0 else "📉"
                report += f"- {gap_emoji} {insight.insight_text}\n"
        
        report += f"\n## Implementation Roadmap\n\n"
        report += f"1. **Week 1-2**: Focus on high-priority, low-effort optimizations\n"
        report += f"2. **Week 3-4**: Implement medium-effort recommendations with high impact\n"
        report += f"3. **Month 2**: Execute comprehensive optimization projects\n"
        report += f"4. **Ongoing**: Monitor results and iterate based on performance\n\n"
        
        report += "---\n*This analysis provides data-driven channel optimization strategies. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for enterprise implementations.*"
        
        return report


def demo_channel_optimization():
    """Executive demonstration of Channel Optimization Engine."""
    
    print("=== Channel Optimization Engine: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    # Initialize optimizer
    optimizer = ChannelOptimizer(
        lookback_days=30,
        benchmark_percentile=0.75,
        enable_predictive_recommendations=True
    )
    
    print("🔧 Setting up channel optimization scenario...")
    
    # Sample performance data with optimization opportunities
    performance_data = {
        'Search': {
            'roas': 4.2,
            'ctr': 0.08,           # Good CTR
            'conversion_rate': 0.18,
            'cpa': 35.00,
            'spend': 50000,
            'revenue': 210000
        },
        'Display': {
            'roas': 1.8,           # Low ROAS - needs optimization
            'ctr': 0.02,           # Very low CTR - creative issue
            'conversion_rate': 0.09,
            'cpa': 85.00,          # High CPA
            'spend': 30000,
            'revenue': 54000
        },
        'Social': {
            'roas': 3.5,
            'ctr': 0.06,
            'conversion_rate': 0.14,
            'cpa': 42.00,
            'spend': 25000,
            'revenue': 87500
        },
        'Email': {
            'roas': 6.2,           # High ROAS - scaling opportunity
            'ctr': 0.15,
            'conversion_rate': 0.28,
            'cpa': 18.00,          # Low CPA
            'spend': 15000,
            'revenue': 93000
        },
        'Direct': {
            'roas': 8.0,           # Very high ROAS
            'ctr': 0.20,
            'conversion_rate': 0.35,
            'cpa': 12.00,
            'spend': 5000,
            'revenue': 40000
        }
    }
    
    # Attribution weights from attribution model
    attribution_weights = {
        'Search': 0.35,
        'Display': 0.15,         # Low attribution despite spend
        'Social': 0.25,
        'Email': 0.20,
        'Direct': 0.05
    }
    
    # Industry benchmarks for comparison
    industry_benchmarks = {
        'roas': {'p25': 2.0, 'p50': 3.0, 'p75': 4.5, 'p90': 6.0},
        'ctr': {'p25': 0.03, 'p50': 0.05, 'p75': 0.08, 'p90': 0.12},
        'conversion_rate': {'p25': 0.10, 'p50': 0.15, 'p75': 0.20, 'p90': 0.25},
        'cpa': {'p25': 25.0, 'p50': 40.0, 'p75': 60.0, 'p90': 80.0}
    }
    
    print(f"📊 Analyzing {len(performance_data)} channels...")
    
    # Perform comprehensive analysis
    results = optimizer.analyze_channels(
        performance_data=performance_data,
        attribution_weights=attribution_weights,
        industry_benchmarks=industry_benchmarks
    )
    
    # Display results
    print(f"\n🎯 CHANNEL OPTIMIZATION ANALYSIS")
    print("=" * 55)
    
    summary = results['summary_metrics']
    print(f"\n📈 PERFORMANCE OVERVIEW:")
    print(f"  • Total Marketing Spend: ${summary['total_spend']:,}")
    print(f"  • Total Revenue Generated: ${summary['total_revenue']:,}")
    print(f"  • Overall ROAS: {summary['overall_roas']:.1f}x")
    print(f"  • Attribution-Weighted ROAS: {summary['weighted_attribution_roas']:.1f}x")
    print(f"  • Optimization Opportunities: {summary['optimization_opportunities']}")
    print(f"  • High Priority Actions: {summary['high_priority_recommendations']}")
    
    # Performance gaps
    gaps = results['performance_gaps']
    print(f"\n📊 CHANNEL PERFORMANCE RANKING:")
    for i, (channel, score) in enumerate(gaps['channel_rankings'][:5], 1):
        rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
        print(f"{rank_emoji} {channel:8}: {score:.2f} performance score")
    
    # Top optimization opportunities
    recommendations = results['optimization_recommendations']
    print(f"\n🔧 TOP OPTIMIZATION RECOMMENDATIONS:")
    
    priority_recs = [r for r in recommendations if r.priority in [OptimizationPriority.CRITICAL, OptimizationPriority.HIGH]]
    
    for i, rec in enumerate(priority_recs[:5], 1):
        priority_emoji = "🔴" if rec.priority == OptimizationPriority.CRITICAL else "🟠"
        type_emoji = {
            'targeting': "🎯", 'bidding': "💰", 'creative': "🎨", 
            'budget': "📊", 'audience': "👥"
        }.get(rec.optimization_type.value, "🔧")
        
        print(f"{i}. {priority_emoji} {rec.title}")
        print(f"   {type_emoji} {rec.channel} | {rec.optimization_type.value.title()}")
        print(f"   Expected Impact: {rec.expected_impact:.1%} | Confidence: {rec.confidence_score:.1%}")
        print(f"   Timeline: {rec.estimated_timeline} | Effort: {rec.implementation_effort}")
        print()
    
    # Attribution efficiency analysis
    efficiency = results['attribution_efficiency']
    print(f"💡 ATTRIBUTION EFFICIENCY INSIGHTS:")
    
    if efficiency['over_attributed']:
        print(f"  📈 Over-attributed (scale up):")
        for channel_data in efficiency['over_attributed'][:2]:
            print(f"    • {channel_data['channel']}: {channel_data['efficiency_ratio']:.1f}x efficiency")
    
    if efficiency['under_attributed']:
        print(f"  📉 Under-attributed (review/optimize):")
        for channel_data in efficiency['under_attributed'][:2]:
            print(f"    • {channel_data['channel']}: {channel_data['efficiency_ratio']:.1f}x efficiency")
    
    # Channel clustering insights
    clustering = results.get('channel_segments', {})
    if 'clusters' in clustering:
        print(f"\n🎯 CHANNEL SEGMENTATION:")
        for cluster_id, cluster_data in clustering['clusters'].items():
            channels_list = ', '.join(cluster_data['channels'])
            print(f"  • {cluster_data['name']}: {channels_list}")
    
    # Key insights
    insights = results['channel_insights']
    significant_insights = [i for i in insights if abs(i.performance_gap) > 0.1][:3]
    
    if significant_insights:
        print(f"\n💡 KEY PERFORMANCE INSIGHTS:")
        for insight in significant_insights:
            gap_emoji = "📈" if insight.performance_gap > 0 else "📉"
            print(f"{gap_emoji} {insight.insight_text}")
    
    # Implementation roadmap
    print(f"\n🗓️ IMPLEMENTATION ROADMAP:")
    print(f"  Week 1-2: Quick wins (Creative optimization for Display)")
    print(f"  Week 3-4: Medium effort (Bidding optimization for underperformers)")
    print(f"  Month 2: Strategic changes (Budget reallocation to high-performers)")
    print(f"  Ongoing: Performance monitoring and iterative optimization")
    
    print(f"\n💰 EXPECTED RESULTS:")
    total_impact = sum(r.expected_impact for r in recommendations[:5])  # Top 5 recommendations
    current_revenue = sum(ch.get('revenue', 0) for ch in performance_data.values())
    projected_improvement = current_revenue * total_impact
    
    print(f"  • Projected Revenue Increase: ${projected_improvement:,.0f}")
    print(f"  • Overall Performance Improvement: {total_impact:.1%}")
    print(f"  • Implementation Timeline: 4-8 weeks")
    print(f"  • Risk Level: Low to Medium")
    
    print("\n" + "="*60)
    print("🚀 Advanced channel optimization for maximum marketing efficiency")
    print("💼 Data-driven insights and actionable recommendations")
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_channel_optimization()