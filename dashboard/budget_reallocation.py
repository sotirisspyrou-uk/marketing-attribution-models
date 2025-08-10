"""
Dynamic Budget Reallocation System

Intelligent budget optimization and reallocation system based on real-time
attribution insights, performance metrics, and predictive modeling for
maximizing marketing ROI across channels.

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
from scipy.optimize import minimize, LinearConstraint
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ReallocationStrategy(Enum):
    """Budget reallocation strategies."""
    PERFORMANCE_BASED = "performance_based"
    ATTRIBUTION_WEIGHTED = "attribution_weighted"
    ROI_MAXIMIZATION = "roi_maximization"
    RISK_ADJUSTED = "risk_adjusted"
    OPPORTUNITY_DRIVEN = "opportunity_driven"


@dataclass
class BudgetConstraint:
    """Budget allocation constraint."""
    channel: str
    min_budget: float
    max_budget: float
    min_percentage: Optional[float] = None
    max_percentage: Optional[float] = None


@dataclass
class ReallocationResult:
    """Budget reallocation recommendation result."""
    channel: str
    current_budget: float
    recommended_budget: float
    change_amount: float
    change_percentage: float
    expected_roi_impact: float
    confidence_score: float
    rationale: str


class BudgetReallocationEngine:
    """
    Advanced budget reallocation system for marketing attribution optimization.
    
    Uses machine learning, optimization algorithms, and business constraints
    to recommend intelligent budget reallocations across marketing channels.
    """
    
    def __init__(self,
                 reallocation_strategy: ReallocationStrategy = ReallocationStrategy.ROI_MAXIMIZATION,
                 lookback_days: int = 30,
                 min_reallocation_threshold: float = 0.05,  # 5%
                 max_single_channel_percentage: float = 0.6,  # 60%
                 risk_tolerance: float = 0.2):
        """
        Initialize Budget Reallocation Engine.
        
        Args:
            reallocation_strategy: Strategy for budget optimization
            lookback_days: Historical data window for analysis
            min_reallocation_threshold: Minimum change to recommend
            max_single_channel_percentage: Maximum budget for single channel
            risk_tolerance: Risk tolerance for reallocation decisions
        """
        self.strategy = reallocation_strategy
        self.lookback_days = lookback_days
        self.min_reallocation_threshold = min_reallocation_threshold
        self.max_single_channel_percentage = max_single_channel_percentage
        self.risk_tolerance = risk_tolerance
        
        # Models and analysis
        self.performance_model = None
        self.scaler = StandardScaler()
        self.historical_performance = {}
        self.attribution_history = []
        self.constraints = []
        
        # Reallocation results
        self.last_recommendations = []
        self.reallocation_history = []
        
        logger.info(f"Budget reallocation engine initialized with {self.strategy.value} strategy")
    
    def add_constraint(self, constraint: BudgetConstraint):
        """Add budget allocation constraint."""
        self.constraints.append(constraint)
        logger.info(f"Added budget constraint for {constraint.channel}")
    
    def analyze_reallocation_opportunities(self,
                                         current_budgets: Dict[str, float],
                                         performance_data: Dict[str, Dict[str, float]],
                                         attribution_weights: Dict[str, float],
                                         total_budget: Optional[float] = None) -> List[ReallocationResult]:
        """
        Analyze and recommend budget reallocation opportunities.
        
        Args:
            current_budgets: Current budget allocation by channel
            performance_data: Performance metrics by channel
            attribution_weights: Attribution weights by channel
            total_budget: Total budget constraint (optional)
            
        Returns:
            List of reallocation recommendations
        """
        logger.info("Analyzing budget reallocation opportunities")
        
        if total_budget is None:
            total_budget = sum(current_budgets.values())
        
        # Update historical data
        self._update_historical_data(current_budgets, performance_data, attribution_weights)
        
        # Train performance prediction model
        self._train_performance_model(performance_data)
        
        # Calculate reallocation recommendations based on strategy
        if self.strategy == ReallocationStrategy.PERFORMANCE_BASED:
            recommendations = self._performance_based_reallocation(
                current_budgets, performance_data, total_budget
            )
        elif self.strategy == ReallocationStrategy.ATTRIBUTION_WEIGHTED:
            recommendations = self._attribution_weighted_reallocation(
                current_budgets, attribution_weights, total_budget
            )
        elif self.strategy == ReallocationStrategy.ROI_MAXIMIZATION:
            recommendations = self._roi_maximization_reallocation(
                current_budgets, performance_data, attribution_weights, total_budget
            )
        elif self.strategy == ReallocationStrategy.RISK_ADJUSTED:
            recommendations = self._risk_adjusted_reallocation(
                current_budgets, performance_data, attribution_weights, total_budget
            )
        else:  # OPPORTUNITY_DRIVEN
            recommendations = self._opportunity_driven_reallocation(
                current_budgets, performance_data, attribution_weights, total_budget
            )
        
        # Apply constraints and filters
        recommendations = self._apply_constraints(recommendations, total_budget)
        recommendations = self._filter_recommendations(recommendations)
        
        # Store results
        self.last_recommendations = recommendations
        self.reallocation_history.append({
            'timestamp': datetime.now(),
            'recommendations': recommendations,
            'total_budget': total_budget,
            'strategy': self.strategy.value
        })
        
        logger.info(f"Generated {len(recommendations)} reallocation recommendations")
        return recommendations
    
    def _update_historical_data(self,
                              current_budgets: Dict[str, float],
                              performance_data: Dict[str, Dict[str, float]],
                              attribution_weights: Dict[str, float]):
        """Update historical performance and attribution data."""
        
        timestamp = datetime.now()
        
        # Update performance history
        for channel, budget in current_budgets.items():
            if channel not in self.historical_performance:
                self.historical_performance[channel] = []
            
            channel_performance = performance_data.get(channel, {})
            
            self.historical_performance[channel].append({
                'timestamp': timestamp,
                'budget': budget,
                'performance': channel_performance,
                'attribution_weight': attribution_weights.get(channel, 0)
            })
            
            # Keep only recent history
            cutoff_date = timestamp - timedelta(days=self.lookback_days * 2)
            self.historical_performance[channel] = [
                h for h in self.historical_performance[channel]
                if h['timestamp'] > cutoff_date
            ]
        
        # Update attribution history
        self.attribution_history.append({
            'timestamp': timestamp,
            'weights': attribution_weights.copy()
        })
        
        # Keep recent attribution history
        cutoff_date = timestamp - timedelta(days=self.lookback_days)
        self.attribution_history = [
            h for h in self.attribution_history
            if h['timestamp'] > cutoff_date
        ]
    
    def _train_performance_model(self, performance_data: Dict[str, Dict[str, float]]):
        """Train ML model to predict performance based on budget allocation."""
        
        training_data = []
        targets = []
        
        for channel, history in self.historical_performance.items():
            if len(history) < 5:  # Need minimum data
                continue
            
            for entry in history:
                budget = entry['budget']
                performance = entry['performance']
                attribution = entry['attribution_weight']
                
                # Features: budget, attribution weight, historical averages
                features = [
                    budget,
                    attribution,
                    performance.get('roas', 0),
                    performance.get('conversion_rate', 0),
                    performance.get('ctr', 0)
                ]
                
                # Target: overall efficiency score
                roas = performance.get('roas', 0)
                cvr = performance.get('conversion_rate', 0)
                efficiency_score = roas * cvr if roas and cvr else 0
                
                training_data.append(features)
                targets.append(efficiency_score)
        
        if len(training_data) < 10:  # Need sufficient training data
            logger.warning("Insufficient data for performance model training")
            return
        
        # Train model
        X = np.array(training_data)
        y = np.array(targets)
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.performance_model = RandomForestRegressor(
            n_estimators=50,
            random_state=42
        )
        self.performance_model.fit(X_scaled, y)
        
        logger.info("Performance prediction model trained successfully")
    
    def _performance_based_reallocation(self,
                                      current_budgets: Dict[str, float],
                                      performance_data: Dict[str, Dict[str, float]],
                                      total_budget: float) -> List[ReallocationResult]:
        """Calculate performance-based budget reallocation."""
        
        recommendations = []
        
        # Calculate performance scores
        performance_scores = {}
        for channel, metrics in performance_data.items():
            roas = metrics.get('roas', 0)
            cvr = metrics.get('conversion_rate', 0)
            ctr = metrics.get('ctr', 0)
            
            # Composite performance score
            performance_scores[channel] = (roas * 0.5) + (cvr * 0.3) + (ctr * 0.2)
        
        # Calculate optimal allocation based on performance
        total_score = sum(performance_scores.values())
        
        if total_score == 0:
            return recommendations
        
        for channel, current_budget in current_budgets.items():
            performance_ratio = performance_scores.get(channel, 0) / total_score
            optimal_budget = total_budget * performance_ratio
            
            change_amount = optimal_budget - current_budget
            change_percentage = (change_amount / current_budget) if current_budget > 0 else 0
            
            # Calculate expected ROI impact
            current_roas = performance_data.get(channel, {}).get('roas', 0)
            expected_roi_impact = change_amount * current_roas if current_roas > 0 else 0
            
            # Confidence based on historical consistency
            confidence = self._calculate_confidence(channel, 'performance')
            
            recommendation = ReallocationResult(
                channel=channel,
                current_budget=current_budget,
                recommended_budget=optimal_budget,
                change_amount=change_amount,
                change_percentage=change_percentage,
                expected_roi_impact=expected_roi_impact,
                confidence_score=confidence,
                rationale=f"Performance-based allocation: {channel} has {performance_scores.get(channel, 0):.2f} performance score"
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _attribution_weighted_reallocation(self,
                                         current_budgets: Dict[str, float],
                                         attribution_weights: Dict[str, float],
                                         total_budget: float) -> List[ReallocationResult]:
        """Calculate attribution-weighted budget reallocation."""
        
        recommendations = []
        
        # Normalize attribution weights
        total_attribution = sum(attribution_weights.values())
        if total_attribution == 0:
            return recommendations
        
        for channel, current_budget in current_budgets.items():
            attribution_ratio = attribution_weights.get(channel, 0) / total_attribution
            optimal_budget = total_budget * attribution_ratio
            
            change_amount = optimal_budget - current_budget
            change_percentage = (change_amount / current_budget) if current_budget > 0 else 0
            
            # Expected ROI impact based on attribution efficiency
            attribution_efficiency = attribution_weights.get(channel, 0) / (current_budget / total_budget) if current_budget > 0 else 0
            expected_roi_impact = change_amount * attribution_efficiency
            
            confidence = self._calculate_confidence(channel, 'attribution')
            
            recommendation = ReallocationResult(
                channel=channel,
                current_budget=current_budget,
                recommended_budget=optimal_budget,
                change_amount=change_amount,
                change_percentage=change_percentage,
                expected_roi_impact=expected_roi_impact,
                confidence_score=confidence,
                rationale=f"Attribution-weighted allocation: {channel} has {attribution_weights.get(channel, 0):.1%} attribution weight"
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _roi_maximization_reallocation(self,
                                     current_budgets: Dict[str, float],
                                     performance_data: Dict[str, Dict[str, float]],
                                     attribution_weights: Dict[str, float],
                                     total_budget: float) -> List[ReallocationResult]:
        """Calculate ROI-maximizing budget reallocation using optimization."""
        
        channels = list(current_budgets.keys())
        n_channels = len(channels)
        
        if n_channels == 0:
            return []
        
        # Define objective function (negative because we minimize)
        def objective(budget_allocation):
            total_roi = 0
            for i, channel in enumerate(channels):
                budget = budget_allocation[i]
                metrics = performance_data.get(channel, {})
                roas = metrics.get('roas', 0)
                attribution_weight = attribution_weights.get(channel, 0)
                
                # ROI with diminishing returns and attribution weighting
                channel_roi = budget * roas * attribution_weight * (1 - 0.1 * (budget / total_budget))
                total_roi += channel_roi
            
            return -total_roi  # Negative for minimization
        
        # Constraints
        constraints = []
        
        # Budget constraint: sum equals total budget
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - total_budget
        })
        
        # Individual channel constraints
        bounds = []
        for i, channel in enumerate(channels):
            current_budget = current_budgets[channel]
            
            # Default bounds: 10% to 200% of current budget
            lower_bound = max(0, current_budget * 0.1)
            upper_bound = current_budget * 2.0
            
            # Apply custom constraints if defined
            for constraint in self.constraints:
                if constraint.channel == channel:
                    if constraint.min_budget:
                        lower_bound = max(lower_bound, constraint.min_budget)
                    if constraint.max_budget:
                        upper_bound = min(upper_bound, constraint.max_budget)
                    if constraint.min_percentage:
                        lower_bound = max(lower_bound, total_budget * constraint.min_percentage)
                    if constraint.max_percentage:
                        upper_bound = min(upper_bound, total_budget * constraint.max_percentage)
            
            bounds.append((lower_bound, upper_bound))
        
        # Initial guess: current allocation
        x0 = np.array([current_budgets[ch] for ch in channels])
        
        # Optimize
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-6, 'maxiter': 1000}
            )
            
            optimal_allocation = result.x
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return []
        
        # Generate recommendations
        recommendations = []
        for i, channel in enumerate(channels):
            current_budget = current_budgets[channel]
            optimal_budget = optimal_allocation[i]
            
            change_amount = optimal_budget - current_budget
            change_percentage = (change_amount / current_budget) if current_budget > 0 else 0
            
            # Calculate expected ROI impact
            metrics = performance_data.get(channel, {})
            roas = metrics.get('roas', 0)
            expected_roi_impact = change_amount * roas
            
            confidence = self._calculate_confidence(channel, 'roi')
            
            recommendation = ReallocationResult(
                channel=channel,
                current_budget=current_budget,
                recommended_budget=optimal_budget,
                change_amount=change_amount,
                change_percentage=change_percentage,
                expected_roi_impact=expected_roi_impact,
                confidence_score=confidence,
                rationale=f"ROI optimization: Mathematical optimization suggests {change_percentage:+.1%} change for maximum ROI"
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _risk_adjusted_reallocation(self,
                                  current_budgets: Dict[str, float],
                                  performance_data: Dict[str, Dict[str, float]],
                                  attribution_weights: Dict[str, float],
                                  total_budget: float) -> List[ReallocationResult]:
        """Calculate risk-adjusted budget reallocation."""
        
        recommendations = []
        
        # Calculate risk scores based on performance volatility
        risk_scores = {}
        for channel in current_budgets.keys():
            if channel in self.historical_performance:
                history = self.historical_performance[channel]
                if len(history) >= 3:
                    roas_values = [h['performance'].get('roas', 0) for h in history]
                    volatility = np.std(roas_values) / np.mean(roas_values) if np.mean(roas_values) > 0 else 1
                    risk_scores[channel] = volatility
                else:
                    risk_scores[channel] = self.risk_tolerance
            else:
                risk_scores[channel] = self.risk_tolerance
        
        # Risk-adjusted performance scores
        risk_adjusted_scores = {}
        for channel, budget in current_budgets.items():
            metrics = performance_data.get(channel, {})
            roas = metrics.get('roas', 0)
            attribution = attribution_weights.get(channel, 0)
            risk = risk_scores.get(channel, self.risk_tolerance)
            
            # Penalize high-risk channels
            risk_penalty = 1 - (risk * self.risk_tolerance)
            risk_adjusted_scores[channel] = (roas * attribution * risk_penalty)
        
        # Allocate based on risk-adjusted scores
        total_score = sum(risk_adjusted_scores.values())
        
        if total_score == 0:
            return recommendations
        
        for channel, current_budget in current_budgets.items():
            score_ratio = risk_adjusted_scores[channel] / total_score
            optimal_budget = total_budget * score_ratio
            
            change_amount = optimal_budget - current_budget
            change_percentage = (change_amount / current_budget) if current_budget > 0 else 0
            
            expected_roi_impact = change_amount * performance_data.get(channel, {}).get('roas', 0)
            confidence = self._calculate_confidence(channel, 'risk_adjusted')
            
            risk_level = "High" if risk_scores[channel] > self.risk_tolerance * 2 else "Medium" if risk_scores[channel] > self.risk_tolerance else "Low"
            
            recommendation = ReallocationResult(
                channel=channel,
                current_budget=current_budget,
                recommended_budget=optimal_budget,
                change_amount=change_amount,
                change_percentage=change_percentage,
                expected_roi_impact=expected_roi_impact,
                confidence_score=confidence,
                rationale=f"Risk-adjusted allocation: {channel} has {risk_level} risk profile with {risk_adjusted_scores[channel]:.2f} adjusted score"
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _opportunity_driven_reallocation(self,
                                       current_budgets: Dict[str, float],
                                       performance_data: Dict[str, Dict[str, float]],
                                       attribution_weights: Dict[str, float],
                                       total_budget: float) -> List[ReallocationResult]:
        """Calculate opportunity-driven budget reallocation."""
        
        recommendations = []
        
        # Identify high-opportunity channels
        opportunity_scores = {}
        for channel, budget in current_budgets.items():
            metrics = performance_data.get(channel, {})
            roas = metrics.get('roas', 0)
            attribution = attribution_weights.get(channel, 0)
            
            # Current budget efficiency
            budget_share = budget / total_budget if total_budget > 0 else 0
            attribution_efficiency = attribution / budget_share if budget_share > 0 else 0
            
            # Growth potential based on recent trends
            growth_potential = self._calculate_growth_potential(channel)
            
            # Opportunity score combines efficiency and growth potential
            opportunity_scores[channel] = (attribution_efficiency * 0.6) + (growth_potential * 0.4)
        
        # Calculate opportunity-driven allocation
        total_opportunity = sum(opportunity_scores.values())
        
        if total_opportunity == 0:
            return recommendations
        
        for channel, current_budget in current_budgets.items():
            opportunity_ratio = opportunity_scores[channel] / total_opportunity
            optimal_budget = total_budget * opportunity_ratio
            
            change_amount = optimal_budget - current_budget
            change_percentage = (change_amount / current_budget) if current_budget > 0 else 0
            
            expected_roi_impact = change_amount * performance_data.get(channel, {}).get('roas', 0)
            confidence = self._calculate_confidence(channel, 'opportunity')
            
            recommendation = ReallocationResult(
                channel=channel,
                current_budget=current_budget,
                recommended_budget=optimal_budget,
                change_amount=change_amount,
                change_percentage=change_percentage,
                expected_roi_impact=expected_roi_impact,
                confidence_score=confidence,
                rationale=f"Opportunity-driven allocation: {channel} has {opportunity_scores[channel]:.2f} opportunity score"
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _calculate_confidence(self, channel: str, strategy: str) -> float:
        """Calculate confidence score for reallocation recommendation."""
        
        base_confidence = 0.5
        
        # Historical data availability
        if channel in self.historical_performance:
            history_length = len(self.historical_performance[channel])
            data_confidence = min(history_length / 20, 1.0)  # Max confidence at 20 data points
            base_confidence += data_confidence * 0.3
        
        # Performance consistency
        if channel in self.historical_performance and len(self.historical_performance[channel]) >= 3:
            roas_values = [h['performance'].get('roas', 0) for h in self.historical_performance[channel]]
            if len(roas_values) > 1:
                consistency = 1 - (np.std(roas_values) / np.mean(roas_values)) if np.mean(roas_values) > 0 else 0
                base_confidence += max(0, consistency) * 0.2
        
        return min(base_confidence, 1.0)
    
    def _calculate_growth_potential(self, channel: str) -> float:
        """Calculate growth potential for a channel."""
        
        if channel not in self.historical_performance:
            return 0.5  # Neutral potential for new channels
        
        history = self.historical_performance[channel][-10:]  # Last 10 data points
        
        if len(history) < 3:
            return 0.5
        
        # Calculate trend in ROAS
        roas_values = [h['performance'].get('roas', 0) for h in history]
        
        if len(roas_values) < 2:
            return 0.5
        
        # Simple linear trend
        x = np.arange(len(roas_values))
        trend = np.polyfit(x, roas_values, 1)[0]  # Slope
        
        # Normalize trend to 0-1 scale
        growth_potential = 0.5 + np.tanh(trend * 5) * 0.5
        
        return max(0, min(1, growth_potential))
    
    def _apply_constraints(self,
                          recommendations: List[ReallocationResult],
                          total_budget: float) -> List[ReallocationResult]:
        """Apply business constraints to recommendations."""
        
        # Apply individual channel constraints
        for recommendation in recommendations:
            for constraint in self.constraints:
                if constraint.channel == recommendation.channel:
                    if constraint.min_budget and recommendation.recommended_budget < constraint.min_budget:
                        recommendation.recommended_budget = constraint.min_budget
                    if constraint.max_budget and recommendation.recommended_budget > constraint.max_budget:
                        recommendation.recommended_budget = constraint.max_budget
                    if constraint.min_percentage:
                        min_budget = total_budget * constraint.min_percentage
                        if recommendation.recommended_budget < min_budget:
                            recommendation.recommended_budget = min_budget
                    if constraint.max_percentage:
                        max_budget = total_budget * constraint.max_percentage
                        if recommendation.recommended_budget > max_budget:
                            recommendation.recommended_budget = max_budget
                    
                    # Recalculate changes
                    recommendation.change_amount = recommendation.recommended_budget - recommendation.current_budget
                    recommendation.change_percentage = (recommendation.change_amount / recommendation.current_budget) if recommendation.current_budget > 0 else 0
        
        # Ensure total budget is maintained
        total_recommended = sum(r.recommended_budget for r in recommendations)
        if total_recommended != total_budget and total_recommended > 0:
            adjustment_factor = total_budget / total_recommended
            for recommendation in recommendations:
                recommendation.recommended_budget *= adjustment_factor
                recommendation.change_amount = recommendation.recommended_budget - recommendation.current_budget
                recommendation.change_percentage = (recommendation.change_amount / recommendation.current_budget) if recommendation.current_budget > 0 else 0
        
        return recommendations
    
    def _filter_recommendations(self, recommendations: List[ReallocationResult]) -> List[ReallocationResult]:
        """Filter recommendations based on thresholds and business logic."""
        
        filtered = []
        
        for recommendation in recommendations:
            # Skip small changes
            if abs(recommendation.change_percentage) < self.min_reallocation_threshold:
                continue
            
            # Skip if confidence is too low
            if recommendation.confidence_score < 0.3:
                continue
            
            filtered.append(recommendation)
        
        return filtered
    
    def get_reallocation_summary(self) -> Dict[str, Any]:
        """Get summary of latest reallocation recommendations."""
        
        if not self.last_recommendations:
            return {'no_recommendations': True}
        
        total_reallocation = sum(abs(r.change_amount) for r in self.last_recommendations)
        positive_changes = [r for r in self.last_recommendations if r.change_amount > 0]
        negative_changes = [r for r in self.last_recommendations if r.change_amount < 0]
        
        expected_roi_impact = sum(r.expected_roi_impact for r in self.last_recommendations)
        avg_confidence = np.mean([r.confidence_score for r in self.last_recommendations])
        
        return {
            'total_recommendations': len(self.last_recommendations),
            'total_reallocation_amount': total_reallocation,
            'channels_increasing_budget': len(positive_changes),
            'channels_decreasing_budget': len(negative_changes),
            'expected_roi_impact': expected_roi_impact,
            'average_confidence': avg_confidence,
            'strategy_used': self.strategy.value,
            'top_increase': max(positive_changes, key=lambda x: x.change_amount) if positive_changes else None,
            'top_decrease': min(negative_changes, key=lambda x: x.change_amount) if negative_changes else None
        }
    
    def generate_reallocation_report(self) -> str:
        """Generate comprehensive budget reallocation report."""
        
        report = "# Budget Reallocation Analysis Report\n\n"
        report += "**Strategic Marketing Budget Optimization by Sotiris Spyrou**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        if not self.last_recommendations:
            report += "## No Recommendations Available\n\n"
            report += "Run budget analysis first to generate reallocation recommendations.\n"
            return report
        
        summary = self.get_reallocation_summary()
        
        # Executive Summary
        report += f"## Executive Summary\n\n"
        report += f"- **Strategy Used**: {summary['strategy_used'].replace('_', ' ').title()}\n"
        report += f"- **Total Recommendations**: {summary['total_recommendations']}\n"
        report += f"- **Expected ROI Impact**: ${summary['expected_roi_impact']:,.2f}\n"
        report += f"- **Average Confidence**: {summary['average_confidence']:.1%}\n\n"
        
        # Budget Changes
        report += f"## Recommended Budget Changes\n\n"
        report += "| Channel | Current Budget | Recommended | Change | Change % | ROI Impact | Confidence |\n"
        report += "|---------|---------------|-------------|--------|----------|------------|------------|\n"
        
        # Sort by change amount (largest increases first)
        sorted_recommendations = sorted(
            self.last_recommendations, 
            key=lambda x: x.change_amount, 
            reverse=True
        )
        
        for rec in sorted_recommendations:
            change_emoji = "📈" if rec.change_amount > 0 else "📉"
            confidence_emoji = "🟢" if rec.confidence_score > 0.7 else "🟡" if rec.confidence_score > 0.5 else "🔴"
            
            report += f"| {rec.channel} | ${rec.current_budget:,.0f} | ${rec.recommended_budget:,.0f} | "
            report += f"{change_emoji} ${rec.change_amount:+,.0f} | {rec.change_percentage:+.1%} | "
            report += f"${rec.expected_roi_impact:+,.0f} | {confidence_emoji} {rec.confidence_score:.1%} |\n"
        
        # Key Insights
        report += f"\n## Key Insights\n\n"
        
        if summary['top_increase']:
            top_inc = summary['top_increase']
            report += f"- **Top Growth Opportunity**: {top_inc.channel} (+${top_inc.change_amount:,.0f}, +{top_inc.change_percentage:.1%})\n"
            report += f"  - {top_inc.rationale}\n"
        
        if summary['top_decrease']:
            top_dec = summary['top_decrease']
            report += f"- **Reallocation Source**: {top_dec.channel} (${top_dec.change_amount:,.0f}, {top_dec.change_percentage:.1%})\n"
            report += f"  - {top_dec.rationale}\n"
        
        report += f"- **Net Efficiency Gain**: Expected ${summary['expected_roi_impact']:+,.2f} ROI improvement\n"
        report += f"- **Implementation Risk**: {'Low' if summary['average_confidence'] > 0.7 else 'Medium' if summary['average_confidence'] > 0.5 else 'High'}\n\n"
        
        # Implementation Recommendations
        report += f"## Implementation Guidelines\n\n"
        report += f"1. **Phase Implementation**: Start with highest-confidence recommendations\n"
        report += f"2. **Monitor Closely**: Track performance for 2-4 weeks before full implementation\n"
        report += f"3. **Risk Management**: Maintain 10-15% buffer for performance fluctuations\n"
        report += f"4. **Review Frequency**: Re-evaluate allocation weekly during implementation\n\n"
        
        report += "---\n*This analysis provides data-driven budget optimization recommendations. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for custom implementation.*"
        
        return report


def demo_budget_reallocation():
    """Executive demonstration of Budget Reallocation System."""
    
    print("=== Budget Reallocation System: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    # Initialize reallocation engine
    engine = BudgetReallocationEngine(
        reallocation_strategy=ReallocationStrategy.ROI_MAXIMIZATION,
        lookback_days=30,
        min_reallocation_threshold=0.05,
        risk_tolerance=0.2
    )
    
    print("💰 Setting up budget reallocation scenario...")
    
    np.random.seed(42)
    
    # Current budget allocation (suboptimal)
    current_budgets = {
        'Search': 50000,
        'Display': 30000,
        'Social': 25000,
        'Email': 15000,
        'Direct': 5000
    }
    
    total_budget = sum(current_budgets.values())
    print(f"📊 Total monthly budget: ${total_budget:,}")
    
    # Performance data showing some channels outperforming others
    performance_data = {
        'Search': {
            'roas': 4.2,
            'conversion_rate': 0.18,
            'ctr': 0.06,
            'cvr': 0.22
        },
        'Display': {
            'roas': 2.1,
            'conversion_rate': 0.12,
            'ctr': 0.04,
            'cvr': 0.15
        },
        'Social': {
            'roas': 3.8,  # High performing
            'conversion_rate': 0.16,
            'ctr': 0.07,
            'cvr': 0.19
        },
        'Email': {
            'roas': 5.5,  # Highest ROAS but low budget
            'conversion_rate': 0.25,
            'ctr': 0.12,
            'cvr': 0.28
        },
        'Direct': {
            'roas': 6.0,  # Very high ROAS, very low budget
            'conversion_rate': 0.30,
            'ctr': 0.15,
            'cvr': 0.35
        }
    }
    
    # Attribution weights from attribution model
    attribution_weights = {
        'Search': 0.30,
        'Display': 0.20,
        'Social': 0.25,
        'Email': 0.15,
        'Direct': 0.10
    }
    
    # Add some business constraints
    engine.add_constraint(BudgetConstraint(
        channel='Search',
        min_budget=40000,  # Must maintain minimum search spend
        max_budget=70000
    ))
    
    engine.add_constraint(BudgetConstraint(
        channel='Email',
        min_percentage=0.15,  # Must be at least 15% of budget
        max_percentage=0.35
    ))
    
    print(f"⚙️ Added budget constraints for Search and Email channels")
    
    # Simulate historical data for model training
    print(f"📈 Simulating historical performance data...")
    
    for i in range(20):  # 20 historical data points
        # Simulate slight variations in performance
        historical_performance = {}
        for channel in current_budgets.keys():
            base_metrics = performance_data[channel]
            historical_performance[channel] = {
                'roas': base_metrics['roas'] + np.random.normal(0, 0.3),
                'conversion_rate': base_metrics['conversion_rate'] + np.random.normal(0, 0.02),
                'ctr': base_metrics['ctr'] + np.random.normal(0, 0.005),
                'cvr': base_metrics['cvr'] + np.random.normal(0, 0.02)
            }
        
        # Simulate attribution variation
        attribution_variation = {}
        for channel, weight in attribution_weights.items():
            attribution_variation[channel] = weight + np.random.normal(0, 0.02)
        
        # Normalize attribution
        total_attr = sum(attribution_variation.values())
        attribution_variation = {k: v/total_attr for k, v in attribution_variation.items()}
        
        # Update historical data
        engine._update_historical_data(current_budgets, historical_performance, attribution_variation)
    
    # Analyze reallocation opportunities
    print(f"\n🔍 Analyzing budget reallocation opportunities...")
    recommendations = engine.analyze_reallocation_opportunities(
        current_budgets=current_budgets,
        performance_data=performance_data,
        attribution_weights=attribution_weights,
        total_budget=total_budget
    )
    
    # Display results
    print(f"\n📊 BUDGET REALLOCATION RECOMMENDATIONS")
    print("=" * 60)
    
    summary = engine.get_reallocation_summary()
    print(f"\n💡 OPTIMIZATION SUMMARY:")
    print(f"  • Strategy: {summary['strategy_used'].replace('_', ' ').title()}")
    print(f"  • Total Recommendations: {summary['total_recommendations']}")
    print(f"  • Expected ROI Impact: ${summary['expected_roi_impact']:+,.0f}")
    print(f"  • Average Confidence: {summary['average_confidence']:.1%}")
    
    print(f"\n📈 RECOMMENDED BUDGET CHANGES:")
    
    # Sort recommendations by change amount
    sorted_recs = sorted(recommendations, key=lambda x: x.change_amount, reverse=True)
    
    for rec in sorted_recs:
        change_emoji = "📈" if rec.change_amount > 0 else "📉"
        confidence_emoji = "🟢" if rec.confidence_score > 0.7 else "🟡" if rec.confidence_score > 0.5 else "🔴"
        
        print(f"{change_emoji} {rec.channel:8}: ${rec.current_budget:6,.0f} → ${rec.recommended_budget:6,.0f} ({rec.change_percentage:+5.1%}) {confidence_emoji}")
        print(f"         Expected ROI Impact: ${rec.expected_roi_impact:+8,.0f}")
        print(f"         Confidence: {rec.confidence_score:.1%} | {rec.rationale}")
        print()
    
    # Show current vs optimal efficiency
    print(f"📊 EFFICIENCY COMPARISON:")
    
    current_efficiency = sum(
        current_budgets[ch] * performance_data[ch]['roas'] * attribution_weights[ch]
        for ch in current_budgets.keys()
    )
    
    optimal_efficiency = sum(
        rec.recommended_budget * performance_data[rec.channel]['roas'] * attribution_weights[rec.channel]
        for rec in recommendations
    )
    
    efficiency_improvement = (optimal_efficiency - current_efficiency) / current_efficiency * 100
    
    print(f"  • Current Allocation Efficiency: {current_efficiency:,.0f}")
    print(f"  • Optimal Allocation Efficiency: {optimal_efficiency:,.0f}")
    print(f"  • Improvement Potential: {efficiency_improvement:+.1f}%")
    
    # Top recommendations
    if summary['top_increase'] and summary['top_decrease']:
        print(f"\n🎯 KEY RECOMMENDATIONS:")
        top_inc = summary['top_increase']
        top_dec = summary['top_decrease']
        
        print(f"  🚀 Scale Up: {top_inc.channel} (+${top_inc.change_amount:,.0f})")
        print(f"     {top_inc.rationale}")
        
        print(f"  📉 Scale Down: {top_dec.channel} (${top_dec.change_amount:,.0f})")
        print(f"     {top_dec.rationale}")
    
    # Implementation guidance
    print(f"\n🛠️ IMPLEMENTATION GUIDANCE:")
    high_confidence = [r for r in recommendations if r.confidence_score > 0.7]
    medium_confidence = [r for r in recommendations if 0.5 < r.confidence_score <= 0.7]
    
    if high_confidence:
        print(f"  • Phase 1 (High Confidence): {', '.join(r.channel for r in high_confidence[:3])}")
    if medium_confidence:
        print(f"  • Phase 2 (Medium Confidence): {', '.join(r.channel for r in medium_confidence[:2])}")
    
    print(f"  • Monitor performance weekly during 4-week implementation")
    print(f"  • Maintain 10% contingency buffer for rapid adjustments")
    
    print("\n" + "="*60)
    print("🚀 Data-driven budget optimization for maximum marketing ROI")
    print("💼 Enterprise-grade allocation intelligence and risk management")
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_budget_reallocation()