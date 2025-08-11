"""
Multi-Channel Budget Optimization Engine

Advanced budget allocation system using mathematical optimization, machine learning,
and constraint-based modeling to maximize ROI across marketing channels.

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
from collections import defaultdict
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from scipy.stats import beta, gamma, norm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class ChannelConstraints:
    """Channel-specific budget constraints."""
    channel: str
    min_budget: float
    max_budget: float
    min_percentage: float = 0.0
    max_percentage: float = 1.0
    required_budget: Optional[float] = None
    priority_level: int = 1  # 1=high, 2=medium, 3=low
    seasonality_factor: float = 1.0
    competitive_pressure: float = 1.0
    brand_alignment: float = 1.0


@dataclass
class OptimizationObjective:
    """Optimization objective configuration."""
    primary_metric: str  # 'roas', 'revenue', 'conversions', 'profit'
    secondary_metrics: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    time_horizon: int = 30  # days
    risk_tolerance: str = 'medium'  # 'low', 'medium', 'high'
    diversification_bonus: float = 0.1
    incrementality_weight: float = 0.3


@dataclass
class BudgetAllocation:
    """Optimized budget allocation result."""
    allocation_id: str
    channel_budgets: Dict[str, float]
    total_budget: float
    expected_metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    optimization_score: float
    risk_score: float
    diversification_score: float
    scenario_analysis: Dict[str, Dict[str, float]]
    recommendations: List[str]
    sensitivity_analysis: Dict[str, Dict[str, float]]
    allocation_rationale: Dict[str, str]
    performance_bounds: Dict[str, Tuple[float, float]]


class BudgetOptimizationEngine:
    """
    Advanced multi-channel budget optimization engine.
    
    Uses mathematical optimization, machine learning models, and constraint-based
    allocation to maximize marketing ROI across channels with configurable
    objectives and risk parameters.
    """
    
    def __init__(self,
                 optimization_method: str = 'hybrid',
                 risk_tolerance: str = 'medium',
                 enable_ml_predictions: bool = True,
                 seasonality_adjustment: bool = True,
                 competitive_adjustment: bool = True):
        """
        Initialize Budget Optimization Engine.
        
        Args:
            optimization_method: 'scipy', 'evolutionary', 'ml_assisted', 'hybrid'
            risk_tolerance: 'conservative', 'medium', 'aggressive'
            enable_ml_predictions: Use ML models for performance prediction
            seasonality_adjustment: Apply seasonal adjustments
            competitive_adjustment: Apply competitive pressure adjustments
        """
        self.optimization_method = optimization_method
        self.risk_tolerance = risk_tolerance
        self.enable_ml_predictions = enable_ml_predictions
        self.seasonality_adjustment = seasonality_adjustment
        self.competitive_adjustment = competitive_adjustment
        
        # Historical performance data
        self.channel_performance_history: Dict[str, List[Dict]] = defaultdict(list)
        self.market_conditions: Dict[str, Any] = {}
        self.seasonality_patterns: Dict[str, Dict] = {}
        self.competitive_landscape: Dict[str, Dict] = {}
        
        # ML models for performance prediction
        self.performance_models: Dict[str, Any] = {}
        self.saturation_models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        
        # Optimization results
        self.optimization_history: List[BudgetAllocation] = []
        self.channel_constraints: Dict[str, ChannelConstraints] = {}
        self.optimization_objectives: Optional[OptimizationObjective] = None
        
        # Risk and uncertainty modeling
        self.uncertainty_models: Dict[str, Any] = {}
        self.risk_factors: Dict[str, float] = {}
        
        logger.info("Budget Optimization Engine initialized")
    
    def add_performance_data(self, performance_data: List[Dict[str, Any]]) -> 'BudgetOptimizationEngine':
        """
        Add historical performance data for optimization.
        
        Args:
            performance_data: List of performance dictionaries with channel metrics
            
        Returns:
            Self for method chaining
        """
        for data_point in performance_data:
            channel = data_point.get('channel', '')
            if channel:
                performance_record = {
                    'date': data_point.get('date', datetime.now()),
                    'budget': data_point.get('budget', 0.0),
                    'spend': data_point.get('spend', 0.0),
                    'impressions': data_point.get('impressions', 0),
                    'clicks': data_point.get('clicks', 0),
                    'conversions': data_point.get('conversions', 0),
                    'revenue': data_point.get('revenue', 0.0),
                    'roas': data_point.get('roas', 0.0),
                    'cpc': data_point.get('cpc', 0.0),
                    'ctr': data_point.get('ctr', 0.0),
                    'cvr': data_point.get('cvr', 0.0),
                    'brand_metrics': data_point.get('brand_metrics', {}),
                    'competitive_metrics': data_point.get('competitive_metrics', {}),
                    'external_factors': data_point.get('external_factors', {})
                }
                
                self.channel_performance_history[channel].append(performance_record)
        
        logger.info(f"Added performance data for {len(set(d.get('channel') for d in performance_data))} channels")
        return self
    
    def set_channel_constraints(self, constraints: List[ChannelConstraints]) -> 'BudgetOptimizationEngine':
        """
        Set budget constraints for channels.
        
        Args:
            constraints: List of channel constraint objects
            
        Returns:
            Self for method chaining
        """
        for constraint in constraints:
            self.channel_constraints[constraint.channel] = constraint
        
        logger.info(f"Set constraints for {len(constraints)} channels")
        return self
    
    def set_optimization_objective(self, objective: OptimizationObjective) -> 'BudgetOptimizationEngine':
        """
        Set optimization objectives and weights.
        
        Args:
            objective: Optimization objective configuration
            
        Returns:
            Self for method chaining
        """
        self.optimization_objectives = objective
        logger.info(f"Set optimization objective: {objective.primary_metric}")
        return self
    
    def train_performance_models(self) -> 'BudgetOptimizationEngine':
        """
        Train machine learning models for performance prediction.
        
        Returns:
            Self for method chaining
        """
        if not self.enable_ml_predictions:
            return self
        
        for channel, history in self.channel_performance_history.items():
            if len(history) < 10:  # Need minimum data points
                continue
            
            # Prepare training data
            features, targets = self._prepare_ml_training_data(history)
            
            if len(features) < 5:
                continue
            
            # Train multiple models for ensemble
            models = {}
            
            try:
                # Random Forest for non-linear relationships
                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                rf_model.fit(features, targets['revenue'])
                models['random_forest'] = rf_model
                
                # Gradient Boosting for sequential patterns
                gb_model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
                gb_model.fit(features, targets['revenue'])
                models['gradient_boosting'] = gb_model
                
                # Elastic Net for regularized linear relationships
                elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
                elastic_features = self.scaler.fit_transform(features)
                elastic_model.fit(elastic_features, targets['revenue'])
                models['elastic_net'] = elastic_model
                
                # Store models and evaluate performance
                model_scores = {}
                tscv = TimeSeriesSplit(n_splits=3)
                
                for model_name, model in models.items():
                    if model_name == 'elastic_net':
                        scores = cross_val_score(model, elastic_features, targets['revenue'], 
                                               cv=tscv, scoring='neg_mean_absolute_error')
                    else:
                        scores = cross_val_score(model, features, targets['revenue'], 
                                               cv=tscv, scoring='neg_mean_absolute_error')
                    model_scores[model_name] = -np.mean(scores)
                
                # Select best performing model
                best_model_name = min(model_scores.items(), key=lambda x: x[1])[0]
                self.performance_models[channel] = {
                    'model': models[best_model_name],
                    'model_type': best_model_name,
                    'score': model_scores[best_model_name],
                    'features_scaler': self.scaler if best_model_name == 'elastic_net' else None
                }
                
                # Train saturation model
                saturation_model = self._train_saturation_model(history)
                if saturation_model:
                    self.saturation_models[channel] = saturation_model
                
                logger.info(f"Trained {best_model_name} model for {channel} (MAE: {model_scores[best_model_name]:.2f})")
                
            except Exception as e:
                logger.warning(f"Model training failed for {channel}: {e}")
        
        return self
    
    def analyze_seasonality(self) -> 'BudgetOptimizationEngine':
        """
        Analyze seasonal patterns in channel performance.
        
        Returns:
            Self for method chaining
        """
        if not self.seasonality_adjustment:
            return self
        
        for channel, history in self.channel_performance_history.items():
            if len(history) < 30:  # Need sufficient data for seasonality
                continue
            
            # Extract seasonal patterns
            seasonality_analysis = self._analyze_channel_seasonality(history)
            self.seasonality_patterns[channel] = seasonality_analysis
        
        logger.info(f"Analyzed seasonality for {len(self.seasonality_patterns)} channels")
        return self
    
    def optimize_budget_allocation(self,
                                 total_budget: float,
                                 time_horizon: int = 30,
                                 scenario_count: int = 5) -> BudgetAllocation:
        """
        Optimize budget allocation across channels.
        
        Args:
            total_budget: Total budget to allocate
            time_horizon: Optimization time horizon in days
            scenario_count: Number of scenarios for uncertainty analysis
            
        Returns:
            Optimized budget allocation
        """
        if not self.optimization_objectives:
            raise ValueError("Optimization objectives must be set before optimization")
        
        # Get available channels
        channels = list(self.channel_performance_history.keys())
        if not channels:
            raise ValueError("No channel performance data available")
        
        # Set up optimization problem
        n_channels = len(channels)
        
        # Initial allocation (equal distribution with constraints)
        initial_allocation = self._get_initial_allocation(channels, total_budget)
        
        # Define bounds based on constraints
        bounds = self._get_optimization_bounds(channels, total_budget)
        
        # Define constraints
        constraints = self._get_optimization_constraints(channels, total_budget)
        
        # Run optimization
        if self.optimization_method == 'scipy':
            result = self._scipy_optimization(initial_allocation, bounds, constraints, channels, total_budget)
        elif self.optimization_method == 'evolutionary':
            result = self._evolutionary_optimization(bounds, constraints, channels, total_budget)
        elif self.optimization_method == 'ml_assisted':
            result = self._ml_assisted_optimization(initial_allocation, bounds, constraints, channels, total_budget)
        else:  # hybrid
            result = self._hybrid_optimization(initial_allocation, bounds, constraints, channels, total_budget)
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
            # Fall back to constraint-based allocation
            optimal_allocation = self._constraint_based_allocation(channels, total_budget)
        else:
            optimal_allocation = result.x
        
        # Create allocation dictionary
        channel_budgets = {channel: max(0, budget) for channel, budget in zip(channels, optimal_allocation)}
        
        # Ensure budget constraint
        actual_total = sum(channel_budgets.values())
        if actual_total > 0:
            scale_factor = total_budget / actual_total
            channel_budgets = {ch: budget * scale_factor for ch, budget in channel_budgets.items()}
        
        # Calculate expected performance
        expected_metrics = self._calculate_expected_performance(channel_budgets, time_horizon)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(channel_budgets, time_horizon)
        
        # Performance scenario analysis
        scenario_analysis = self._run_scenario_analysis(channel_budgets, scenario_count, time_horizon)
        
        # Calculate optimization scores
        optimization_score = self._calculate_optimization_score(channel_budgets, expected_metrics)
        risk_score = self._calculate_risk_score(channel_budgets, scenario_analysis)
        diversification_score = self._calculate_diversification_score(channel_budgets)
        
        # Generate recommendations
        recommendations = self._generate_allocation_recommendations(channel_budgets, expected_metrics, scenario_analysis)
        
        # Sensitivity analysis
        sensitivity_analysis = self._run_sensitivity_analysis(channel_budgets, time_horizon)
        
        # Allocation rationale
        allocation_rationale = self._generate_allocation_rationale(channel_budgets, expected_metrics)
        
        # Performance bounds
        performance_bounds = self._calculate_performance_bounds(channel_budgets, scenario_analysis)
        
        # Create allocation result
        allocation = BudgetAllocation(
            allocation_id=f"allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            channel_budgets=channel_budgets,
            total_budget=sum(channel_budgets.values()),
            expected_metrics=expected_metrics,
            confidence_intervals=confidence_intervals,
            optimization_score=optimization_score,
            risk_score=risk_score,
            diversification_score=diversification_score,
            scenario_analysis=scenario_analysis,
            recommendations=recommendations,
            sensitivity_analysis=sensitivity_analysis,
            allocation_rationale=allocation_rationale,
            performance_bounds=performance_bounds
        )
        
        self.optimization_history.append(allocation)
        
        logger.info(f"Optimized budget allocation: {optimization_score:.3f} optimization score, {risk_score:.3f} risk score")
        return allocation
    
    def _prepare_ml_training_data(self, history: List[Dict]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare training data for ML models."""
        features = []
        targets = defaultdict(list)
        
        for record in history:
            # Feature vector
            feature_vec = [
                record['budget'],
                record['spend'],
                record.get('cpc', 0),
                record.get('ctr', 0) * 100,  # Convert to percentage
                record.get('cvr', 0) * 100,  # Convert to percentage
                record['date'].weekday(),  # Day of week
                record['date'].month,      # Month
                int(record['date'].strftime('%U')),  # Week of year
                record.get('external_factors', {}).get('seasonality', 1.0),
                record.get('competitive_metrics', {}).get('competition_level', 1.0)
            ]
            
            features.append(feature_vec)
            
            # Targets
            targets['revenue'].append(record['revenue'])
            targets['conversions'].append(record['conversions'])
            targets['roas'].append(record['roas'])
        
        return np.array(features), {k: np.array(v) for k, v in targets.items()}
    
    def _train_saturation_model(self, history: List[Dict]) -> Optional[Dict]:
        """Train saturation curve model for channel."""
        try:
            # Extract budget and performance data
            budgets = np.array([r['budget'] for r in history])
            revenues = np.array([r['revenue'] for r in history])
            
            # Fit adstock saturation model: Revenue = a * Budget^b / (c + Budget^b)
            from scipy.optimize import curve_fit
            
            def saturation_curve(budget, a, b, c):
                return a * np.power(budget, b) / (c + np.power(budget, b))
            
            # Initial parameter guesses
            p0 = [max(revenues), 0.8, np.mean(budgets)]
            
            popt, pcov = curve_fit(
                saturation_curve, budgets, revenues,
                p0=p0, maxfev=2000,
                bounds=([0, 0.1, 0], [np.inf, 2.0, np.inf])
            )
            
            return {
                'function': saturation_curve,
                'parameters': popt,
                'covariance': pcov,
                'r_squared': self._calculate_r_squared(revenues, saturation_curve(budgets, *popt))
            }
            
        except Exception as e:
            logger.warning(f"Saturation model training failed: {e}")
            return None
    
    def _analyze_channel_seasonality(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze seasonal patterns in channel performance."""
        # Group by month and calculate seasonal factors
        monthly_performance = defaultdict(list)
        
        for record in history:
            month = record['date'].month
            monthly_performance[month].append({
                'roas': record['roas'],
                'revenue': record['revenue'],
                'conversions': record['conversions']
            })
        
        # Calculate seasonal factors
        seasonal_factors = {}
        overall_avg_roas = np.mean([r['roas'] for r in history])
        
        for month, records in monthly_performance.items():
            if records:
                avg_roas = np.mean([r['roas'] for r in records])
                seasonal_factors[month] = avg_roas / overall_avg_roas if overall_avg_roas > 0 else 1.0
        
        # Fill missing months with average
        for month in range(1, 13):
            if month not in seasonal_factors:
                seasonal_factors[month] = 1.0
        
        return {
            'seasonal_factors': seasonal_factors,
            'peak_months': sorted(seasonal_factors.keys(), key=lambda x: seasonal_factors[x], reverse=True)[:3],
            'low_months': sorted(seasonal_factors.keys(), key=lambda x: seasonal_factors[x])[:3]
        }
    
    def _get_initial_allocation(self, channels: List[str], total_budget: float) -> np.ndarray:
        """Get initial budget allocation."""
        n_channels = len(channels)
        allocation = np.full(n_channels, total_budget / n_channels)
        
        # Adjust for constraints
        for i, channel in enumerate(channels):
            if channel in self.channel_constraints:
                constraint = self.channel_constraints[channel]
                
                if constraint.required_budget:
                    allocation[i] = constraint.required_budget
                else:
                    min_budget = max(constraint.min_budget, total_budget * constraint.min_percentage)
                    max_budget = min(constraint.max_budget, total_budget * constraint.max_percentage)
                    allocation[i] = np.clip(allocation[i], min_budget, max_budget)
        
        # Normalize to total budget
        if np.sum(allocation) > 0:
            allocation = allocation * (total_budget / np.sum(allocation))
        
        return allocation
    
    def _get_optimization_bounds(self, channels: List[str], total_budget: float) -> List[Tuple[float, float]]:
        """Get optimization bounds for each channel."""
        bounds = []
        
        for channel in channels:
            if channel in self.channel_constraints:
                constraint = self.channel_constraints[channel]
                min_bound = max(constraint.min_budget, total_budget * constraint.min_percentage)
                max_bound = min(constraint.max_budget, total_budget * constraint.max_percentage)
            else:
                min_bound = 0.0
                max_bound = total_budget
            
            bounds.append((min_bound, max_bound))
        
        return bounds
    
    def _get_optimization_constraints(self, channels: List[str], total_budget: float) -> List[Dict]:
        """Get optimization constraints."""
        constraints = []
        
        # Budget constraint
        def budget_constraint(x):
            return total_budget - np.sum(x)
        
        constraints.append({
            'type': 'eq',
            'fun': budget_constraint
        })
        
        # Required budget constraints
        for i, channel in enumerate(channels):
            if channel in self.channel_constraints:
                constraint = self.channel_constraints[channel]
                if constraint.required_budget:
                    def required_constraint(x, idx=i, req_budget=constraint.required_budget):
                        return x[idx] - req_budget
                    
                    constraints.append({
                        'type': 'eq',
                        'fun': required_constraint
                    })
        
        return constraints
    
    def _objective_function(self, allocation: np.ndarray, channels: List[str], time_horizon: int) -> float:
        """Objective function to maximize."""
        channel_budgets = {channel: budget for channel, budget in zip(channels, allocation)}
        
        # Calculate expected performance
        expected_metrics = self._calculate_expected_performance(channel_budgets, time_horizon)
        
        # Primary objective
        primary_value = expected_metrics.get(self.optimization_objectives.primary_metric, 0.0)
        
        # Secondary objectives
        secondary_value = 0.0
        for metric in self.optimization_objectives.secondary_metrics:
            weight = self.optimization_objectives.weights.get(metric, 0.1)
            secondary_value += weight * expected_metrics.get(metric, 0.0)
        
        # Diversification bonus
        diversification_bonus = 0.0
        if self.optimization_objectives.diversification_bonus > 0:
            # Calculate Herfindahl-Hirschman Index (lower is more diversified)
            total_budget = np.sum(allocation)
            if total_budget > 0:
                proportions = allocation / total_budget
                hhi = np.sum(proportions ** 2)
                # Convert to diversification score (higher is better)
                diversification_score = 1.0 - hhi
                diversification_bonus = self.optimization_objectives.diversification_bonus * diversification_score
        
        # Risk penalty
        risk_penalty = self._calculate_allocation_risk_penalty(allocation, channels)
        
        total_objective = primary_value + secondary_value + diversification_bonus - risk_penalty
        
        return -total_objective  # Negative because we're minimizing
    
    def _scipy_optimization(self, initial_allocation: np.ndarray, bounds: List[Tuple], 
                          constraints: List[Dict], channels: List[str], total_budget: float):
        """Run scipy-based optimization."""
        return minimize(
            fun=self._objective_function,
            x0=initial_allocation,
            args=(channels, self.optimization_objectives.time_horizon),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
    
    def _evolutionary_optimization(self, bounds: List[Tuple], constraints: List[Dict], 
                                 channels: List[str], total_budget: float):
        """Run evolutionary optimization."""
        # Convert constraints to penalty function
        def constrained_objective(allocation):
            # Budget constraint penalty
            budget_violation = abs(np.sum(allocation) - total_budget)
            penalty = 1000 * budget_violation
            
            # Required budget penalties
            for i, channel in enumerate(channels):
                if channel in self.channel_constraints:
                    constraint = self.channel_constraints[channel]
                    if constraint.required_budget:
                        penalty += 1000 * abs(allocation[i] - constraint.required_budget)
            
            objective = self._objective_function(allocation, channels, self.optimization_objectives.time_horizon)
            return objective + penalty
        
        return differential_evolution(
            func=constrained_objective,
            bounds=bounds,
            seed=42,
            maxiter=300,
            popsize=15,
            atol=1e-8,
            tol=1e-8
        )
    
    def _ml_assisted_optimization(self, initial_allocation: np.ndarray, bounds: List[Tuple],
                                constraints: List[Dict], channels: List[str], total_budget: float):
        """Run ML-assisted optimization."""
        # Use ML models to predict performance more accurately
        best_allocation = initial_allocation.copy()
        best_score = float('inf')
        
        # Grid search with ML predictions
        n_iterations = 50
        
        for _ in range(n_iterations):
            # Generate candidate allocation
            candidate = np.array([
                np.random.uniform(bound[0], bound[1]) 
                for bound in bounds
            ])
            
            # Normalize to budget constraint
            candidate = candidate * (total_budget / np.sum(candidate))
            
            # Evaluate using ML models
            score = self._objective_function(candidate, channels, self.optimization_objectives.time_horizon)
            
            if score < best_score:
                best_score = score
                best_allocation = candidate
        
        # Create mock result object
        class MockResult:
            def __init__(self, x, success=True, message="ML-assisted optimization completed"):
                self.x = x
                self.success = success
                self.message = message
        
        return MockResult(best_allocation)
    
    def _hybrid_optimization(self, initial_allocation: np.ndarray, bounds: List[Tuple],
                           constraints: List[Dict], channels: List[str], total_budget: float):
        """Run hybrid optimization combining multiple methods."""
        # Start with scipy
        scipy_result = self._scipy_optimization(initial_allocation, bounds, constraints, channels, total_budget)
        
        # Improve with evolutionary if scipy didn't converge well
        if not scipy_result.success or scipy_result.fun > -1000:
            evolutionary_result = self._evolutionary_optimization(bounds, constraints, channels, total_budget)
            
            if evolutionary_result.success:
                return evolutionary_result
        
        return scipy_result
    
    def _constraint_based_allocation(self, channels: List[str], total_budget: float) -> np.ndarray:
        """Fall-back constraint-based allocation."""
        allocation = np.zeros(len(channels))
        remaining_budget = total_budget
        
        # First, allocate required budgets
        for i, channel in enumerate(channels):
            if channel in self.channel_constraints:
                constraint = self.channel_constraints[channel]
                if constraint.required_budget:
                    allocation[i] = constraint.required_budget
                    remaining_budget -= constraint.required_budget
        
        # Distribute remaining budget based on priority and performance
        remaining_channels = [
            (i, channel) for i, channel in enumerate(channels)
            if channel not in self.channel_constraints or not self.channel_constraints[channel].required_budget
        ]
        
        if remaining_channels and remaining_budget > 0:
            # Simple equal allocation with priority weighting
            weights = []
            for i, channel in remaining_channels:
                if channel in self.channel_constraints:
                    priority = self.channel_constraints[channel].priority_level
                    weight = 1.0 / priority  # Higher priority = higher weight
                else:
                    weight = 1.0
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            for j, (i, channel) in enumerate(remaining_channels):
                allocation[i] = (weights[j] / total_weight) * remaining_budget
        
        return allocation
    
    def _calculate_expected_performance(self, channel_budgets: Dict[str, float], 
                                      time_horizon: int) -> Dict[str, float]:
        """Calculate expected performance metrics."""
        total_metrics = defaultdict(float)
        
        for channel, budget in channel_budgets.items():
            if budget <= 0:
                continue
            
            # Use ML models if available
            if channel in self.performance_models and self.enable_ml_predictions:
                channel_metrics = self._predict_ml_performance(channel, budget, time_horizon)
            else:
                # Fall back to historical averages
                channel_metrics = self._predict_historical_performance(channel, budget, time_horizon)
            
            # Apply seasonality adjustment
            if self.seasonality_adjustment and channel in self.seasonality_patterns:
                current_month = datetime.now().month
                seasonal_factor = self.seasonality_patterns[channel]['seasonal_factors'].get(current_month, 1.0)
                for metric in channel_metrics:
                    channel_metrics[metric] *= seasonal_factor
            
            # Aggregate metrics
            for metric, value in channel_metrics.items():
                total_metrics[metric] += value
        
        return dict(total_metrics)
    
    def _predict_ml_performance(self, channel: str, budget: float, time_horizon: int) -> Dict[str, float]:
        """Predict performance using ML models."""
        model_info = self.performance_models[channel]
        model = model_info['model']
        
        # Create feature vector for prediction
        current_date = datetime.now()
        features = np.array([[
            budget,
            budget * 0.95,  # Assume 95% spend rate
            0.5,  # Default CPC
            2.0,  # Default CTR
            1.5,  # Default CVR
            current_date.weekday(),
            current_date.month,
            int(current_date.strftime('%U')),
            self.seasonality_patterns.get(channel, {}).get('seasonal_factors', {}).get(current_date.month, 1.0),
            1.0   # Default competition level
        ]])
        
        # Scale features if needed
        if model_info['features_scaler']:
            features = model_info['features_scaler'].transform(features)
        
        # Predict revenue
        predicted_revenue = model.predict(features)[0]
        
        # Scale by time horizon
        predicted_revenue *= (time_horizon / 30)  # Assume model trained on monthly data
        
        # Calculate other metrics based on revenue
        predicted_roas = predicted_revenue / budget if budget > 0 else 0.0
        predicted_conversions = predicted_revenue / 100  # Assume $100 average order value
        
        return {
            'revenue': max(0, predicted_revenue),
            'roas': predicted_roas,
            'conversions': max(0, predicted_conversions),
            'profit': max(0, predicted_revenue - budget)
        }
    
    def _predict_historical_performance(self, channel: str, budget: float, time_horizon: int) -> Dict[str, float]:
        """Predict performance using historical averages."""
        if channel not in self.channel_performance_history:
            return {'revenue': 0.0, 'roas': 0.0, 'conversions': 0.0, 'profit': 0.0}
        
        history = self.channel_performance_history[channel]
        
        # Calculate average performance metrics
        avg_roas = np.mean([r['roas'] for r in history if r['budget'] > 0])
        avg_revenue_per_budget = np.mean([r['revenue'] / r['budget'] for r in history if r['budget'] > 0])
        avg_conversions_per_budget = np.mean([r['conversions'] / r['budget'] for r in history if r['budget'] > 0])
        
        # Apply saturation if model available
        if channel in self.saturation_models:
            saturation_model = self.saturation_models[channel]
            predicted_revenue = saturation_model['function'](budget, *saturation_model['parameters'])
            predicted_revenue *= (time_horizon / 30)
        else:
            predicted_revenue = budget * avg_revenue_per_budget * (time_horizon / 30)
        
        predicted_roas = predicted_revenue / budget if budget > 0 else 0.0
        predicted_conversions = budget * avg_conversions_per_budget * (time_horizon / 30)
        
        return {
            'revenue': max(0, predicted_revenue),
            'roas': predicted_roas,
            'conversions': max(0, predicted_conversions),
            'profit': max(0, predicted_revenue - budget)
        }
    
    def _calculate_confidence_intervals(self, channel_budgets: Dict[str, float], 
                                      time_horizon: int) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for performance predictions."""
        confidence_intervals = {}
        
        for metric in ['revenue', 'roas', 'conversions', 'profit']:
            # Monte Carlo simulation for uncertainty
            simulated_values = []
            
            for _ in range(1000):  # 1000 simulations
                simulated_metrics = defaultdict(float)
                
                for channel, budget in channel_budgets.items():
                    if budget <= 0:
                        continue
                    
                    # Add noise to predictions
                    if channel in self.performance_models:
                        # Use model uncertainty
                        base_prediction = self._predict_ml_performance(channel, budget, time_horizon)
                        noise_factor = np.random.normal(1.0, 0.15)  # 15% uncertainty
                    else:
                        base_prediction = self._predict_historical_performance(channel, budget, time_horizon)
                        noise_factor = np.random.normal(1.0, 0.25)  # 25% uncertainty for historical
                    
                    channel_value = base_prediction.get(metric, 0.0) * noise_factor
                    simulated_metrics[metric] += max(0, channel_value)
                
                simulated_values.append(simulated_metrics[metric])
            
            # Calculate 90% confidence interval
            lower_bound = np.percentile(simulated_values, 5)
            upper_bound = np.percentile(simulated_values, 95)
            confidence_intervals[metric] = (lower_bound, upper_bound)
        
        return confidence_intervals
    
    def _run_scenario_analysis(self, channel_budgets: Dict[str, float], 
                             scenario_count: int, time_horizon: int) -> Dict[str, Dict[str, float]]:
        """Run scenario analysis with different market conditions."""
        scenarios = {}
        
        # Base case
        scenarios['base_case'] = self._calculate_expected_performance(channel_budgets, time_horizon)
        
        # Optimistic scenario (+20% performance)
        optimistic_budgets = {ch: budget * 1.0 for ch, budget in channel_budgets.items()}  # Same budget
        optimistic_performance = self._calculate_expected_performance(optimistic_budgets, time_horizon)
        scenarios['optimistic'] = {k: v * 1.2 for k, v in optimistic_performance.items()}
        
        # Pessimistic scenario (-20% performance)
        pessimistic_budgets = {ch: budget * 1.0 for ch, budget in channel_budgets.items()}
        pessimistic_performance = self._calculate_expected_performance(pessimistic_budgets, time_horizon)
        scenarios['pessimistic'] = {k: v * 0.8 for k, v in pessimistic_performance.items()}
        
        # Economic downturn scenario (-35% performance)
        downturn_performance = self._calculate_expected_performance(channel_budgets, time_horizon)
        scenarios['economic_downturn'] = {k: v * 0.65 for k, v in downturn_performance.items()}
        
        # Competitive pressure scenario
        competitive_budgets = {ch: budget * 0.9 for ch, budget in channel_budgets.items()}  # 10% budget cut
        competitive_performance = self._calculate_expected_performance(competitive_budgets, time_horizon)
        scenarios['high_competition'] = competitive_performance
        
        return scenarios
    
    def _calculate_optimization_score(self, channel_budgets: Dict[str, float], 
                                    expected_metrics: Dict[str, float]) -> float:
        """Calculate overall optimization score."""
        primary_metric = self.optimization_objectives.primary_metric
        primary_value = expected_metrics.get(primary_metric, 0.0)
        
        # Normalize by budget
        total_budget = sum(channel_budgets.values())
        if total_budget > 0:
            efficiency = primary_value / total_budget
        else:
            efficiency = 0.0
        
        # Add diversification bonus
        diversification_score = self._calculate_diversification_score(channel_budgets)
        
        return efficiency + 0.1 * diversification_score
    
    def _calculate_risk_score(self, channel_budgets: Dict[str, float], 
                            scenario_analysis: Dict[str, Dict[str, float]]) -> float:
        """Calculate risk score based on scenario variance."""
        primary_metric = self.optimization_objectives.primary_metric
        
        scenario_values = [
            scenario.get(primary_metric, 0.0) 
            for scenario in scenario_analysis.values()
        ]
        
        if len(scenario_values) > 1:
            risk_score = np.std(scenario_values) / np.mean(scenario_values) if np.mean(scenario_values) > 0 else 1.0
        else:
            risk_score = 0.5  # Default moderate risk
        
        return min(risk_score, 1.0)  # Cap at 1.0
    
    def _calculate_diversification_score(self, channel_budgets: Dict[str, float]) -> float:
        """Calculate diversification score (higher is more diversified)."""
        total_budget = sum(channel_budgets.values())
        if total_budget <= 0:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index
        proportions = [budget / total_budget for budget in channel_budgets.values()]
        hhi = sum(p ** 2 for p in proportions)
        
        # Convert to diversification score (0 to 1, higher is better)
        diversification_score = 1.0 - hhi
        
        return diversification_score
    
    def _calculate_allocation_risk_penalty(self, allocation: np.ndarray, channels: List[str]) -> float:
        """Calculate risk penalty for allocation."""
        if self.risk_tolerance == 'aggressive':
            return 0.0
        
        penalty = 0.0
        total_budget = np.sum(allocation)
        
        if total_budget <= 0:
            return penalty
        
        # Concentration penalty
        max_allocation = np.max(allocation) / total_budget
        if max_allocation > 0.6:  # More than 60% in one channel
            concentration_penalty = (max_allocation - 0.6) * 100
            penalty += concentration_penalty
        
        # Untested channel penalty
        for i, channel in enumerate(channels):
            channel_share = allocation[i] / total_budget
            if channel not in self.channel_performance_history and channel_share > 0.2:
                penalty += channel_share * 50  # Penalty for large allocation to untested channel
        
        return penalty
    
    def _run_sensitivity_analysis(self, channel_budgets: Dict[str, float], 
                                time_horizon: int) -> Dict[str, Dict[str, float]]:
        """Run sensitivity analysis for budget changes."""
        sensitivity = {}
        base_performance = self._calculate_expected_performance(channel_budgets, time_horizon)
        base_value = base_performance.get(self.optimization_objectives.primary_metric, 0.0)
        
        for channel in channel_budgets.keys():
            channel_sensitivity = {}
            
            # Test +/- 10% budget changes
            for change in [-0.1, 0.1]:
                test_budgets = channel_budgets.copy()
                original_budget = test_budgets[channel]
                test_budgets[channel] = original_budget * (1 + change)
                
                # Adjust other channels proportionally to maintain total budget
                total_original = sum(channel_budgets.values())
                total_test = sum(test_budgets.values())
                if total_test != total_original:
                    adjustment_factor = total_original / total_test
                    test_budgets = {ch: budget * adjustment_factor for ch, budget in test_budgets.items()}
                
                test_performance = self._calculate_expected_performance(test_budgets, time_horizon)
                test_value = test_performance.get(self.optimization_objectives.primary_metric, 0.0)
                
                if base_value != 0:
                    sensitivity_ratio = (test_value - base_value) / base_value
                else:
                    sensitivity_ratio = 0.0
                
                change_label = f"{change:+.0%}"
                channel_sensitivity[change_label] = sensitivity_ratio
            
            sensitivity[channel] = channel_sensitivity
        
        return sensitivity
    
    def _generate_allocation_recommendations(self, channel_budgets: Dict[str, float],
                                           expected_metrics: Dict[str, float],
                                           scenario_analysis: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate allocation recommendations."""
        recommendations = []
        
        total_budget = sum(channel_budgets.values())
        
        # Budget concentration check
        max_channel = max(channel_budgets.items(), key=lambda x: x[1])
        max_share = max_channel[1] / total_budget if total_budget > 0 else 0
        
        if max_share > 0.7:
            recommendations.append(f"Consider reducing concentration in {max_channel[0]} ({max_share:.1%} of budget)")
        
        # Low budget channels
        low_budget_channels = [ch for ch, budget in channel_budgets.items() if budget / total_budget < 0.05]
        if low_budget_channels:
            recommendations.append(f"Consider consolidating or eliminating low-budget channels: {', '.join(low_budget_channels)}")
        
        # Performance-based recommendations
        primary_metric = self.optimization_objectives.primary_metric
        primary_value = expected_metrics.get(primary_metric, 0.0)
        
        if primary_metric == 'roas' and primary_value < 2.0:
            recommendations.append("Overall ROAS below 2.0 - consider reallocating to higher-performing channels")
        
        # Risk-based recommendations
        worst_case = scenario_analysis.get('pessimistic', {}).get(primary_metric, 0.0)
        base_case = scenario_analysis.get('base_case', {}).get(primary_metric, 1.0)
        
        if worst_case < base_case * 0.6:
            recommendations.append("High downside risk detected - consider more conservative allocation")
        
        # Diversification recommendations
        diversification_score = self._calculate_diversification_score(channel_budgets)
        if diversification_score < 0.3:
            recommendations.append("Low diversification - consider spreading budget across more channels")
        
        return recommendations
    
    def _generate_allocation_rationale(self, channel_budgets: Dict[str, float],
                                     expected_metrics: Dict[str, float]) -> Dict[str, str]:
        """Generate rationale for each channel allocation."""
        rationale = {}
        
        total_budget = sum(channel_budgets.values())
        
        for channel, budget in channel_budgets.items():
            if budget <= 0:
                rationale[channel] = "No budget allocated due to poor expected performance"
                continue
            
            share = budget / total_budget if total_budget > 0 else 0
            
            # Base rationale on historical performance and constraints
            reasons = []
            
            if channel in self.channel_constraints:
                constraint = self.channel_constraints[channel]
                if constraint.required_budget:
                    reasons.append("fixed budget requirement")
                elif constraint.priority_level == 1:
                    reasons.append("high priority channel")
            
            if channel in self.performance_models:
                reasons.append("strong ML-predicted performance")
            
            if channel in self.channel_performance_history:
                history = self.channel_performance_history[channel]
                avg_roas = np.mean([r['roas'] for r in history if r['roas'] > 0])
                if avg_roas > 3.0:
                    reasons.append("historically high ROAS")
                elif avg_roas < 1.5:
                    reasons.append("limited allocation due to low historical ROAS")
            
            if share > 0.3:
                reasons.append("major budget allocation for high-impact channel")
            elif share < 0.05:
                reasons.append("minimal test allocation")
            
            rationale[channel] = f"{share:.1%} allocation: " + "; ".join(reasons)
        
        return rationale
    
    def _calculate_performance_bounds(self, channel_budgets: Dict[str, float],
                                    scenario_analysis: Dict[str, Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        """Calculate performance bounds across scenarios."""
        bounds = {}
        
        for metric in ['revenue', 'roas', 'conversions', 'profit']:
            scenario_values = [
                scenario.get(metric, 0.0)
                for scenario in scenario_analysis.values()
            ]
            
            if scenario_values:
                bounds[metric] = (min(scenario_values), max(scenario_values))
            else:
                bounds[metric] = (0.0, 0.0)
        
        return bounds
    
    def _calculate_r_squared(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate R-squared value."""
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def generate_optimization_report(self, allocation: BudgetAllocation) -> str:
        """Generate comprehensive optimization report."""
        report = "# Budget Optimization Analysis\n\n"
        report += "**Multi-Channel Budget Optimization by Sotiris Spyrou**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        # Executive Summary
        report += "## Executive Summary\n\n"
        report += f"- **Total Budget**: ${allocation.total_budget:,.2f}\n"
        report += f"- **Optimization Score**: {allocation.optimization_score:.3f}\n"
        report += f"- **Risk Score**: {allocation.risk_score:.3f} ({'Low' if allocation.risk_score < 0.3 else 'Medium' if allocation.risk_score < 0.7 else 'High'})\n"
        report += f"- **Diversification Score**: {allocation.diversification_score:.3f}\n\n"
        
        # Expected Performance
        report += "## Expected Performance\n\n"
        for metric, value in allocation.expected_metrics.items():
            confidence = allocation.confidence_intervals.get(metric, (0, 0))
            report += f"- **{metric.upper()}**: {value:,.2f} (90% CI: {confidence[0]:,.2f} - {confidence[1]:,.2f})\n"
        report += "\n"
        
        # Budget Allocation
        report += "## Budget Allocation\n\n"
        report += "| Channel | Budget | Share | Rationale |\n"
        report += "|---------|--------|-------|----------|\n"
        
        sorted_channels = sorted(allocation.channel_budgets.items(), key=lambda x: x[1], reverse=True)
        for channel, budget in sorted_channels:
            share = budget / allocation.total_budget if allocation.total_budget > 0 else 0
            rationale = allocation.allocation_rationale.get(channel, "Optimized allocation")
            report += f"| {channel} | ${budget:,.2f} | {share:.1%} | {rationale} |\n"
        
        report += "\n"
        
        # Scenario Analysis
        if allocation.scenario_analysis:
            report += "## Scenario Analysis\n\n"
            primary_metric = self.optimization_objectives.primary_metric if self.optimization_objectives else 'revenue'
            
            report += f"**{primary_metric.upper()} Projections:**\n\n"
            for scenario, metrics in allocation.scenario_analysis.items():
                value = metrics.get(primary_metric, 0)
                report += f"- **{scenario.replace('_', ' ').title()}**: {value:,.2f}\n"
            report += "\n"
        
        # Recommendations
        if allocation.recommendations:
            report += "## Strategic Recommendations\n\n"
            for i, recommendation in enumerate(allocation.recommendations, 1):
                report += f"{i}. {recommendation}\n"
            report += "\n"
        
        # Sensitivity Analysis
        if allocation.sensitivity_analysis:
            report += "## Sensitivity Analysis\n\n"
            report += "**Budget Change Impact on Primary Metric:**\n\n"
            
            for channel, sensitivity in allocation.sensitivity_analysis.items():
                report += f"### {channel}\n"
                for change, impact in sensitivity.items():
                    report += f"- {change} budget change: {impact:+.1%} performance impact\n"
                report += "\n"
        
        report += "---\n"
        report += "*Advanced budget optimization maximizes ROI through data-driven allocation. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for enterprise implementations.*"
        
        return report


def demo_budget_optimization():
    """Executive demonstration of Budget Optimization Engine."""
    
    print("=== Multi-Channel Budget Optimization Engine: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    # Initialize optimization engine
    optimizer = BudgetOptimizationEngine(
        optimization_method='hybrid',
        risk_tolerance='medium',
        enable_ml_predictions=True,
        seasonality_adjustment=True,
        competitive_adjustment=True
    )
    
    print("💰 Generating realistic marketing performance data...")
    
    # Generate realistic demo data
    np.random.seed(42)
    channels = ['Search', 'Display', 'Social', 'Email', 'Video', 'Affiliate']
    
    # Generate 6 months of performance data
    performance_data = []
    base_date = datetime.now() - timedelta(days=180)
    
    for channel in channels:
        # Channel-specific performance characteristics
        channel_params = {
            'Search': {'base_roas': 4.5, 'efficiency': 0.8, 'variance': 0.3},
            'Display': {'base_roas': 2.2, 'efficiency': 0.6, 'variance': 0.4},
            'Social': {'base_roas': 3.1, 'efficiency': 0.7, 'variance': 0.5},
            'Email': {'base_roas': 6.2, 'efficiency': 0.9, 'variance': 0.2},
            'Video': {'base_roas': 2.8, 'efficiency': 0.5, 'variance': 0.6},
            'Affiliate': {'base_roas': 3.8, 'efficiency': 0.7, 'variance': 0.4}
        }
        
        params = channel_params[channel]
        
        for days_ago in range(0, 180, 7):  # Weekly data points
            date = base_date + timedelta(days=days_ago)
            
            # Seasonal adjustment
            seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
            
            # Budget varies with some trend and noise
            base_budget = np.random.uniform(5000, 25000)
            budget = base_budget * seasonal_factor * np.random.uniform(0.7, 1.3)
            
            # Spend rate (95-100% of budget)
            spend = budget * np.random.uniform(0.95, 1.0)
            
            # Performance metrics with saturation effects
            budget_efficiency = params['efficiency'] * (1 - np.exp(-budget / 10000))
            roas = params['base_roas'] * budget_efficiency * seasonal_factor * np.random.uniform(1 - params['variance'], 1 + params['variance'])
            
            revenue = spend * max(0.1, roas)  # Ensure minimum positive revenue
            conversions = revenue / np.random.uniform(80, 150)  # Variable AOV
            
            # Other metrics
            impressions = int(spend * np.random.uniform(15, 35))
            clicks = int(impressions * np.random.uniform(0.01, 0.08))
            cpc = spend / clicks if clicks > 0 else 0
            ctr = clicks / impressions if impressions > 0 else 0
            cvr = conversions / clicks if clicks > 0 else 0
            
            performance_data.append({
                'channel': channel,
                'date': date,
                'budget': budget,
                'spend': spend,
                'impressions': impressions,
                'clicks': clicks,
                'conversions': conversions,
                'revenue': revenue,
                'roas': roas,
                'cpc': cpc,
                'ctr': ctr,
                'cvr': cvr,
                'brand_metrics': {'awareness_lift': np.random.uniform(0.01, 0.05)},
                'competitive_metrics': {'competition_level': np.random.uniform(0.8, 1.2)},
                'external_factors': {'seasonality': seasonal_factor}
            })
    
    print(f"📊 Generated {len(performance_data)} performance data points across {len(channels)} channels")
    
    # Add performance data to optimizer
    optimizer.add_performance_data(performance_data)
    
    # Set channel constraints
    constraints = [
        ChannelConstraints('Search', min_budget=15000, max_budget=50000, min_percentage=0.15, priority_level=1),
        ChannelConstraints('Display', min_budget=8000, max_budget=30000, min_percentage=0.10, priority_level=2),
        ChannelConstraints('Social', min_budget=10000, max_budget=35000, min_percentage=0.12, priority_level=1),
        ChannelConstraints('Email', min_budget=3000, max_budget=15000, min_percentage=0.05, priority_level=1),
        ChannelConstraints('Video', min_budget=5000, max_budget=25000, min_percentage=0.08, priority_level=3),
        ChannelConstraints('Affiliate', min_budget=2000, max_budget=20000, min_percentage=0.03, priority_level=2)
    ]
    
    optimizer.set_channel_constraints(constraints)
    
    # Set optimization objective
    objective = OptimizationObjective(
        primary_metric='revenue',
        secondary_metrics=['conversions', 'profit'],
        weights={'conversions': 0.2, 'profit': 0.3},
        time_horizon=30,
        risk_tolerance='medium',
        diversification_bonus=0.15,
        incrementality_weight=0.25
    )
    
    optimizer.set_optimization_objective(objective)
    
    print("\n🔄 Training performance models and analyzing patterns...")
    
    # Train ML models and analyze patterns
    optimizer.train_performance_models()
    optimizer.analyze_seasonality()
    
    print("\n🎯 Optimizing budget allocation...")
    
    # Run budget optimization
    total_budget = 120000  # $120K monthly budget
    allocation = optimizer.optimize_budget_allocation(
        total_budget=total_budget,
        time_horizon=30,
        scenario_count=5
    )
    
    print("\n💰 BUDGET OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"\n📈 OPTIMIZATION SUMMARY:")
    print(f"  • Total Budget: ${allocation.total_budget:,.2f}")
    print(f"  • Optimization Score: {allocation.optimization_score:.3f}")
    print(f"  • Risk Score: {allocation.risk_score:.3f} ({'Low' if allocation.risk_score < 0.3 else 'Medium' if allocation.risk_score < 0.7 else 'High'} risk)")
    print(f"  • Diversification Score: {allocation.diversification_score:.3f}")
    
    # Expected performance
    print(f"\n🎯 EXPECTED PERFORMANCE:")
    for metric, value in allocation.expected_metrics.items():
        confidence = allocation.confidence_intervals.get(metric, (0, 0))
        print(f"  • {metric.upper()}: {value:,.2f} (CI: {confidence[0]:,.0f} - {confidence[1]:,.0f})")
    
    # Budget allocation
    print(f"\n💸 BUDGET ALLOCATION:")
    sorted_channels = sorted(allocation.channel_budgets.items(), key=lambda x: x[1], reverse=True)
    for channel, budget in sorted_channels:
        share = budget / allocation.total_budget if allocation.total_budget > 0 else 0
        print(f"  • {channel}: ${budget:,.2f} ({share:.1%})")
    
    # Scenario analysis
    if allocation.scenario_analysis:
        print(f"\n📊 SCENARIO ANALYSIS (Revenue):")
        for scenario, metrics in allocation.scenario_analysis.items():
            revenue = metrics.get('revenue', 0)
            print(f"  • {scenario.replace('_', ' ').title()}: ${revenue:,.2f}")
    
    # Top recommendations
    if allocation.recommendations:
        print(f"\n💡 KEY RECOMMENDATIONS:")
        for i, recommendation in enumerate(allocation.recommendations[:3], 1):
            print(f"  {i}. {recommendation}")
    
    # Sensitivity analysis for top 3 channels
    print(f"\n📈 SENSITIVITY ANALYSIS (Top Channels):")
    top_channels = sorted_channels[:3]
    for channel, budget in top_channels:
        if channel in allocation.sensitivity_analysis:
            sensitivity = allocation.sensitivity_analysis[channel]
            print(f"  • {channel}:")
            for change, impact in sensitivity.items():
                print(f"    {change} budget → {impact:+.1%} performance")
    
    # Performance bounds
    print(f"\n🎯 PERFORMANCE BOUNDS:")
    primary_metric = 'revenue'
    bounds = allocation.performance_bounds.get(primary_metric, (0, 0))
    print(f"  • Revenue Range: ${bounds[0]:,.2f} - ${bounds[1]:,.2f}")
    print(f"  • Expected Upside: ${bounds[1] - allocation.expected_metrics.get(primary_metric, 0):,.2f}")
    print(f"  • Downside Protection: ${allocation.expected_metrics.get(primary_metric, 0) - bounds[0]:,.2f}")
    
    # Allocation rationale for top channels
    print(f"\n🎯 ALLOCATION RATIONALE:")
    for channel, budget in top_channels:
        if channel in allocation.allocation_rationale:
            print(f"  • {channel}: {allocation.allocation_rationale[channel]}")
    
    print("\n" + "="*70)
    print("🚀 Advanced mathematical optimization for maximum marketing ROI")
    print("💼 Multi-constraint budget allocation with risk management")
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_budget_optimization()