"""
Ensemble Attribution Model

Combines multiple attribution models to create a robust, consensus-based
attribution approach that leverages the strengths of different methodologies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
import logging

logger = logging.getLogger(__name__)


class EnsembleAttribution:
    """
    Ensemble Attribution Model combining multiple attribution approaches.
    
    Integrates rule-based models (first-touch, last-touch, linear) with
    algorithmic models (Markov, Shapley) and data-driven approaches to
    provide robust attribution results.
    """
    
    def __init__(self, 
                 models: Optional[Dict[str, Any]] = None,
                 weights: Optional[Dict[str, float]] = None,
                 meta_learner: str = 'linear',
                 validation_method: str = 'cross_validation'):
        """
        Initialize Ensemble Attribution model.
        
        Args:
            models: Dictionary of attribution models to ensemble
            weights: Manual weights for each model (if None, learns automatically)
            meta_learner: Meta-learning algorithm ('linear', 'rf', 'gbm')
            validation_method: Method for model validation
        """
        self.models = models or {}
        self.manual_weights = weights
        self.meta_learner = meta_learner
        self.validation_method = validation_method
        
        # Model performance tracking
        self.model_performances = {}
        self.learned_weights = {}
        self.ensemble_results = {}
        
        # Meta-learner
        self.meta_model = None
        self._initialize_meta_learner()
    
    def _initialize_meta_learner(self):
        """Initialize the meta-learning model."""
        if self.meta_learner == 'linear':
            self.meta_model = LinearRegression()
        elif self.meta_learner == 'rf':
            self.meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.meta_learner == 'gbm':
            self.meta_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown meta_learner: {self.meta_learner}")
    
    def add_model(self, name: str, model: Any, weight: Optional[float] = None):
        """
        Add an attribution model to the ensemble.
        
        Args:
            name: Model identifier
            model: Attribution model instance
            weight: Optional manual weight for this model
        """
        self.models[name] = model
        if weight is not None and self.manual_weights is None:
            self.manual_weights = {}
        if weight is not None:
            self.manual_weights[name] = weight
        
        logger.info(f"Added model '{name}' to ensemble")
    
    def fit(self, journey_data: pd.DataFrame, 
            validation_data: Optional[pd.DataFrame] = None) -> 'EnsembleAttribution':
        """
        Fit the ensemble attribution model.
        
        Args:
            journey_data: Customer journey data for training
            validation_data: Optional validation data for meta-learning
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting ensemble attribution model")
        
        if not self.models:
            self._initialize_default_models()
        
        # Fit individual models
        model_predictions = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"Fitting model: {name}")
                
                # Fit the model
                if hasattr(model, 'fit'):
                    model.fit(journey_data)
                
                # Get predictions/attributions
                if hasattr(model, 'get_attribution_results'):
                    results = model.get_attribution_results()
                    model_predictions[name] = self._extract_attribution_weights(results)
                elif hasattr(model, 'predict'):
                    # For custom prediction methods
                    pred = model.predict(journey_data)
                    model_predictions[name] = pred
                else:
                    # Rule-based models
                    pred = self._calculate_rule_based_attribution(model, journey_data, name)
                    model_predictions[name] = pred
                
                # Evaluate model performance
                if validation_data is not None:
                    perf = self._evaluate_model_performance(model, validation_data, name)
                    self.model_performances[name] = perf
                
            except Exception as e:
                logger.warning(f"Failed to fit model '{name}': {e}")
                continue
        
        # Learn ensemble weights
        if self.manual_weights is None:
            self.learned_weights = self._learn_ensemble_weights(
                model_predictions, journey_data, validation_data
            )
        else:
            self.learned_weights = self.manual_weights.copy()
        
        # Calculate ensemble results
        self.ensemble_results = self._calculate_ensemble_attribution(model_predictions)
        
        logger.info("Ensemble model fitting completed")
        return self
    
    def _initialize_default_models(self):
        """Initialize default attribution models if none provided."""
        logger.info("Initializing default attribution models")
        
        # Rule-based models
        self.models.update({
            'first_touch': self._first_touch_attribution,
            'last_touch': self._last_touch_attribution,
            'linear': self._linear_attribution,
            'position_based': self._position_based_attribution,
            'time_decay': self._time_decay_attribution
        })
    
    def _extract_attribution_weights(self, results: pd.DataFrame) -> Dict[str, float]:
        """Extract attribution weights from model results."""
        if 'attribution_weight' in results.columns:
            return dict(zip(results['channel'], results['attribution_weight']))
        elif 'shapley_value' in results.columns:
            # Normalize Shapley values
            total = results['shapley_value'].sum()
            if total > 0:
                return dict(zip(results['channel'], results['shapley_value'] / total))
        elif 'removal_effect' in results.columns:
            # Normalize removal effects
            total = results['removal_effect'].sum()
            if total > 0:
                return dict(zip(results['channel'], results['removal_effect'] / total))
        
        # Fallback: equal attribution
        return {ch: 1/len(results) for ch in results['channel']}
    
    def _calculate_rule_based_attribution(self, model_func: Callable, 
                                        data: pd.DataFrame, 
                                        model_name: str) -> Dict[str, float]:
        """Calculate attribution using rule-based models."""
        try:
            return model_func(data)
        except Exception as e:
            logger.warning(f"Rule-based model '{model_name}' failed: {e}")
            # Return equal attribution as fallback
            channels = data['touchpoint'].unique()
            return {ch: 1/len(channels) for ch in channels}
    
    def _first_touch_attribution(self, data: pd.DataFrame) -> Dict[str, float]:
        """First-touch attribution model."""
        first_touches = data.groupby('customer_id')['touchpoint'].first()
        channel_counts = first_touches.value_counts()
        total_customers = len(first_touches)
        
        return {ch: count/total_customers for ch, count in channel_counts.items()}
    
    def _last_touch_attribution(self, data: pd.DataFrame) -> Dict[str, float]:
        """Last-touch attribution model."""
        last_touches = data.groupby('customer_id')['touchpoint'].last()
        channel_counts = last_touches.value_counts()
        total_customers = len(last_touches)
        
        return {ch: count/total_customers for ch, count in channel_counts.items()}
    
    def _linear_attribution(self, data: pd.DataFrame) -> Dict[str, float]:
        """Linear attribution model."""
        customer_journeys = data.groupby('customer_id')['touchpoint'].apply(list)
        
        channel_credits = {}
        total_credits = 0
        
        for journey in customer_journeys:
            if len(journey) > 0:
                credit_per_touch = 1 / len(journey)
                for touchpoint in journey:
                    channel_credits[touchpoint] = channel_credits.get(touchpoint, 0) + credit_per_touch
                total_credits += 1
        
        if total_credits > 0:
            return {ch: credit/total_credits for ch, credit in channel_credits.items()}
        return {}
    
    def _position_based_attribution(self, data: pd.DataFrame) -> Dict[str, float]:
        """Position-based (U-shaped) attribution model."""
        customer_journeys = data.groupby('customer_id')['touchpoint'].apply(list)
        
        channel_credits = {}
        total_journeys = 0
        
        for journey in customer_journeys:
            if len(journey) == 1:
                # Single touch gets full credit
                channel_credits[journey[0]] = channel_credits.get(journey[0], 0) + 1.0
            elif len(journey) == 2:
                # Two touches split equally
                for touch in journey:
                    channel_credits[touch] = channel_credits.get(touch, 0) + 0.5
            else:
                # First and last get 40% each, middle touches split 20%
                channel_credits[journey[0]] = channel_credits.get(journey[0], 0) + 0.4
                channel_credits[journey[-1]] = channel_credits.get(journey[-1], 0) + 0.4
                
                middle_credit = 0.2 / (len(journey) - 2)
                for touch in journey[1:-1]:
                    channel_credits[touch] = channel_credits.get(touch, 0) + middle_credit
            
            total_journeys += 1
        
        if total_journeys > 0:
            return {ch: credit/total_journeys for ch, credit in channel_credits.items()}
        return {}
    
    def _time_decay_attribution(self, data: pd.DataFrame, decay_rate: float = 0.7) -> Dict[str, float]:
        """Time-decay attribution model."""
        data_sorted = data.sort_values(['customer_id', 'timestamp'])
        customer_journeys = data_sorted.groupby('customer_id').apply(
            lambda x: list(zip(x['touchpoint'], x['timestamp']))
        )
        
        channel_credits = {}
        total_credits = 0
        
        for journey in customer_journeys:
            if len(journey) == 0:
                continue
                
            # Calculate time-based weights
            timestamps = [t[1] for t in journey]
            max_time = max(timestamps)
            
            journey_credits = {}
            total_journey_credit = 0
            
            for touchpoint, timestamp in journey:
                # Time decay: more recent touches get higher weight
                days_ago = (max_time - timestamp).total_seconds() / (24 * 3600)
                weight = decay_rate ** days_ago
                
                journey_credits[touchpoint] = journey_credits.get(touchpoint, 0) + weight
                total_journey_credit += weight
            
            # Normalize journey credits
            if total_journey_credit > 0:
                for touchpoint, credit in journey_credits.items():
                    normalized_credit = credit / total_journey_credit
                    channel_credits[touchpoint] = channel_credits.get(touchpoint, 0) + normalized_credit
                
                total_credits += 1
        
        if total_credits > 0:
            return {ch: credit/total_credits for ch, credit in channel_credits.items()}
        return {}
    
    def _evaluate_model_performance(self, model: Any, validation_data: pd.DataFrame, 
                                  model_name: str) -> Dict[str, float]:
        """Evaluate individual model performance."""
        try:
            # This is a simplified evaluation - in practice, you'd want
            # to use business metrics like incremental conversions
            
            if hasattr(model, 'get_model_statistics'):
                stats = model.get_model_statistics()
                return {
                    'model_score': stats.get('total_removal_effect', 0.5),
                    'num_channels': stats.get('num_channels', 0),
                    'concentration': stats.get('channel_concentration', 0.5)
                }
            else:
                # Basic evaluation for rule-based models
                pred = self._calculate_rule_based_attribution(model, validation_data, model_name)
                
                # Evaluate diversity and completeness
                num_channels = len(pred)
                completeness = abs(1.0 - sum(pred.values()))  # How close to 1.0
                diversity = 1.0 - sum(w**2 for w in pred.values())  # Herfindahl index
                
                return {
                    'model_score': 1.0 - completeness,  # Higher is better
                    'num_channels': num_channels,
                    'concentration': 1.0 - diversity
                }
        
        except Exception as e:
            logger.warning(f"Failed to evaluate model '{model_name}': {e}")
            return {'model_score': 0.0, 'num_channels': 0, 'concentration': 1.0}
    
    def _learn_ensemble_weights(self, model_predictions: Dict[str, Dict[str, float]], 
                              journey_data: pd.DataFrame,
                              validation_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Learn optimal ensemble weights."""
        
        if validation_data is None:
            # Use performance-based weighting
            return self._performance_based_weights()
        
        try:
            # Use meta-learning approach
            return self._meta_learning_weights(model_predictions, validation_data)
        except Exception as e:
            logger.warning(f"Meta-learning failed: {e}, falling back to performance-based weights")
            return self._performance_based_weights()
    
    def _performance_based_weights(self) -> Dict[str, float]:
        """Calculate weights based on individual model performance."""
        if not self.model_performances:
            # Equal weights if no performance data
            num_models = len(self.models)
            return {name: 1.0/num_models for name in self.models.keys()}
        
        # Weight by model score
        total_score = sum(perf.get('model_score', 0) for perf in self.model_performances.values())
        
        if total_score == 0:
            num_models = len(self.models)
            return {name: 1.0/num_models for name in self.models.keys()}
        
        weights = {}
        for name, perf in self.model_performances.items():
            score = perf.get('model_score', 0)
            weights[name] = score / total_score
        
        return weights
    
    def _meta_learning_weights(self, model_predictions: Dict[str, Dict[str, float]], 
                             validation_data: pd.DataFrame) -> Dict[str, float]:
        """Learn ensemble weights using meta-learning."""
        
        # Create feature matrix from model predictions
        all_channels = set()
        for pred in model_predictions.values():
            all_channels.update(pred.keys())
        
        all_channels = sorted(list(all_channels))
        
        # Create feature matrix (models x channels)
        X = []
        model_names = []
        
        for model_name, pred in model_predictions.items():
            features = [pred.get(channel, 0) for channel in all_channels]
            X.append(features)
            model_names.append(model_name)
        
        X = np.array(X).T  # Transpose: channels x models
        
        if X.shape[0] == 0 or X.shape[1] == 0:
            # Fallback to equal weights
            num_models = len(self.models)
            return {name: 1.0/num_models for name in self.models.keys()}
        
        # Create target variable (simplified - use conversion rates by channel)
        y = []
        validation_performance = validation_data.groupby('touchpoint')['converted'].mean()
        
        for channel in all_channels:
            y.append(validation_performance.get(channel, 0.1))  # Default 10% conversion
        
        y = np.array(y)
        
        try:
            # Fit meta-learner
            self.meta_model.fit(X, y)
            
            # Get feature importance or coefficients as weights
            if hasattr(self.meta_model, 'feature_importances_'):
                importances = self.meta_model.feature_importances_
            elif hasattr(self.meta_model, 'coef_'):
                importances = np.abs(self.meta_model.coef_)
            else:
                # Fallback to equal weights
                importances = np.ones(len(model_names))
            
            # Normalize to sum to 1
            total_importance = np.sum(importances)
            if total_importance > 0:
                weights = importances / total_importance
            else:
                weights = np.ones(len(model_names)) / len(model_names)
            
            return dict(zip(model_names, weights))
        
        except Exception as e:
            logger.warning(f"Meta-learning failed: {e}")
            num_models = len(self.models)
            return {name: 1.0/num_models for name in self.models.keys()}
    
    def _calculate_ensemble_attribution(self, model_predictions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate final ensemble attribution."""
        
        # Get all channels
        all_channels = set()
        for pred in model_predictions.values():
            all_channels.update(pred.keys())
        
        ensemble_attribution = {}
        
        for channel in all_channels:
            weighted_sum = 0
            total_weight = 0
            
            for model_name, predictions in model_predictions.items():
                if model_name in self.learned_weights:
                    weight = self.learned_weights[model_name]
                    attribution = predictions.get(channel, 0)
                    
                    weighted_sum += weight * attribution
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_attribution[channel] = weighted_sum / total_weight
            else:
                ensemble_attribution[channel] = 0
        
        return ensemble_attribution
    
    def get_attribution_results(self) -> pd.DataFrame:
        """
        Get ensemble attribution results as DataFrame.
        
        Returns:
            DataFrame with ensemble attribution results
        """
        if not self.ensemble_results:
            raise ValueError("Model not fitted. Call fit() first.")
        
        results = []
        for channel, weight in self.ensemble_results.items():
            results.append({
                'channel': channel,
                'ensemble_attribution': weight,
                'attribution_weight': weight  # Compatibility
            })
        
        df = pd.DataFrame(results)
        return df.sort_values('ensemble_attribution', ascending=False)
    
    def get_model_contributions(self) -> pd.DataFrame:
        """Get individual model contributions to ensemble."""
        if not self.models:
            return pd.DataFrame()
        
        contributions = []
        
        for model_name in self.models.keys():
            weight = self.learned_weights.get(model_name, 0)
            performance = self.model_performances.get(model_name, {})
            
            contributions.append({
                'model': model_name,
                'ensemble_weight': weight,
                'model_score': performance.get('model_score', 0),
                'num_channels': performance.get('num_channels', 0),
                'concentration': performance.get('concentration', 0)
            })
        
        return pd.DataFrame(contributions).sort_values('ensemble_weight', ascending=False)
    
    def predict(self, journey: List[str]) -> Dict[str, float]:
        """
        Predict attribution for a single journey.
        
        Args:
            journey: List of touchpoints
            
        Returns:
            Attribution weights for journey channels
        """
        if not self.ensemble_results:
            raise ValueError("Model not fitted. Call fit() first.")
        
        journey_channels = set(journey)
        
        # Get ensemble attributions for journey channels
        journey_attribution = {}
        total_attribution = 0
        
        for channel in journey_channels:
            attribution = self.ensemble_results.get(channel, 0)
            journey_attribution[channel] = attribution
            total_attribution += attribution
        
        # Normalize to sum to 1
        if total_attribution > 0:
            journey_attribution = {
                ch: attr / total_attribution 
                for ch, attr in journey_attribution.items()
            }
        else:
            # Equal attribution if no learned weights
            equal_weight = 1 / len(journey_channels) if journey_channels else 0
            journey_attribution = {ch: equal_weight for ch in journey_channels}
        
        return journey_attribution
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get ensemble model statistics."""
        return {
            'num_models': len(self.models),
            'model_weights': self.learned_weights.copy(),
            'ensemble_channels': len(self.ensemble_results),
            'total_attribution': sum(self.ensemble_results.values()),
            'attribution_concentration': sum(w**2 for w in self.ensemble_results.values()),
            'model_performances': self.model_performances.copy()
        }


def demo_ensemble_attribution():
    """Demonstration of Ensemble Attribution."""
    
    np.random.seed(42)
    
    # Sample journey data
    sample_data = pd.DataFrame({
        'customer_id': np.repeat(range(1, 501), 4),
        'touchpoint': np.random.choice(['Search', 'Display', 'Email', 'Social', 'Direct'], 2000),
        'timestamp': pd.date_range('2024-01-01', periods=2000, freq='H'),
        'converted': np.random.choice([True, False], 2000, p=[0.2, 0.8])
    })
    
    # Initialize ensemble
    ensemble = EnsembleAttribution(meta_learner='linear')
    
    # Fit ensemble
    ensemble.fit(sample_data)
    
    # Get results
    results = ensemble.get_attribution_results()
    print("Ensemble Attribution Results:")
    print(results)
    
    # Get model contributions
    contributions = ensemble.get_model_contributions()
    print("\nModel Contributions:")
    print(contributions)
    
    # Get ensemble statistics
    stats = ensemble.get_ensemble_statistics()
    print("\nEnsemble Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        else:
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    demo_ensemble_attribution()