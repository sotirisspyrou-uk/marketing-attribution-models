"""
Advanced Channel Saturation Models

Sophisticated modeling framework for identifying channel saturation points,
diminishing returns, and optimal spend allocation using statistical and
machine learning approaches.

Author: Sotiris Spyrou
Portfolio: https://verityai.co
LinkedIn: https://www.linkedin.com/in/sspyrou/

DISCLAIMER: This is demonstration code for portfolio purposes only.
Not intended for production use without proper testing and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.optimize import curve_fit, minimize, differential_evolution
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class SaturationPoint:
    """Channel saturation analysis result."""
    channel: str
    saturation_spend: float
    saturation_response: float
    efficiency_at_saturation: float
    marginal_roi: float
    confidence_score: float
    recommended_spend: float
    current_efficiency: float
    potential_gain: float


@dataclass
class SaturationModel:
    """Fitted saturation model with parameters."""
    model_id: str
    channel: str
    model_type: str
    function: Callable
    parameters: List[float]
    r_squared: float
    rmse: float
    confidence_intervals: List[Tuple[float, float]]
    spend_range: Tuple[float, float]
    response_range: Tuple[float, float]
    optimal_spend: float
    saturation_point: SaturationPoint
    model_equation: str
    goodness_of_fit: Dict[str, float]


class ChannelSaturationModels:
    """
    Advanced channel saturation modeling framework.
    
    Identifies saturation points, diminishing returns, and optimal spend
    allocation using multiple statistical and ML approaches including
    Adstock, S-curves, and custom response functions.
    """
    
    def __init__(self,
                 enable_adstock: bool = True,
                 adstock_decay_rate: float = 0.5,
                 saturation_threshold: float = 0.95,
                 min_data_points: int = 20):
        """
        Initialize Channel Saturation Models.
        
        Args:
            enable_adstock: Enable adstock (carryover) effects modeling
            adstock_decay_rate: Default adstock decay rate
            saturation_threshold: Threshold for saturation detection (95% of asymptote)
            min_data_points: Minimum data points required for modeling
        """
        self.enable_adstock = enable_adstock
        self.adstock_decay_rate = adstock_decay_rate
        self.saturation_threshold = saturation_threshold
        self.min_data_points = min_data_points
        
        # Data storage
        self.channel_data: Dict[str, pd.DataFrame] = {}
        self.processed_data: Dict[str, pd.DataFrame] = {}
        
        # Fitted models
        self.saturation_models: Dict[str, SaturationModel] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        # Model functions
        self.model_functions = {
            'hill': self._hill_saturation,
            'exponential': self._exponential_saturation,
            'logistic': self._logistic_saturation,
            'power': self._power_saturation,
            'adstock_hill': self._adstock_hill_saturation,
            'diminishing_returns': self._diminishing_returns,
            'michaelis_menten': self._michaelis_menten
        }
        
        # Scalers for normalization
        self.spend_scalers: Dict[str, MinMaxScaler] = {}
        self.response_scalers: Dict[str, MinMaxScaler] = {}
        
        logger.info("Channel Saturation Models initialized")
    
    def add_channel_data(self, channel: str, data: List[Dict[str, Any]]) -> 'ChannelSaturationModels':
        """
        Add channel performance data for saturation analysis.
        
        Args:
            channel: Channel name
            data: List of performance data points with spend and response metrics
            
        Returns:
            Self for method chaining
        """
        df = pd.DataFrame(data)
        
        # Ensure required columns
        required_columns = ['spend', 'response']
        for col in required_columns:
            if col not in df.columns:
                # Try to infer from common column names
                if col == 'response':
                    if 'revenue' in df.columns:
                        df['response'] = df['revenue']
                    elif 'conversions' in df.columns:
                        df['response'] = df['conversions']
                    elif 'clicks' in df.columns:
                        df['response'] = df['clicks']
                    else:
                        raise ValueError(f"Could not find response metric for {channel}")
        
        # Sort by spend for proper curve fitting
        df = df.sort_values('spend').reset_index(drop=True)
        
        # Store channel data
        self.channel_data[channel] = df
        
        logger.info(f"Added {len(df)} data points for channel: {channel}")
        return self
    
    def prepare_data(self, channel: str) -> 'ChannelSaturationModels':
        """
        Prepare and clean data for saturation modeling.
        
        Args:
            channel: Channel name
            
        Returns:
            Self for method chaining
        """
        if channel not in self.channel_data:
            raise ValueError(f"No data found for channel: {channel}")
        
        df = self.channel_data[channel].copy()
        
        # Remove zero spend and response data points
        df = df[(df['spend'] > 0) & (df['response'] > 0)]
        
        # Apply adstock transformation if enabled
        if self.enable_adstock:
            df['adstock_spend'] = self._apply_adstock(df['spend'], self.adstock_decay_rate)
        else:
            df['adstock_spend'] = df['spend']
        
        # Calculate efficiency metrics
        df['efficiency'] = df['response'] / df['spend']
        df['marginal_efficiency'] = df['response'].diff() / df['spend'].diff()
        
        # Smooth outliers (optional)
        df = self._smooth_outliers(df)
        
        # Normalize data for better convergence
        spend_scaler = MinMaxScaler()
        response_scaler = MinMaxScaler()
        
        df['spend_normalized'] = spend_scaler.fit_transform(df[['spend']])
        df['response_normalized'] = response_scaler.fit_transform(df[['response']])
        df['adstock_spend_normalized'] = spend_scaler.transform(df[['adstock_spend']])
        
        # Store scalers
        self.spend_scalers[channel] = spend_scaler
        self.response_scalers[channel] = response_scaler
        
        # Store processed data
        self.processed_data[channel] = df
        
        logger.info(f"Prepared data for {channel}: {len(df)} clean data points")
        return self
    
    def fit_saturation_models(self, channel: str, models: Optional[List[str]] = None) -> 'ChannelSaturationModels':
        """
        Fit multiple saturation models to channel data.
        
        Args:
            channel: Channel name
            models: List of model types to fit (default: all available)
            
        Returns:
            Self for method chaining
        """
        if channel not in self.processed_data:
            raise ValueError(f"No processed data for {channel}. Run prepare_data() first.")
        
        df = self.processed_data[channel]
        
        if len(df) < self.min_data_points:
            logger.warning(f"Insufficient data for {channel}: {len(df)} points")
            return self
        
        models = models or list(self.model_functions.keys())
        
        # Prepare data for fitting
        spend = df['spend'].values
        response = df['response'].values
        adstock_spend = df['adstock_spend'].values
        
        model_results = {}
        
        for model_type in models:
            try:
                # Fit the model
                fitted_model = self._fit_model(
                    model_type, spend, response, adstock_spend, channel
                )
                
                if fitted_model:
                    model_results[model_type] = fitted_model
                    logger.info(f"Successfully fitted {model_type} model for {channel}")
                
            except Exception as e:
                logger.warning(f"Failed to fit {model_type} model for {channel}: {e}")
        
        # Select best model based on performance
        if model_results:
            best_model = self._select_best_model(model_results, spend, response)
            self.saturation_models[channel] = best_model
            
            # Calculate model performance metrics
            self.model_performance[channel] = {
                'best_model': best_model.model_type,
                'r_squared': best_model.r_squared,
                'rmse': best_model.rmse,
                'models_fitted': len(model_results)
            }
            
            logger.info(f"Best model for {channel}: {best_model.model_type} (R² = {best_model.r_squared:.3f})")
        
        return self
    
    def analyze_saturation_point(self, channel: str) -> Optional[SaturationPoint]:
        """
        Analyze saturation point for a channel.
        
        Args:
            channel: Channel name
            
        Returns:
            SaturationPoint analysis or None if no model available
        """
        if channel not in self.saturation_models:
            logger.warning(f"No saturation model for {channel}")
            return None
        
        model = self.saturation_models[channel]
        
        # Find saturation point (95% of asymptote for most models)
        saturation_analysis = self._calculate_saturation_metrics(model)
        
        return saturation_analysis
    
    def predict_response(self, channel: str, spend_values: List[float]) -> List[float]:
        """
        Predict response for given spend values.
        
        Args:
            channel: Channel name
            spend_values: List of spend values to predict for
            
        Returns:
            List of predicted response values
        """
        if channel not in self.saturation_models:
            raise ValueError(f"No saturation model for {channel}")
        
        model = self.saturation_models[channel]
        predictions = []
        
        for spend in spend_values:
            if self.enable_adstock:
                # Apply adstock transformation
                adstock_spend = spend * (1 + self.adstock_decay_rate)
                pred = model.function(adstock_spend, *model.parameters)
            else:
                pred = model.function(spend, *model.parameters)
            
            predictions.append(max(0, pred))  # Ensure non-negative
        
        return predictions
    
    def optimize_spend_allocation(self, 
                                 channels: List[str],
                                 total_budget: float,
                                 constraints: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, float]:
        """
        Optimize spend allocation across channels considering saturation.
        
        Args:
            channels: List of channels to optimize
            total_budget: Total budget to allocate
            constraints: Optional min/max constraints per channel
            
        Returns:
            Optimal spend allocation dictionary
        """
        available_models = [ch for ch in channels if ch in self.saturation_models]
        
        if not available_models:
            # Fall back to equal allocation
            return {ch: total_budget / len(channels) for ch in channels}
        
        # Set up optimization problem
        def objective_function(allocation):
            total_response = 0
            for i, channel in enumerate(available_models):
                spend = allocation[i]
                if spend > 0:
                    predicted_response = self.predict_response(channel, [spend])[0]
                    total_response += predicted_response
            return -total_response  # Minimize negative (maximize positive)
        
        # Constraints
        constraints_list = []
        
        # Budget constraint
        def budget_constraint(allocation):
            return total_budget - sum(allocation)
        
        constraints_list.append({'type': 'eq', 'fun': budget_constraint})
        
        # Bounds
        bounds = []
        for channel in available_models:
            if constraints and channel in constraints:
                bounds.append(constraints[channel])
            else:
                bounds.append((0, total_budget))
        
        # Initial guess (equal allocation)
        initial_allocation = [total_budget / len(available_models)] * len(available_models)
        
        # Optimize
        try:
            result = minimize(
                objective_function,
                initial_allocation,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 1000}
            )
            
            if result.success:
                allocation = {ch: result.x[i] for i, ch in enumerate(available_models)}
                # Add zero allocation for channels without models
                for ch in channels:
                    if ch not in allocation:
                        allocation[ch] = 0
                return allocation
            
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
        
        # Fall back to proportional allocation based on efficiency
        return self._efficiency_based_allocation(available_models, total_budget)
    
    def _apply_adstock(self, spend_series: pd.Series, decay_rate: float) -> pd.Series:
        """Apply adstock transformation to spend data."""
        adstock_spend = np.zeros(len(spend_series))
        adstock_spend[0] = spend_series.iloc[0]
        
        for i in range(1, len(spend_series)):
            adstock_spend[i] = spend_series.iloc[i] + decay_rate * adstock_spend[i-1]
        
        return pd.Series(adstock_spend, index=spend_series.index)
    
    def _smooth_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove or smooth outliers in the data."""
        # Use IQR method for outlier detection
        Q1 = df['efficiency'].quantile(0.25)
        Q3 = df['efficiency'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing them
        df['efficiency'] = df['efficiency'].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _hill_saturation(self, spend: float, max_response: float, half_saturation: float, shape: float) -> float:
        """Hill saturation function (Michaelis-Menten with shape parameter)."""
        return max_response * (spend ** shape) / (half_saturation ** shape + spend ** shape)
    
    def _exponential_saturation(self, spend: float, max_response: float, decay_rate: float) -> float:
        """Exponential saturation function."""
        return max_response * (1 - np.exp(-decay_rate * spend))
    
    def _logistic_saturation(self, spend: float, max_response: float, midpoint: float, growth_rate: float) -> float:
        """Logistic (S-curve) saturation function."""
        return max_response / (1 + np.exp(-growth_rate * (spend - midpoint)))
    
    def _power_saturation(self, spend: float, coefficient: float, exponent: float) -> float:
        """Power law saturation function."""
        return coefficient * (spend ** exponent)
    
    def _adstock_hill_saturation(self, spend: float, max_response: float, half_saturation: float, 
                                shape: float, adstock_rate: float) -> float:
        """Hill saturation with built-in adstock."""
        effective_spend = spend * (1 + adstock_rate)
        return self._hill_saturation(effective_spend, max_response, half_saturation, shape)
    
    def _diminishing_returns(self, spend: float, base_response: float, diminishing_factor: float) -> float:
        """Diminishing returns function."""
        return base_response * np.log(1 + diminishing_factor * spend)
    
    def _michaelis_menten(self, spend: float, vmax: float, km: float) -> float:
        """Michaelis-Menten saturation function."""
        return (vmax * spend) / (km + spend)
    
    def _fit_model(self, model_type: str, spend: np.ndarray, response: np.ndarray, 
                   adstock_spend: np.ndarray, channel: str) -> Optional[SaturationModel]:
        """Fit a specific model type to the data."""
        model_func = self.model_functions[model_type]
        
        # Initial parameter guesses based on model type
        if model_type == 'hill':
            p0 = [np.max(response), np.median(spend), 1.0]
            bounds = ([0, 0, 0.1], [np.inf, np.inf, 5.0])
        elif model_type == 'exponential':
            p0 = [np.max(response), 0.001]
            bounds = ([0, 0], [np.inf, 1.0])
        elif model_type == 'logistic':
            p0 = [np.max(response), np.median(spend), 0.001]
            bounds = ([0, 0, 0], [np.inf, np.inf, 1.0])
        elif model_type == 'power':
            p0 = [1.0, 0.5]
            bounds = ([0, 0], [np.inf, 1.0])
        elif model_type == 'adstock_hill':
            p0 = [np.max(response), np.median(spend), 1.0, self.adstock_decay_rate]
            bounds = ([0, 0, 0.1, 0], [np.inf, np.inf, 5.0, 1.0])
        elif model_type == 'diminishing_returns':
            p0 = [np.max(response), 0.001]
            bounds = ([0, 0], [np.inf, 1.0])
        elif model_type == 'michaelis_menten':
            p0 = [np.max(response), np.median(spend)]
            bounds = ([0, 0], [np.inf, np.inf])
        else:
            return None
        
        try:
            # Use adstock spend for adstock models
            if 'adstock' in model_type:
                x_data = adstock_spend
            else:
                x_data = spend
            
            # Fit the curve
            popt, pcov = curve_fit(
                model_func, x_data, response,
                p0=p0, bounds=bounds, maxfev=5000,
                method='trf'
            )
            
            # Calculate predictions and metrics
            predictions = model_func(x_data, *popt)
            r_squared = r2_score(response, predictions)
            rmse = np.sqrt(mean_squared_error(response, predictions))
            
            # Calculate confidence intervals
            try:
                parameter_errors = np.sqrt(np.diag(pcov))
                confidence_intervals = [
                    (param - 1.96 * error, param + 1.96 * error)
                    for param, error in zip(popt, parameter_errors)
                ]
            except:
                confidence_intervals = [(0, 0)] * len(popt)
            
            # Find optimal spend
            optimal_spend = self._find_optimal_spend(model_func, popt, spend)
            
            # Calculate saturation point
            saturation_point = self._calculate_saturation_point(
                channel, model_func, popt, spend, response
            )
            
            # Generate model equation string
            model_equation = self._generate_model_equation(model_type, popt)
            
            # Goodness of fit metrics
            goodness_of_fit = {
                'r_squared': r_squared,
                'rmse': rmse,
                'aic': len(response) * np.log(rmse**2) + 2 * len(popt),
                'bic': len(response) * np.log(rmse**2) + len(popt) * np.log(len(response))
            }
            
            return SaturationModel(
                model_id=f"{channel}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                channel=channel,
                model_type=model_type,
                function=model_func,
                parameters=popt.tolist(),
                r_squared=r_squared,
                rmse=rmse,
                confidence_intervals=confidence_intervals,
                spend_range=(np.min(spend), np.max(spend)),
                response_range=(np.min(response), np.max(response)),
                optimal_spend=optimal_spend,
                saturation_point=saturation_point,
                model_equation=model_equation,
                goodness_of_fit=goodness_of_fit
            )
            
        except Exception as e:
            logger.debug(f"Failed to fit {model_type} for {channel}: {e}")
            return None
    
    def _select_best_model(self, model_results: Dict[str, SaturationModel], 
                          spend: np.ndarray, response: np.ndarray) -> SaturationModel:
        """Select the best model based on performance metrics."""
        # Weight different metrics
        weights = {'r_squared': 0.4, 'rmse': 0.3, 'aic': 0.2, 'bic': 0.1}
        
        best_model = None
        best_score = float('-inf')
        
        for model in model_results.values():
            # Normalize metrics
            r_sq_score = model.r_squared * weights['r_squared']
            rmse_score = (1 / (1 + model.rmse)) * weights['rmse']  # Lower RMSE is better
            aic_score = (1 / (1 + abs(model.goodness_of_fit['aic']))) * weights['aic']
            bic_score = (1 / (1 + abs(model.goodness_of_fit['bic']))) * weights['bic']
            
            total_score = r_sq_score + rmse_score + aic_score + bic_score
            
            if total_score > best_score:
                best_score = total_score
                best_model = model
        
        return best_model or list(model_results.values())[0]
    
    def _find_optimal_spend(self, model_func: Callable, parameters: np.ndarray, 
                           spend_range: np.ndarray) -> float:
        """Find optimal spend that maximizes ROI."""
        spend_min, spend_max = np.min(spend_range), np.max(spend_range)
        
        def roi_objective(spend):
            if spend <= 0:
                return 0
            response = model_func(spend, *parameters)
            roi = response / spend
            return -roi  # Minimize negative ROI
        
        try:
            result = minimize(
                roi_objective,
                x0=np.median(spend_range),
                bounds=[(spend_min, spend_max)],
                method='L-BFGS-B'
            )
            
            if result.success:
                return result.x[0]
        except:
            pass
        
        # Fall back to maximum efficiency point
        test_spends = np.linspace(spend_min, spend_max, 100)
        efficiencies = []
        
        for spend in test_spends:
            response = model_func(spend, *parameters)
            efficiency = response / spend if spend > 0 else 0
            efficiencies.append(efficiency)
        
        optimal_idx = np.argmax(efficiencies)
        return test_spends[optimal_idx]
    
    def _calculate_saturation_point(self, channel: str, model_func: Callable, 
                                  parameters: np.ndarray, spend: np.ndarray, 
                                  response: np.ndarray) -> SaturationPoint:
        """Calculate detailed saturation point analysis."""
        # Find asymptotic maximum for the model
        spend_max = np.max(spend)
        test_spends = np.linspace(0, spend_max * 3, 1000)  # Test beyond current range
        test_responses = [model_func(s, *parameters) for s in test_spends]
        
        # Find 95% of maximum response
        max_response = np.max(test_responses)
        saturation_response = max_response * self.saturation_threshold
        
        # Find spend level that achieves saturation
        saturation_idx = np.argmax(np.array(test_responses) >= saturation_response)
        saturation_spend = test_spends[saturation_idx] if saturation_idx > 0 else spend_max
        
        # Calculate efficiency at saturation
        efficiency_at_saturation = saturation_response / saturation_spend if saturation_spend > 0 else 0
        
        # Calculate marginal ROI at saturation
        delta_spend = saturation_spend * 0.01  # 1% increase
        response_at_delta = model_func(saturation_spend + delta_spend, *parameters)
        marginal_roi = (response_at_delta - saturation_response) / delta_spend
        
        # Current efficiency (at maximum spend)
        current_response = model_func(spend_max, *parameters)
        current_efficiency = current_response / spend_max if spend_max > 0 else 0
        
        # Potential gain
        potential_gain = (saturation_response - current_response) / current_response if current_response > 0 else 0
        
        # Confidence score based on model fit
        confidence_score = min(self.saturation_models[channel].r_squared, 1.0) if channel in self.saturation_models else 0.5
        
        # Recommended spend (between current max and saturation)
        recommended_spend = min(spend_max * 1.2, saturation_spend * 0.8)
        
        return SaturationPoint(
            channel=channel,
            saturation_spend=saturation_spend,
            saturation_response=saturation_response,
            efficiency_at_saturation=efficiency_at_saturation,
            marginal_roi=marginal_roi,
            confidence_score=confidence_score,
            recommended_spend=recommended_spend,
            current_efficiency=current_efficiency,
            potential_gain=potential_gain
        )
    
    def _calculate_saturation_metrics(self, model: SaturationModel) -> SaturationPoint:
        """Calculate saturation metrics from fitted model."""
        return model.saturation_point
    
    def _generate_model_equation(self, model_type: str, parameters: np.ndarray) -> str:
        """Generate human-readable model equation."""
        if model_type == 'hill':
            return f"Response = {parameters[0]:.2f} * Spend^{parameters[2]:.2f} / ({parameters[1]:.2f}^{parameters[2]:.2f} + Spend^{parameters[2]:.2f})"
        elif model_type == 'exponential':
            return f"Response = {parameters[0]:.2f} * (1 - exp(-{parameters[1]:.4f} * Spend))"
        elif model_type == 'logistic':
            return f"Response = {parameters[0]:.2f} / (1 + exp(-{parameters[2]:.4f} * (Spend - {parameters[1]:.2f})))"
        elif model_type == 'power':
            return f"Response = {parameters[0]:.2f} * Spend^{parameters[1]:.2f}"
        elif model_type == 'michaelis_menten':
            return f"Response = {parameters[0]:.2f} * Spend / ({parameters[1]:.2f} + Spend)"
        else:
            return f"{model_type} model with parameters: {[f'{p:.3f}' for p in parameters]}"
    
    def _efficiency_based_allocation(self, channels: List[str], total_budget: float) -> Dict[str, float]:
        """Allocate budget based on channel efficiency at current spend levels."""
        efficiencies = {}
        
        for channel in channels:
            if channel in self.processed_data:
                df = self.processed_data[channel]
                current_efficiency = df['efficiency'].iloc[-1]  # Latest efficiency
                efficiencies[channel] = current_efficiency
            else:
                efficiencies[channel] = 1.0  # Default
        
        # Proportional allocation based on efficiency
        total_efficiency = sum(efficiencies.values())
        allocation = {}
        
        for channel in channels:
            share = efficiencies[channel] / total_efficiency if total_efficiency > 0 else 1.0 / len(channels)
            allocation[channel] = total_budget * share
        
        return allocation
    
    def generate_saturation_report(self, channels: Optional[List[str]] = None) -> str:
        """Generate comprehensive saturation analysis report."""
        channels = channels or list(self.saturation_models.keys())
        
        report = "# Channel Saturation Analysis\n\n"
        report += "**Advanced Saturation Modeling by Sotiris Spyrou**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        # Executive Summary
        report += "## Executive Summary\n\n"
        report += f"- **Channels Analyzed**: {len(channels)}\n"
        report += f"- **Models Fitted**: {len(self.saturation_models)}\n"
        report += f"- **Adstock Modeling**: {'Enabled' if self.enable_adstock else 'Disabled'}\n\n"
        
        # Channel Analysis
        for channel in channels:
            if channel not in self.saturation_models:
                continue
                
            model = self.saturation_models[channel]
            saturation = model.saturation_point
            
            report += f"## {channel} Channel Analysis\n\n"
            report += f"- **Best Model**: {model.model_type.replace('_', ' ').title()}\n"
            report += f"- **Model Accuracy**: R² = {model.r_squared:.3f}\n"
            report += f"- **Model Equation**: {model.model_equation}\n\n"
            
            # Saturation Metrics
            report += f"### Saturation Analysis\n\n"
            report += f"- **Saturation Spend**: ${saturation.saturation_spend:,.2f}\n"
            report += f"- **Saturation Response**: {saturation.saturation_response:,.2f}\n"
            report += f"- **Current Efficiency**: {saturation.current_efficiency:.3f}\n"
            report += f"- **Efficiency at Saturation**: {saturation.efficiency_at_saturation:.3f}\n"
            report += f"- **Marginal ROI**: {saturation.marginal_roi:.3f}\n"
            report += f"- **Recommended Spend**: ${saturation.recommended_spend:,.2f}\n\n"
            
            # Optimization Insights
            if saturation.potential_gain > 0.1:
                report += f"💡 **Opportunity**: {saturation.potential_gain:.1%} potential gain available\n\n"
            elif saturation.marginal_roi < 0.1:
                report += f"⚠️ **Warning**: Low marginal ROI indicates approaching saturation\n\n"
        
        # Portfolio Optimization
        if len(channels) > 1:
            report += "## Portfolio Optimization\n\n"
            
            # Calculate total recommended budget
            total_recommended = sum(
                self.saturation_models[ch].saturation_point.recommended_spend 
                for ch in channels if ch in self.saturation_models
            )
            
            report += f"- **Total Recommended Budget**: ${total_recommended:,.2f}\n"
            
            # Allocation recommendations
            report += "\n### Recommended Allocation\n\n"
            report += "| Channel | Recommended Spend | Share | Current Efficiency |\n"
            report += "|---------|------------------|-------|-------------------|\n"
            
            for channel in channels:
                if channel in self.saturation_models:
                    saturation = self.saturation_models[channel].saturation_point
                    share = saturation.recommended_spend / total_recommended if total_recommended > 0 else 0
                    report += f"| {channel} | ${saturation.recommended_spend:,.2f} | {share:.1%} | {saturation.current_efficiency:.3f} |\n"
            
            report += "\n"
        
        report += "---\n"
        report += "*Advanced saturation modeling optimizes spend allocation across channels. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for enterprise implementations.*"
        
        return report
    
    def get_saturation_insights(self) -> Dict[str, Any]:
        """Get comprehensive saturation insights across all channels."""
        insights = {
            'total_channels': len(self.saturation_models),
            'model_performance': {},
            'saturation_summary': {},
            'optimization_opportunities': []
        }
        
        for channel, model in self.saturation_models.items():
            saturation = model.saturation_point
            
            # Model performance
            insights['model_performance'][channel] = {
                'model_type': model.model_type,
                'r_squared': model.r_squared,
                'confidence_score': saturation.confidence_score
            }
            
            # Saturation summary
            insights['saturation_summary'][channel] = {
                'saturation_spend': saturation.saturation_spend,
                'recommended_spend': saturation.recommended_spend,
                'potential_gain': saturation.potential_gain,
                'marginal_roi': saturation.marginal_roi
            }
            
            # Optimization opportunities
            if saturation.potential_gain > 0.1:
                insights['optimization_opportunities'].append({
                    'channel': channel,
                    'type': 'scale_up',
                    'opportunity': f"{saturation.potential_gain:.1%} potential gain",
                    'recommended_action': f"Increase spend to ${saturation.recommended_spend:,.2f}"
                })
            elif saturation.marginal_roi < 0.1:
                insights['optimization_opportunities'].append({
                    'channel': channel,
                    'type': 'scale_down',
                    'opportunity': f"Low marginal ROI ({saturation.marginal_roi:.3f})",
                    'recommended_action': f"Consider reducing spend or reallocating budget"
                })
        
        return insights


def demo_saturation_models():
    """Executive demonstration of Channel Saturation Models."""
    
    print("=== Advanced Channel Saturation Models: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    # Initialize saturation models
    saturation_analyzer = ChannelSaturationModels(
        enable_adstock=True,
        adstock_decay_rate=0.3,
        saturation_threshold=0.95,
        min_data_points=15
    )
    
    print("📊 Generating realistic channel saturation data...")
    
    # Generate demo data for multiple channels
    np.random.seed(42)
    
    channels = ['Search', 'Display', 'Social', 'Video']
    
    for channel in channels:
        # Channel-specific saturation characteristics
        if channel == 'Search':
            # High efficiency, clear saturation point
            max_response = 50000
            saturation_spend = 80000
            noise_level = 0.1
        elif channel == 'Display':
            # Lower efficiency, gradual saturation
            max_response = 30000
            saturation_spend = 120000
            noise_level = 0.15
        elif channel == 'Social':
            # Variable efficiency, S-curve pattern
            max_response = 40000
            saturation_spend = 60000
            noise_level = 0.2
        else:  # Video
            # Power law with diminishing returns
            max_response = 35000
            saturation_spend = 100000
            noise_level = 0.12
        
        # Generate spend points
        spend_points = np.logspace(3, np.log10(saturation_spend * 1.5), 30)  # $1K to beyond saturation
        
        channel_data = []
        for spend in spend_points:
            # Different response curves for different channels
            if channel == 'Search':
                # Hill saturation curve
                response = max_response * (spend ** 1.5) / (saturation_spend ** 1.5 + spend ** 1.5)
            elif channel == 'Display':
                # Exponential saturation
                response = max_response * (1 - np.exp(-spend / 40000))
            elif channel == 'Social':
                # Logistic S-curve
                response = max_response / (1 + np.exp(-0.00008 * (spend - saturation_spend)))
            else:  # Video
                # Power law with diminishing returns
                response = max_response * (spend ** 0.6) / (spend ** 0.6 + 30000 ** 0.6)
            
            # Add noise and ensure positive values
            noise = np.random.normal(1, noise_level)
            response = max(100, response * noise)
            
            # Add some additional metrics
            channel_data.append({
                'spend': spend,
                'response': response,
                'revenue': response,  # Using response as revenue for simplicity
                'date': datetime.now() - timedelta(days=30 - len(channel_data))
            })
        
        # Add channel data
        saturation_analyzer.add_channel_data(channel, channel_data)
    
    print(f"📈 Generated saturation data for {len(channels)} channels")
    
    print("\n🔄 Analyzing channel saturation patterns...")
    
    # Process and fit models for each channel
    results = {}
    for channel in channels:
        print(f"  • Processing {channel}...")
        saturation_analyzer.prepare_data(channel)
        saturation_analyzer.fit_saturation_models(channel)
        
        # Analyze saturation point
        saturation_point = saturation_analyzer.analyze_saturation_point(channel)
        if saturation_point:
            results[channel] = saturation_point
    
    print("\n📊 CHANNEL SATURATION ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\n📈 SATURATION SUMMARY:")
    print(f"  • Channels Analyzed: {len(results)}")
    print(f"  • Models Successfully Fitted: {len(saturation_analyzer.saturation_models)}")
    print(f"  • Adstock Modeling: {'Enabled' if saturation_analyzer.enable_adstock else 'Disabled'}")
    
    # Show detailed results for each channel
    total_current_spend = 0
    total_recommended_spend = 0
    
    print(f"\n🎯 CHANNEL-BY-CHANNEL ANALYSIS:")
    
    for channel, saturation in results.items():
        model = saturation_analyzer.saturation_models[channel]
        current_spend = saturation_analyzer.processed_data[channel]['spend'].max()
        
        total_current_spend += current_spend
        total_recommended_spend += saturation.recommended_spend
        
        print(f"\n• {channel.upper()} CHANNEL:")
        print(f"  📊 Model: {model.model_type.replace('_', ' ').title()} (R² = {model.r_squared:.3f})")
        print(f"  💰 Current Spend: ${current_spend:,.2f}")
        print(f"  🎯 Recommended Spend: ${saturation.recommended_spend:,.2f}")
        print(f"  📈 Saturation Point: ${saturation.saturation_spend:,.2f}")
        print(f"  ⚡ Current Efficiency: {saturation.current_efficiency:.3f}")
        print(f"  🔄 Marginal ROI: {saturation.marginal_roi:.3f}")
        
        if saturation.potential_gain > 0.1:
            print(f"  💡 Opportunity: {saturation.potential_gain:.1%} potential gain available")
        elif saturation.marginal_roi < 0.1:
            print(f"  ⚠️  Warning: Approaching saturation (low marginal ROI)")
    
    # Portfolio optimization
    print(f"\n💼 PORTFOLIO OPTIMIZATION:")
    print(f"  • Current Total Spend: ${total_current_spend:,.2f}")
    print(f"  • Recommended Total Spend: ${total_recommended_spend:,.2f}")
    print(f"  • Budget Change: {((total_recommended_spend - total_current_spend) / total_current_spend * 100):+.1f}%")
    
    # Test spend optimization
    print(f"\n🎯 OPTIMAL SPEND ALLOCATION:")
    test_budget = 200000  # $200K test budget
    optimal_allocation = saturation_analyzer.optimize_spend_allocation(
        channels=channels,
        total_budget=test_budget
    )
    
    for channel, allocated_spend in optimal_allocation.items():
        share = allocated_spend / test_budget if test_budget > 0 else 0
        print(f"  • {channel}: ${allocated_spend:,.2f} ({share:.1%})")
    
    # Predict response at optimal allocation
    print(f"\n📈 PREDICTED PERFORMANCE AT OPTIMAL ALLOCATION:")
    total_predicted_response = 0
    
    for channel, spend in optimal_allocation.items():
        if spend > 0 and channel in saturation_analyzer.saturation_models:
            predicted_response = saturation_analyzer.predict_response(channel, [spend])[0]
            efficiency = predicted_response / spend
            total_predicted_response += predicted_response
            print(f"  • {channel}: {predicted_response:,.0f} response (efficiency: {efficiency:.3f})")
    
    overall_efficiency = total_predicted_response / test_budget if test_budget > 0 else 0
    print(f"  • Overall Portfolio Efficiency: {overall_efficiency:.3f}")
    
    # Saturation insights
    insights = saturation_analyzer.get_saturation_insights()
    
    print(f"\n💡 KEY OPTIMIZATION OPPORTUNITIES:")
    for opportunity in insights['optimization_opportunities']:
        action_emoji = "🚀" if opportunity['type'] == 'scale_up' else "⚠️"
        print(f"  {action_emoji} {opportunity['channel']}: {opportunity['opportunity']}")
        print(f"     → {opportunity['recommended_action']}")
    
    # Model performance summary
    print(f"\n🎯 MODEL PERFORMANCE SUMMARY:")
    avg_r_squared = np.mean([perf['r_squared'] for perf in insights['model_performance'].values()])
    print(f"  • Average Model Accuracy (R²): {avg_r_squared:.3f}")
    
    best_model_channel = max(insights['model_performance'].items(), 
                           key=lambda x: x[1]['r_squared'])[0]
    best_r_squared = insights['model_performance'][best_model_channel]['r_squared']
    print(f"  • Best Model: {best_model_channel} (R² = {best_r_squared:.3f})")
    
    print("\n" + "="*70)
    print("🚀 Advanced saturation modeling optimizes channel spend allocation")
    print("💼 Identify saturation points and maximize marketing ROI")
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_saturation_models()