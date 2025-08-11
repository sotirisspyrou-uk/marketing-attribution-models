"""
Advanced Campaign Performance Forecaster

Predictive analytics engine for campaign performance forecasting using
time series analysis, machine learning, and scenario modeling.

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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.cluster import KMeans
from scipy import stats
from scipy.optimize import curve_fit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


@dataclass
class CampaignForecast:
    """Campaign performance forecast result."""
    forecast_id: str
    campaign_name: str
    forecast_period: int  # days
    forecast_start_date: datetime
    forecast_end_date: datetime
    predictions: Dict[str, List[float]]
    confidence_intervals: Dict[str, List[Tuple[float, float]]]
    model_accuracy: Dict[str, float]
    forecast_summary: Dict[str, float]
    risk_assessment: Dict[str, Any]
    scenario_analysis: Dict[str, Dict[str, float]]
    optimization_recommendations: List[str]
    feature_importance: Dict[str, float]
    forecast_assumptions: List[str]


@dataclass
class MarketConditions:
    """Market conditions affecting campaign performance."""
    seasonality_factor: float = 1.0
    competitive_pressure: float = 1.0
    economic_indicator: float = 1.0
    trend_factor: float = 1.0
    external_events: List[str] = field(default_factory=list)


class CampaignPerformanceForecaster:
    """
    Advanced campaign performance forecasting engine.
    
    Uses time series analysis, machine learning, and scenario modeling
    to predict campaign performance with confidence intervals and
    risk assessment.
    """
    
    def __init__(self,
                 forecasting_horizon: int = 30,
                 confidence_level: float = 0.9,
                 enable_seasonality: bool = True,
                 enable_trend_analysis: bool = True,
                 model_selection: str = 'auto'):
        """
        Initialize Campaign Performance Forecaster.
        
        Args:
            forecasting_horizon: Default forecasting horizon in days
            confidence_level: Confidence level for prediction intervals
            enable_seasonality: Enable seasonal decomposition
            enable_trend_analysis: Enable trend analysis
            model_selection: 'auto', 'ml', 'time_series', 'ensemble'
        """
        self.forecasting_horizon = forecasting_horizon
        self.confidence_level = confidence_level
        self.enable_seasonality = enable_seasonality
        self.enable_trend_analysis = enable_trend_analysis
        self.model_selection = model_selection
        
        # Historical data storage
        self.campaign_data: Dict[str, pd.DataFrame] = {}
        self.market_data: Dict[str, List[Dict]] = defaultdict(list)
        
        # Trained models
        self.models: Dict[str, Dict[str, Any]] = {}
        self.seasonal_components: Dict[str, Dict] = {}
        self.trend_components: Dict[str, Dict] = {}
        
        # Feature engineering
        self.feature_scalers: Dict[str, StandardScaler] = {}
        self.polynomial_features: Dict[str, PolynomialFeatures] = {}
        
        # Forecasting results
        self.forecasts: Dict[str, CampaignForecast] = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            },
            'elastic_net': {
                'alpha': 0.1,
                'l1_ratio': 0.5,
                'random_state': 42
            }
        }
        
        logger.info("Campaign Performance Forecaster initialized")
    
    def add_campaign_data(self, campaign_name: str, data: List[Dict[str, Any]]) -> 'CampaignPerformanceForecaster':
        """
        Add historical campaign performance data.
        
        Args:
            campaign_name: Name of the campaign
            data: List of daily performance data points
            
        Returns:
            Self for method chaining
        """
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure datetime index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Store campaign data
        self.campaign_data[campaign_name] = df
        
        logger.info(f"Added {len(df)} data points for campaign: {campaign_name}")
        return self
    
    def add_market_conditions(self, date: datetime, conditions: MarketConditions) -> 'CampaignPerformanceForecaster':
        """
        Add market conditions data.
        
        Args:
            date: Date for market conditions
            conditions: Market conditions data
            
        Returns:
            Self for method chaining
        """
        market_record = {
            'date': date,
            'seasonality_factor': conditions.seasonality_factor,
            'competitive_pressure': conditions.competitive_pressure,
            'economic_indicator': conditions.economic_indicator,
            'trend_factor': conditions.trend_factor,
            'external_events': conditions.external_events
        }
        
        self.market_data[date.strftime('%Y-%m-%d')] = market_record
        return self
    
    def train_forecasting_models(self, campaign_name: str) -> 'CampaignPerformanceForecaster':
        """
        Train forecasting models for a specific campaign.
        
        Args:
            campaign_name: Name of the campaign to train models for
            
        Returns:
            Self for method chaining
        """
        if campaign_name not in self.campaign_data:
            raise ValueError(f"No data found for campaign: {campaign_name}")
        
        df = self.campaign_data[campaign_name]
        
        if len(df) < 14:  # Need at least 2 weeks of data
            logger.warning(f"Insufficient data for {campaign_name}: {len(df)} days")
            return self
        
        # Prepare features and targets
        features, targets = self._prepare_training_data(df, campaign_name)
        
        if len(features) == 0:
            logger.warning(f"No valid features for {campaign_name}")
            return self
        
        # Initialize models storage
        self.models[campaign_name] = {}
        
        # Train different model types based on selection
        if self.model_selection in ['auto', 'ml', 'ensemble']:
            self._train_ml_models(campaign_name, features, targets)
        
        if self.model_selection in ['auto', 'time_series', 'ensemble']:
            self._train_time_series_models(campaign_name, df)
        
        # Seasonal decomposition
        if self.enable_seasonality:
            self._analyze_seasonality(campaign_name, df)
        
        # Trend analysis
        if self.enable_trend_analysis:
            self._analyze_trends(campaign_name, df)
        
        logger.info(f"Trained forecasting models for {campaign_name}")
        return self
    
    def generate_forecast(self,
                         campaign_name: str,
                         forecast_days: Optional[int] = None,
                         budget_scenario: Optional[float] = None,
                         market_conditions: Optional[MarketConditions] = None) -> CampaignForecast:
        """
        Generate performance forecast for a campaign.
        
        Args:
            campaign_name: Name of the campaign
            forecast_days: Number of days to forecast (default: forecasting_horizon)
            budget_scenario: Optional budget scenario multiplier
            market_conditions: Optional market conditions override
            
        Returns:
            Campaign forecast result
        """
        if campaign_name not in self.models:
            raise ValueError(f"No trained models for campaign: {campaign_name}")
        
        forecast_days = forecast_days or self.forecasting_horizon
        df = self.campaign_data[campaign_name]
        
        # Generate forecast dates
        last_date = df.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        # Generate predictions
        predictions = {}
        confidence_intervals = {}
        model_accuracy = {}
        
        # Key metrics to forecast
        forecast_metrics = ['spend', 'impressions', 'clicks', 'conversions', 'revenue', 'roas']
        
        for metric in forecast_metrics:
            if metric not in df.columns:
                continue
            
            # Generate predictions using best available model
            pred_results = self._generate_metric_predictions(
                campaign_name, metric, forecast_dates, budget_scenario, market_conditions
            )
            
            predictions[metric] = pred_results['predictions']
            confidence_intervals[metric] = pred_results['confidence_intervals']
            model_accuracy[metric] = pred_results['accuracy']
        
        # Generate forecast summary
        forecast_summary = self._calculate_forecast_summary(predictions)
        
        # Risk assessment
        risk_assessment = self._assess_forecast_risk(campaign_name, predictions, confidence_intervals)
        
        # Scenario analysis
        scenario_analysis = self._run_forecast_scenarios(
            campaign_name, forecast_dates, budget_scenario, market_conditions
        )
        
        # Optimization recommendations
        recommendations = self._generate_optimization_recommendations(
            campaign_name, predictions, risk_assessment
        )
        
        # Feature importance
        feature_importance = self._calculate_feature_importance(campaign_name)
        
        # Forecast assumptions
        assumptions = self._document_forecast_assumptions(
            campaign_name, budget_scenario, market_conditions
        )
        
        # Create forecast result
        forecast = CampaignForecast(
            forecast_id=f"{campaign_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            campaign_name=campaign_name,
            forecast_period=forecast_days,
            forecast_start_date=forecast_dates[0].to_pydatetime(),
            forecast_end_date=forecast_dates[-1].to_pydatetime(),
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            model_accuracy=model_accuracy,
            forecast_summary=forecast_summary,
            risk_assessment=risk_assessment,
            scenario_analysis=scenario_analysis,
            optimization_recommendations=recommendations,
            feature_importance=feature_importance,
            forecast_assumptions=assumptions
        )
        
        # Store forecast
        self.forecasts[forecast.forecast_id] = forecast
        
        logger.info(f"Generated {forecast_days}-day forecast for {campaign_name}")
        return forecast
    
    def _prepare_training_data(self, df: pd.DataFrame, campaign_name: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare training data with feature engineering."""
        features = []
        targets = defaultdict(list)
        
        # Create time-based features
        for i, (date, row) in enumerate(df.iterrows()):
            feature_vector = []
            
            # Basic metrics
            feature_vector.extend([
                row.get('budget', 0),
                row.get('spend', 0),
                row.get('impressions', 0),
                row.get('clicks', 0),
                row.get('ctr', 0) * 100,  # Convert to percentage
                row.get('cpc', 0),
                row.get('cvr', 0) * 100   # Convert to percentage
            ])
            
            # Time features
            feature_vector.extend([
                date.dayofweek,
                date.month,
                date.day,
                date.weekofyear if hasattr(date, 'weekofyear') else date.isocalendar()[1],
                int(date.strftime('%j'))  # Day of year
            ])
            
            # Lagged features (if available)
            if i >= 1:
                prev_row = df.iloc[i-1]
                feature_vector.extend([
                    prev_row.get('roas', 0),
                    prev_row.get('conversions', 0),
                    prev_row.get('revenue', 0)
                ])
            else:
                feature_vector.extend([0, 0, 0])
            
            # Rolling averages (if enough data)
            if i >= 7:
                week_data = df.iloc[max(0, i-6):i+1]
                feature_vector.extend([
                    week_data.get('roas', pd.Series([0])).mean(),
                    week_data.get('conversions', pd.Series([0])).mean(),
                    week_data.get('revenue', pd.Series([0])).mean()
                ])
            else:
                feature_vector.extend([row.get('roas', 0), row.get('conversions', 0), row.get('revenue', 0)])
            
            # Market conditions (if available)
            date_str = date.strftime('%Y-%m-%d')
            if date_str in self.market_data:
                market_info = self.market_data[date_str]
                feature_vector.extend([
                    market_info.get('seasonality_factor', 1.0),
                    market_info.get('competitive_pressure', 1.0),
                    market_info.get('economic_indicator', 1.0),
                    market_info.get('trend_factor', 1.0)
                ])
            else:
                feature_vector.extend([1.0, 1.0, 1.0, 1.0])
            
            features.append(feature_vector)
            
            # Target variables
            targets['spend'].append(row.get('spend', 0))
            targets['impressions'].append(row.get('impressions', 0))
            targets['clicks'].append(row.get('clicks', 0))
            targets['conversions'].append(row.get('conversions', 0))
            targets['revenue'].append(row.get('revenue', 0))
            targets['roas'].append(row.get('roas', 0))
        
        features_array = np.array(features)
        targets_dict = {k: np.array(v) for k, v in targets.items()}
        
        return features_array, targets_dict
    
    def _train_ml_models(self, campaign_name: str, features: np.ndarray, targets: Dict[str, np.ndarray]):
        """Train machine learning models."""
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        self.feature_scalers[campaign_name] = scaler
        
        # Polynomial features for non-linear relationships
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        features_poly = poly.fit_transform(features_scaled)
        self.polynomial_features[campaign_name] = poly
        
        # Train models for each target metric
        for metric, target_values in targets.items():
            if len(target_values) < 10:  # Need sufficient data
                continue
            
            metric_models = {}
            metric_scores = {}
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            try:
                # Random Forest
                rf_model = RandomForestRegressor(**self.model_configs['random_forest'])
                rf_scores = cross_val_score(
                    rf_model, features_scaled, target_values,
                    cv=tscv, scoring='neg_mean_absolute_error'
                )
                rf_model.fit(features_scaled, target_values)
                metric_models['random_forest'] = rf_model
                metric_scores['random_forest'] = -np.mean(rf_scores)
                
                # Gradient Boosting
                gb_model = GradientBoostingRegressor(**self.model_configs['gradient_boosting'])
                gb_scores = cross_val_score(
                    gb_model, features_scaled, target_values,
                    cv=tscv, scoring='neg_mean_absolute_error'
                )
                gb_model.fit(features_scaled, target_values)
                metric_models['gradient_boosting'] = gb_model
                metric_scores['gradient_boosting'] = -np.mean(gb_scores)
                
                # Elastic Net (with polynomial features)
                en_model = ElasticNet(**self.model_configs['elastic_net'])
                en_scores = cross_val_score(
                    en_model, features_poly, target_values,
                    cv=tscv, scoring='neg_mean_absolute_error'
                )
                en_model.fit(features_poly, target_values)
                metric_models['elastic_net'] = en_model
                metric_scores['elastic_net'] = -np.mean(en_scores)
                
                # Select best model
                best_model_name = min(metric_scores.items(), key=lambda x: x[1])[0]
                
                self.models[campaign_name][f'{metric}_ml'] = {
                    'model': metric_models[best_model_name],
                    'model_type': best_model_name,
                    'score': metric_scores[best_model_name],
                    'all_scores': metric_scores,
                    'feature_importance': self._get_feature_importance(
                        metric_models[best_model_name], best_model_name
                    )
                }
                
            except Exception as e:
                logger.warning(f"ML model training failed for {campaign_name}.{metric}: {e}")
    
    def _train_time_series_models(self, campaign_name: str, df: pd.DataFrame):
        """Train time series models."""
        for column in ['spend', 'impressions', 'clicks', 'conversions', 'revenue', 'roas']:
            if column not in df.columns or df[column].isnull().all():
                continue
            
            try:
                series = df[column].fillna(method='ffill').fillna(method='bfill')
                
                if len(series) < 14:  # Need minimum data for time series
                    continue
                
                # ARIMA model
                try:
                    # Simple ARIMA(1,1,1) for demonstration
                    arima_model = ARIMA(series, order=(1, 1, 1))
                    arima_fitted = arima_model.fit()
                    
                    # Calculate AIC for model selection
                    arima_aic = arima_fitted.aic
                    
                    self.models[campaign_name][f'{column}_arima'] = {
                        'model': arima_fitted,
                        'model_type': 'arima',
                        'aic': arima_aic
                    }
                    
                except Exception as e:
                    logger.debug(f"ARIMA failed for {column}: {e}")
                
                # Exponential Smoothing
                try:
                    # Determine seasonality
                    seasonal_periods = min(7, len(series) // 3)  # Weekly or adaptive
                    
                    if len(series) >= seasonal_periods * 2:
                        es_model = ExponentialSmoothing(
                            series,
                            seasonal_periods=seasonal_periods,
                            trend='add',
                            seasonal='add'
                        )
                    else:
                        es_model = ExponentialSmoothing(series, trend='add')
                    
                    es_fitted = es_model.fit()
                    
                    self.models[campaign_name][f'{column}_exponential_smoothing'] = {
                        'model': es_fitted,
                        'model_type': 'exponential_smoothing'
                    }
                    
                except Exception as e:
                    logger.debug(f"Exponential Smoothing failed for {column}: {e}")
                
            except Exception as e:
                logger.warning(f"Time series training failed for {column}: {e}")
    
    def _analyze_seasonality(self, campaign_name: str, df: pd.DataFrame):
        """Analyze seasonal patterns in the data."""
        seasonal_components = {}
        
        for column in ['spend', 'impressions', 'clicks', 'conversions', 'revenue', 'roas']:
            if column not in df.columns or len(df) < 14:
                continue
            
            try:
                series = df[column].fillna(method='ffill').fillna(method='bfill')
                
                # Perform seasonal decomposition
                if len(series) >= 14:  # Need at least 2 weeks
                    decomposition = seasonal_decompose(
                        series, model='additive', period=min(7, len(series)//2)
                    )
                    
                    seasonal_components[column] = {
                        'seasonal': decomposition.seasonal,
                        'trend': decomposition.trend,
                        'residual': decomposition.resid,
                        'seasonal_strength': np.var(decomposition.seasonal.dropna()) / np.var(series.dropna())
                    }
                
            except Exception as e:
                logger.debug(f"Seasonality analysis failed for {column}: {e}")
        
        self.seasonal_components[campaign_name] = seasonal_components
    
    def _analyze_trends(self, campaign_name: str, df: pd.DataFrame):
        """Analyze trend patterns in the data."""
        trend_components = {}
        
        for column in ['spend', 'impressions', 'clicks', 'conversions', 'revenue', 'roas']:
            if column not in df.columns:
                continue
            
            try:
                series = df[column].fillna(method='ffill').fillna(method='bfill')
                
                # Linear trend analysis
                x = np.arange(len(series))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
                
                trend_components[column] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'flat'
                }
                
            except Exception as e:
                logger.debug(f"Trend analysis failed for {column}: {e}")
        
        self.trend_components[campaign_name] = trend_components
    
    def _generate_metric_predictions(self,
                                   campaign_name: str,
                                   metric: str,
                                   forecast_dates: pd.DatetimeIndex,
                                   budget_scenario: Optional[float] = None,
                                   market_conditions: Optional[MarketConditions] = None) -> Dict[str, Any]:
        """Generate predictions for a specific metric."""
        predictions = []
        confidence_intervals = []
        
        # Try different model types and select best
        model_predictions = {}
        
        # ML model predictions
        if f'{metric}_ml' in self.models[campaign_name]:
            ml_pred = self._predict_with_ml_model(
                campaign_name, metric, forecast_dates, budget_scenario, market_conditions
            )
            model_predictions['ml'] = ml_pred
        
        # Time series model predictions
        for ts_model_name in [f'{metric}_arima', f'{metric}_exponential_smoothing']:
            if ts_model_name in self.models[campaign_name]:
                ts_pred = self._predict_with_ts_model(
                    campaign_name, ts_model_name, len(forecast_dates)
                )
                model_predictions[ts_model_name.split('_')[-1]] = ts_pred
        
        # Ensemble prediction if multiple models available
        if len(model_predictions) > 1:
            predictions, confidence_intervals = self._ensemble_predictions(model_predictions)
        elif len(model_predictions) == 1:
            pred_data = list(model_predictions.values())[0]
            predictions = pred_data.get('predictions', [0] * len(forecast_dates))
            confidence_intervals = pred_data.get('confidence_intervals', [(0, 0)] * len(forecast_dates))
        else:
            # Fallback to simple trend extrapolation
            predictions, confidence_intervals = self._fallback_prediction(
                campaign_name, metric, len(forecast_dates)
            )
        
        # Calculate model accuracy
        accuracy = self._calculate_model_accuracy(campaign_name, metric)
        
        return {
            'predictions': predictions,
            'confidence_intervals': confidence_intervals,
            'accuracy': accuracy
        }
    
    def _predict_with_ml_model(self,
                             campaign_name: str,
                             metric: str,
                             forecast_dates: pd.DatetimeIndex,
                             budget_scenario: Optional[float],
                             market_conditions: Optional[MarketConditions]) -> Dict[str, Any]:
        """Generate predictions using ML model."""
        model_info = self.models[campaign_name][f'{metric}_ml']
        model = model_info['model']
        model_type = model_info['model_type']
        
        # Get last known values from historical data
        df = self.campaign_data[campaign_name]
        last_row = df.iloc[-1]
        
        predictions = []
        confidence_intervals = []
        
        for i, date in enumerate(forecast_dates):
            # Create feature vector for prediction
            feature_vector = []
            
            # Basic metrics (using previous predictions or scenarios)
            if budget_scenario:
                budget = last_row.get('budget', 0) * budget_scenario
                spend = budget * 0.95  # Assume 95% spend rate
            else:
                budget = last_row.get('budget', 0)
                spend = last_row.get('spend', 0)
            
            feature_vector.extend([
                budget,
                spend,
                last_row.get('impressions', 0),
                last_row.get('clicks', 0),
                last_row.get('ctr', 0) * 100,
                last_row.get('cpc', 0),
                last_row.get('cvr', 0) * 100
            ])
            
            # Time features
            feature_vector.extend([
                date.dayofweek,
                date.month,
                date.day,
                date.isocalendar()[1],
                int(date.strftime('%j'))
            ])
            
            # Lagged features (use last known or previous prediction)
            if i == 0:
                feature_vector.extend([
                    last_row.get('roas', 0),
                    last_row.get('conversions', 0),
                    last_row.get('revenue', 0)
                ])
            else:
                # Use previous predictions
                prev_pred = predictions[i-1] if predictions else last_row.get(metric, 0)
                feature_vector.extend([prev_pred, prev_pred * 0.1, prev_pred * 1.5])
            
            # Rolling averages (use last known)
            feature_vector.extend([
                last_row.get('roas', 0),
                last_row.get('conversions', 0),
                last_row.get('revenue', 0)
            ])
            
            # Market conditions
            if market_conditions:
                feature_vector.extend([
                    market_conditions.seasonality_factor,
                    market_conditions.competitive_pressure,
                    market_conditions.economic_indicator,
                    market_conditions.trend_factor
                ])
            else:
                feature_vector.extend([1.0, 1.0, 1.0, 1.0])
            
            # Scale features
            feature_array = np.array([feature_vector])
            scaler = self.feature_scalers[campaign_name]
            feature_scaled = scaler.transform(feature_array)
            
            # Use polynomial features for elastic net
            if model_type == 'elastic_net':
                poly = self.polynomial_features[campaign_name]
                feature_scaled = poly.transform(feature_scaled)
            
            # Make prediction
            prediction = model.predict(feature_scaled)[0]
            predictions.append(max(0, prediction))  # Ensure non-negative
            
            # Simple confidence interval (using model variance)
            prediction_std = model_info.get('score', prediction * 0.1)
            ci_lower = max(0, prediction - 1.96 * prediction_std)
            ci_upper = prediction + 1.96 * prediction_std
            confidence_intervals.append((ci_lower, ci_upper))
        
        return {
            'predictions': predictions,
            'confidence_intervals': confidence_intervals
        }
    
    def _predict_with_ts_model(self, campaign_name: str, model_name: str, periods: int) -> Dict[str, Any]:
        """Generate predictions using time series model."""
        model_info = self.models[campaign_name][model_name]
        model = model_info['model']
        model_type = model_info['model_type']
        
        try:
            if model_type == 'arima':
                # ARIMA forecast
                forecast_result = model.forecast(steps=periods)
                predictions = forecast_result.tolist()
                
                # Get confidence intervals
                conf_int = model.get_forecast(steps=periods).conf_int()
                confidence_intervals = [(max(0, lower), upper) for lower, upper in conf_int.values]
                
            elif model_type == 'exponential_smoothing':
                # Exponential smoothing forecast
                forecast_result = model.forecast(periods)
                predictions = forecast_result.tolist()
                
                # Simple confidence intervals for exponential smoothing
                forecast_std = np.std(model.resid.dropna()) if hasattr(model, 'resid') else np.std(predictions) * 0.1
                confidence_intervals = [
                    (max(0, pred - 1.96 * forecast_std), pred + 1.96 * forecast_std)
                    for pred in predictions
                ]
            
            else:
                raise ValueError(f"Unknown time series model type: {model_type}")
            
            return {
                'predictions': [max(0, p) for p in predictions],
                'confidence_intervals': confidence_intervals
            }
            
        except Exception as e:
            logger.warning(f"Time series prediction failed for {model_name}: {e}")
            # Fallback to simple prediction
            return {
                'predictions': [0] * periods,
                'confidence_intervals': [(0, 0)] * periods
            }
    
    def _ensemble_predictions(self, model_predictions: Dict[str, Dict]) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Combine predictions from multiple models."""
        all_predictions = []
        all_confidence_intervals = []
        
        # Extract predictions from all models
        for model_name, pred_data in model_predictions.items():
            all_predictions.append(pred_data.get('predictions', []))
            all_confidence_intervals.append(pred_data.get('confidence_intervals', []))
        
        if not all_predictions:
            return [], []
        
        # Average predictions
        ensemble_predictions = []
        ensemble_confidence_intervals = []
        
        for i in range(len(all_predictions[0])):
            # Average predictions
            period_predictions = [preds[i] for preds in all_predictions if i < len(preds)]
            avg_prediction = np.mean(period_predictions) if period_predictions else 0
            ensemble_predictions.append(avg_prediction)
            
            # Average confidence intervals
            period_intervals = [intervals[i] for intervals in all_confidence_intervals if i < len(intervals)]
            if period_intervals:
                avg_lower = np.mean([interval[0] for interval in period_intervals])
                avg_upper = np.mean([interval[1] for interval in period_intervals])
                ensemble_confidence_intervals.append((avg_lower, avg_upper))
            else:
                ensemble_confidence_intervals.append((avg_prediction * 0.8, avg_prediction * 1.2))
        
        return ensemble_predictions, ensemble_confidence_intervals
    
    def _fallback_prediction(self, campaign_name: str, metric: str, periods: int) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Fallback prediction using simple trend extrapolation."""
        df = self.campaign_data[campaign_name]
        
        if metric not in df.columns:
            return [0] * periods, [(0, 0)] * periods
        
        series = df[metric].fillna(method='ffill').fillna(method='bfill')
        
        if len(series) == 0:
            return [0] * periods, [(0, 0)] * periods
        
        # Simple linear trend
        x = np.arange(len(series))
        slope, intercept, _, _, _ = stats.linregress(x, series.values)
        
        predictions = []
        confidence_intervals = []
        
        for i in range(periods):
            pred = intercept + slope * (len(series) + i)
            pred = max(0, pred)  # Ensure non-negative
            predictions.append(pred)
            
            # Simple confidence interval
            std_dev = np.std(series.values) if len(series) > 1 else pred * 0.1
            ci_lower = max(0, pred - 1.96 * std_dev)
            ci_upper = pred + 1.96 * std_dev
            confidence_intervals.append((ci_lower, ci_upper))
        
        return predictions, confidence_intervals
    
    def _calculate_model_accuracy(self, campaign_name: str, metric: str) -> float:
        """Calculate model accuracy for a metric."""
        if f'{metric}_ml' in self.models[campaign_name]:
            return 1.0 / (1.0 + self.models[campaign_name][f'{metric}_ml'].get('score', 1.0))
        return 0.7  # Default moderate accuracy
    
    def _calculate_forecast_summary(self, predictions: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate summary statistics for the forecast."""
        summary = {}
        
        for metric, values in predictions.items():
            if values:
                summary[f'{metric}_total'] = sum(values)
                summary[f'{metric}_average'] = np.mean(values)
                summary[f'{metric}_trend'] = (values[-1] - values[0]) / len(values) if len(values) > 1 else 0
        
        # Calculate derived metrics
        if 'revenue' in predictions and 'spend' in predictions:
            total_revenue = sum(predictions['revenue'])
            total_spend = sum(predictions['spend'])
            summary['forecast_roas'] = total_revenue / total_spend if total_spend > 0 else 0
        
        return summary
    
    def _assess_forecast_risk(self,
                            campaign_name: str,
                            predictions: Dict[str, List[float]],
                            confidence_intervals: Dict[str, List[Tuple[float, float]]]) -> Dict[str, Any]:
        """Assess risk factors in the forecast."""
        risk_assessment = {
            'overall_risk': 'medium',
            'risk_factors': [],
            'confidence_score': 0.5,
            'volatility_score': 0.5
        }
        
        # Calculate prediction volatility
        volatilities = []
        for metric, values in predictions.items():
            if len(values) > 1:
                volatility = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                volatilities.append(volatility)
        
        if volatilities:
            avg_volatility = np.mean(volatilities)
            risk_assessment['volatility_score'] = min(avg_volatility, 1.0)
            
            if avg_volatility > 0.3:
                risk_assessment['risk_factors'].append("High prediction volatility")
        
        # Assess confidence intervals width
        ci_widths = []
        for metric, intervals in confidence_intervals.items():
            for lower, upper in intervals:
                if upper > lower:
                    width = (upper - lower) / ((upper + lower) / 2) if (upper + lower) > 0 else 0
                    ci_widths.append(width)
        
        if ci_widths:
            avg_ci_width = np.mean(ci_widths)
            risk_assessment['confidence_score'] = max(0, 1.0 - avg_ci_width)
            
            if avg_ci_width > 0.5:
                risk_assessment['risk_factors'].append("Wide confidence intervals indicate high uncertainty")
        
        # Model quality assessment
        model_accuracies = []
        for metric in predictions.keys():
            accuracy = self._calculate_model_accuracy(campaign_name, metric)
            model_accuracies.append(accuracy)
        
        if model_accuracies:
            avg_accuracy = np.mean(model_accuracies)
            if avg_accuracy < 0.6:
                risk_assessment['risk_factors'].append("Low model accuracy")
        
        # Overall risk classification
        risk_score = (risk_assessment['volatility_score'] + (1 - risk_assessment['confidence_score'])) / 2
        
        if risk_score < 0.3:
            risk_assessment['overall_risk'] = 'low'
        elif risk_score > 0.7:
            risk_assessment['overall_risk'] = 'high'
        
        return risk_assessment
    
    def _run_forecast_scenarios(self,
                              campaign_name: str,
                              forecast_dates: pd.DatetimeIndex,
                              budget_scenario: Optional[float],
                              market_conditions: Optional[MarketConditions]) -> Dict[str, Dict[str, float]]:
        """Run scenario analysis for forecast."""
        scenarios = {}
        
        # Base scenario (current conditions)
        scenarios['base'] = self._calculate_forecast_summary(
            self.generate_forecast(campaign_name, len(forecast_dates)).predictions
        )
        
        # Optimistic scenario (+20% performance)
        optimistic_conditions = MarketConditions(
            seasonality_factor=1.2,
            competitive_pressure=0.9,
            economic_indicator=1.1,
            trend_factor=1.15
        )
        try:
            optimistic_forecast = self.generate_forecast(
                campaign_name, len(forecast_dates), 
                budget_scenario, optimistic_conditions
            )
            scenarios['optimistic'] = self._calculate_forecast_summary(optimistic_forecast.predictions)
        except:
            scenarios['optimistic'] = {k: v * 1.2 for k, v in scenarios['base'].items()}
        
        # Pessimistic scenario (-20% performance)
        pessimistic_conditions = MarketConditions(
            seasonality_factor=0.8,
            competitive_pressure=1.2,
            economic_indicator=0.9,
            trend_factor=0.85
        )
        try:
            pessimistic_forecast = self.generate_forecast(
                campaign_name, len(forecast_dates),
                budget_scenario, pessimistic_conditions
            )
            scenarios['pessimistic'] = self._calculate_forecast_summary(pessimistic_forecast.predictions)
        except:
            scenarios['pessimistic'] = {k: v * 0.8 for k, v in scenarios['base'].items()}
        
        return scenarios
    
    def _generate_optimization_recommendations(self,
                                             campaign_name: str,
                                             predictions: Dict[str, List[float]],
                                             risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on forecast."""
        recommendations = []
        
        # Budget recommendations
        if 'spend' in predictions and 'revenue' in predictions:
            total_spend = sum(predictions['spend'])
            total_revenue = sum(predictions['revenue'])
            forecast_roas = total_revenue / total_spend if total_spend > 0 else 0
            
            if forecast_roas < 2.0:
                recommendations.append("Consider reducing budget allocation - forecast ROAS below 2.0")
            elif forecast_roas > 4.0:
                recommendations.append("Consider increasing budget - forecast shows strong ROAS potential")
        
        # Risk-based recommendations
        if risk_assessment['overall_risk'] == 'high':
            recommendations.append("High forecast uncertainty - consider conservative budget approach")
        
        if 'High prediction volatility' in risk_assessment['risk_factors']:
            recommendations.append("Expected performance volatility - implement flexible budget allocation")
        
        # Trend-based recommendations
        trend_components = self.trend_components.get(campaign_name, {})
        for metric, trend_info in trend_components.items():
            if trend_info.get('trend_direction') == 'decreasing' and trend_info.get('p_value', 1) < 0.05:
                recommendations.append(f"{metric} shows declining trend - review campaign strategy")
        
        return recommendations
    
    def _calculate_feature_importance(self, campaign_name: str) -> Dict[str, float]:
        """Calculate feature importance across models."""
        importance_scores = defaultdict(float)
        model_count = 0
        
        for model_key, model_info in self.models[campaign_name].items():
            if '_ml' in model_key and 'feature_importance' in model_info:
                importance = model_info['feature_importance']
                for feature, score in importance.items():
                    importance_scores[feature] += score
                model_count += 1
        
        # Average importance across models
        if model_count > 0:
            for feature in importance_scores:
                importance_scores[feature] /= model_count
        
        return dict(importance_scores)
    
    def _document_forecast_assumptions(self,
                                     campaign_name: str,
                                     budget_scenario: Optional[float],
                                     market_conditions: Optional[MarketConditions]) -> List[str]:
        """Document key assumptions made in the forecast."""
        assumptions = []
        
        # Data assumptions
        df = self.campaign_data[campaign_name]
        assumptions.append(f"Forecast based on {len(df)} days of historical data")
        
        # Budget assumptions
        if budget_scenario:
            assumptions.append(f"Budget scenario assumes {budget_scenario:.0%} of current spend levels")
        else:
            assumptions.append("Budget assumed to remain at current levels")
        
        # Market assumptions
        if market_conditions:
            assumptions.append("Custom market conditions applied to forecast")
        else:
            assumptions.append("Market conditions assumed to remain stable")
        
        # Model assumptions
        model_types = set()
        for model_key in self.models[campaign_name].keys():
            if '_ml' in model_key:
                model_types.add('machine_learning')
            elif 'arima' in model_key:
                model_types.add('time_series')
            elif 'exponential_smoothing' in model_key:
                model_types.add('exponential_smoothing')
        
        if model_types:
            assumptions.append(f"Forecast uses {', '.join(model_types)} modeling approaches")
        
        return assumptions
    
    def _get_feature_importance(self, model, model_type: str) -> Dict[str, float]:
        """Extract feature importance from trained model."""
        if model_type in ['random_forest', 'gradient_boosting'] and hasattr(model, 'feature_importances_'):
            # Create basic feature names
            feature_names = [
                'budget', 'spend', 'impressions', 'clicks', 'ctr', 'cpc', 'cvr',
                'dayofweek', 'month', 'day', 'week', 'dayofyear',
                'prev_roas', 'prev_conversions', 'prev_revenue',
                'rolling_roas', 'rolling_conversions', 'rolling_revenue',
                'seasonality', 'competition', 'economic', 'trend'
            ]
            
            importances = model.feature_importances_
            if len(importances) == len(feature_names):
                return dict(zip(feature_names, importances))
        
        return {}
    
    def generate_forecast_report(self, forecast: CampaignForecast) -> str:
        """Generate comprehensive forecast report."""
        report = "# Campaign Performance Forecast\n\n"
        report += "**Advanced Campaign Forecasting by Sotiris Spyrou**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        # Executive Summary
        report += "## Executive Summary\n\n"
        report += f"- **Campaign**: {forecast.campaign_name}\n"
        report += f"- **Forecast Period**: {forecast.forecast_period} days ({forecast.forecast_start_date.strftime('%Y-%m-%d')} to {forecast.forecast_end_date.strftime('%Y-%m-%d')})\n"
        report += f"- **Overall Risk**: {forecast.risk_assessment.get('overall_risk', 'medium').title()}\n"
        report += f"- **Confidence Score**: {forecast.risk_assessment.get('confidence_score', 0.5):.1%}\n\n"
        
        # Forecast Summary
        if forecast.forecast_summary:
            report += "## Forecast Summary\n\n"
            for metric, value in forecast.forecast_summary.items():
                if 'total' in metric:
                    report += f"- **{metric.replace('_', ' ').title()}**: {value:,.2f}\n"
                elif 'average' in metric:
                    report += f"- **{metric.replace('_', ' ').title()}**: {value:,.2f}\n"
                elif 'roas' in metric:
                    report += f"- **{metric.replace('_', ' ').title()}**: {value:.2f}x\n"
            report += "\n"
        
        # Key Predictions
        report += "## Key Performance Predictions\n\n"
        report += "| Metric | Forecast Total | Daily Average | Confidence Interval |\n"
        report += "|--------|---------------|---------------|---------------------|\n"
        
        for metric in ['spend', 'revenue', 'conversions', 'roas']:
            if metric in forecast.predictions:
                predictions = forecast.predictions[metric]
                total_pred = sum(predictions)
                avg_pred = np.mean(predictions)
                
                # Average confidence interval
                if metric in forecast.confidence_intervals:
                    intervals = forecast.confidence_intervals[metric]
                    avg_ci_lower = np.mean([ci[0] for ci in intervals])
                    avg_ci_upper = np.mean([ci[1] for ci in intervals])
                    ci_text = f"{avg_ci_lower:.2f} - {avg_ci_upper:.2f}"
                else:
                    ci_text = "N/A"
                
                report += f"| {metric.title()} | {total_pred:,.2f} | {avg_pred:,.2f} | {ci_text} |\n"
        
        report += "\n"
        
        # Risk Assessment
        report += "## Risk Assessment\n\n"
        risk = forecast.risk_assessment
        report += f"- **Overall Risk Level**: {risk.get('overall_risk', 'medium').title()}\n"
        report += f"- **Forecast Confidence**: {risk.get('confidence_score', 0.5):.1%}\n"
        report += f"- **Volatility Score**: {risk.get('volatility_score', 0.5):.1%}\n\n"
        
        if risk.get('risk_factors'):
            report += "**Key Risk Factors:**\n\n"
            for factor in risk['risk_factors']:
                report += f"- {factor}\n"
            report += "\n"
        
        # Scenario Analysis
        if forecast.scenario_analysis:
            report += "## Scenario Analysis\n\n"
            
            for scenario, metrics in forecast.scenario_analysis.items():
                report += f"### {scenario.title()} Scenario\n\n"
                for metric, value in metrics.items():
                    if 'roas' in metric:
                        report += f"- **{metric.replace('_', ' ').title()}**: {value:.2f}x\n"
                    else:
                        report += f"- **{metric.replace('_', ' ').title()}**: {value:,.2f}\n"
                report += "\n"
        
        # Optimization Recommendations
        if forecast.optimization_recommendations:
            report += "## Optimization Recommendations\n\n"
            for i, rec in enumerate(forecast.optimization_recommendations, 1):
                report += f"{i}. {rec}\n"
            report += "\n"
        
        # Model Performance
        if forecast.model_accuracy:
            report += "## Model Performance\n\n"
            report += "| Metric | Model Accuracy |\n"
            report += "|--------|----------------|\n"
            
            for metric, accuracy in forecast.model_accuracy.items():
                report += f"| {metric.title()} | {accuracy:.1%} |\n"
            
            report += "\n"
        
        # Forecast Assumptions
        if forecast.forecast_assumptions:
            report += "## Key Assumptions\n\n"
            for assumption in forecast.forecast_assumptions:
                report += f"- {assumption}\n"
            report += "\n"
        
        report += "---\n"
        report += "*Advanced forecasting provides data-driven insights for campaign optimization. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for enterprise implementations.*"
        
        return report


def demo_campaign_forecaster():
    """Executive demonstration of Campaign Performance Forecaster."""
    
    print("=== Advanced Campaign Performance Forecaster: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    # Initialize forecaster
    forecaster = CampaignPerformanceForecaster(
        forecasting_horizon=30,
        confidence_level=0.9,
        enable_seasonality=True,
        enable_trend_analysis=True,
        model_selection='auto'
    )
    
    print("📊 Generating realistic campaign performance data...")
    
    # Generate demo campaign data
    np.random.seed(42)
    
    # Create 60 days of historical data
    campaign_data = []
    base_date = datetime.now() - timedelta(days=60)
    
    # Campaign parameters
    base_budget = 15000
    base_roas = 3.5
    
    for i in range(60):
        date = base_date + timedelta(days=i)
        
        # Seasonal and trend effects
        seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * i / 7)  # Weekly seasonality
        trend_factor = 1.0 + 0.02 * i  # Growing trend
        noise = np.random.normal(1.0, 0.15)
        
        # Budget with some variation
        daily_budget = base_budget * seasonal_factor * noise * np.random.uniform(0.8, 1.2)
        spend = daily_budget * np.random.uniform(0.90, 1.0)
        
        # Performance metrics
        impressions = int(spend * np.random.uniform(18, 28))
        ctr = np.random.uniform(0.018, 0.045) * seasonal_factor
        clicks = int(impressions * ctr)
        cpc = spend / clicks if clicks > 0 else 0
        
        cvr = np.random.uniform(0.015, 0.035) * trend_factor
        conversions = clicks * cvr
        
        revenue = spend * base_roas * seasonal_factor * trend_factor * noise
        roas = revenue / spend if spend > 0 else 0
        
        campaign_data.append({
            'date': date,
            'budget': daily_budget,
            'spend': spend,
            'impressions': impressions,
            'clicks': clicks,
            'ctr': ctr,
            'cpc': cpc,
            'conversions': conversions,
            'cvr': cvr,
            'revenue': revenue,
            'roas': roas
        })
    
    # Add campaign data to forecaster
    forecaster.add_campaign_data("premium_search_campaign", campaign_data)
    
    # Add some market conditions
    for i in range(0, 60, 7):
        date = base_date + timedelta(days=i)
        conditions = MarketConditions(
            seasonality_factor=1.0 + 0.2 * np.sin(2 * np.pi * i / 28),  # Monthly cycle
            competitive_pressure=np.random.uniform(0.9, 1.1),
            economic_indicator=np.random.uniform(0.95, 1.05),
            trend_factor=1.0 + 0.01 * i
        )
        forecaster.add_market_conditions(date, conditions)
    
    print(f"📈 Generated 60 days of campaign performance data")
    
    print("\n🔄 Training forecasting models...")
    
    # Train models
    forecaster.train_forecasting_models("premium_search_campaign")
    
    print("\n🎯 Generating 30-day performance forecast...")
    
    # Generate forecast
    forecast = forecaster.generate_forecast(
        campaign_name="premium_search_campaign",
        forecast_days=30,
        budget_scenario=1.2,  # 20% budget increase scenario
        market_conditions=MarketConditions(
            seasonality_factor=1.1,
            competitive_pressure=1.0,
            economic_indicator=1.05,
            trend_factor=1.02
        )
    )
    
    print("\n📊 CAMPAIGN FORECAST RESULTS")
    print("=" * 60)
    
    print(f"\n📈 FORECAST SUMMARY:")
    print(f"  • Campaign: {forecast.campaign_name}")
    print(f"  • Forecast Period: {forecast.forecast_period} days")
    print(f"  • Risk Level: {forecast.risk_assessment.get('overall_risk', 'medium').title()}")
    print(f"  • Confidence Score: {forecast.risk_assessment.get('confidence_score', 0.5):.1%}")
    
    # Key performance metrics
    print(f"\n🎯 EXPECTED PERFORMANCE:")
    for metric in ['spend', 'revenue', 'conversions', 'roas']:
        if metric in forecast.predictions:
            total_pred = sum(forecast.predictions[metric])
            avg_pred = np.mean(forecast.predictions[metric])
            
            if metric == 'roas':
                print(f"  • {metric.upper()}: {avg_pred:.2f}x average")
            else:
                print(f"  • Total {metric.title()}: {total_pred:,.2f}")
    
    # Forecast summary metrics
    if forecast.forecast_summary:
        print(f"\n📊 FORECAST HIGHLIGHTS:")
        if 'forecast_roas' in forecast.forecast_summary:
            print(f"  • Forecast ROAS: {forecast.forecast_summary['forecast_roas']:.2f}x")
        
        for key, value in forecast.forecast_summary.items():
            if 'trend' in key and value != 0:
                trend_direction = "↗️ Increasing" if value > 0 else "↘️ Decreasing"
                print(f"  • {key.replace('_', ' ').title()}: {trend_direction} ({value:+.2f}/day)")
    
    # Model accuracy
    if forecast.model_accuracy:
        print(f"\n🎯 MODEL ACCURACY:")
        for metric, accuracy in forecast.model_accuracy.items():
            print(f"  • {metric.title()}: {accuracy:.1%}")
    
    # Risk assessment
    print(f"\n⚠️ RISK ASSESSMENT:")
    print(f"  • Overall Risk: {forecast.risk_assessment.get('overall_risk', 'medium').title()}")
    print(f"  • Volatility Score: {forecast.risk_assessment.get('volatility_score', 0.5):.1%}")
    
    if forecast.risk_assessment.get('risk_factors'):
        print(f"  • Risk Factors: {len(forecast.risk_assessment['risk_factors'])}")
        for factor in forecast.risk_assessment['risk_factors'][:2]:
            print(f"    - {factor}")
    
    # Scenario analysis
    if forecast.scenario_analysis:
        print(f"\n📊 SCENARIO ANALYSIS (Revenue):")
        for scenario, metrics in forecast.scenario_analysis.items():
            revenue = metrics.get('revenue_total', 0)
            print(f"  • {scenario.title()}: ${revenue:,.2f}")
    
    # Optimization recommendations
    if forecast.optimization_recommendations:
        print(f"\n💡 KEY RECOMMENDATIONS:")
        for i, rec in enumerate(forecast.optimization_recommendations[:3], 1):
            print(f"  {i}. {rec}")
    
    # Feature importance (if available)
    if forecast.feature_importance:
        print(f"\n📈 TOP PERFORMANCE DRIVERS:")
        sorted_features = sorted(forecast.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:5]:
            print(f"  • {feature.title()}: {importance:.3f}")
    
    # Confidence intervals for key metrics
    print(f"\n📊 CONFIDENCE INTERVALS (Daily Average):")
    for metric in ['spend', 'revenue', 'conversions']:
        if metric in forecast.confidence_intervals:
            intervals = forecast.confidence_intervals[metric]
            avg_lower = np.mean([ci[0] for ci in intervals])
            avg_upper = np.mean([ci[1] for ci in intervals])
            avg_pred = np.mean(forecast.predictions[metric])
            print(f"  • {metric.title()}: {avg_pred:,.2f} (CI: {avg_lower:,.2f} - {avg_upper:,.2f})")
    
    print("\n" + "="*70)
    print("🚀 Advanced forecasting for data-driven campaign optimization")
    print("💼 Predictive analytics with confidence intervals and risk assessment")
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_campaign_forecaster()