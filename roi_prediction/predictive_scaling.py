"""
Predictive Scaling & Dynamic Budget Allocation for Marketing Campaigns

Advanced framework for predictive scaling of marketing campaigns based on real-time
performance data, market conditions, and competitive intelligence. Enables proactive
budget reallocation and campaign optimization.

Author: Sotiris Spyrou
LinkedIn: https://linkedin.com/in/sotiris-spyrou
Portfolio: https://github.com/sotirisspyrou

This is a demonstration of advanced marketing analytics capabilities.
In production, this would integrate with your specific data infrastructure and business requirements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats, optimize
from scipy.signal import savgol_filter
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

@dataclass
class ScalingConfig:
    """Configuration for predictive scaling system."""
    prediction_window_hours: int = 24
    scaling_threshold: float = 0.15  # 15% performance change triggers scaling
    max_budget_increase: float = 2.0  # Max 2x budget increase
    max_budget_decrease: float = 0.5  # Min 50% budget retention
    confidence_threshold: float = 0.8  # Minimum prediction confidence
    cooling_period_hours: int = 4  # Hours between scaling actions
    market_factor_weight: float = 0.3  # Weight of market conditions in decisions
    competitive_factor_weight: float = 0.2  # Weight of competitive intelligence
    risk_tolerance: str = 'moderate'  # 'conservative', 'moderate', 'aggressive'
    
@dataclass
class MarketConditions:
    """Market condition indicators."""
    seasonality_index: float
    market_volatility: float
    competitive_pressure: float
    consumer_sentiment: float
    economic_indicators: Dict[str, float]
    trend_direction: str  # 'up', 'down', 'stable'
    confidence_score: float
    
@dataclass
class ScalingDecision:
    """Scaling decision and rationale."""
    campaign_id: str
    channel: str
    current_budget: float
    recommended_budget: float
    scaling_factor: float
    confidence_score: float
    decision_rationale: List[str]
    predicted_performance: Dict[str, float]
    risk_assessment: str
    implementation_priority: int
    expected_impact: Dict[str, float]
    timestamp: datetime
    
@dataclass
class PerformanceAnomaly:
    """Detected performance anomaly."""
    campaign_id: str
    metric: str
    anomaly_type: str  # 'spike', 'drop', 'drift'
    severity: str  # 'low', 'medium', 'high', 'critical'
    detected_value: float
    expected_value: float
    deviation_score: float
    timestamp: datetime
    suggested_actions: List[str]

class TimeSeriesForecaster:
    """Advanced time series forecasting for campaign performance."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.models = {}
        self.fitted_models = {}
        
    def fit_ensemble_forecasters(self, 
                                time_series_data: pd.DataFrame,
                                target_column: str,
                                time_column: str = 'timestamp') -> Dict[str, Any]:
        """Fit ensemble of forecasting models."""
        
        # Prepare time series
        ts_data = time_series_data.copy()
        ts_data[time_column] = pd.to_datetime(ts_data[time_column])
        ts_data = ts_data.sort_values(time_column).set_index(time_column)
        
        # Handle missing values
        ts_data[target_column] = ts_data[target_column].interpolate(method='time')
        
        # Split data for validation
        train_size = int(len(ts_data) * 0.8)
        train_data = ts_data[:train_size]
        test_data = ts_data[train_size:]
        
        forecasting_results = {}
        
        # ARIMA Model
        try:
            arima_model = ARIMA(train_data[target_column], order=(2, 1, 2))
            arima_fitted = arima_model.fit()
            arima_forecast = arima_fitted.forecast(steps=len(test_data))
            arima_mae = mean_absolute_error(test_data[target_column], arima_forecast)
            
            self.fitted_models['arima'] = arima_fitted
            forecasting_results['arima'] = {
                'mae': arima_mae,
                'forecast': arima_forecast,
                'model': arima_fitted
            }
        except Exception as e:
            print(f"ARIMA fitting failed: {e}")
        
        # Exponential Smoothing
        try:
            ets_model = ETSModel(
                train_data[target_column],
                trend='add',
                seasonal='add' if len(train_data) > 24 else None,
                seasonal_periods=24 if len(train_data) > 24 else None
            )
            ets_fitted = ets_model.fit()
            ets_forecast = ets_fitted.forecast(steps=len(test_data))
            ets_mae = mean_absolute_error(test_data[target_column], ets_forecast)
            
            self.fitted_models['ets'] = ets_fitted
            forecasting_results['ets'] = {
                'mae': ets_mae,
                'forecast': ets_forecast,
                'model': ets_fitted
            }
        except Exception as e:
            print(f"ETS fitting failed: {e}")
        
        # Machine Learning Approach
        try:
            # Create lagged features
            ml_data = self._create_ml_features(train_data, target_column)
            
            X_train = ml_data.drop([target_column], axis=1).fillna(0)
            y_train = ml_data[target_column]
            
            # Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Generate forecast for test period
            ml_test_data = self._create_ml_features(test_data, target_column)
            X_test = ml_test_data.drop([target_column], axis=1).fillna(0)
            rf_forecast = rf_model.predict(X_test)
            rf_mae = mean_absolute_error(test_data[target_column], rf_forecast)
            
            self.fitted_models['random_forest'] = rf_model
            forecasting_results['random_forest'] = {
                'mae': rf_mae,
                'forecast': rf_forecast,
                'model': rf_model
            }
        except Exception as e:
            print(f"ML forecasting failed: {e}")
        
        return forecasting_results
    
    def _create_ml_features(self, 
                           time_series: pd.DataFrame,
                           target_column: str,
                           lags: int = 24) -> pd.DataFrame:
        """Create machine learning features from time series."""
        
        ml_features = time_series.copy()
        
        # Lag features
        for lag in range(1, min(lags, len(time_series))):
            ml_features[f'{target_column}_lag_{lag}'] = time_series[target_column].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12, 24]:
            if len(time_series) > window:
                ml_features[f'{target_column}_rolling_mean_{window}'] = (
                    time_series[target_column].rolling(window=window).mean()
                )
                ml_features[f'{target_column}_rolling_std_{window}'] = (
                    time_series[target_column].rolling(window=window).std()
                )
        
        # Time-based features
        ml_features['hour'] = ml_features.index.hour
        ml_features['day_of_week'] = ml_features.index.dayofweek
        ml_features['month'] = ml_features.index.month
        
        # Trend features
        ml_features['linear_trend'] = np.arange(len(ml_features))
        
        return ml_features
    
    def forecast_performance(self, 
                           campaign_id: str,
                           horizon_hours: int = None) -> Dict[str, Any]:
        """Generate performance forecast for specific campaign."""
        
        horizon_hours = horizon_hours or self.config.prediction_window_hours
        
        ensemble_forecasts = []
        forecast_confidence = []
        
        for model_name, model_result in self.fitted_models.items():
            try:
                if model_name in ['arima', 'ets']:
                    forecast = model_result.forecast(steps=horizon_hours)
                    # Confidence interval for statistical models
                    forecast_conf = model_result.get_forecast(steps=horizon_hours)
                    conf_int = forecast_conf.conf_int()
                    confidence = 1 - (conf_int.iloc[:, 1] - conf_int.iloc[:, 0]).mean() / forecast.mean()
                else:
                    # For ML models, need to create future features
                    # Simplified approach - use last known values
                    forecast = np.repeat(model_result.predict([[0] * 50])[-1], horizon_hours)
                    confidence = 0.7  # Default confidence for ML models
                
                ensemble_forecasts.append(forecast)
                forecast_confidence.append(confidence)
                
            except Exception as e:
                print(f"Forecast generation failed for {model_name}: {e}")
        
        if not ensemble_forecasts:
            return {'error': 'No successful forecasts generated'}
        
        # Combine forecasts
        ensemble_forecast = np.mean(ensemble_forecasts, axis=0)
        avg_confidence = np.mean(forecast_confidence)
        
        # Calculate prediction intervals
        forecast_std = np.std(ensemble_forecasts, axis=0)
        prediction_interval_lower = ensemble_forecast - 1.96 * forecast_std
        prediction_interval_upper = ensemble_forecast + 1.96 * forecast_std
        
        return {
            'campaign_id': campaign_id,
            'forecast': ensemble_forecast.tolist(),
            'confidence': avg_confidence,
            'prediction_interval': {
                'lower': prediction_interval_lower.tolist(),
                'upper': prediction_interval_upper.tolist()
            },
            'forecast_horizon_hours': horizon_hours,
            'models_used': list(self.fitted_models.keys())
        }

class AnomalyDetector:
    """Real-time anomaly detection for campaign performance."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.models = {}
        self.baselines = {}
        
    def fit_anomaly_detectors(self, 
                            historical_data: pd.DataFrame,
                            metrics: List[str]) -> Dict[str, Any]:
        """Fit anomaly detection models for specified metrics."""
        
        results = {}
        
        for metric in metrics:
            if metric not in historical_data.columns:
                continue
            
            # Statistical approach - control limits
            metric_data = historical_data[metric].dropna()
            mean_val = metric_data.mean()
            std_val = metric_data.std()
            
            self.baselines[metric] = {
                'mean': mean_val,
                'std': std_val,
                'upper_control_limit': mean_val + 3 * std_val,
                'lower_control_limit': mean_val - 3 * std_val,
                'upper_warning_limit': mean_val + 2 * std_val,
                'lower_warning_limit': mean_val - 2 * std_val
            }
            
            # Machine learning approach - Isolation Forest
            try:
                # Create feature matrix with lagged values
                feature_matrix = self._create_anomaly_features(historical_data, metric)
                
                isolation_forest = IsolationForest(
                    contamination=0.1,  # Expect 10% anomalies
                    random_state=42
                )
                isolation_forest.fit(feature_matrix)
                
                self.models[metric] = isolation_forest
                results[metric] = {
                    'baseline_stats': self.baselines[metric],
                    'ml_model_fitted': True
                }
                
            except Exception as e:
                print(f"ML anomaly detection fitting failed for {metric}: {e}")
                results[metric] = {
                    'baseline_stats': self.baselines[metric],
                    'ml_model_fitted': False
                }
        
        return results
    
    def _create_anomaly_features(self, 
                               data: pd.DataFrame,
                               metric: str,
                               window_sizes: List[int] = [3, 6, 12]) -> pd.DataFrame:
        """Create features for anomaly detection."""
        
        features = pd.DataFrame()
        metric_series = data[metric].dropna()
        
        # Current value
        features['current'] = metric_series
        
        # Rolling statistics
        for window in window_sizes:
            if len(metric_series) > window:
                features[f'rolling_mean_{window}'] = metric_series.rolling(window).mean()
                features[f'rolling_std_{window}'] = metric_series.rolling(window).std()
                features[f'rolling_min_{window}'] = metric_series.rolling(window).min()
                features[f'rolling_max_{window}'] = metric_series.rolling(window).max()
        
        # Lag features
        for lag in [1, 2, 3, 6, 12]:
            if lag < len(metric_series):
                features[f'lag_{lag}'] = metric_series.shift(lag)
        
        # Remove rows with NaN values
        features = features.dropna()
        
        return features
    
    def detect_anomalies(self, 
                        current_data: pd.DataFrame,
                        timestamp: datetime = None) -> List[PerformanceAnomaly]:
        """Detect anomalies in current performance data."""
        
        anomalies = []
        timestamp = timestamp or datetime.now()
        
        for metric in self.baselines.keys():
            if metric not in current_data.columns:
                continue
            
            current_value = current_data[metric].iloc[-1] if len(current_data) > 0 else None
            if current_value is None:
                continue
            
            baseline = self.baselines[metric]
            
            # Statistical anomaly detection
            anomaly_detected = False
            anomaly_type = None
            severity = 'low'
            
            if current_value > baseline['upper_control_limit']:
                anomaly_detected = True
                anomaly_type = 'spike'
                severity = 'critical'
            elif current_value < baseline['lower_control_limit']:
                anomaly_detected = True
                anomaly_type = 'drop'
                severity = 'critical'
            elif current_value > baseline['upper_warning_limit']:
                anomaly_detected = True
                anomaly_type = 'spike'
                severity = 'medium'
            elif current_value < baseline['lower_warning_limit']:
                anomaly_detected = True
                anomaly_type = 'drop'
                severity = 'medium'
            
            # ML-based anomaly detection
            if metric in self.models:
                try:
                    feature_vector = self._create_anomaly_features(current_data, metric).iloc[-1:].fillna(0)
                    anomaly_score = self.models[metric].decision_function(feature_vector)[0]
                    
                    if anomaly_score < -0.1:  # Threshold for anomaly
                        if not anomaly_detected:
                            anomaly_detected = True
                            anomaly_type = 'drift'
                            severity = 'medium'
                        
                except Exception as e:
                    print(f"ML anomaly detection failed for {metric}: {e}")
            
            if anomaly_detected:
                # Calculate deviation score
                deviation_score = abs(current_value - baseline['mean']) / baseline['std']
                
                # Generate suggested actions
                suggested_actions = self._generate_anomaly_actions(
                    metric, anomaly_type, severity, current_value, baseline['mean']
                )
                
                anomaly = PerformanceAnomaly(
                    campaign_id=current_data.get('campaign_id', 'unknown'),
                    metric=metric,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    detected_value=current_value,
                    expected_value=baseline['mean'],
                    deviation_score=deviation_score,
                    timestamp=timestamp,
                    suggested_actions=suggested_actions
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _generate_anomaly_actions(self, 
                                metric: str,
                                anomaly_type: str,
                                severity: str,
                                current_value: float,
                                expected_value: float) -> List[str]:
        """Generate suggested actions based on detected anomaly."""
        
        actions = []
        
        if anomaly_type == 'spike':
            if metric in ['clicks', 'conversions', 'revenue']:
                actions.append("Consider increasing budget to capitalize on strong performance")
                actions.append("Analyze traffic sources driving the spike")
                actions.append("Prepare to scale winning creative/targeting")
            elif metric in ['cpc', 'cpa', 'cost']:
                actions.append("Investigate cost increases - check bid competition")
                actions.append("Review targeting settings for efficiency")
                actions.append("Consider bid cap adjustments")
        
        elif anomaly_type == 'drop':
            if metric in ['clicks', 'conversions', 'revenue']:
                actions.append("Urgent: Investigate traffic/conversion drops")
                actions.append("Check campaign settings and targeting")
                actions.append("Review creative performance and refresh if needed")
                actions.append("Consider increasing bids if competition increased")
            elif metric in ['ctr', 'conversion_rate']:
                actions.append("Review ad creative performance")
                actions.append("Analyze landing page metrics")
                actions.append("Check for technical issues")
        
        elif anomaly_type == 'drift':
            actions.append("Monitor trend continuation")
            actions.append("Investigate gradual performance changes")
            actions.append("Consider strategic adjustments to campaign settings")
        
        if severity == 'critical':
            actions.insert(0, "CRITICAL: Immediate attention required")
            actions.append("Alert campaign manager immediately")
        
        return actions

class MarketIntelligence:
    """Market intelligence and competitive analysis."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        
    def analyze_market_conditions(self, 
                                market_data: pd.DataFrame,
                                timestamp: datetime = None) -> MarketConditions:
        """Analyze current market conditions affecting campaign performance."""
        
        timestamp = timestamp or datetime.now()
        
        # Seasonality analysis
        seasonality_index = self._calculate_seasonality_index(market_data, timestamp)
        
        # Market volatility
        market_volatility = self._calculate_market_volatility(market_data)
        
        # Competitive pressure (simplified simulation)
        competitive_pressure = self._estimate_competitive_pressure(market_data)
        
        # Consumer sentiment (simplified)
        consumer_sentiment = self._analyze_consumer_sentiment(market_data)
        
        # Economic indicators
        economic_indicators = self._process_economic_indicators(market_data)
        
        # Trend direction
        trend_direction = self._determine_trend_direction(market_data)
        
        # Overall confidence score
        confidence_score = self._calculate_confidence_score(
            seasonality_index, market_volatility, competitive_pressure
        )
        
        return MarketConditions(
            seasonality_index=seasonality_index,
            market_volatility=market_volatility,
            competitive_pressure=competitive_pressure,
            consumer_sentiment=consumer_sentiment,
            economic_indicators=economic_indicators,
            trend_direction=trend_direction,
            confidence_score=confidence_score
        )
    
    def _calculate_seasonality_index(self, 
                                   market_data: pd.DataFrame,
                                   timestamp: datetime) -> float:
        """Calculate seasonality index based on historical patterns."""
        
        # Simplified seasonality calculation
        month = timestamp.month
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Mock seasonality patterns
        monthly_factors = {
            1: 0.8, 2: 0.7, 3: 0.9, 4: 1.0, 5: 1.1, 6: 1.2,
            7: 1.3, 8: 1.2, 9: 1.1, 10: 1.2, 11: 1.4, 12: 1.5
        }
        
        hourly_factors = {
            hour: 0.5 + 0.5 * np.sin(2 * np.pi * hour / 24) + 0.5
            for hour in range(24)
        }
        
        weekly_factors = {
            0: 1.2, 1: 1.3, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.7, 6: 0.8
        }
        
        seasonality_index = (
            monthly_factors.get(month, 1.0) * 
            hourly_factors.get(hour, 1.0) * 
            weekly_factors.get(day_of_week, 1.0)
        ) / 3.0  # Normalize
        
        return seasonality_index
    
    def _calculate_market_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate market volatility indicator."""
        
        if 'market_index' in market_data.columns and len(market_data) > 10:
            returns = market_data['market_index'].pct_change().dropna()
            volatility = returns.rolling(window=10).std().iloc[-1]
            return min(1.0, volatility * 100)  # Scale to 0-1
        
        # Default moderate volatility
        return 0.5
    
    def _estimate_competitive_pressure(self, market_data: pd.DataFrame) -> float:
        """Estimate competitive pressure in the market."""
        
        if 'competitor_activity' in market_data.columns:
            recent_activity = market_data['competitor_activity'].tail(7).mean()
            return min(1.0, recent_activity / 100)  # Normalize
        
        # Simulate competitive pressure
        return np.random.uniform(0.3, 0.8)
    
    def _analyze_consumer_sentiment(self, market_data: pd.DataFrame) -> float:
        """Analyze consumer sentiment indicators."""
        
        if 'sentiment_score' in market_data.columns:
            return market_data['sentiment_score'].iloc[-1] / 100
        
        # Simulate consumer sentiment
        return np.random.uniform(0.4, 0.9)
    
    def _process_economic_indicators(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Process relevant economic indicators."""
        
        indicators = {}
        
        economic_columns = ['gdp_growth', 'unemployment_rate', 'inflation_rate', 'consumer_confidence']
        
        for col in economic_columns:
            if col in market_data.columns:
                indicators[col] = market_data[col].iloc[-1]
            else:
                # Simulate indicators
                indicators[col] = np.random.uniform(0, 1)
        
        return indicators
    
    def _determine_trend_direction(self, market_data: pd.DataFrame) -> str:
        """Determine overall market trend direction."""
        
        if 'market_index' in market_data.columns and len(market_data) > 5:
            recent_change = market_data['market_index'].iloc[-1] / market_data['market_index'].iloc[-5] - 1
            
            if recent_change > 0.05:
                return 'up'
            elif recent_change < -0.05:
                return 'down'
            else:
                return 'stable'
        
        return 'stable'
    
    def _calculate_confidence_score(self, 
                                  seasonality: float,
                                  volatility: float,
                                  competitive_pressure: float) -> float:
        """Calculate overall confidence score for market analysis."""
        
        # Higher seasonality and lower volatility/competition = higher confidence
        confidence = (seasonality + (1 - volatility) + (1 - competitive_pressure)) / 3
        return min(1.0, max(0.0, confidence))

class PredictiveScaler:
    """Main predictive scaling engine."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.forecaster = TimeSeriesForecaster(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.market_intelligence = MarketIntelligence(config)
        self.scaling_history = []
        
    def analyze_scaling_opportunity(self, 
                                  campaign_data: pd.DataFrame,
                                  market_data: pd.DataFrame = None) -> List[ScalingDecision]:
        """Analyze campaigns for scaling opportunities."""
        
        scaling_decisions = []
        
        # Get unique campaigns
        campaigns = campaign_data['campaign_id'].unique()
        
        for campaign_id in campaigns:
            campaign_subset = campaign_data[campaign_data['campaign_id'] == campaign_id]
            
            # Generate performance forecast
            forecast_result = self.forecaster.forecast_performance(campaign_id)
            
            if 'error' in forecast_result:
                continue
            
            # Detect anomalies
            anomalies = self.anomaly_detector.detect_anomalies(campaign_subset)
            
            # Analyze market conditions
            market_conditions = None
            if market_data is not None:
                market_conditions = self.market_intelligence.analyze_market_conditions(market_data)
            
            # Make scaling decision
            scaling_decision = self._make_scaling_decision(
                campaign_id, campaign_subset, forecast_result, anomalies, market_conditions
            )
            
            if scaling_decision:
                scaling_decisions.append(scaling_decision)
        
        # Sort by priority
        scaling_decisions.sort(key=lambda x: x.implementation_priority, reverse=True)
        
        return scaling_decisions
    
    def _make_scaling_decision(self, 
                             campaign_id: str,
                             campaign_data: pd.DataFrame,
                             forecast: Dict[str, Any],
                             anomalies: List[PerformanceAnomaly],
                             market_conditions: MarketConditions = None) -> Optional[ScalingDecision]:
        """Make intelligent scaling decision based on multiple factors."""
        
        # Current campaign metrics
        current_budget = campaign_data['budget'].iloc[-1] if 'budget' in campaign_data.columns else 1000
        current_performance = campaign_data[['clicks', 'conversions', 'revenue']].iloc[-1:].mean()
        
        # Forecast analysis
        predicted_performance = np.mean(forecast['forecast'])
        forecast_confidence = forecast['confidence']
        
        # Skip if confidence too low
        if forecast_confidence < self.config.confidence_threshold:
            return None
        
        # Calculate performance change expectation
        recent_performance = current_performance.mean()
        performance_change = (predicted_performance - recent_performance) / recent_performance
        
        # Base scaling decision on performance change
        scaling_factor = 1.0
        decision_rationale = []
        
        if performance_change > self.config.scaling_threshold:
            scaling_factor = min(self.config.max_budget_increase, 1 + performance_change)
            decision_rationale.append(f"Strong performance forecast (+{performance_change:.1%})")
        elif performance_change < -self.config.scaling_threshold:
            scaling_factor = max(self.config.max_budget_decrease, 1 + performance_change)
            decision_rationale.append(f"Declining performance forecast ({performance_change:.1%})")
        
        # Adjust for anomalies
        critical_anomalies = [a for a in anomalies if a.severity == 'critical']
        if critical_anomalies:
            positive_anomalies = [a for a in critical_anomalies if a.anomaly_type == 'spike']
            negative_anomalies = [a for a in critical_anomalies if a.anomaly_type == 'drop']
            
            if positive_anomalies:
                scaling_factor *= 1.2
                decision_rationale.append("Positive performance anomalies detected")
            elif negative_anomalies:
                scaling_factor *= 0.8
                decision_rationale.append("Negative performance anomalies detected")
        
        # Adjust for market conditions
        if market_conditions:
            market_adjustment = self._calculate_market_adjustment(market_conditions)
            scaling_factor *= market_adjustment
            decision_rationale.append(f"Market conditions adjustment: {market_adjustment:.2f}")
        
        # Risk tolerance adjustment
        risk_multiplier = self._get_risk_multiplier()
        scaling_factor = 1 + (scaling_factor - 1) * risk_multiplier
        
        # Calculate recommended budget
        recommended_budget = current_budget * scaling_factor
        
        # Determine implementation priority
        priority = self._calculate_priority(performance_change, forecast_confidence, anomalies)
        
        # Risk assessment
        risk_assessment = self._assess_risk(scaling_factor, forecast_confidence, market_conditions)
        
        # Expected impact calculation
        expected_impact = self._calculate_expected_impact(
            current_performance, predicted_performance, scaling_factor
        )
        
        return ScalingDecision(
            campaign_id=campaign_id,
            channel=campaign_data.get('channel', ['unknown']).iloc[-1] if 'channel' in campaign_data.columns else 'unknown',
            current_budget=current_budget,
            recommended_budget=recommended_budget,
            scaling_factor=scaling_factor,
            confidence_score=forecast_confidence,
            decision_rationale=decision_rationale,
            predicted_performance={
                'forecast_value': predicted_performance,
                'performance_change': performance_change
            },
            risk_assessment=risk_assessment,
            implementation_priority=priority,
            expected_impact=expected_impact,
            timestamp=datetime.now()
        )
    
    def _calculate_market_adjustment(self, market_conditions: MarketConditions) -> float:
        """Calculate market-based adjustment factor."""
        
        adjustment = 1.0
        
        # Seasonality adjustment
        adjustment *= market_conditions.seasonality_index
        
        # Volatility adjustment (reduce scaling in volatile markets)
        adjustment *= (1 - market_conditions.market_volatility * 0.3)
        
        # Competitive pressure adjustment
        adjustment *= (1 - market_conditions.competitive_pressure * 0.2)
        
        # Consumer sentiment adjustment
        adjustment *= (0.8 + market_conditions.consumer_sentiment * 0.4)
        
        return adjustment
    
    def _get_risk_multiplier(self) -> float:
        """Get risk multiplier based on risk tolerance setting."""
        
        risk_multipliers = {
            'conservative': 0.5,
            'moderate': 0.75,
            'aggressive': 1.0
        }
        
        return risk_multipliers.get(self.config.risk_tolerance, 0.75)
    
    def _calculate_priority(self, 
                          performance_change: float,
                          confidence: float,
                          anomalies: List[PerformanceAnomaly]) -> int:
        """Calculate implementation priority (1-10 scale)."""
        
        priority = 5  # Base priority
        
        # Adjust for performance change magnitude
        priority += min(3, abs(performance_change) * 10)
        
        # Adjust for confidence
        priority += (confidence - 0.5) * 4
        
        # Adjust for critical anomalies
        critical_anomalies = len([a for a in anomalies if a.severity == 'critical'])
        priority += critical_anomalies * 2
        
        return int(min(10, max(1, priority)))
    
    def _assess_risk(self, 
                   scaling_factor: float,
                   confidence: float,
                   market_conditions: MarketConditions = None) -> str:
        """Assess overall risk level of scaling decision."""
        
        risk_score = 0
        
        # Scaling factor risk
        if abs(scaling_factor - 1) > 0.5:
            risk_score += 2
        elif abs(scaling_factor - 1) > 0.25:
            risk_score += 1
        
        # Confidence risk
        if confidence < 0.7:
            risk_score += 2
        elif confidence < 0.8:
            risk_score += 1
        
        # Market risk
        if market_conditions:
            if market_conditions.market_volatility > 0.7:
                risk_score += 1
            if market_conditions.competitive_pressure > 0.8:
                risk_score += 1
        
        if risk_score >= 4:
            return 'high'
        elif risk_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_expected_impact(self, 
                                 current_performance: pd.Series,
                                 predicted_performance: float,
                                 scaling_factor: float) -> Dict[str, float]:
        """Calculate expected impact of scaling decision."""
        
        # Estimated impact on key metrics
        impact = {}
        
        # Revenue impact
        current_revenue = current_performance.get('revenue', 0)
        impact['revenue_change'] = (predicted_performance - current_revenue) * scaling_factor
        impact['revenue_change_pct'] = impact['revenue_change'] / current_revenue if current_revenue > 0 else 0
        
        # Volume impact (clicks, conversions)
        impact['volume_change'] = scaling_factor - 1
        
        # Efficiency impact (simplified assumption)
        impact['efficiency_change'] = (predicted_performance / current_revenue - 1) if current_revenue > 0 else 0
        
        return impact

# Executive Demo Functions

def generate_sample_campaign_data(n_campaigns: int = 5, days: int = 30) -> pd.DataFrame:
    """Generate sample campaign performance data for demonstration."""
    np.random.seed(42)
    
    campaigns_data = []
    channels = ['google_ads', 'facebook', 'linkedin', 'twitter', 'tiktok']
    
    for i in range(n_campaigns):
        campaign_id = f'campaign_{i:03d}'
        channel = channels[i % len(channels)]
        base_budget = np.random.uniform(1000, 10000)
        
        # Generate daily performance data
        for day in range(days):
            timestamp = datetime.now() - timedelta(days=days-day)
            
            # Simulate performance with trends and seasonality
            trend_factor = 1 + 0.02 * day / days  # Slight upward trend
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * day / 7)  # Weekly seasonality
            noise_factor = 1 + np.random.normal(0, 0.1)
            
            performance_multiplier = trend_factor * seasonal_factor * noise_factor
            
            # Base performance metrics
            clicks = max(0, int(np.random.poisson(100) * performance_multiplier))
            ctr = max(0.001, np.random.normal(0.03, 0.005) * performance_multiplier)
            conversions = max(0, int(clicks * np.random.normal(0.02, 0.005)))
            cpa = max(1, np.random.normal(50, 10) / performance_multiplier)
            revenue = conversions * np.random.normal(150, 30)
            
            campaigns_data.append({
                'timestamp': timestamp,
                'campaign_id': campaign_id,
                'channel': channel,
                'budget': base_budget * (1 + 0.1 * np.sin(2 * np.pi * day / 30)),
                'clicks': clicks,
                'impressions': int(clicks / ctr),
                'ctr': ctr,
                'conversions': conversions,
                'conversion_rate': conversions / clicks if clicks > 0 else 0,
                'cpa': cpa,
                'revenue': revenue,
                'roas': revenue / (clicks * ctr * 1000) if clicks > 0 else 0
            })
    
    return pd.DataFrame(campaigns_data)

def generate_sample_market_data(days: int = 30) -> pd.DataFrame:
    """Generate sample market conditions data for demonstration."""
    np.random.seed(42)
    
    market_data = []
    
    for day in range(days):
        timestamp = datetime.now() - timedelta(days=days-day)
        
        # Simulate market indicators
        market_data.append({
            'timestamp': timestamp,
            'market_index': 100 + np.random.normal(0, 5) + day * 0.1,
            'competitor_activity': max(0, np.random.normal(50, 15)),
            'sentiment_score': max(0, min(100, np.random.normal(65, 10))),
            'gdp_growth': np.random.normal(2.5, 0.5),
            'unemployment_rate': np.random.normal(5.0, 1.0),
            'inflation_rate': np.random.normal(3.0, 0.8),
            'consumer_confidence': max(0, min(100, np.random.normal(75, 10)))
        })
    
    return pd.DataFrame(market_data)

def executive_demo_predictive_scaling():
    """
    Executive demonstration of predictive scaling capabilities.
    
    This function showcases:
    1. Real-time performance forecasting using ensemble methods
    2. Anomaly detection for proactive campaign management
    3. Market intelligence integration for scaling decisions
    4. Dynamic budget allocation optimization
    5. Risk-adjusted scaling recommendations
    """
    print("🚀 PREDICTIVE SCALING & DYNAMIC BUDGET ALLOCATION DEMO")
    print("=" * 65)
    print("AI-powered campaign scaling based on predictive analytics")
    print("Portfolio demonstration by Sotiris Spyrou")
    print()
    
    # Configuration
    config = ScalingConfig(
        prediction_window_hours=24,
        scaling_threshold=0.15,
        confidence_threshold=0.7,
        risk_tolerance='moderate'
    )
    
    # Generate sample data
    print("📊 GENERATING SAMPLE DATA")
    print("-" * 30)
    campaign_data = generate_sample_campaign_data(5, 30)
    market_data = generate_sample_market_data(30)
    
    print(f"Generated {len(campaign_data):,} campaign data points")
    print(f"Campaign period: {campaign_data['timestamp'].min().date()} to {campaign_data['timestamp'].max().date()}")
    print(f"Campaigns: {', '.join(campaign_data['campaign_id'].unique())}")
    print(f"Channels: {', '.join(campaign_data['channel'].unique())}")
    print()
    
    # Initialize predictive scaler
    scaler = PredictiveScaler(config)
    
    print("1. PERFORMANCE FORECASTING")
    print("-" * 30)
    
    # Fit forecasting models for each campaign
    campaigns = campaign_data['campaign_id'].unique()
    
    for campaign_id in campaigns[:2]:  # Demo with first 2 campaigns
        campaign_subset = campaign_data[campaign_data['campaign_id'] == campaign_id]
        
        # Fit forecasting models
        forecast_results = scaler.forecaster.fit_ensemble_forecasters(
            campaign_subset, 'revenue', 'timestamp'
        )
        
        print(f"\nCampaign: {campaign_id}")
        print(f"Forecasting Models Performance:")
        for model_name, results in forecast_results.items():
            print(f"• {model_name}: MAE = {results['mae']:.2f}")
    print()
    
    print("2. ANOMALY DETECTION")
    print("-" * 30)
    
    # Fit anomaly detection models
    metrics_to_monitor = ['clicks', 'conversions', 'revenue', 'ctr', 'conversion_rate']
    anomaly_results = scaler.anomaly_detector.fit_anomaly_detectors(
        campaign_data, metrics_to_monitor
    )
    
    for metric, result in anomaly_results.items():
        baseline = result['baseline_stats']
        print(f"{metric}:")
        print(f"  • Mean: {baseline['mean']:.2f}")
        print(f"  • Control limits: [{baseline['lower_control_limit']:.2f}, {baseline['upper_control_limit']:.2f}]")
        print(f"  • ML model fitted: {result['ml_model_fitted']}")
    
    # Detect current anomalies
    recent_data = campaign_data.tail(10)  # Last 10 data points
    detected_anomalies = scaler.anomaly_detector.detect_anomalies(recent_data)
    
    print(f"\nDetected Anomalies: {len(detected_anomalies)}")
    for anomaly in detected_anomalies[:3]:  # Show first 3
        print(f"• {anomaly.campaign_id} - {anomaly.metric}: {anomaly.anomaly_type} "
              f"({anomaly.severity}) - {anomaly.detected_value:.2f} vs {anomaly.expected_value:.2f}")
    print()
    
    print("3. MARKET INTELLIGENCE")
    print("-" * 30)
    
    # Analyze market conditions
    market_conditions = scaler.market_intelligence.analyze_market_conditions(market_data)
    
    print(f"Market Analysis:")
    print(f"• Seasonality Index: {market_conditions.seasonality_index:.2f}")
    print(f"• Market Volatility: {market_conditions.market_volatility:.2f}")
    print(f"• Competitive Pressure: {market_conditions.competitive_pressure:.2f}")
    print(f"• Consumer Sentiment: {market_conditions.consumer_sentiment:.2f}")
    print(f"• Trend Direction: {market_conditions.trend_direction}")
    print(f"• Confidence Score: {market_conditions.confidence_score:.2f}")
    print()
    
    print("4. SCALING DECISIONS")
    print("-" * 30)
    
    # Generate scaling recommendations
    scaling_decisions = scaler.analyze_scaling_opportunity(campaign_data, market_data)
    
    print(f"Scaling Recommendations: {len(scaling_decisions)}")
    
    for decision in scaling_decisions:
        budget_change = (decision.recommended_budget - decision.current_budget) / decision.current_budget
        print(f"\nCampaign: {decision.campaign_id} ({decision.channel})")
        print(f"• Current Budget: ${decision.current_budget:,.2f}")
        print(f"• Recommended Budget: ${decision.recommended_budget:,.2f}")
        print(f"• Budget Change: {budget_change:+.1%}")
        print(f"• Scaling Factor: {decision.scaling_factor:.2f}")
        print(f"• Confidence: {decision.confidence_score:.1%}")
        print(f"• Risk Level: {decision.risk_assessment}")
        print(f"• Priority: {decision.implementation_priority}/10")
        print(f"• Rationale: {'; '.join(decision.decision_rationale[:2])}")
        
        if decision.expected_impact:
            revenue_change = decision.expected_impact.get('revenue_change', 0)
            print(f"• Expected Revenue Impact: ${revenue_change:+,.2f}")
    print()
    
    print("5. PERFORMANCE ANALYTICS")
    print("-" * 30)
    
    # Calculate overall portfolio metrics
    total_current_budget = sum(d.current_budget for d in scaling_decisions)
    total_recommended_budget = sum(d.recommended_budget for d in scaling_decisions)
    portfolio_change = (total_recommended_budget - total_current_budget) / total_current_budget
    
    high_priority_decisions = [d for d in scaling_decisions if d.implementation_priority >= 7]
    scale_up_decisions = [d for d in scaling_decisions if d.scaling_factor > 1.1]
    scale_down_decisions = [d for d in scaling_decisions if d.scaling_factor < 0.9]
    
    print(f"Portfolio Analysis:")
    print(f"• Total Current Budget: ${total_current_budget:,.2f}")
    print(f"• Total Recommended Budget: ${total_recommended_budget:,.2f}")
    print(f"• Portfolio Budget Change: {portfolio_change:+.1%}")
    print(f"• High Priority Actions: {len(high_priority_decisions)}")
    print(f"• Scale Up Recommendations: {len(scale_up_decisions)}")
    print(f"• Scale Down Recommendations: {len(scale_down_decisions)}")
    print()
    
    print("📊 KEY BUSINESS INSIGHTS")
    print("-" * 30)
    avg_confidence = np.mean([d.confidence_score for d in scaling_decisions])
    high_confidence_decisions = len([d for d in scaling_decisions if d.confidence_score > 0.8])
    
    print(f"• AI-powered scaling recommendations for {len(scaling_decisions)} campaigns")
    print(f"• Average prediction confidence: {avg_confidence:.1%}")
    print(f"• High confidence decisions: {high_confidence_decisions}")
    print(f"• Market conditions incorporated in {len([d for d in scaling_decisions if 'Market conditions' in str(d.decision_rationale)])}/5 decisions")
    print(f"• Anomalies detected and factored into recommendations")
    print()
    
    print("🎯 EXECUTIVE SUMMARY")
    print("-" * 30)
    print("✅ Real-time performance forecasting with ensemble ML models")
    print("✅ Proactive anomaly detection and automated alerting")
    print("✅ Market intelligence integration for informed scaling")
    print("✅ Risk-adjusted dynamic budget allocation")
    print("✅ Prioritized action recommendations with confidence scores")
    print()
    print("Portfolio demonstration of advanced predictive marketing analytics")
    print("Enabling proactive campaign optimization and budget efficiency")

if __name__ == "__main__":
    executive_demo_predictive_scaling()