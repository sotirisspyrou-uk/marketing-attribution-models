"""
Customer Lifetime Value Prediction for Marketing Attribution

Advanced CLV prediction framework that integrates with attribution models to optimize 
customer acquisition and retention strategies. Combines statistical methods with 
machine learning for accurate revenue forecasting.

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
from scipy.special import gamma, gammaln
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, XGBRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

@dataclass
class CLVModelConfig:
    """Configuration for CLV modeling."""
    prediction_horizon_months: int = 24
    confidence_level: float = 0.95
    discount_rate: float = 0.1  # Annual discount rate
    model_type: str = 'btyd'  # 'btyd', 'ml_ensemble', 'hybrid'
    btyd_model: str = 'bg_nbd'  # 'bg_nbd', 'pareto_nbd', 'modified_bg_nbd'
    gamma_gamma_model: bool = True  # For monetary value prediction
    cohort_analysis: bool = True
    segmentation_method: str = 'rfm'  # 'rfm', 'behavioral', 'clustering'
    
@dataclass
class CustomerSegment:
    """Customer segment definition."""
    segment_id: str
    segment_name: str
    customers: List[str]
    characteristics: Dict[str, Any]
    avg_clv: float
    clv_confidence_interval: Tuple[float, float]
    acquisition_cost: float
    retention_strategies: List[str]
    
@dataclass
class CLVPrediction:
    """Individual customer CLV prediction result."""
    customer_id: str
    predicted_clv: float
    clv_confidence_interval: Tuple[float, float]
    prediction_components: Dict[str, float]
    segment: str
    acquisition_channel: str
    days_since_first_purchase: int
    predicted_transactions: float
    predicted_monetary_value: float
    churn_probability: float
    model_used: str
    prediction_date: datetime
    
@dataclass
class CohortAnalysis:
    """Cohort analysis results."""
    cohort_table: pd.DataFrame
    retention_rates: pd.DataFrame
    revenue_cohorts: pd.DataFrame
    clv_by_cohort: pd.DataFrame
    cohort_insights: Dict[str, Any]

class BuyTillYouDieModels:
    """Buy Till You Die (BTYD) models for CLV prediction."""
    
    def __init__(self, config: CLVModelConfig):
        self.config = config
        self.bg_nbd_params = None
        self.gamma_gamma_params = None
        self.fitted = False
        
    def fit_bg_nbd_model(self, 
                        frequency: np.ndarray,
                        recency: np.ndarray,
                        T: np.ndarray) -> Dict[str, float]:
        """Fit BG/NBD (Beta Geometric/Negative Binomial Distribution) model."""
        
        def log_likelihood(params):
            r, alpha, a, b = params
            
            # Ensure parameters are positive
            if any(p <= 0 for p in params):
                return -np.inf
            
            # Log-likelihood calculation for BG/NBD model
            ll = 0
            
            for i in range(len(frequency)):
                x, t_x, T_i = frequency[i], recency[i], T[i]
                
                if x == 0:
                    # Customer made no repeat purchases
                    ll += np.log((a / (a + b)) * ((alpha / (alpha + T_i)) ** r))
                else:
                    # Customer made x repeat purchases
                    term1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
                    term2 = gammaln(a + 1) - gammaln(a) + gammaln(b + x) - gammaln(b)
                    term3 = -gammaln(a + b + x + 1) + gammaln(a + b)
                    term4 = np.log((alpha + T_i) ** (-r - x))
                    
                    # Probability of being alive
                    prob_alive = 1 / (1 + (b / (b + x)) * ((alpha + T_i) / alpha) ** (r + x))
                    term5 = np.log(prob_alive + (1 - prob_alive) * ((alpha + t_x) / (alpha + T_i)) ** (r + x))
                    
                    ll += term1 + term2 + term3 + term4 + term5
            
            return ll
        
        # Initial parameter guesses
        initial_params = [1.0, 1.0, 1.0, 1.0]
        
        # Optimize parameters
        result = optimize.minimize(
            lambda params: -log_likelihood(params),
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 10000}
        )
        
        if result.success:
            self.bg_nbd_params = result.x
            return {
                'r': result.x[0],
                'alpha': result.x[1], 
                'a': result.x[2],
                'b': result.x[3],
                'log_likelihood': -result.fun
            }
        else:
            raise ValueError("BG/NBD model fitting failed")
    
    def fit_gamma_gamma_model(self, 
                            frequency: np.ndarray,
                            monetary_value: np.ndarray) -> Dict[str, float]:
        """Fit Gamma-Gamma model for monetary value prediction."""
        
        # Filter customers with at least one repeat purchase
        valid_customers = frequency > 0
        freq_filtered = frequency[valid_customers]
        monetary_filtered = monetary_value[valid_customers]
        
        def log_likelihood(params):
            p, q, v = params
            
            if any(p <= 0 for p in params):
                return -np.inf
            
            ll = 0
            for i in range(len(freq_filtered)):
                x, m = freq_filtered[i], monetary_filtered[i]
                
                term1 = gammaln(p * x + q) - gammaln(p) - gammaln(q)
                term2 = (q - 1) * np.log(x) + (p * x - 1) * np.log(m)
                term3 = (p * x) * np.log(v) - (p * x + q) * np.log(v + x * m)
                
                ll += term1 + term2 + term3
            
            return ll
        
        # Initial parameter guesses  
        initial_params = [1.0, 1.0, 1.0]
        
        result = optimize.minimize(
            lambda params: -log_likelihood(params),
            initial_params,
            method='Nelder-Mead'
        )
        
        if result.success:
            self.gamma_gamma_params = result.x
            return {
                'p': result.x[0],
                'q': result.x[1],
                'v': result.x[2],
                'log_likelihood': -result.fun
            }
        else:
            raise ValueError("Gamma-Gamma model fitting failed")
    
    def predict_future_transactions(self, 
                                  frequency: np.ndarray,
                                  recency: np.ndarray,
                                  T: np.ndarray,
                                  future_t: float) -> np.ndarray:
        """Predict number of transactions in future period."""
        if self.bg_nbd_params is None:
            raise ValueError("BG/NBD model not fitted")
        
        r, alpha, a, b = self.bg_nbd_params
        
        predictions = []
        for i in range(len(frequency)):
            x, t_x, T_i = frequency[i], recency[i], T[i]
            
            if x == 0:
                # No repeat purchases yet
                prob_alive = a / (a + b)
                expected_purchases = prob_alive * (r / alpha) * future_t
            else:
                # Has made repeat purchases
                prob_alive = 1 / (1 + (b / (b + x)) * ((alpha + T_i) / alpha) ** (r + x))
                expected_purchases = prob_alive * (r + x) / (alpha + T_i) * future_t
            
            predictions.append(max(0, expected_purchases))
        
        return np.array(predictions)
    
    def predict_average_order_value(self, 
                                   frequency: np.ndarray,
                                   monetary_value: np.ndarray) -> np.ndarray:
        """Predict average order value using Gamma-Gamma model."""
        if self.gamma_gamma_params is None:
            raise ValueError("Gamma-Gamma model not fitted")
        
        p, q, v = self.gamma_gamma_params
        
        predictions = []
        for i in range(len(frequency)):
            x, m = frequency[i], monetary_value[i]
            
            if x > 0:
                # Expected monetary value for repeat customers
                predicted_aov = (q - 1) / (p * x + q - 1) * (v + x * m) / x
            else:
                # Use overall average for customers with no repeat purchases
                predicted_aov = np.mean(monetary_value[frequency > 0])
            
            predictions.append(max(0, predicted_aov))
        
        return np.array(predictions)

class MLBasedCLVModel:
    """Machine learning-based CLV prediction model."""
    
    def __init__(self, config: CLVModelConfig):
        self.config = config
        self.models = {}
        self.feature_scaler = StandardScaler()
        self.fitted = False
        
    def engineer_features(self, 
                         customer_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML-based CLV prediction."""
        features = customer_data.copy()
        
        # Recency, Frequency, Monetary features
        features['days_since_last_purchase'] = (
            datetime.now() - pd.to_datetime(features['last_purchase_date'])
        ).dt.days
        
        features['days_since_first_purchase'] = (
            datetime.now() - pd.to_datetime(features['first_purchase_date'])
        ).dt.days
        
        features['purchase_frequency'] = (
            features['total_transactions'] / 
            (features['days_since_first_purchase'] + 1) * 365
        )
        
        features['avg_order_value'] = (
            features['total_revenue'] / features['total_transactions']
        ).fillna(0)
        
        features['customer_lifetime_days'] = features['days_since_first_purchase']
        
        # Behavioral features
        features['avg_days_between_purchases'] = (
            features['days_since_first_purchase'] / 
            features['total_transactions'].clip(lower=1)
        )
        
        features['purchase_acceleration'] = (
            features['purchase_frequency'] - features['purchase_frequency'].rolling(3).mean()
        ).fillna(0)
        
        features['monetary_trend'] = (
            features['avg_order_value'] - features['avg_order_value'].rolling(3).mean()
        ).fillna(0)
        
        # Channel features
        if 'acquisition_channel' in features.columns:
            channel_encoder = LabelEncoder()
            features['acquisition_channel_encoded'] = channel_encoder.fit_transform(
                features['acquisition_channel'].astype(str)
            )
        
        # Seasonal features
        features['acquisition_month'] = pd.to_datetime(features['first_purchase_date']).dt.month
        features['acquisition_quarter'] = pd.to_datetime(features['first_purchase_date']).dt.quarter
        
        # Cohort features  
        features['cohort_month'] = (
            pd.to_datetime(features['first_purchase_date']).dt.to_period('M')
        )
        
        # Interaction features
        features['freq_x_monetary'] = features['purchase_frequency'] * features['avg_order_value']
        features['recency_x_frequency'] = (
            features['days_since_last_purchase'] * features['purchase_frequency']
        )
        
        return features
    
    def fit_ensemble_models(self, 
                          X: pd.DataFrame,
                          y: pd.Series) -> Dict[str, Any]:
        """Fit ensemble of ML models for CLV prediction."""
        
        # Define models
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'xgboost': XGBRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5)
        }
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.feature_scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Train and evaluate models
        model_scores = {}
        
        for name, model in models.items():
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')
            model_scores[name] = scores.mean()
            
            # Fit on full data
            model.fit(X_scaled, y)
            self.models[name] = model
        
        self.fitted = True
        
        return {
            'model_scores': model_scores,
            'best_model': max(model_scores, key=model_scores.get),
            'feature_names': X.columns.tolist()
        }
    
    def predict_clv(self, 
                   X: pd.DataFrame,
                   ensemble_method: str = 'weighted_average') -> np.ndarray:
        """Predict CLV using ensemble of models."""
        if not self.fitted:
            raise ValueError("Models not fitted yet")
        
        X_scaled = pd.DataFrame(
            self.feature_scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)
        
        if ensemble_method == 'simple_average':
            return np.mean(list(predictions.values()), axis=0)
        elif ensemble_method == 'weighted_average':
            # Weight by model performance
            weights = np.array([
                0.25, 0.25, 0.25, 0.15, 0.10  # RF, GB, XGB, Ridge, ElasticNet
            ])
            weighted_preds = sum(
                w * pred for w, pred in zip(weights, predictions.values())
            )
            return weighted_preds
        else:
            # Return best single model prediction
            best_model_name = max(predictions.keys(), key=lambda k: len(predictions[k]))
            return predictions[best_model_name]

class CustomerSegmentation:
    """Customer segmentation for targeted CLV strategies."""
    
    def __init__(self, config: CLVModelConfig):
        self.config = config
        self.segments = {}
        
    def rfm_segmentation(self, 
                        customer_data: pd.DataFrame) -> pd.DataFrame:
        """Perform RFM (Recency, Frequency, Monetary) segmentation."""
        
        # Calculate RFM metrics
        rfm_data = customer_data.copy()
        
        # Recency (days since last purchase)
        rfm_data['recency'] = (
            datetime.now() - pd.to_datetime(rfm_data['last_purchase_date'])
        ).dt.days
        
        # Frequency (number of transactions)  
        rfm_data['frequency'] = rfm_data['total_transactions']
        
        # Monetary (total revenue)
        rfm_data['monetary'] = rfm_data['total_revenue']
        
        # Create RFM scores (quintiles)
        rfm_data['r_score'] = pd.qcut(
            rfm_data['recency'].rank(method='first'), 5, labels=[5,4,3,2,1]
        )
        rfm_data['f_score'] = pd.qcut(
            rfm_data['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]
        )
        rfm_data['m_score'] = pd.qcut(
            rfm_data['monetary'].rank(method='first'), 5, labels=[1,2,3,4,5]
        )
        
        # Combine RFM scores
        rfm_data['rfm_score'] = (
            rfm_data['r_score'].astype(str) + 
            rfm_data['f_score'].astype(str) + 
            rfm_data['m_score'].astype(str)
        )
        
        # Define segments based on RFM scores
        def rfm_segment(row):
            if row['rfm_score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif row['rfm_score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers' 
            elif row['rfm_score'] in ['512', '511', '422', '421', '412', '411', '311']:
                return 'Potential Loyalists'
            elif row['rfm_score'] in ['533', '532', '531', '523', '522', '521', '515', '514', '513', '425', '424', '413', '414', '415', '315', '314', '313']:
                return 'New Customers'
            elif row['rfm_score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return 'Promising'
            elif row['rfm_score'] in ['155', '251', '252', '253', '254', '255', '245', '244', '253', '243', '242', '241']:
                return 'Need Attention'
            elif row['rfm_score'] in ['135', '125', '231', '241', '251', '124', '123', '122', '132', '231', '241']:
                return 'About to Sleep'
            elif row['rfm_score'] in ['155', '144', '214', '215', '115', '114', '113']:
                return 'At Risk'
            elif row['rfm_score'] in ['155', '144', '134', '135', '124', '125']:
                return 'Cannot Lose Them'
            elif row['rfm_score'] in ['332', '333', '231', '241', '251', '233', '232', '223', '222', '231']:
                return 'Hibernating'
            else:
                return 'Lost'
        
        rfm_data['segment'] = rfm_data.apply(rfm_segment, axis=1)
        
        return rfm_data
    
    def behavioral_clustering(self, 
                            customer_data: pd.DataFrame,
                            n_clusters: int = 5) -> pd.DataFrame:
        """Perform behavioral clustering for segmentation."""
        
        # Select behavioral features
        behavioral_features = [
            'total_transactions', 'total_revenue', 'avg_order_value',
            'days_since_first_purchase', 'days_since_last_purchase'
        ]
        
        # Handle missing values and scale features
        feature_data = customer_data[behavioral_features].fillna(0)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Add cluster labels to data
        segmented_data = customer_data.copy()
        segmented_data['cluster'] = cluster_labels
        segmented_data['segment'] = [f'Cluster_{i}' for i in cluster_labels]
        
        # Analyze cluster characteristics
        cluster_summary = segmented_data.groupby('cluster').agg({
            'total_transactions': ['mean', 'median'],
            'total_revenue': ['mean', 'median'],
            'avg_order_value': ['mean', 'median'],
            'days_since_first_purchase': ['mean', 'median'],
            'days_since_last_purchase': ['mean', 'median']
        }).round(2)
        
        return segmented_data, cluster_summary

class LifetimeValuePredictor:
    """Main CLV prediction engine combining multiple approaches."""
    
    def __init__(self, config: CLVModelConfig):
        self.config = config
        self.btyd_model = BuyTillYouDieModels(config)
        self.ml_model = MLBasedCLVModel(config)
        self.segmentation = CustomerSegmentation(config)
        self.fitted_models = {}
        
    def prepare_customer_data(self, 
                            transaction_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare customer-level data from transaction data."""
        
        # Aggregate transaction data to customer level
        customer_summary = transaction_data.groupby('customer_id').agg({
            'transaction_date': ['min', 'max', 'count'],
            'revenue': ['sum', 'mean'],
            'order_id': 'count'
        }).reset_index()
        
        # Flatten column names
        customer_summary.columns = [
            'customer_id', 'first_purchase_date', 'last_purchase_date', 
            'total_transactions', 'total_revenue', 'avg_order_value', 'order_count'
        ]
        
        # Calculate RFM metrics
        customer_summary['days_since_first_purchase'] = (
            datetime.now() - pd.to_datetime(customer_summary['first_purchase_date'])
        ).dt.days
        
        customer_summary['days_since_last_purchase'] = (
            datetime.now() - pd.to_datetime(customer_summary['last_purchase_date'])
        ).dt.days
        
        # Calculate BTYD metrics
        customer_summary['frequency'] = customer_summary['total_transactions'] - 1  # Repeat purchases
        customer_summary['recency'] = customer_summary['days_since_last_purchase']
        customer_summary['T'] = customer_summary['days_since_first_purchase']
        
        return customer_summary
    
    def fit_models(self, 
                  customer_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit all CLV prediction models."""
        
        results = {}
        
        # Fit BTYD models
        if self.config.model_type in ['btyd', 'hybrid']:
            frequency = customer_data['frequency'].values
            recency = customer_data['recency'].values  
            T = customer_data['T'].values
            monetary = customer_data['avg_order_value'].values
            
            # Fit BG/NBD model
            bg_nbd_results = self.btyd_model.fit_bg_nbd_model(frequency, recency, T)
            results['bg_nbd'] = bg_nbd_results
            
            # Fit Gamma-Gamma model if configured
            if self.config.gamma_gamma_model:
                gg_results = self.btyd_model.fit_gamma_gamma_model(frequency, monetary)
                results['gamma_gamma'] = gg_results
        
        # Fit ML models
        if self.config.model_type in ['ml_ensemble', 'hybrid']:
            # Engineer features
            feature_data = self.ml_model.engineer_features(customer_data)
            
            # Define feature columns (exclude target and ID columns)
            feature_cols = [
                col for col in feature_data.columns 
                if col not in ['customer_id', 'total_revenue', 'first_purchase_date', 
                              'last_purchase_date', 'cohort_month']
            ]
            
            X = feature_data[feature_cols].select_dtypes(include=[np.number])
            y = feature_data['total_revenue']  # Using current total revenue as proxy for historical CLV
            
            ml_results = self.ml_model.fit_ensemble_models(X, y)
            results['ml_ensemble'] = ml_results
        
        self.fitted_models = results
        return results
    
    def predict_customer_clv(self, 
                           customer_data: pd.DataFrame) -> List[CLVPrediction]:
        """Predict CLV for individual customers."""
        
        predictions = []
        
        for _, customer in customer_data.iterrows():
            customer_id = customer['customer_id']
            
            # BTYD prediction
            btyd_clv = 0
            if 'bg_nbd' in self.fitted_models:
                # Predict future transactions
                future_transactions = self.btyd_model.predict_future_transactions(
                    np.array([customer['frequency']]),
                    np.array([customer['recency']]),
                    np.array([customer['T']]),
                    self.config.prediction_horizon_months * 30
                )[0]
                
                # Predict average order value
                if 'gamma_gamma' in self.fitted_models:
                    predicted_aov = self.btyd_model.predict_average_order_value(
                        np.array([customer['frequency']]),
                        np.array([customer['avg_order_value']])
                    )[0]
                else:
                    predicted_aov = customer['avg_order_value']
                
                btyd_clv = future_transactions * predicted_aov
            
            # ML prediction
            ml_clv = 0
            if 'ml_ensemble' in self.fitted_models:
                feature_data = self.ml_model.engineer_features(
                    pd.DataFrame([customer])
                )
                feature_cols = [
                    col for col in feature_data.columns 
                    if col not in ['customer_id', 'total_revenue', 'first_purchase_date', 
                                  'last_purchase_date', 'cohort_month']
                ]
                X = feature_data[feature_cols].select_dtypes(include=[np.number])
                ml_clv = self.ml_model.predict_clv(X)[0]
            
            # Combine predictions based on model type
            if self.config.model_type == 'btyd':
                final_clv = btyd_clv
                model_used = 'BG/NBD + Gamma-Gamma'
            elif self.config.model_type == 'ml_ensemble':
                final_clv = ml_clv
                model_used = 'ML Ensemble'
            else:  # hybrid
                final_clv = 0.6 * btyd_clv + 0.4 * ml_clv
                model_used = 'Hybrid (BTYD + ML)'
            
            # Apply discount rate
            discount_factor = 1 / (1 + self.config.discount_rate) ** (
                self.config.prediction_horizon_months / 12
            )
            discounted_clv = final_clv * discount_factor
            
            # Calculate confidence interval (simplified approach)
            clv_std = discounted_clv * 0.2  # Assume 20% standard deviation
            ci_lower = discounted_clv - 1.96 * clv_std
            ci_upper = discounted_clv + 1.96 * clv_std
            
            # Determine segment (simplified)
            segment = self._determine_segment(customer)
            
            # Calculate churn probability (simplified)
            churn_prob = min(0.9, customer['days_since_last_purchase'] / 365)
            
            prediction = CLVPrediction(
                customer_id=customer_id,
                predicted_clv=discounted_clv,
                clv_confidence_interval=(ci_lower, ci_upper),
                prediction_components={
                    'btyd_clv': btyd_clv,
                    'ml_clv': ml_clv,
                    'discount_factor': discount_factor
                },
                segment=segment,
                acquisition_channel=customer.get('acquisition_channel', 'unknown'),
                days_since_first_purchase=customer['days_since_first_purchase'],
                predicted_transactions=future_transactions if 'future_transactions' in locals() else 0,
                predicted_monetary_value=predicted_aov if 'predicted_aov' in locals() else customer['avg_order_value'],
                churn_probability=churn_prob,
                model_used=model_used,
                prediction_date=datetime.now()
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def _determine_segment(self, customer: pd.Series) -> str:
        """Determine customer segment based on basic criteria."""
        if customer['total_revenue'] > 1000 and customer['total_transactions'] > 10:
            return 'High Value'
        elif customer['total_revenue'] > 500 and customer['total_transactions'] > 5:
            return 'Medium Value'
        elif customer['days_since_last_purchase'] > 365:
            return 'At Risk'
        else:
            return 'Standard'
    
    def perform_cohort_analysis(self, 
                              transaction_data: pd.DataFrame) -> CohortAnalysis:
        """Perform cohort analysis to understand CLV patterns."""
        
        # Prepare cohort data
        transaction_data['transaction_date'] = pd.to_datetime(transaction_data['transaction_date'])
        transaction_data['order_period'] = transaction_data['transaction_date'].dt.to_period('M')
        
        # Get customer's first purchase month
        customer_cohorts = transaction_data.groupby('customer_id')['transaction_date'].min().reset_index()
        customer_cohorts['cohort_group'] = customer_cohorts['transaction_date'].dt.to_period('M')
        
        # Merge cohort info back to transactions
        df = transaction_data.merge(customer_cohorts[['customer_id', 'cohort_group']], on='customer_id')
        
        # Calculate period number (months since first purchase)
        df['period_number'] = (df['order_period'] - df['cohort_group']).apply(attrgetter('n'))
        
        # Create cohort table
        cohort_data = df.groupby(['cohort_group', 'period_number'])['customer_id'].nunique().reset_index()
        cohort_sizes = df.groupby('cohort_group')['customer_id'].nunique()
        
        cohort_table = cohort_data.pivot(index='cohort_group', 
                                        columns='period_number', 
                                        values='customer_id')
        
        # Calculate retention rates
        cohort_sizes = cohort_table.iloc[:, 0]
        retention_table = cohort_table.divide(cohort_sizes, axis=0)
        
        # Revenue cohorts
        revenue_data = df.groupby(['cohort_group', 'period_number'])['revenue'].sum().reset_index()
        revenue_cohorts = revenue_data.pivot(index='cohort_group',
                                           columns='period_number', 
                                           values='revenue')
        
        # CLV by cohort (simplified calculation)
        clv_by_cohort = revenue_cohorts.sum(axis=1).reset_index()
        clv_by_cohort.columns = ['cohort_group', 'total_clv']
        clv_by_cohort['avg_clv'] = clv_by_cohort['total_clv'] / cohort_sizes
        
        # Generate insights
        insights = {
            'avg_retention_1_month': retention_table.iloc[:, 1].mean(),
            'avg_retention_3_months': retention_table.iloc[:, 3].mean() if retention_table.shape[1] > 3 else None,
            'avg_retention_6_months': retention_table.iloc[:, 6].mean() if retention_table.shape[1] > 6 else None,
            'best_cohort': clv_by_cohort.loc[clv_by_cohort['avg_clv'].idxmax(), 'cohort_group'],
            'worst_cohort': clv_by_cohort.loc[clv_by_cohort['avg_clv'].idxmin(), 'cohort_group']
        }
        
        return CohortAnalysis(
            cohort_table=cohort_table,
            retention_rates=retention_table,
            revenue_cohorts=revenue_cohorts,
            clv_by_cohort=clv_by_cohort,
            cohort_insights=insights
        )

# Executive Demo Functions

def generate_sample_customer_data(n_customers: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate sample customer and transaction data for demonstration."""
    np.random.seed(42)
    
    # Generate customer data
    customers = []
    transactions = []
    
    channels = ['organic', 'paid_search', 'social', 'email', 'direct', 'referral']
    
    for i in range(n_customers):
        customer_id = f'customer_{i:04d}'
        acquisition_channel = np.random.choice(channels)
        first_purchase = datetime.now() - timedelta(days=np.random.randint(1, 1095))  # Up to 3 years ago
        
        # Customer characteristics influence behavior
        if acquisition_channel == 'paid_search':
            transaction_multiplier = 1.2
            aov_multiplier = 1.1
        elif acquisition_channel == 'organic':
            transaction_multiplier = 1.0
            aov_multiplier = 1.0
        else:
            transaction_multiplier = 0.8
            aov_multiplier = 0.9
        
        # Generate transaction patterns
        n_transactions = max(1, int(np.random.poisson(5) * transaction_multiplier))
        base_aov = np.random.lognormal(4, 0.5) * aov_multiplier
        
        customer_transactions = []
        current_date = first_purchase
        
        for j in range(n_transactions):
            transaction_id = f'{customer_id}_txn_{j}'
            
            # Revenue varies around base AOV
            revenue = max(10, np.random.normal(base_aov, base_aov * 0.3))
            
            customer_transactions.append({
                'transaction_id': transaction_id,
                'customer_id': customer_id,
                'transaction_date': current_date,
                'revenue': revenue,
                'order_id': transaction_id
            })
            
            # Next transaction timing (if any)
            if j < n_transactions - 1:
                days_gap = np.random.exponential(60)  # Average 60 days between purchases
                current_date += timedelta(days=days_gap)
        
        transactions.extend(customer_transactions)
        
        # Customer summary
        total_revenue = sum(t['revenue'] for t in customer_transactions)
        customers.append({
            'customer_id': customer_id,
            'acquisition_channel': acquisition_channel,
            'first_purchase_date': first_purchase,
            'last_purchase_date': customer_transactions[-1]['transaction_date'],
            'total_transactions': n_transactions,
            'total_revenue': total_revenue,
            'avg_order_value': total_revenue / n_transactions
        })
    
    customer_df = pd.DataFrame(customers)
    transaction_df = pd.DataFrame(transactions)
    
    return customer_df, transaction_df

def executive_demo_lifetime_value_prediction():
    """
    Executive demonstration of Customer Lifetime Value prediction capabilities.
    
    This function showcases:
    1. BTYD (Buy Till You Die) models for behavioral prediction
    2. Machine learning ensemble for CLV forecasting
    3. Customer segmentation and targeting strategies
    4. Cohort analysis and retention insights
    5. ROI optimization through CLV-based decision making
    """
    print("🚀 CUSTOMER LIFETIME VALUE PREDICTION DEMO")
    print("=" * 60)
    print("Advanced CLV modeling for marketing optimization")
    print("Portfolio demonstration by Sotiris Spyrou")
    print()
    
    # Configuration
    config = CLVModelConfig(
        prediction_horizon_months=24,
        model_type='hybrid',
        btyd_model='bg_nbd',
        gamma_gamma_model=True,
        segmentation_method='rfm'
    )
    
    # Generate sample data
    print("📊 GENERATING SAMPLE DATA")
    print("-" * 30)
    customer_data, transaction_data = generate_sample_customer_data(1000)
    print(f"Generated {len(customer_data):,} customers with {len(transaction_data):,} transactions")
    print(f"Date range: {transaction_data['transaction_date'].min().date()} to {transaction_data['transaction_date'].max().date()}")
    print()
    
    # Initialize predictor
    predictor = LifetimeValuePredictor(config)
    
    # Prepare customer data
    prepared_data = predictor.prepare_customer_data(transaction_data)
    print("1. DATA PREPARATION")
    print("-" * 30)
    print(f"Customer metrics calculated:")
    print(f"• Average transactions per customer: {prepared_data['total_transactions'].mean():.1f}")
    print(f"• Average customer lifetime: {prepared_data['days_since_first_purchase'].mean():.0f} days")
    print(f"• Average order value: ${prepared_data['avg_order_value'].mean():.2f}")
    print(f"• Average total revenue per customer: ${prepared_data['total_revenue'].mean():.2f}")
    print()
    
    # Fit models
    print("2. MODEL TRAINING")
    print("-" * 30)
    model_results = predictor.fit_models(prepared_data)
    
    if 'bg_nbd' in model_results:
        bg_params = model_results['bg_nbd']
        print(f"BG/NBD Model Parameters:")
        print(f"• r (purchase rate): {bg_params['r']:.3f}")
        print(f"• alpha (purchase rate heterogeneity): {bg_params['alpha']:.3f}")
        print(f"• a (dropout rate): {bg_params['a']:.3f}")
        print(f"• b (dropout rate heterogeneity): {bg_params['b']:.3f}")
        print(f"• Log-likelihood: {bg_params['log_likelihood']:.2f}")
    
    if 'gamma_gamma' in model_results:
        gg_params = model_results['gamma_gamma']
        print(f"\nGamma-Gamma Model Parameters:")
        print(f"• p: {gg_params['p']:.3f}")
        print(f"• q: {gg_params['q']:.3f}")
        print(f"• v: {gg_params['v']:.3f}")
    
    if 'ml_ensemble' in model_results:
        ml_results = model_results['ml_ensemble']
        print(f"\nML Ensemble Performance:")
        for model, score in ml_results['model_scores'].items():
            print(f"• {model}: R² = {score:.3f}")
        print(f"• Best model: {ml_results['best_model']}")
    print()
    
    # Generate CLV predictions
    print("3. CLV PREDICTIONS")
    print("-" * 30)
    clv_predictions = predictor.predict_customer_clv(prepared_data.head(100))  # Predict for first 100 customers
    
    # Analyze predictions
    predicted_clvs = [p.predicted_clv for p in clv_predictions]
    
    print(f"CLV Predictions (24-month horizon):")
    print(f"• Average predicted CLV: ${np.mean(predicted_clvs):.2f}")
    print(f"• Median predicted CLV: ${np.median(predicted_clvs):.2f}")
    print(f"• CLV range: ${min(predicted_clvs):.2f} - ${max(predicted_clvs):.2f}")
    print(f"• Standard deviation: ${np.std(predicted_clvs):.2f}")
    print()
    
    # Top customers analysis
    top_customers = sorted(clv_predictions, key=lambda x: x.predicted_clv, reverse=True)[:10]
    print("Top 10 Customers by Predicted CLV:")
    for i, customer in enumerate(top_customers[:5], 1):
        print(f"{i}. {customer.customer_id}: ${customer.predicted_clv:.2f} "
              f"(CI: ${customer.clv_confidence_interval[0]:.2f}-${customer.clv_confidence_interval[1]:.2f})")
    print()
    
    # Customer segmentation
    print("4. CUSTOMER SEGMENTATION")
    print("-" * 30)
    segment_counts = {}
    segment_clvs = {}
    
    for pred in clv_predictions:
        segment = pred.segment
        if segment not in segment_counts:
            segment_counts[segment] = 0
            segment_clvs[segment] = []
        segment_counts[segment] += 1
        segment_clvs[segment].append(pred.predicted_clv)
    
    for segment, count in segment_counts.items():
        avg_clv = np.mean(segment_clvs[segment])
        print(f"• {segment}: {count} customers (${avg_clv:.2f} avg CLV)")
    print()
    
    # Cohort analysis
    print("5. COHORT ANALYSIS")
    print("-" * 30)
    cohort_analysis = predictor.perform_cohort_analysis(transaction_data)
    
    insights = cohort_analysis.cohort_insights
    print(f"Cohort Insights:")
    print(f"• 1-month retention rate: {insights['avg_retention_1_month']:.1%}")
    if insights['avg_retention_3_months']:
        print(f"• 3-month retention rate: {insights['avg_retention_3_months']:.1%}")
    if insights['avg_retention_6_months']:
        print(f"• 6-month retention rate: {insights['avg_retention_6_months']:.1%}")
    print(f"• Best performing cohort: {insights['best_cohort']}")
    print(f"• Worst performing cohort: {insights['worst_cohort']}")
    print()
    
    # Channel analysis
    print("6. CHANNEL PERFORMANCE")
    print("-" * 30)
    channel_performance = {}
    
    for pred in clv_predictions:
        channel = pred.acquisition_channel
        if channel not in channel_performance:
            channel_performance[channel] = []
        channel_performance[channel].append(pred.predicted_clv)
    
    # Sort channels by average CLV
    sorted_channels = sorted(
        channel_performance.items(),
        key=lambda x: np.mean(x[1]),
        reverse=True
    )
    
    for channel, clvs in sorted_channels:
        avg_clv = np.mean(clvs)
        count = len(clvs)
        print(f"• {channel}: ${avg_clv:.2f} avg CLV ({count} customers)")
    print()
    
    print("📊 KEY BUSINESS INSIGHTS")
    print("-" * 30)
    total_predicted_clv = sum(predicted_clvs)
    best_channel = sorted_channels[0][0] if sorted_channels else 'N/A'
    high_value_customers = len([p for p in clv_predictions if p.predicted_clv > np.mean(predicted_clvs) * 1.5])
    
    print(f"• Total portfolio CLV (100 customers): ${total_predicted_clv:,.2f}")
    print(f"• Best acquisition channel: {best_channel}")
    print(f"• High-value customers (>150% of avg): {high_value_customers}")
    print(f"• Churn risk customers: {len([p for p in clv_predictions if p.churn_probability > 0.7])}")
    print()
    
    print("🎯 EXECUTIVE SUMMARY")
    print("-" * 30)
    print("✅ Hybrid BTYD + ML modeling for accurate CLV prediction")
    print("✅ Customer segmentation enabling targeted strategies")
    print("✅ Cohort analysis revealing retention patterns")
    print("✅ Channel performance optimization opportunities")
    print("✅ Churn risk identification for proactive retention")
    print()
    print("Portfolio demonstration of advanced customer analytics")
    print("Driving data-driven customer acquisition and retention strategies")

if __name__ == "__main__":
    executive_demo_lifetime_value_prediction()