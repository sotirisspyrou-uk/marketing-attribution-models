"""
Data-Driven Attribution Model

Machine learning-based attribution model that learns from historical data
to determine optimal attribution weights across marketing touchpoints.

Author: Sotiris Spyrou
Portfolio: https://verityai.co
LinkedIn: https://www.linkedin.com/in/sspyrou/

DISCLAIMER: This is demonstration code for portfolio purposes only.
Not intended for production use without proper testing and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
import warnings
import logging

logger = logging.getLogger(__name__)


class DataDrivenAttribution:
    """
    Advanced data-driven attribution model using machine learning.
    
    Uses historical customer journey data to train models that predict
    conversion probability and attribution weights for each touchpoint.
    """
    
    def __init__(self,
                 model_type: str = 'gradient_boosting',
                 attribution_method: str = 'feature_importance',
                 lookback_window: int = 30,
                 min_touchpoints: int = 2):
        """
        Initialize Data-Driven Attribution model.
        
        Args:
            model_type: ML model type ('gradient_boosting', 'random_forest', 'logistic')
            attribution_method: How to extract attribution ('feature_importance', 'shapley')
            lookback_window: Days to look back for journey analysis
            min_touchpoints: Minimum touchpoints required for attribution
        """
        self.model_type = model_type
        self.attribution_method = attribution_method
        self.lookback_window = lookback_window
        self.min_touchpoints = min_touchpoints
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Attribution results
        self.channel_attribution = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Feature engineering
        self.feature_columns = []
        self.channel_list = []
        
    def fit(self, journey_data: pd.DataFrame) -> 'DataDrivenAttribution':
        """
        Fit the data-driven attribution model.
        
        Args:
            journey_data: Customer journey data with timestamps and conversions
            
        Returns:
            Self for method chaining
        """
        logger.info("Training data-driven attribution model")
        
        # Validate input data
        required_columns = ['customer_id', 'touchpoint', 'timestamp', 'converted']
        if not all(col in journey_data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        # Prepare training data
        X, y, customer_features = self._prepare_training_data(journey_data)
        
        if len(X) == 0:
            raise ValueError("No valid training data after preprocessing")
        
        # Initialize and train model
        self._initialize_model()
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        self._evaluate_model_performance(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Calculate attribution weights
        self._calculate_attribution_weights(X_train_scaled, customer_features)
        
        logger.info(f"Model trained with {self.model_performance.get('accuracy', 0):.3f} accuracy")
        return self
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Prepare training data with feature engineering."""
        
        # Sort by customer and timestamp
        data = data.sort_values(['customer_id', 'timestamp'])
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Get unique channels
        self.channel_list = sorted(data['touchpoint'].unique())
        
        features = []
        targets = []
        customer_features = []
        
        # Group by customer
        for customer_id, customer_data in data.groupby('customer_id'):
            customer_data = customer_data.sort_values('timestamp')
            
            # Skip customers with too few touchpoints
            if len(customer_data) < self.min_touchpoints:
                continue
            
            # Create features for this customer journey
            customer_features_dict = self._create_customer_features(customer_data)
            
            # Target: did customer convert?
            converted = customer_data['converted'].any()
            
            features.append(list(customer_features_dict.values()))
            targets.append(int(converted))
            customer_features.append(customer_features_dict)
        
        if not features:
            return np.array([]), np.array([]), pd.DataFrame()
        
        # Store feature names
        self.feature_columns = list(customer_features[0].keys())
        
        X = np.array(features)
        y = np.array(targets)
        customer_features_df = pd.DataFrame(customer_features)
        
        logger.info(f"Prepared {len(X)} customer journeys with {X.shape[1]} features")
        return X, y, customer_features_df
    
    def _create_customer_features(self, customer_data: pd.DataFrame) -> Dict[str, float]:
        """Create features for a single customer journey."""
        
        features = {}
        
        # Basic journey metrics
        features['journey_length'] = len(customer_data)
        features['unique_channels'] = customer_data['touchpoint'].nunique()
        
        # Time-based features
        time_span = (customer_data['timestamp'].max() - customer_data['timestamp'].min()).total_seconds()
        features['journey_duration_hours'] = time_span / 3600
        features['avg_time_between_touches'] = time_span / (len(customer_data) - 1) if len(customer_data) > 1 else 0
        
        # Channel frequency features
        channel_counts = customer_data['touchpoint'].value_counts()
        for channel in self.channel_list:
            features[f'{channel}_count'] = channel_counts.get(channel, 0)
            features[f'{channel}_frequency'] = channel_counts.get(channel, 0) / len(customer_data)
        
        # Position features (first/last touch indicators)
        first_touch = customer_data.iloc[0]['touchpoint']
        last_touch = customer_data.iloc[-1]['touchpoint']
        
        for channel in self.channel_list:
            features[f'{channel}_first_touch'] = 1 if first_touch == channel else 0
            features[f'{channel}_last_touch'] = 1 if last_touch == channel else 0
        
        # Sequential patterns
        touches = customer_data['touchpoint'].tolist()
        
        # Channel transition features
        for i in range(len(self.channel_list)):
            for j in range(len(self.channel_list)):
                if i != j:
                    channel_i = self.channel_list[i]
                    channel_j = self.channel_list[j]
                    transition_count = 0
                    
                    for k in range(len(touches) - 1):
                        if touches[k] == channel_i and touches[k+1] == channel_j:
                            transition_count += 1
                    
                    features[f'{channel_i}_to_{channel_j}'] = transition_count
        
        # Time decay features
        current_time = customer_data['timestamp'].max()
        for i, (_, touch) in enumerate(customer_data.iterrows()):
            time_diff = (current_time - touch['timestamp']).total_seconds() / 3600  # hours
            decay_weight = np.exp(-time_diff / 24)  # 24-hour half-life
            
            channel = touch['touchpoint']
            features[f'{channel}_time_weighted'] = features.get(f'{channel}_time_weighted', 0) + decay_weight
        
        # Journey complexity features
        features['channel_diversity'] = len(set(touches)) / len(self.channel_list)
        features['repeat_touches'] = len(touches) - len(set(touches))
        
        return features
    
    def _initialize_model(self):
        """Initialize the ML model based on model_type."""
        
        if self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def _evaluate_model_performance(self, X_train: np.ndarray, X_test: np.ndarray,
                                  y_train: np.ndarray, y_test: np.ndarray):
        """Evaluate model performance."""
        
        # Predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        if self.model_type == 'logistic':
            # Classification metrics
            train_auc = roc_auc_score(y_train, train_pred)
            test_auc = roc_auc_score(y_test, test_pred)
            
            self.model_performance = {
                'train_auc': train_auc,
                'test_auc': test_auc,
                'accuracy': test_auc,
                'model_type': 'classification'
            }
        else:
            # Regression metrics
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            self.model_performance = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'accuracy': max(0, test_r2),
                'model_type': 'regression'
            }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2' if self.model_type != 'logistic' else 'roc_auc')
        self.model_performance['cv_mean'] = cv_scores.mean()
        self.model_performance['cv_std'] = cv_scores.std()
        
        logger.info(f"Model performance: {self.model_performance['accuracy']:.3f}")
    
    def _calculate_attribution_weights(self, X: np.ndarray, customer_features: pd.DataFrame):
        """Calculate attribution weights using the trained model."""
        
        if self.attribution_method == 'feature_importance':
            self._calculate_feature_importance_attribution(X, customer_features)
        elif self.attribution_method == 'shapley':
            self._calculate_shapley_attribution(X, customer_features)
        else:
            raise ValueError(f"Unknown attribution_method: {self.attribution_method}")
    
    def _calculate_feature_importance_attribution(self, X: np.ndarray, customer_features: pd.DataFrame):
        """Calculate attribution using feature importance."""
        
        # Get feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_).flatten()
        else:
            # Fallback: equal importance
            importances = np.ones(len(self.feature_columns)) / len(self.feature_columns)
        
        # Store feature importance
        self.feature_importance = dict(zip(self.feature_columns, importances))
        
        # Calculate channel attribution
        channel_attribution = {}
        
        for channel in self.channel_list:
            # Sum importance of all features related to this channel
            channel_importance = 0
            feature_count = 0
            
            for feature_name, importance in self.feature_importance.items():
                if channel.lower() in feature_name.lower():
                    channel_importance += importance
                    feature_count += 1
            
            # Normalize by feature count to avoid bias toward channels with more features
            if feature_count > 0:
                channel_attribution[channel] = channel_importance / feature_count
            else:
                channel_attribution[channel] = 0
        
        # Normalize to sum to 1
        total_attribution = sum(channel_attribution.values())
        if total_attribution > 0:
            self.channel_attribution = {
                ch: attr / total_attribution 
                for ch, attr in channel_attribution.items()
            }
        else:
            # Fallback: equal attribution
            equal_weight = 1 / len(self.channel_list)
            self.channel_attribution = {ch: equal_weight for ch in self.channel_list}
    
    def _calculate_shapley_attribution(self, X: np.ndarray, customer_features: pd.DataFrame):
        """Calculate attribution using approximate Shapley values."""
        
        # Simplified Shapley approximation using feature permutation
        n_samples = min(1000, len(X))  # Limit for computational efficiency
        sample_indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[sample_indices]
        
        shapley_values = np.zeros(len(self.feature_columns))
        
        # For each feature
        for i, feature_name in enumerate(self.feature_columns):
            marginal_contributions = []
            
            # Sample random subsets
            for _ in range(50):  # Number of random subsets
                # Create random subset excluding current feature
                subset = np.random.choice(len(self.feature_columns), 
                                       size=np.random.randint(1, len(self.feature_columns)), 
                                       replace=False)
                subset = subset[subset != i]
                
                # Calculate marginal contribution
                if len(subset) > 0:
                    # Prediction with subset
                    X_subset = np.zeros_like(X_sample)
                    X_subset[:, subset] = X_sample[:, subset]
                    pred_without = self.model.predict(X_subset)
                    
                    # Prediction with subset + current feature
                    X_with_feature = X_subset.copy()
                    X_with_feature[:, i] = X_sample[:, i]
                    pred_with = self.model.predict(X_with_feature)
                    
                    # Marginal contribution
                    marginal_contribution = np.mean(pred_with - pred_without)
                    marginal_contributions.append(marginal_contribution)
            
            shapley_values[i] = np.mean(marginal_contributions)
        
        # Store Shapley values
        self.feature_importance = dict(zip(self.feature_columns, shapley_values))
        
        # Calculate channel attribution from Shapley values
        self._calculate_feature_importance_attribution(X, customer_features)
    
    def predict(self, journey: List[str]) -> Dict[str, float]:
        """
        Predict attribution for a single journey.
        
        Args:
            journey: List of touchpoints in order
            
        Returns:
            Attribution weights for each touchpoint
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Create dummy customer data for prediction
        timestamps = pd.date_range('2024-01-01', periods=len(journey), freq='D')
        dummy_data = pd.DataFrame({
            'customer_id': [1] * len(journey),
            'touchpoint': journey,
            'timestamp': timestamps,
            'converted': [True] * len(journey)  # Dummy conversion
        })
        
        # Create features
        customer_features = self._create_customer_features(dummy_data)
        
        # Predict using model
        features_array = np.array([list(customer_features.values())])
        features_scaled = self.scaler.transform(features_array)
        
        prediction = self.model.predict(features_scaled)[0]
        
        # Calculate attribution for this journey
        journey_channels = set(journey)
        journey_attribution = {}
        
        total_attribution = 0
        for channel in journey_channels:
            channel_attr = self.channel_attribution.get(channel, 0)
            journey_attribution[channel] = channel_attr
            total_attribution += channel_attr
        
        # Normalize
        if total_attribution > 0:
            journey_attribution = {
                ch: attr / total_attribution 
                for ch, attr in journey_attribution.items()
            }
        else:
            # Equal attribution fallback
            equal_weight = 1 / len(journey_channels)
            journey_attribution = {ch: equal_weight for ch in journey_channels}
        
        return journey_attribution
    
    def get_attribution_results(self) -> pd.DataFrame:
        """
        Get attribution results as DataFrame.
        
        Returns:
            DataFrame with channel attribution weights and insights
        """
        if not self.channel_attribution:
            raise ValueError("Model not fitted. Call fit() first.")
        
        results = []
        
        for channel, weight in self.channel_attribution.items():
            # Calculate related feature importance
            related_importance = sum(
                imp for feature_name, imp in self.feature_importance.items()
                if channel.lower() in feature_name.lower()
            )
            
            results.append({
                'channel': channel,
                'attribution_weight': weight,
                'feature_importance_sum': related_importance,
                'rank': 0  # Will be filled after sorting
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('attribution_weight', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance analysis."""
        
        if not self.feature_importance:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': importance}
            for feature, importance in self.feature_importance.items()
        ]).sort_values('importance', ascending=False)
        
        # Add feature categories
        importance_df['category'] = importance_df['feature'].apply(self._categorize_feature)
        
        return importance_df
    
    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize features for better interpretation."""
        
        if '_count' in feature_name or '_frequency' in feature_name:
            return 'Channel Frequency'
        elif '_first_touch' in feature_name or '_last_touch' in feature_name:
            return 'Position Features'
        elif '_to_' in feature_name:
            return 'Channel Transitions'
        elif '_time_weighted' in feature_name:
            return 'Time Decay'
        elif feature_name in ['journey_length', 'unique_channels', 'journey_duration_hours']:
            return 'Journey Metrics'
        else:
            return 'Other'
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        
        return {
            'model_type': self.model_type,
            'attribution_method': self.attribution_method,
            'num_features': len(self.feature_columns),
            'num_channels': len(self.channel_list),
            'model_performance': self.model_performance,
            'attribution_concentration': sum(w**2 for w in self.channel_attribution.values()),
            'top_features': dict(list(sorted(
                self.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]))
        }
    
    def generate_insights_report(self) -> str:
        """Generate executive insights report."""
        
        report = "# Data-Driven Attribution Analysis\n\n"
        report += "**Advanced Marketing Analytics by Sotiris Spyrou**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*DISCLAIMER: This is demonstration code for portfolio purposes.*\n\n"
        
        # Model Performance
        perf = self.model_performance
        report += f"## Model Performance\n"
        report += f"- **Model Type**: {self.model_type.title()}\n"
        report += f"- **Accuracy**: {perf.get('accuracy', 0):.1%}\n"
        report += f"- **Cross-Validation Score**: {perf.get('cv_mean', 0):.3f} (±{perf.get('cv_std', 0):.3f})\n\n"
        
        # Attribution Results
        attribution_df = self.get_attribution_results()
        report += f"## Channel Attribution Results\n\n"
        report += "| Rank | Channel | Attribution | Feature Importance |\n"
        report += "|------|---------|-------------|--------------------|\n"
        
        for _, row in attribution_df.head(10).iterrows():
            report += f"| {row['rank']} | {row['channel']} | {row['attribution_weight']:.1%} | {row['feature_importance_sum']:.3f} |\n"
        
        # Key Features
        feature_df = self.get_feature_importance()
        if not feature_df.empty:
            report += f"\n## Top Predictive Features\n\n"
            for _, row in feature_df.head(5).iterrows():
                report += f"- **{row['feature']}**: {row['importance']:.3f} ({row['category']})\n"
        
        # Business Recommendations
        top_channel = attribution_df.iloc[0]['channel']
        report += f"\n## Strategic Recommendations\n\n"
        report += f"1. **Prioritize {top_channel}**: Highest attribution weight suggests strong conversion influence\n"
        report += f"2. **Optimize Journey Length**: Model learns optimal touchpoint sequences\n"
        report += f"3. **Time-Based Targeting**: Leverage time decay insights for campaign timing\n"
        report += f"4. **Cross-Channel Synergies**: Utilize transition patterns for journey optimization\n\n"
        
        report += "---\n*This analysis demonstrates advanced ML-driven attribution capabilities. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for custom implementations.*"
        
        return report


def demo_data_driven_attribution():
    """Executive demonstration of Data-Driven Attribution."""
    
    print("=== Data-Driven Attribution: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    np.random.seed(42)
    
    # Generate realistic customer journey data
    customers = []
    channels = ['Search', 'Display', 'Social', 'Email', 'Direct']
    channel_conversion_rates = {
        'Search': 0.25, 'Direct': 0.35, 'Email': 0.20, 'Social': 0.15, 'Display': 0.10
    }
    
    # Generate 1000 customer journeys
    for customer_id in range(1, 1001):
        # Random journey length (1-8 touchpoints)
        journey_length = np.random.choice(range(1, 9), p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02])
        
        # Generate journey
        journey_channels = np.random.choice(channels, journey_length, 
                                          p=[0.3, 0.2, 0.2, 0.15, 0.15])
        
        # Generate timestamps (spread over 30 days)
        start_date = pd.Timestamp('2024-01-01')
        timestamps = [start_date + pd.Timedelta(days=np.random.randint(0, 30)) + 
                     pd.Timedelta(hours=np.random.randint(0, 24)) for _ in range(journey_length)]
        timestamps.sort()
        
        # Determine conversion (influenced by channels in journey)
        conversion_prob = np.mean([channel_conversion_rates[ch] for ch in journey_channels])
        converted = np.random.random() < conversion_prob
        
        # Add to dataset
        for i, (channel, timestamp) in enumerate(zip(journey_channels, timestamps)):
            customers.append({
                'customer_id': customer_id,
                'touchpoint': channel,
                'timestamp': timestamp,
                'converted': converted
            })
    
    journey_data = pd.DataFrame(customers)
    
    print(f"📊 Generated {len(journey_data)} touchpoints across {journey_data['customer_id'].nunique()} customers")
    print(f"📈 Overall conversion rate: {journey_data.groupby('customer_id')['converted'].first().mean():.1%}")
    
    # Initialize and train model
    model = DataDrivenAttribution(
        model_type='gradient_boosting',
        attribution_method='feature_importance'
    )
    
    print("\n🤖 Training ML attribution model...")
    model.fit(journey_data)
    
    # Display results
    print("\n📊 DATA-DRIVEN ATTRIBUTION RESULTS")
    print("=" * 50)
    
    attribution_results = model.get_attribution_results()
    print(f"\n🏆 Channel Attribution Rankings:")
    for _, row in attribution_results.iterrows():
        rank_emoji = "🥇" if row['rank'] == 1 else "🥈" if row['rank'] == 2 else "🥉" if row['rank'] == 3 else "📊"
        print(f"{rank_emoji} {row['channel']:8}: {row['attribution_weight']:.1%} attribution")
    
    # Model performance
    stats = model.get_model_statistics()
    performance = stats['model_performance']
    print(f"\n🎯 Model Performance:")
    print(f"  • Model Accuracy: {performance.get('accuracy', 0):.1%}")
    print(f"  • Cross-Validation: {performance.get('cv_mean', 0):.3f}")
    print(f"  • Features Used: {stats['num_features']}")
    
    # Top predictive features
    feature_importance = model.get_feature_importance()
    print(f"\n🔍 Top Predictive Features:")
    for _, row in feature_importance.head(3).iterrows():
        print(f"  • {row['feature']}: {row['importance']:.3f} ({row['category']})")
    
    # Journey prediction example
    print(f"\n🧭 Journey Attribution Example:")
    sample_journey = ['Social', 'Search', 'Email', 'Direct']
    journey_attribution = model.predict(sample_journey)
    print(f"Journey: {' → '.join(sample_journey)}")
    for channel, weight in sorted(journey_attribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {channel}: {weight:.1%}")
    
    print("\n" + "="*60)
    print("🚀 Advanced ML-driven attribution for strategic marketing")
    print("💼 Enterprise-grade data science for marketing optimization") 
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_data_driven_attribution()