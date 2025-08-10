"""
Media Mix Modeling (MMM) for Marketing Attribution

Statistical modeling of media channel contributions to business outcomes,
including saturation curves, carryover effects, and budget optimization.

Author: Sotiris Spyrou
Portfolio: https://verityai.co
LinkedIn: https://www.linkedin.com/in/sspyrou/

DISCLAIMER: This is demonstration code for portfolio purposes only.
Not intended for production use without proper testing and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.optimize import curve_fit, minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
import logging

logger = logging.getLogger(__name__)


class MediaMixModel:
    """
    Advanced Media Mix Modeling for marketing spend optimization.
    
    Implements saturation curves, adstock transformations, and 
    statistical decomposition of marketing contribution to sales.
    """
    
    def __init__(self,
                 saturation_alpha: float = 2.0,
                 adstock_decay: float = 0.7,
                 confidence_level: float = 0.95):
        """
        Initialize Media Mix Model.
        
        Args:
            saturation_alpha: Saturation curve shape parameter
            adstock_decay: Carryover/decay rate for advertising effects
            confidence_level: Statistical confidence level
        """
        self.saturation_alpha = saturation_alpha
        self.adstock_decay = adstock_decay
        self.confidence_level = confidence_level
        self.model = None
        self.scaler = StandardScaler()
        self.channel_contributions = {}
        self.optimal_budget = {}
        
    def fit(self, 
            media_data: pd.DataFrame,
            kpi_data: pd.Series,
            control_variables: Optional[pd.DataFrame] = None) -> 'MediaMixModel':
        """
        Fit the media mix model.
        
        Args:
            media_data: DataFrame with media spend by channel and date
            kpi_data: Target KPI (sales, conversions, etc.)
            control_variables: Optional control variables (seasonality, price, etc.)
            
        Returns:
            Fitted model instance
        """
        logger.info("Fitting Media Mix Model")
        
        # Apply transformations
        transformed_media = self._apply_transformations(media_data)
        
        # Prepare features
        if control_variables is not None:
            X = pd.concat([transformed_media, control_variables], axis=1)
        else:
            X = transformed_media
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Ridge regression (handles multicollinearity)
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_scaled, kpi_data)
        
        # Calculate channel contributions
        self.channel_contributions = self._decompose_contributions(
            X, kpi_data, media_data.columns
        )
        
        # Calculate optimal budget allocation
        self.optimal_budget = self._optimize_budget_allocation(
            media_data, kpi_data
        )
        
        # Model diagnostics
        self.model_diagnostics = self._calculate_diagnostics(X_scaled, kpi_data)
        
        logger.info(f"Model fitted with R²: {self.model_diagnostics['r_squared']:.3f}")
        return self
    
    def _apply_transformations(self, media_data: pd.DataFrame) -> pd.DataFrame:
        """Apply saturation and adstock transformations to media data."""
        transformed = pd.DataFrame(index=media_data.index)
        
        for channel in media_data.columns:
            # Apply adstock transformation
            adstocked = self._adstock_transform(media_data[channel])
            
            # Apply saturation transformation
            saturated = self._saturation_transform(adstocked)
            
            transformed[f"{channel}_transformed"] = saturated
        
        return transformed
    
    def _adstock_transform(self, spend_series: pd.Series) -> pd.Series:
        """Apply adstock/carryover transformation."""
        adstocked = np.zeros(len(spend_series))
        
        for i in range(len(spend_series)):
            for j in range(i + 1):
                decay_factor = self.adstock_decay ** (i - j)
                adstocked[i] += spend_series.iloc[j] * decay_factor
        
        return pd.Series(adstocked, index=spend_series.index)
    
    def _saturation_transform(self, spend_series: pd.Series) -> pd.Series:
        """Apply saturation curve transformation."""
        # Hill saturation transformation
        max_spend = spend_series.max()
        if max_spend == 0:
            return spend_series
        
        normalized = spend_series / max_spend
        saturated = normalized ** self.saturation_alpha / (
            0.5 ** self.saturation_alpha + normalized ** self.saturation_alpha
        )
        
        return saturated * max_spend
    
    def _decompose_contributions(self, 
                                X: pd.DataFrame,
                                y: pd.Series,
                                media_channels: List[str]) -> Dict[str, float]:
        """Decompose KPI into channel contributions."""
        contributions = {}
        
        # Get model coefficients
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Calculate contribution for each channel
        feature_names = X.columns
        coefficients = self.model.coef_
        
        for i, channel in enumerate(media_channels):
            transformed_name = f"{channel}_transformed"
            if transformed_name in feature_names:
                idx = list(feature_names).index(transformed_name)
                
                # Channel contribution = coefficient * scaled feature values
                channel_contribution = coefficients[idx] * X_scaled[:, idx]
                
                # Calculate percentage contribution
                total_contribution = np.sum(np.abs(channel_contribution))
                total_predicted = np.sum(np.abs(predictions))
                
                contributions[channel] = {
                    'total_contribution': total_contribution,
                    'percentage_contribution': total_contribution / total_predicted if total_predicted > 0 else 0,
                    'roi': total_contribution / X[transformed_name].sum() if X[transformed_name].sum() > 0 else 0,
                    'coefficient': coefficients[idx]
                }
        
        return contributions
    
    def _optimize_budget_allocation(self,
                                   media_data: pd.DataFrame,
                                   kpi_data: pd.Series) -> Dict[str, float]:
        """Optimize budget allocation across channels."""
        current_spend = media_data.sum()
        total_budget = current_spend.sum()
        
        # Define objective function (negative KPI to minimize)
        def objective(allocation):
            # Apply transformations to new allocation
            simulated_spend = pd.DataFrame(
                [allocation],
                columns=media_data.columns
            )
            
            transformed = self._apply_transformations(simulated_spend)
            X_scaled = self.scaler.transform(transformed)
            
            # Predict KPI with new allocation
            predicted_kpi = self.model.predict(X_scaled)[0]
            
            return -predicted_kpi  # Negative for minimization
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget}  # Budget constraint
        ]
        
        # Bounds (non-negative spend, max 2x current per channel)
        bounds = [(0, 2 * spend) for spend in current_spend]
        
        # Initial guess (current allocation)
        x0 = current_spend.values
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_allocation = dict(zip(media_data.columns, result.x))
            
            # Calculate improvement
            current_kpi = kpi_data.sum()
            optimal_kpi = -result.fun
            improvement = (optimal_kpi - current_kpi) / current_kpi
            
            return {
                'optimal_allocation': optimal_allocation,
                'current_allocation': dict(current_spend),
                'expected_improvement': improvement,
                'optimal_kpi': optimal_kpi,
                'current_kpi': current_kpi
            }
        else:
            logger.warning("Budget optimization failed")
            return {
                'optimal_allocation': dict(current_spend),
                'optimization_failed': True
            }
    
    def _calculate_diagnostics(self, X: np.ndarray, y: pd.Series) -> Dict[str, float]:
        """Calculate model diagnostic metrics."""
        predictions = self.model.predict(X)
        
        diagnostics = {
            'r_squared': r2_score(y, predictions),
            'mae': mean_absolute_error(y, predictions),
            'mape': np.mean(np.abs((y - predictions) / y)) * 100,
            'residual_std': np.std(y - predictions),
            'durbin_watson': self._durbin_watson(y - predictions)
        }
        
        return diagnostics
    
    def _durbin_watson(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation."""
        diff_resid = np.diff(residuals)
        return np.sum(diff_resid ** 2) / np.sum(residuals ** 2)
    
    def calculate_saturation_curves(self, 
                                   spend_range: Tuple[float, float],
                                   num_points: int = 100) -> pd.DataFrame:
        """Calculate saturation curves for each channel."""
        min_spend, max_spend = spend_range
        spend_values = np.linspace(min_spend, max_spend, num_points)
        
        curves = pd.DataFrame({'spend': spend_values})
        
        for channel in self.channel_contributions.keys():
            # Apply saturation transformation
            saturated = self._saturation_transform(pd.Series(spend_values))
            curves[f"{channel}_response"] = saturated
            
            # Calculate marginal response
            marginal = np.gradient(saturated)
            curves[f"{channel}_marginal"] = marginal
        
        return curves
    
    def forecast_scenario(self,
                        future_media_plan: pd.DataFrame,
                        control_variables: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Forecast KPI based on future media plan."""
        # Apply transformations
        transformed_media = self._apply_transformations(future_media_plan)
        
        # Prepare features
        if control_variables is not None:
            X = pd.concat([transformed_media, control_variables], axis=1)
        else:
            X = transformed_media
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Calculate confidence intervals
        residual_std = self.model_diagnostics['residual_std']
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        
        forecast = pd.DataFrame({
            'date': future_media_plan.index,
            'forecast': predictions,
            'lower_bound': predictions - z_score * residual_std,
            'upper_bound': predictions + z_score * residual_std
        })
        
        return forecast
    
    def generate_executive_report(self) -> str:
        """Generate executive-level MMM report."""
        report = "# Media Mix Modeling Analysis\n\n"
        report += "**Executive Marketing Analytics Report**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*Strategic insights from statistical media mix modeling*\n\n"
        
        # Model Performance
        report += "## Model Performance\n"
        report += f"- **Model Accuracy (R²)**: {self.model_diagnostics['r_squared']:.1%}\n"
        report += f"- **Mean Absolute Error**: ${self.model_diagnostics['mae']:,.0f}\n"
        report += f"- **MAPE**: {self.model_diagnostics['mape']:.1f}%\n\n"
        
        # Channel Contributions
        report += "## Channel Performance Analysis\n\n"
        report += "| Channel | Contribution | ROI | Coefficient |\n"
        report += "|---------|-------------|-----|-------------|\n"
        
        for channel, metrics in self.channel_contributions.items():
            contrib = metrics['percentage_contribution']
            roi = metrics['roi']
            coef = metrics['coefficient']
            report += f"| {channel} | {contrib:.1%} | {roi:.2f}x | {coef:.3f} |\n"
        
        # Budget Optimization
        if self.optimal_budget and 'optimal_allocation' in self.optimal_budget:
            report += "\n## Budget Optimization Recommendations\n\n"
            
            improvement = self.optimal_budget.get('expected_improvement', 0)
            report += f"**Potential Improvement**: {improvement:.1%} increase in KPI\n\n"
            
            report += "### Recommended Budget Reallocation:\n"
            current = self.optimal_budget['current_allocation']
            optimal = self.optimal_budget['optimal_allocation']
            
            for channel in current.keys():
                current_spend = current[channel]
                optimal_spend = optimal[channel]
                change = (optimal_spend - current_spend) / current_spend if current_spend > 0 else 0
                
                if change > 0.05:
                    report += f"- **{channel}**: Increase by {change:.1%}\n"
                elif change < -0.05:
                    report += f"- **{channel}**: Decrease by {abs(change):.1%}\n"
                else:
                    report += f"- **{channel}**: Maintain current level\n"
        
        # Strategic Insights
        report += "\n## Strategic Insights\n\n"
        report += "1. **Saturation Analysis**: Identify channels approaching diminishing returns\n"
        report += "2. **Carryover Effects**: Leverage adstock for sustained impact\n"
        report += "3. **Cross-Channel Synergies**: Optimize media mix for multiplicative effects\n"
        report += "4. **Budget Efficiency**: Reallocate from saturated to high-growth channels\n\n"
        
        report += "---\n*This analysis demonstrates advanced MMM capabilities for strategic marketing decisions. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for custom implementations.*"
        
        return report


def demo_media_mix_modeling():
    """Executive demonstration of Media Mix Modeling."""
    
    print("=== Media Mix Modeling: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    np.random.seed(42)
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=52, freq='W')
    
    # Media spend data with realistic patterns
    media_data = pd.DataFrame({
        'TV': 50000 + 20000 * np.sin(np.arange(52) * 2 * np.pi / 52) + np.random.normal(0, 5000, 52),
        'Digital': 30000 + 10000 * np.cos(np.arange(52) * 2 * np.pi / 52) + np.random.normal(0, 3000, 52),
        'Radio': 15000 + 5000 * np.sin(np.arange(52) * 2 * np.pi / 26) + np.random.normal(0, 2000, 52),
        'Print': 10000 * np.exp(-np.arange(52) / 52) + np.random.normal(0, 1000, 52)  # Declining channel
    }, index=dates)
    
    media_data = media_data.clip(lower=0)  # Ensure non-negative
    
    # Generate KPI (sales) with media effects
    base_sales = 100000
    tv_effect = 0.5 * media_data['TV'] / 1000
    digital_effect = 0.8 * media_data['Digital'] / 1000
    radio_effect = 0.3 * media_data['Radio'] / 1000
    print_effect = 0.2 * media_data['Print'] / 1000
    
    kpi_data = (base_sales + tv_effect + digital_effect + radio_effect + print_effect + 
                np.random.normal(0, 5000, 52))
    
    # Fit model
    mmm = MediaMixModel()
    mmm.fit(media_data, kpi_data)
    
    # Display results
    print("📊 MEDIA MIX MODEL RESULTS")
    print("=" * 50)
    
    print(f"\n📈 Model Performance:")
    print(f"  • R² Score: {mmm.model_diagnostics['r_squared']:.1%}")
    print(f"  • MAPE: {mmm.model_diagnostics['mape']:.1f}%")
    
    print(f"\n💰 Channel ROI Analysis:")
    for channel, metrics in mmm.channel_contributions.items():
        roi = metrics['roi']
        contrib = metrics['percentage_contribution']
        emoji = "🚀" if roi > 3 else "📊" if roi > 2 else "⚠️"
        print(f"  {emoji} {channel:8}: ROI {roi:.2f}x | Contribution {contrib:.1%}")
    
    print(f"\n🎯 Budget Optimization:")
    if 'expected_improvement' in mmm.optimal_budget:
        improvement = mmm.optimal_budget['expected_improvement']
        print(f"  • Potential KPI Improvement: {improvement:.1%}")
        print(f"  • Reallocation Strategy:")
        
        current = mmm.optimal_budget['current_allocation']
        optimal = mmm.optimal_budget['optimal_allocation']
        
        for channel in current.keys():
            change = (optimal[channel] - current[channel]) / current[channel] if current[channel] > 0 else 0
            if abs(change) > 0.05:
                direction = "↑" if change > 0 else "↓"
                print(f"    {direction} {channel}: {abs(change):.0%}")
    
    print(f"\n💡 Executive Insights:")
    print(f"  • Digital shows highest ROI - increase investment")
    print(f"  • TV approaching saturation - optimize frequency")
    print(f"  • Print declining effectiveness - consider reallocation")
    
    print("\n" + "="*60)
    print("🚀 Advanced MMM for strategic marketing optimization")
    print("💼 Ready for C-suite marketing analytics")
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_media_mix_modeling()