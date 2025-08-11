"""
Incrementality Testing & Causal Inference for Marketing Attribution

Advanced framework for measuring true incremental lift and causal impact of marketing activities.
Combines statistical rigor with practical business application for accurate ROI measurement.

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
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.power import ttest_power
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

@dataclass
class IncrementalityTestConfig:
    """Configuration for incrementality testing."""
    test_duration_days: int = 14
    pre_period_days: int = 28
    post_period_days: int = 14
    confidence_level: float = 0.95
    minimum_detectable_effect: float = 0.05
    power: float = 0.8
    alpha: float = 0.05
    test_groups: List[str] = field(default_factory=lambda: ['control', 'treatment'])
    randomization_unit: str = 'user_id'
    stratification_variables: List[str] = field(default_factory=list)
    
@dataclass
class CausalInferenceConfig:
    """Configuration for causal inference methods."""
    propensity_score_method: str = 'logistic'  # 'logistic', 'random_forest', 'gradient_boosting'
    matching_method: str = 'nearest'  # 'nearest', 'caliper', 'stratification'
    caliper_width: float = 0.1
    synthetic_control_method: str = 'elastic_net'  # 'elastic_net', 'ridge', 'lasso'
    instrumental_variables: List[str] = field(default_factory=list)
    regression_discontinuity_cutoff: Optional[float] = None
    difference_in_differences_periods: int = 4

@dataclass
class IncrementalityResult:
    """Results from incrementality testing."""
    incremental_lift: float
    lift_confidence_interval: Tuple[float, float]
    statistical_significance: bool
    p_value: float
    effect_size: float
    control_mean: float
    treatment_mean: float
    sample_sizes: Dict[str, int]
    test_power: float
    methodology: str
    confidence_level: float
    test_duration: str
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SyntheticControlResult:
    """Results from synthetic control analysis."""
    treatment_effect: float
    pre_period_fit: float
    post_period_gap: List[float]
    donor_weights: Dict[str, float]
    control_units: List[str]
    treatment_unit: str
    statistical_significance: bool
    placebo_test_results: Dict[str, float]
    mspe_ratio: float

class IncrementalityTestDesigner:
    """Design and power calculations for incrementality tests."""
    
    def __init__(self, config: IncrementalityTestConfig):
        self.config = config
        
    def calculate_required_sample_size(self, 
                                     baseline_conversion_rate: float,
                                     expected_lift: float,
                                     power: float = None,
                                     alpha: float = None) -> Dict[str, Any]:
        """Calculate required sample size for detecting expected lift."""
        power = power or self.config.power
        alpha = alpha or self.config.alpha
        
        # Calculate effect size (Cohen's h for proportions)
        p1 = baseline_conversion_rate
        p2 = baseline_conversion_rate * (1 + expected_lift)
        
        effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        
        # Calculate required sample size per group
        n_per_group = ((stats.norm.ppf(1 - alpha/2) + stats.norm.ppf(power)) / effect_size) ** 2
        total_sample_size = n_per_group * 2  # For two groups
        
        return {
            'n_per_group': int(np.ceil(n_per_group)),
            'total_sample_size': int(np.ceil(total_sample_size)),
            'effect_size': effect_size,
            'baseline_rate': p1,
            'treatment_rate': p2,
            'expected_lift': expected_lift,
            'power': power,
            'alpha': alpha
        }
    
    def calculate_minimum_detectable_effect(self,
                                          sample_size_per_group: int,
                                          baseline_conversion_rate: float,
                                          power: float = None,
                                          alpha: float = None) -> float:
        """Calculate minimum detectable effect given sample size."""
        power = power or self.config.power
        alpha = alpha or self.config.alpha
        
        # Calculate detectable effect size
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        effect_size = (z_alpha + z_beta) / np.sqrt(sample_size_per_group / 2)
        
        # Convert effect size back to lift percentage
        p1 = baseline_conversion_rate
        arcsin_p1 = np.arcsin(np.sqrt(p1))
        arcsin_p2 = arcsin_p1 + effect_size / 2
        p2 = (np.sin(arcsin_p2)) ** 2
        
        lift = (p2 - p1) / p1
        
        return lift
    
    def design_randomization_strategy(self, 
                                    population_size: int,
                                    stratification_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Design randomization strategy for the experiment."""
        if stratification_data is None:
            # Simple randomization
            treatment_size = population_size // 2
            control_size = population_size - treatment_size
            
            return {
                'randomization_type': 'simple',
                'treatment_size': treatment_size,
                'control_size': control_size,
                'stratification': None
            }
        
        # Stratified randomization
        strata_allocation = {}
        for stratum in stratification_data[self.config.stratification_variables].drop_duplicates().itertuples():
            stratum_data = stratification_data[
                (stratification_data[self.config.stratification_variables] == 
                 pd.Series(stratum[1:], index=self.config.stratification_variables)).all(axis=1)
            ]
            stratum_size = len(stratum_data)
            treatment_size = stratum_size // 2
            control_size = stratum_size - treatment_size
            
            strata_allocation[str(stratum)] = {
                'treatment_size': treatment_size,
                'control_size': control_size,
                'total_size': stratum_size
            }
        
        return {
            'randomization_type': 'stratified',
            'strata_allocation': strata_allocation,
            'stratification_variables': self.config.stratification_variables
        }

class PropensityScoreMatching:
    """Propensity score matching for causal inference."""
    
    def __init__(self, config: CausalInferenceConfig):
        self.config = config
        self.propensity_model = None
        self.matched_pairs = None
        
    def fit_propensity_model(self, 
                           data: pd.DataFrame,
                           treatment_column: str,
                           feature_columns: List[str]) -> Dict[str, Any]:
        """Fit propensity score model."""
        X = data[feature_columns]
        y = data[treatment_column]
        
        if self.config.propensity_score_method == 'logistic':
            from sklearn.linear_model import LogisticRegression
            self.propensity_model = LogisticRegression()
        elif self.config.propensity_score_method == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            self.propensity_model = RandomForestClassifier()
        elif self.config.propensity_score_method == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            self.propensity_model = GradientBoostingClassifier()
        
        self.propensity_model.fit(X, y)
        
        # Calculate propensity scores
        propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        data = data.copy()
        data['propensity_score'] = propensity_scores
        
        # Assess balance
        balance_stats = self._assess_covariate_balance(data, treatment_column, feature_columns)
        
        return {
            'model_type': self.config.propensity_score_method,
            'propensity_scores': propensity_scores,
            'balance_stats': balance_stats,
            'common_support': self._assess_common_support(data, treatment_column)
        }
    
    def perform_matching(self, 
                        data: pd.DataFrame,
                        treatment_column: str) -> pd.DataFrame:
        """Perform propensity score matching."""
        if self.propensity_model is None:
            raise ValueError("Must fit propensity model first")
        
        treatment_data = data[data[treatment_column] == 1].copy()
        control_data = data[data[treatment_column] == 0].copy()
        
        matched_pairs = []
        
        if self.config.matching_method == 'nearest':
            for _, treatment_unit in treatment_data.iterrows():
                distances = np.abs(control_data['propensity_score'] - treatment_unit['propensity_score'])
                nearest_control_idx = distances.idxmin()
                
                if distances[nearest_control_idx] <= self.config.caliper_width:
                    matched_pairs.append({
                        'treatment_idx': treatment_unit.name,
                        'control_idx': nearest_control_idx,
                        'distance': distances[nearest_control_idx]
                    })
                    # Remove matched control unit
                    control_data = control_data.drop(nearest_control_idx)
        
        self.matched_pairs = matched_pairs
        
        # Create matched dataset
        treatment_indices = [pair['treatment_idx'] for pair in matched_pairs]
        control_indices = [pair['control_idx'] for pair in matched_pairs]
        
        matched_data = pd.concat([
            data.loc[treatment_indices],
            data.loc[control_indices]
        ])
        
        return matched_data
    
    def _assess_covariate_balance(self, 
                                 data: pd.DataFrame,
                                 treatment_column: str,
                                 feature_columns: List[str]) -> Dict[str, float]:
        """Assess covariate balance between treatment and control groups."""
        balance_stats = {}
        
        for feature in feature_columns:
            treatment_mean = data[data[treatment_column] == 1][feature].mean()
            control_mean = data[data[treatment_column] == 0][feature].mean()
            pooled_std = data[feature].std()
            
            standardized_diff = (treatment_mean - control_mean) / pooled_std
            balance_stats[feature] = standardized_diff
        
        return balance_stats
    
    def _assess_common_support(self, 
                              data: pd.DataFrame,
                              treatment_column: str) -> Dict[str, float]:
        """Assess common support region."""
        treatment_scores = data[data[treatment_column] == 1]['propensity_score']
        control_scores = data[data[treatment_column] == 0]['propensity_score']
        
        common_min = max(treatment_scores.min(), control_scores.min())
        common_max = min(treatment_scores.max(), control_scores.max())
        
        treatment_in_support = ((treatment_scores >= common_min) & 
                               (treatment_scores <= common_max)).sum()
        control_in_support = ((control_scores >= common_min) & 
                             (control_scores <= common_max)).sum()
        
        return {
            'common_support_min': common_min,
            'common_support_max': common_max,
            'treatment_in_support_pct': treatment_in_support / len(treatment_scores),
            'control_in_support_pct': control_in_support / len(control_scores)
        }

class SyntheticControlMethod:
    """Synthetic control method for causal inference."""
    
    def __init__(self, config: CausalInferenceConfig):
        self.config = config
        self.synthetic_weights = None
        self.donor_pool = None
        
    def fit_synthetic_control(self, 
                            data: pd.DataFrame,
                            treatment_unit: str,
                            donor_units: List[str],
                            outcome_column: str,
                            time_column: str,
                            treatment_start_time: Any) -> SyntheticControlResult:
        """Fit synthetic control model."""
        # Prepare data
        pre_treatment_data = data[data[time_column] < treatment_start_time]
        post_treatment_data = data[data[time_column] >= treatment_start_time]
        
        # Get treatment unit outcomes
        treatment_pre = pre_treatment_data[pre_treatment_data['unit'] == treatment_unit][outcome_column].values
        treatment_post = post_treatment_data[post_treatment_data['unit'] == treatment_unit][outcome_column].values
        
        # Get donor pool outcomes
        donor_pre_outcomes = []
        donor_post_outcomes = []
        
        for donor in donor_units:
            donor_pre = pre_treatment_data[pre_treatment_data['unit'] == donor][outcome_column].values
            donor_post = post_treatment_data[post_treatment_data['unit'] == donor][outcome_column].values
            
            if len(donor_pre) == len(treatment_pre) and len(donor_post) == len(treatment_post):
                donor_pre_outcomes.append(donor_pre)
                donor_post_outcomes.append(donor_post)
        
        donor_pre_matrix = np.array(donor_pre_outcomes).T
        donor_post_matrix = np.array(donor_post_outcomes).T
        
        # Fit synthetic control weights
        if self.config.synthetic_control_method == 'elastic_net':
            from sklearn.linear_model import ElasticNet
            model = ElasticNet(positive=True, fit_intercept=False)
            model.fit(donor_pre_matrix, treatment_pre)
            weights = model.coef_
        elif self.config.synthetic_control_method == 'ridge':
            model = Ridge(positive=True, fit_intercept=False)
            model.fit(donor_pre_matrix, treatment_pre)
            weights = model.coef_
        elif self.config.synthetic_control_method == 'lasso':
            model = Lasso(positive=True, fit_intercept=False)
            model.fit(donor_pre_matrix, treatment_pre)
            weights = model.coef_
        
        # Normalize weights
        weights = weights / weights.sum()
        self.synthetic_weights = weights
        
        # Calculate synthetic control outcomes
        synthetic_pre = donor_pre_matrix @ weights
        synthetic_post = donor_post_matrix @ weights
        
        # Calculate treatment effect
        post_period_gaps = treatment_post - synthetic_post
        treatment_effect = np.mean(post_period_gaps)
        
        # Pre-period fit quality
        pre_period_fit = r2_score(treatment_pre, synthetic_pre)
        
        # Create donor weights dictionary
        donor_weights = dict(zip(donor_units[:len(weights)], weights))
        
        # Perform placebo tests
        placebo_results = self._perform_placebo_tests(
            data, donor_units, outcome_column, time_column, treatment_start_time
        )
        
        # Calculate MSPE ratio for significance
        pre_mspe_treated = np.mean((treatment_pre - synthetic_pre) ** 2)
        post_mspe_treated = np.mean((treatment_post - synthetic_post) ** 2)
        mspe_ratio = post_mspe_treated / pre_mspe_treated if pre_mspe_treated > 0 else float('inf')
        
        # Statistical significance based on placebo distribution
        placebo_mspe_ratios = list(placebo_results.values())
        p_value = (sum(ratio >= mspe_ratio for ratio in placebo_mspe_ratios) + 1) / (len(placebo_mspe_ratios) + 1)
        statistical_significance = p_value < 0.05
        
        return SyntheticControlResult(
            treatment_effect=treatment_effect,
            pre_period_fit=pre_period_fit,
            post_period_gap=post_period_gaps.tolist(),
            donor_weights=donor_weights,
            control_units=donor_units[:len(weights)],
            treatment_unit=treatment_unit,
            statistical_significance=statistical_significance,
            placebo_test_results=placebo_results,
            mspe_ratio=mspe_ratio
        )
    
    def _perform_placebo_tests(self, 
                              data: pd.DataFrame,
                              donor_units: List[str],
                              outcome_column: str,
                              time_column: str,
                              treatment_start_time: Any) -> Dict[str, float]:
        """Perform placebo tests using donor units as fake treatment units."""
        placebo_results = {}
        
        for fake_treatment_unit in donor_units[:10]:  # Limit for computational efficiency
            remaining_donors = [unit for unit in donor_units if unit != fake_treatment_unit]
            
            try:
                result = self.fit_synthetic_control(
                    data, fake_treatment_unit, remaining_donors,
                    outcome_column, time_column, treatment_start_time
                )
                placebo_results[fake_treatment_unit] = result.mspe_ratio
            except:
                continue
        
        return placebo_results

class DifferenceInDifferences:
    """Difference-in-differences causal inference method."""
    
    def __init__(self):
        self.model = None
        self.results = None
        
    def estimate_treatment_effect(self, 
                                data: pd.DataFrame,
                                outcome_column: str,
                                treatment_column: str,
                                time_column: str,
                                unit_column: str,
                                control_variables: List[str] = None) -> Dict[str, Any]:
        """Estimate treatment effect using difference-in-differences."""
        # Create interaction term
        data = data.copy()
        data['treatment_post'] = data[treatment_column] * data[time_column]
        
        # Prepare regression formula
        formula_parts = [outcome_column, '~', treatment_column, '+', time_column, '+', 'treatment_post']
        
        if control_variables:
            formula_parts.extend(['+'] + ['+'.join(control_variables)])
        
        # Add unit fixed effects
        formula_parts.extend(['+', f'C({unit_column})'])
        
        formula = ' '.join(formula_parts)
        
        # Fit regression model
        self.model = sm.formula.ols(formula, data=data).fit()
        
        # Extract treatment effect (coefficient on interaction term)
        treatment_effect = self.model.params['treatment_post']
        se = self.model.bse['treatment_post']
        t_stat = self.model.tvalues['treatment_post']
        p_value = self.model.pvalues['treatment_post']
        
        # Confidence interval
        conf_int = self.model.conf_int().loc['treatment_post']
        
        return {
            'treatment_effect': treatment_effect,
            'standard_error': se,
            't_statistic': t_stat,
            'p_value': p_value,
            'confidence_interval': (conf_int[0], conf_int[1]),
            'statistical_significance': p_value < 0.05,
            'r_squared': self.model.rsquared,
            'model_summary': str(self.model.summary())
        }

class IncrementalityTester:
    """Main class for running incrementality tests and causal inference."""
    
    def __init__(self, 
                 test_config: IncrementalityTestConfig,
                 causal_config: CausalInferenceConfig):
        self.test_config = test_config
        self.causal_config = causal_config
        self.designer = IncrementalityTestDesigner(test_config)
        
    def run_ab_test_analysis(self, 
                           test_data: pd.DataFrame,
                           outcome_column: str,
                           treatment_column: str,
                           user_id_column: str = None) -> IncrementalityResult:
        """Run A/B test analysis with statistical significance testing."""
        # Split data by treatment group
        control_data = test_data[test_data[treatment_column] == 0]
        treatment_data = test_data[test_data[treatment_column] == 1]
        
        # Calculate basic statistics
        control_outcomes = control_data[outcome_column]
        treatment_outcomes = treatment_data[outcome_column]
        
        control_mean = control_outcomes.mean()
        treatment_mean = treatment_outcomes.mean()
        
        # Calculate lift
        incremental_lift = (treatment_mean - control_mean) / control_mean if control_mean > 0 else 0
        
        # Perform statistical test
        if outcome_column in ['conversion', 'click', 'purchase'] or control_outcomes.nunique() == 2:
            # Binary outcome - use proportions test
            control_successes = control_outcomes.sum()
            treatment_successes = treatment_outcomes.sum()
            control_n = len(control_outcomes)
            treatment_n = len(treatment_outcomes)
            
            # Two-proportion z-test
            p1 = control_successes / control_n
            p2 = treatment_successes / treatment_n
            p_pooled = (control_successes + treatment_successes) / (control_n + treatment_n)
            
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_n + 1/treatment_n))
            z_stat = (p2 - p1) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            # Confidence interval for difference in proportions
            se_diff = np.sqrt(p1*(1-p1)/control_n + p2*(1-p2)/treatment_n)
            margin_error = stats.norm.ppf(1 - (1-self.test_config.confidence_level)/2) * se_diff
            diff = p2 - p1
            ci_lower = diff - margin_error
            ci_upper = diff + margin_error
            
            # Convert to lift confidence interval
            lift_ci_lower = ci_lower / p1 if p1 > 0 else float('-inf')
            lift_ci_upper = ci_upper / p1 if p1 > 0 else float('inf')
            
        else:
            # Continuous outcome - use t-test
            t_stat, p_value = stats.ttest_ind(treatment_outcomes, control_outcomes)
            
            # Confidence interval for difference in means
            se_diff = np.sqrt(control_outcomes.var()/len(control_outcomes) + 
                             treatment_outcomes.var()/len(treatment_outcomes))
            margin_error = stats.t.ppf(1 - (1-self.test_config.confidence_level)/2, 
                                      len(control_outcomes) + len(treatment_outcomes) - 2) * se_diff
            diff = treatment_mean - control_mean
            ci_lower = diff - margin_error
            ci_upper = diff + margin_error
            
            # Convert to lift confidence interval
            lift_ci_lower = ci_lower / control_mean if control_mean > 0 else float('-inf')
            lift_ci_upper = ci_upper / control_mean if control_mean > 0 else float('inf')
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_outcomes)-1)*control_outcomes.var() + 
                             (len(treatment_outcomes)-1)*treatment_outcomes.var()) / 
                            (len(control_outcomes) + len(treatment_outcomes) - 2))
        effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # Calculate achieved power
        achieved_power = ttest_power(effect_size, len(control_outcomes), self.test_config.alpha)
        
        return IncrementalityResult(
            incremental_lift=incremental_lift,
            lift_confidence_interval=(lift_ci_lower, lift_ci_upper),
            statistical_significance=p_value < self.test_config.alpha,
            p_value=p_value,
            effect_size=effect_size,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            sample_sizes={'control': len(control_outcomes), 'treatment': len(treatment_outcomes)},
            test_power=achieved_power,
            methodology='A/B Test',
            confidence_level=self.test_config.confidence_level,
            test_duration=f"{self.test_config.test_duration_days} days"
        )
    
    def run_propensity_score_analysis(self, 
                                    data: pd.DataFrame,
                                    outcome_column: str,
                                    treatment_column: str,
                                    feature_columns: List[str]) -> IncrementalityResult:
        """Run propensity score matching analysis."""
        psm = PropensityScoreMatching(self.causal_config)
        
        # Fit propensity score model
        ps_results = psm.fit_propensity_model(data, treatment_column, feature_columns)
        
        # Perform matching
        matched_data = psm.perform_matching(data, treatment_column)
        
        # Calculate treatment effect on matched data
        control_outcomes = matched_data[matched_data[treatment_column] == 0][outcome_column]
        treatment_outcomes = matched_data[matched_data[treatment_column] == 1][outcome_column]
        
        control_mean = control_outcomes.mean()
        treatment_mean = treatment_outcomes.mean()
        incremental_lift = (treatment_mean - control_mean) / control_mean if control_mean > 0 else 0
        
        # Statistical test on matched sample
        t_stat, p_value = stats.ttest_ind(treatment_outcomes, control_outcomes)
        
        # Confidence interval
        se_diff = np.sqrt(control_outcomes.var()/len(control_outcomes) + 
                         treatment_outcomes.var()/len(treatment_outcomes))
        margin_error = stats.t.ppf(1 - (1-self.test_config.confidence_level)/2, 
                                  len(control_outcomes) + len(treatment_outcomes) - 2) * se_diff
        diff = treatment_mean - control_mean
        lift_ci_lower = (diff - margin_error) / control_mean if control_mean > 0 else float('-inf')
        lift_ci_upper = (diff + margin_error) / control_mean if control_mean > 0 else float('inf')
        
        # Effect size
        pooled_std = np.sqrt(((len(control_outcomes)-1)*control_outcomes.var() + 
                             (len(treatment_outcomes)-1)*treatment_outcomes.var()) / 
                            (len(control_outcomes) + len(treatment_outcomes) - 2))
        effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        return IncrementalityResult(
            incremental_lift=incremental_lift,
            lift_confidence_interval=(lift_ci_lower, lift_ci_upper),
            statistical_significance=p_value < self.test_config.alpha,
            p_value=p_value,
            effect_size=effect_size,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            sample_sizes={'control': len(control_outcomes), 'treatment': len(treatment_outcomes)},
            test_power=ttest_power(effect_size, len(control_outcomes), self.test_config.alpha),
            methodology='Propensity Score Matching',
            confidence_level=self.test_config.confidence_level,
            test_duration="Observational",
            additional_metrics={
                'balance_stats': ps_results['balance_stats'],
                'common_support': ps_results['common_support'],
                'matched_pairs': len(psm.matched_pairs)
            }
        )
    
    def run_synthetic_control_analysis(self, 
                                     data: pd.DataFrame,
                                     treatment_unit: str,
                                     donor_units: List[str],
                                     outcome_column: str,
                                     time_column: str,
                                     treatment_start_time: Any) -> Dict[str, Any]:
        """Run synthetic control analysis."""
        sc_method = SyntheticControlMethod(self.causal_config)
        
        result = sc_method.fit_synthetic_control(
            data, treatment_unit, donor_units, 
            outcome_column, time_column, treatment_start_time
        )
        
        # Calculate some additional statistics for IncrementalityResult format
        post_treatment_data = data[
            (data[time_column] >= treatment_start_time) & 
            (data['unit'] == treatment_unit)
        ]
        
        if len(post_treatment_data) > 0:
            treatment_mean = post_treatment_data[outcome_column].mean()
            # Estimate what the synthetic control mean would be
            synthetic_mean = treatment_mean - result.treatment_effect
            incremental_lift = result.treatment_effect / synthetic_mean if synthetic_mean > 0 else 0
        else:
            treatment_mean = 0
            synthetic_mean = 0
            incremental_lift = 0
        
        return {
            'synthetic_control_result': result,
            'incremental_lift': incremental_lift,
            'treatment_effect': result.treatment_effect,
            'statistical_significance': result.statistical_significance,
            'pre_period_fit': result.pre_period_fit,
            'methodology': 'Synthetic Control Method'
        }
    
    def run_difference_in_differences_analysis(self, 
                                             data: pd.DataFrame,
                                             outcome_column: str,
                                             treatment_column: str,
                                             time_column: str,
                                             unit_column: str,
                                             control_variables: List[str] = None) -> IncrementalityResult:
        """Run difference-in-differences analysis."""
        did = DifferenceInDifferences()
        
        results = did.estimate_treatment_effect(
            data, outcome_column, treatment_column, 
            time_column, unit_column, control_variables
        )
        
        # Calculate some basic statistics for the result
        pre_treatment = data[data[time_column] == 0]
        post_treatment = data[data[time_column] == 1]
        
        control_pre_mean = pre_treatment[pre_treatment[treatment_column] == 0][outcome_column].mean()
        control_post_mean = post_treatment[post_treatment[treatment_column] == 0][outcome_column].mean()
        treatment_pre_mean = pre_treatment[pre_treatment[treatment_column] == 1][outcome_column].mean()
        treatment_post_mean = post_treatment[post_treatment[treatment_column] == 1][outcome_column].mean()
        
        # Calculate lift
        control_change = control_post_mean - control_pre_mean
        treatment_change = treatment_post_mean - treatment_pre_mean
        incremental_lift = (results['treatment_effect'] / control_pre_mean) if control_pre_mean > 0 else 0
        
        return IncrementalityResult(
            incremental_lift=incremental_lift,
            lift_confidence_interval=results['confidence_interval'],
            statistical_significance=results['statistical_significance'],
            p_value=results['p_value'],
            effect_size=abs(results['treatment_effect']) / np.sqrt(data[outcome_column].var()),
            control_mean=control_post_mean,
            treatment_mean=treatment_post_mean,
            sample_sizes={
                'control': len(data[data[treatment_column] == 0]),
                'treatment': len(data[data[treatment_column] == 1])
            },
            test_power=0.8,  # Would need more detailed calculation
            methodology='Difference-in-Differences',
            confidence_level=self.test_config.confidence_level,
            test_duration="Panel data",
            additional_metrics={
                'r_squared': results['r_squared'],
                'treatment_effect': results['treatment_effect'],
                'standard_error': results['standard_error']
            }
        )

# Executive Demo Functions

def generate_sample_experiment_data(n_control: int = 5000, 
                                  n_treatment: int = 5000,
                                  base_conversion_rate: float = 0.05,
                                  true_lift: float = 0.15) -> pd.DataFrame:
    """Generate sample A/B test data for demonstration."""
    np.random.seed(42)
    
    # Control group
    control_conversions = np.random.binomial(1, base_conversion_rate, n_control)
    control_data = pd.DataFrame({
        'user_id': [f'control_{i}' for i in range(n_control)],
        'treatment': 0,
        'conversion': control_conversions,
        'revenue': control_conversions * np.random.lognormal(4, 0.5, n_control),
        'days_since_signup': np.random.exponential(30, n_control),
        'previous_purchases': np.random.poisson(2, n_control)
    })
    
    # Treatment group with lift
    treatment_conversion_rate = base_conversion_rate * (1 + true_lift)
    treatment_conversions = np.random.binomial(1, treatment_conversion_rate, n_treatment)
    treatment_data = pd.DataFrame({
        'user_id': [f'treatment_{i}' for i in range(n_treatment)],
        'treatment': 1,
        'conversion': treatment_conversions,
        'revenue': treatment_conversions * np.random.lognormal(4.1, 0.5, n_treatment),  # Slight revenue boost
        'days_since_signup': np.random.exponential(30, n_treatment),
        'previous_purchases': np.random.poisson(2, n_treatment)
    })
    
    return pd.concat([control_data, treatment_data], ignore_index=True)

def generate_observational_data(n_users: int = 10000) -> pd.DataFrame:
    """Generate observational data for causal inference methods."""
    np.random.seed(42)
    
    # User characteristics that influence both treatment selection and outcome
    age = np.random.normal(35, 10, n_users)
    income = np.random.lognormal(10, 0.5, n_users)
    previous_engagement = np.random.beta(2, 5, n_users)
    
    # Treatment assignment influenced by characteristics (selection bias)
    treatment_propensity = 0.1 + 0.3 * (age > 30) + 0.2 * (income > np.median(income)) + 0.4 * previous_engagement
    treatment = np.random.binomial(1, treatment_propensity, n_users)
    
    # Outcome influenced by both treatment and characteristics
    outcome_base = 10 + 0.5 * age + 0.00001 * income + 20 * previous_engagement
    treatment_effect = 5  # True treatment effect
    outcome = outcome_base + treatment_effect * treatment + np.random.normal(0, 5, n_users)
    
    return pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(n_users)],
        'treatment': treatment,
        'outcome': outcome,
        'age': age,
        'income': income,
        'previous_engagement': previous_engagement
    })

def generate_panel_data(n_units: int = 50, n_periods: int = 20, treatment_start: int = 10) -> pd.DataFrame:
    """Generate panel data for difference-in-differences analysis."""
    np.random.seed(42)
    
    data = []
    
    # Half units get treatment
    treatment_units = n_units // 2
    
    for unit in range(n_units):
        unit_effect = np.random.normal(10, 2)  # Unit fixed effect
        is_treatment_unit = unit < treatment_units
        
        for period in range(n_periods):
            time_trend = 0.5 * period
            treatment = 1 if (is_treatment_unit and period >= treatment_start) else 0
            treatment_effect = 3 if treatment else 0
            
            outcome = (unit_effect + time_trend + treatment_effect + 
                      np.random.normal(0, 1))
            
            data.append({
                'unit': f'unit_{unit}',
                'period': period,
                'time': 1 if period >= treatment_start else 0,
                'treatment_unit': 1 if is_treatment_unit else 0,
                'treatment': treatment,
                'outcome': outcome
            })
    
    return pd.DataFrame(data)

def executive_demo_incrementality_testing():
    """
    Executive demonstration of incrementality testing capabilities.
    
    This function showcases:
    1. A/B test design and analysis with statistical rigor
    2. Propensity score matching for observational data
    3. Difference-in-differences for panel data
    4. Power calculations and sample size determination
    5. Multiple causal inference methods comparison
    """
    print("🚀 INCREMENTALITY TESTING & CAUSAL INFERENCE DEMO")
    print("=" * 60)
    print("Advanced framework for measuring true marketing impact")
    print("Portfolio demonstration by Sotiris Spyrou")
    print()
    
    # Configuration
    test_config = IncrementalityTestConfig(
        test_duration_days=14,
        confidence_level=0.95,
        power=0.8,
        alpha=0.05
    )
    
    causal_config = CausalInferenceConfig(
        propensity_score_method='logistic',
        matching_method='nearest'
    )
    
    tester = IncrementalityTester(test_config, causal_config)
    
    print("1. A/B TEST ANALYSIS")
    print("-" * 30)
    
    # Generate and analyze A/B test data
    ab_data = generate_sample_experiment_data(
        n_control=5000, n_treatment=5000, 
        base_conversion_rate=0.05, true_lift=0.15
    )
    
    ab_result = tester.run_ab_test_analysis(
        ab_data, 'conversion', 'treatment', 'user_id'
    )
    
    print(f"Incremental Lift: {ab_result.incremental_lift:.1%}")
    print(f"Statistical Significance: {ab_result.statistical_significance}")
    print(f"P-value: {ab_result.p_value:.4f}")
    print(f"Confidence Interval: [{ab_result.lift_confidence_interval[0]:.1%}, {ab_result.lift_confidence_interval[1]:.1%}]")
    print(f"Effect Size: {ab_result.effect_size:.3f}")
    print(f"Test Power: {ab_result.test_power:.1%}")
    print()
    
    # Sample size calculation
    print("2. SAMPLE SIZE PLANNING")
    print("-" * 30)
    
    sample_calc = tester.designer.calculate_required_sample_size(
        baseline_conversion_rate=0.05,
        expected_lift=0.10,
        power=0.8,
        alpha=0.05
    )
    
    print(f"Required sample size per group: {sample_calc['n_per_group']:,}")
    print(f"Total required sample size: {sample_calc['total_sample_size']:,}")
    print(f"Expected treatment rate: {sample_calc['treatment_rate']:.3f}")
    print(f"Effect size: {sample_calc['effect_size']:.3f}")
    print()
    
    print("3. PROPENSITY SCORE MATCHING")
    print("-" * 30)
    
    # Generate and analyze observational data
    obs_data = generate_observational_data(10000)
    
    psm_result = tester.run_propensity_score_analysis(
        obs_data, 'outcome', 'treatment', 
        ['age', 'income', 'previous_engagement']
    )
    
    print(f"Treatment Effect: {psm_result.treatment_mean - psm_result.control_mean:.2f}")
    print(f"Statistical Significance: {psm_result.statistical_significance}")
    print(f"P-value: {psm_result.p_value:.4f}")
    print(f"Matched Pairs: {psm_result.additional_metrics['matched_pairs']:,}")
    print(f"Effect Size: {psm_result.effect_size:.3f}")
    print()
    
    print("4. DIFFERENCE-IN-DIFFERENCES")
    print("-" * 30)
    
    # Generate and analyze panel data
    panel_data = generate_panel_data(50, 20, 10)
    
    did_result = tester.run_difference_in_differences_analysis(
        panel_data, 'outcome', 'treatment_unit', 'time', 'unit'
    )
    
    print(f"Treatment Effect: {did_result.additional_metrics['treatment_effect']:.2f}")
    print(f"Statistical Significance: {did_result.statistical_significance}")
    print(f"P-value: {did_result.p_value:.4f}")
    print(f"R-squared: {did_result.additional_metrics['r_squared']:.3f}")
    print(f"Standard Error: {did_result.additional_metrics['standard_error']:.3f}")
    print()
    
    print("5. POWER ANALYSIS")
    print("-" * 30)
    
    # Minimum detectable effect calculation
    mde = tester.designer.calculate_minimum_detectable_effect(
        sample_size_per_group=5000,
        baseline_conversion_rate=0.05,
        power=0.8,
        alpha=0.05
    )
    
    print(f"Minimum Detectable Effect: {mde:.1%}")
    print(f"With sample size: 5,000 per group")
    print(f"Power: 80%, Alpha: 5%")
    print()
    
    print("📊 KEY BUSINESS INSIGHTS")
    print("-" * 30)
    print(f"• A/B test detected {ab_result.incremental_lift:.1%} lift with {ab_result.test_power:.0%} power")
    print(f"• PSM analysis corrected for selection bias in observational data")
    print(f"• DiD method isolated treatment effect from time trends")
    print(f"• Sample size planning ensures adequate statistical power")
    print()
    
    print("🎯 EXECUTIVE SUMMARY")
    print("-" * 30)
    print("✅ Rigorous causal inference methodology")
    print("✅ Multiple approaches for different data scenarios") 
    print("✅ Statistical significance testing and confidence intervals")
    print("✅ Power analysis and experimental design optimization")
    print("✅ Bias correction for observational studies")
    print()
    print("Portfolio demonstration of advanced marketing analytics capabilities")
    print("Connecting statistical rigor with actionable business insights")

if __name__ == "__main__":
    executive_demo_incrementality_testing()