"""
Statistical Significance Testing for Marketing Attribution

Comprehensive statistical testing framework for validating marketing
attribution results, A/B tests, and experimental design.

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
from scipy.stats import chi2_contingency, f_oneway, mannwhitneyu
from statsmodels.stats.power import TTestPower
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
import warnings
import logging

logger = logging.getLogger(__name__)


class StatisticalSignificanceTester:
    """
    Advanced statistical testing for marketing attribution and experiments.
    
    Provides comprehensive testing methods including A/B testing, 
    multiple comparison corrections, and power analysis.
    """
    
    def __init__(self,
                 confidence_level: float = 0.95,
                 minimum_detectable_effect: float = 0.05,
                 power: float = 0.8):
        """
        Initialize Statistical Significance Tester.
        
        Args:
            confidence_level: Statistical confidence level (default 95%)
            minimum_detectable_effect: MDE for power calculations
            power: Statistical power for sample size calculations
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.minimum_detectable_effect = minimum_detectable_effect
        self.power = power
        self.test_results = {}
        
    def run_ab_test(self,
                    control_data: pd.Series,
                    treatment_data: pd.Series,
                    test_type: str = 'conversion') -> Dict[str, Any]:
        """
        Run comprehensive A/B test analysis.
        
        Args:
            control_data: Control group data
            treatment_data: Treatment group data
            test_type: Type of test ('conversion', 'revenue', 'continuous')
            
        Returns:
            Comprehensive test results with significance and confidence intervals
        """
        logger.info(f"Running A/B test for {test_type}")
        
        results = {
            'test_type': test_type,
            'sample_sizes': {
                'control': len(control_data),
                'treatment': len(treatment_data)
            }
        }
        
        if test_type == 'conversion':
            results.update(self._conversion_test(control_data, treatment_data))
        elif test_type == 'revenue':
            results.update(self._revenue_test(control_data, treatment_data))
        else:
            results.update(self._continuous_test(control_data, treatment_data))
        
        # Add practical significance assessment
        results['practical_significance'] = self._assess_practical_significance(results)
        
        # Add sample size adequacy
        results['sample_size_adequate'] = self._check_sample_size_adequacy(
            control_data, treatment_data, test_type
        )
        
        return results
    
    def _conversion_test(self, control: pd.Series, treatment: pd.Series) -> Dict[str, Any]:
        """Test for conversion rate differences."""
        
        # Calculate conversion rates
        control_conversions = control.sum()
        treatment_conversions = treatment.sum()
        control_n = len(control)
        treatment_n = len(treatment)
        
        control_rate = control_conversions / control_n
        treatment_rate = treatment_conversions / treatment_n
        
        # Z-test for proportions
        z_stat, p_value = proportions_ztest(
            [control_conversions, treatment_conversions],
            [control_n, treatment_n]
        )
        
        # Calculate lift
        lift = (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0
        
        # Confidence interval for difference
        pooled_se = np.sqrt(
            control_rate * (1 - control_rate) / control_n +
            treatment_rate * (1 - treatment_rate) / treatment_n
        )
        
        z_critical = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = (treatment_rate - control_rate) - z_critical * pooled_se
        ci_upper = (treatment_rate - control_rate) + z_critical * pooled_se
        
        return {
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'absolute_difference': treatment_rate - control_rate,
            'relative_lift': lift,
            'z_statistic': z_stat,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'statistically_significant': p_value < self.alpha,
            'winner': 'treatment' if treatment_rate > control_rate and p_value < self.alpha else
                     'control' if control_rate > treatment_rate and p_value < self.alpha else
                     'no_winner'
        }
    
    def _revenue_test(self, control: pd.Series, treatment: pd.Series) -> Dict[str, Any]:
        """Test for revenue/value differences with outlier handling."""
        
        # Remove outliers using IQR method
        control_clean = self._remove_outliers(control)
        treatment_clean = self._remove_outliers(treatment)
        
        # Calculate statistics
        control_mean = control_clean.mean()
        treatment_mean = treatment_clean.mean()
        control_median = control_clean.median()
        treatment_median = treatment_clean.median()
        
        # Mann-Whitney U test (robust to outliers)
        u_stat, p_value_mw = mannwhitneyu(control_clean, treatment_clean, alternative='two-sided')
        
        # T-test for comparison
        t_stat, p_value_t = stats.ttest_ind(control_clean, treatment_clean)
        
        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_confidence_interval(
            control_clean, treatment_clean
        )
        
        # Calculate lift
        lift = (treatment_mean - control_mean) / control_mean if control_mean > 0 else 0
        
        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'control_median': control_median,
            'treatment_median': treatment_median,
            'absolute_difference': treatment_mean - control_mean,
            'relative_lift': lift,
            't_statistic': t_stat,
            'p_value_ttest': p_value_t,
            'u_statistic': u_stat,
            'p_value_mannwhitney': p_value_mw,
            'confidence_interval': (ci_lower, ci_upper),
            'statistically_significant': p_value_mw < self.alpha,
            'outliers_removed': {
                'control': len(control) - len(control_clean),
                'treatment': len(treatment) - len(treatment_clean)
            }
        }
    
    def _continuous_test(self, control: pd.Series, treatment: pd.Series) -> Dict[str, Any]:
        """Test for continuous metric differences."""
        
        # Basic statistics
        control_mean = control.mean()
        treatment_mean = treatment.mean()
        control_std = control.std()
        treatment_std = treatment.std()
        
        # Welch's t-test (doesn't assume equal variances)
        t_stat, p_value = stats.ttest_ind(control, treatment, equal_var=False)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((control_std**2 + treatment_std**2) / 2)
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval
        se_diff = np.sqrt(control_std**2/len(control) + treatment_std**2/len(treatment))
        t_critical = stats.t.ppf(1 - self.alpha/2, len(control) + len(treatment) - 2)
        ci_lower = (treatment_mean - control_mean) - t_critical * se_diff
        ci_upper = (treatment_mean - control_mean) + t_critical * se_diff
        
        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'control_std': control_std,
            'treatment_std': treatment_std,
            'absolute_difference': treatment_mean - control_mean,
            'cohens_d': cohens_d,
            't_statistic': t_stat,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'statistically_significant': p_value < self.alpha,
            'effect_size_interpretation': self._interpret_effect_size(cohens_d)
        }
    
    def _remove_outliers(self, data: pd.Series) -> pd.Series:
        """Remove outliers using IQR method."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return data[(data >= lower_bound) & (data <= upper_bound)]
    
    def _bootstrap_confidence_interval(self,
                                     control: pd.Series,
                                     treatment: pd.Series,
                                     n_bootstrap: int = 10000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for difference."""
        
        differences = []
        
        for _ in range(n_bootstrap):
            control_sample = control.sample(len(control), replace=True)
            treatment_sample = treatment.sample(len(treatment), replace=True)
            diff = treatment_sample.mean() - control_sample.mean()
            differences.append(diff)
        
        ci_lower = np.percentile(differences, (self.alpha/2) * 100)
        ci_upper = np.percentile(differences, (1 - self.alpha/2) * 100)
        
        return ci_lower, ci_upper
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _assess_practical_significance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess practical significance beyond statistical significance."""
        
        assessment = {
            'is_practically_significant': False,
            'reasoning': []
        }
        
        # Check if statistically significant first
        if not results.get('statistically_significant', False):
            assessment['reasoning'].append("Not statistically significant")
            return assessment
        
        # Check effect size
        if 'relative_lift' in results:
            lift = abs(results['relative_lift'])
            if lift >= self.minimum_detectable_effect:
                assessment['is_practically_significant'] = True
                assessment['reasoning'].append(f"Lift of {lift:.1%} exceeds MDE of {self.minimum_detectable_effect:.1%}")
            else:
                assessment['reasoning'].append(f"Lift of {lift:.1%} below MDE threshold")
        
        # Check confidence interval
        if 'confidence_interval' in results:
            ci_lower, ci_upper = results['confidence_interval']
            if ci_lower > 0 or ci_upper < 0:
                assessment['reasoning'].append("Confidence interval doesn't include zero")
            else:
                assessment['reasoning'].append("Confidence interval includes zero")
        
        return assessment
    
    def _check_sample_size_adequacy(self,
                                   control: pd.Series,
                                   treatment: pd.Series,
                                   test_type: str) -> Dict[str, Any]:
        """Check if sample size is adequate for detecting MDE."""
        
        power_analysis = TTestPower()
        
        if test_type == 'conversion':
            # For proportions test
            control_rate = control.mean()
            effect_size = self.minimum_detectable_effect * control_rate
            
            required_n = power_analysis.solve_power(
                effect_size=effect_size,
                alpha=self.alpha,
                power=self.power,
                ratio=len(treatment)/len(control),
                alternative='two-sided'
            )
        else:
            # For continuous metrics
            pooled_std = np.sqrt((control.std()**2 + treatment.std()**2) / 2)
            effect_size = self.minimum_detectable_effect * control.mean() / pooled_std
            
            required_n = power_analysis.solve_power(
                effect_size=effect_size,
                alpha=self.alpha,
                power=self.power,
                ratio=len(treatment)/len(control),
                alternative='two-sided'
            )
        
        actual_n = min(len(control), len(treatment))
        
        return {
            'required_sample_size': int(required_n) if not np.isnan(required_n) else 'undefined',
            'actual_sample_size': actual_n,
            'is_adequate': actual_n >= required_n if not np.isnan(required_n) else False,
            'power_achieved': power_analysis.solve_power(
                effect_size=effect_size,
                nobs1=actual_n,
                alpha=self.alpha,
                ratio=len(treatment)/len(control),
                alternative='two-sided'
            ) if not np.isnan(effect_size) else 0
        }
    
    def multiple_testing_correction(self,
                                   p_values: List[float],
                                   method: str = 'bonferroni') -> Dict[str, Any]:
        """Apply multiple testing correction to p-values."""
        
        reject, adjusted_p_values, alpha_sidak, alpha_bonf = multipletests(
            p_values,
            alpha=self.alpha,
            method=method
        )
        
        return {
            'original_p_values': p_values,
            'adjusted_p_values': adjusted_p_values.tolist(),
            'reject_null': reject.tolist(),
            'correction_method': method,
            'adjusted_alpha': alpha_bonf if method == 'bonferroni' else alpha_sidak
        }
    
    def calculate_sample_size(self,
                            baseline_rate: float,
                            minimum_detectable_effect: float,
                            test_type: str = 'conversion') -> Dict[str, int]:
        """Calculate required sample size for experiments."""
        
        power_analysis = TTestPower()
        
        if test_type == 'conversion':
            # Effect size for proportions
            effect_size = minimum_detectable_effect
        else:
            # Assume coefficient of variation of 1 for continuous metrics
            effect_size = minimum_detectable_effect
        
        # Calculate for different power levels
        sample_sizes = {}
        
        for power_level in [0.7, 0.8, 0.9, 0.95]:
            n = power_analysis.solve_power(
                effect_size=effect_size,
                alpha=self.alpha,
                power=power_level,
                ratio=1,  # Equal group sizes
                alternative='two-sided'
            )
            
            sample_sizes[f'power_{int(power_level*100)}'] = int(n) if not np.isnan(n) else 'undefined'
        
        # Add duration estimates
        daily_traffic = 1000  # Assumption for demo
        duration_days = {
            power: n / daily_traffic if isinstance(n, int) else 'undefined'
            for power, n in sample_sizes.items()
        }
        
        return {
            'sample_sizes': sample_sizes,
            'duration_days': duration_days,
            'assumptions': {
                'baseline_rate': baseline_rate,
                'mde': minimum_detectable_effect,
                'alpha': self.alpha,
                'test_type': test_type
            }
        }
    
    def generate_test_report(self, test_results: Dict[str, Any]) -> str:
        """Generate executive-friendly test report."""
        
        report = "# Statistical Significance Test Report\n\n"
        report += "**Marketing Experimentation Analysis**\n"
        report += "- **Portfolio**: https://verityai.co\n"
        report += "- **LinkedIn**: https://www.linkedin.com/in/sspyrou/\n\n"
        report += "*Statistical validation for data-driven marketing decisions*\n\n"
        
        # Test Summary
        report += "## Test Summary\n\n"
        report += f"- **Test Type**: {test_results.get('test_type', 'Unknown')}\n"
        report += f"- **Sample Sizes**: Control={test_results['sample_sizes']['control']:,}, "
        report += f"Treatment={test_results['sample_sizes']['treatment']:,}\n"
        
        # Results
        report += "\n## Results\n\n"
        
        if 'control_rate' in test_results:
            report += f"- **Control Rate**: {test_results['control_rate']:.2%}\n"
            report += f"- **Treatment Rate**: {test_results['treatment_rate']:.2%}\n"
        elif 'control_mean' in test_results:
            report += f"- **Control Mean**: {test_results['control_mean']:.2f}\n"
            report += f"- **Treatment Mean**: {test_results['treatment_mean']:.2f}\n"
        
        if 'relative_lift' in test_results:
            report += f"- **Relative Lift**: {test_results['relative_lift']:.1%}\n"
        
        report += f"- **P-Value**: {test_results.get('p_value', test_results.get('p_value_ttest', 'N/A')):.4f}\n"
        report += f"- **Statistical Significance**: {'✅ Yes' if test_results['statistically_significant'] else '❌ No'}\n"
        
        # Confidence Interval
        if 'confidence_interval' in test_results:
            ci_lower, ci_upper = test_results['confidence_interval']
            report += f"- **95% Confidence Interval**: [{ci_lower:.4f}, {ci_upper:.4f}]\n"
        
        # Practical Significance
        if 'practical_significance' in test_results:
            prac_sig = test_results['practical_significance']
            report += f"\n## Practical Significance\n\n"
            report += f"- **Practically Significant**: {'✅ Yes' if prac_sig['is_practically_significant'] else '❌ No'}\n"
            for reason in prac_sig['reasoning']:
                report += f"- {reason}\n"
        
        # Sample Size Adequacy
        if 'sample_size_adequate' in test_results:
            adequacy = test_results['sample_size_adequate']
            report += f"\n## Sample Size Analysis\n\n"
            report += f"- **Required Sample Size**: {adequacy['required_sample_size']}\n"
            report += f"- **Actual Sample Size**: {adequacy['actual_sample_size']}\n"
            report += f"- **Power Achieved**: {adequacy['power_achieved']:.1%}\n"
            report += f"- **Sample Size Adequate**: {'✅ Yes' if adequacy['is_adequate'] else '❌ No'}\n"
        
        # Recommendations
        report += "\n## Recommendations\n\n"
        
        if test_results.get('winner') == 'treatment':
            report += "🎯 **Implement the treatment** - statistically significant improvement detected\n"
        elif test_results.get('winner') == 'control':
            report += "🛑 **Keep the control** - treatment showed negative impact\n"
        else:
            report += "⏸️ **Continue testing** - no significant difference detected yet\n"
        
        report += "\n---\n*This analysis demonstrates rigorous statistical testing for marketing decisions. "
        report += "Contact [Sotiris Spyrou](https://www.linkedin.com/in/sspyrou/) for expert experimentation.*"
        
        return report


def demo_statistical_testing():
    """Executive demonstration of Statistical Significance Testing."""
    
    print("=== Statistical Significance Testing: Executive Demo ===")
    print("Portfolio: https://verityai.co | LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("DISCLAIMER: Demo code for portfolio purposes only\n")
    
    np.random.seed(42)
    
    # Generate sample A/B test data
    # Control: 10% conversion rate
    control_conversions = np.random.binomial(1, 0.10, 5000)
    # Treatment: 12% conversion rate (20% lift)
    treatment_conversions = np.random.binomial(1, 0.12, 5000)
    
    control_data = pd.Series(control_conversions)
    treatment_data = pd.Series(treatment_conversions)
    
    # Initialize tester
    tester = StatisticalSignificanceTester()
    
    # Run A/B test
    results = tester.run_ab_test(control_data, treatment_data, test_type='conversion')
    
    # Display results
    print("📊 A/B TEST RESULTS")
    print("=" * 50)
    
    print(f"\n📈 Performance Metrics:")
    print(f"  • Control Conversion Rate: {results['control_rate']:.2%}")
    print(f"  • Treatment Conversion Rate: {results['treatment_rate']:.2%}")
    print(f"  • Relative Lift: {results['relative_lift']:.1%}")
    
    print(f"\n📐 Statistical Analysis:")
    print(f"  • P-Value: {results['p_value']:.4f}")
    print(f"  • Z-Statistic: {results['z_statistic']:.2f}")
    ci = results['confidence_interval']
    print(f"  • 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    print(f"\n✅ Decision:")
    if results['statistically_significant']:
        print(f"  • Result: STATISTICALLY SIGNIFICANT")
        print(f"  • Winner: {results['winner'].upper()}")
        print(f"  • Action: Implement the treatment")
    else:
        print(f"  • Result: NOT SIGNIFICANT")
        print(f"  • Action: Continue testing or increase sample size")
    
    # Sample size calculation
    print(f"\n📏 Sample Size Planning:")
    sample_calc = tester.calculate_sample_size(0.10, 0.02, 'conversion')
    print(f"  • For 80% power: {sample_calc['sample_sizes']['power_80']} per group")
    print(f"  • For 90% power: {sample_calc['sample_sizes']['power_90']} per group")
    
    # Revenue test example
    print(f"\n💰 Revenue Test Example:")
    control_revenue = np.random.lognormal(3.5, 1.2, 1000)  # Log-normal for revenue
    treatment_revenue = np.random.lognormal(3.6, 1.2, 1000)  # Slightly higher
    
    revenue_results = tester.run_ab_test(
        pd.Series(control_revenue),
        pd.Series(treatment_revenue),
        test_type='revenue'
    )
    
    print(f"  • Control Mean Revenue: ${revenue_results['control_mean']:.2f}")
    print(f"  • Treatment Mean Revenue: ${revenue_results['treatment_mean']:.2f}")
    print(f"  • Revenue Lift: {revenue_results['relative_lift']:.1%}")
    print(f"  • Significant: {'Yes ✅' if revenue_results['statistically_significant'] else 'No ❌'}")
    
    print("\n" + "="*60)
    print("🚀 Rigorous statistical testing for marketing excellence")
    print("💼 Enterprise-grade experimentation framework")
    print("📞 Contact: https://www.linkedin.com/in/sspyrou/")


if __name__ == "__main__":
    demo_statistical_testing()