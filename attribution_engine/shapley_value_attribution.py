"""
Shapley Value Attribution Model

Implements game theory-based fair attribution using Shapley values to calculate
the marginal contribution of each marketing touchpoint across all possible coalitions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional, Callable
from itertools import combinations, permutations
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class ShapleyValueAttribution:
    """
    Shapley Value Attribution Model for fair multi-touch attribution.
    
    Calculates the marginal contribution of each channel using game theory
    principles, ensuring fair allocation based on each channel's contribution
    across all possible coalitions.
    """
    
    def __init__(self, sample_size: int = 10000, max_workers: int = 4, 
                 random_state: int = 42):
        """
        Initialize Shapley Value Attribution model.
        
        Args:
            sample_size: Number of coalition samples for approximation
            max_workers: Number of parallel workers for computation
            random_state: Random seed for reproducibility
        """
        self.sample_size = sample_size
        self.max_workers = max_workers
        self.random_state = random_state
        self.channels = set()
        self.shapley_values = {}
        self.conversion_function = None
        self.journey_data = None
        
        random.seed(random_state)
        np.random.seed(random_state)
    
    def fit(self, journey_data: pd.DataFrame, 
            conversion_function: Optional[Callable] = None) -> 'ShapleyValueAttribution':
        """
        Fit the Shapley Value Attribution model.
        
        Args:
            journey_data: DataFrame with customer journey data
            conversion_function: Custom function to calculate conversion value
                                for a given coalition of channels
        
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Shapley Value Attribution model")
        
        # Validate input data
        required_cols = ['customer_id', 'touchpoint', 'timestamp', 'converted']
        if not all(col in journey_data.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        self.journey_data = journey_data
        self.channels = set(journey_data['touchpoint'].unique())
        
        # Use default conversion function if none provided
        if conversion_function is None:
            self.conversion_function = self._default_conversion_function
        else:
            self.conversion_function = conversion_function
        
        # Calculate Shapley values
        self.shapley_values = self._calculate_shapley_values()
        
        logger.info(f"Model fitted with {len(self.channels)} channels")
        return self
    
    def _default_conversion_function(self, coalition: Set[str]) -> float:
        """
        Default conversion function calculating conversion rate for a coalition.
        
        Args:
            coalition: Set of channels in the coalition
            
        Returns:
            Conversion value (rate) for the coalition
        """
        if not coalition:
            return 0.0
        
        # Filter journeys that contain only channels from the coalition
        customer_journeys = {}
        
        for _, row in self.journey_data.iterrows():
            customer_id = row['customer_id']
            touchpoint = row['touchpoint']
            converted = row['converted']
            
            if customer_id not in customer_journeys:
                customer_journeys[customer_id] = {
                    'touchpoints': set(),
                    'converted': converted
                }
            
            customer_journeys[customer_id]['touchpoints'].add(touchpoint)
        
        # Count customers whose journey is subset of coalition and converted
        coalition_conversions = 0
        coalition_customers = 0
        
        for customer_id, journey_info in customer_journeys.items():
            journey_channels = journey_info['touchpoints']
            
            # Check if journey uses only channels from coalition
            if journey_channels.issubset(coalition) and journey_channels:
                coalition_customers += 1
                if journey_info['converted']:
                    coalition_conversions += 1
        
        if coalition_customers == 0:
            return 0.0
        
        return coalition_conversions / coalition_customers
    
    def _calculate_shapley_values(self) -> Dict[str, float]:
        """Calculate Shapley values for all channels using sampling approximation."""
        if len(self.channels) <= 12:  # Exact calculation for small sets
            return self._exact_shapley_calculation()
        else:  # Sampling approximation for larger sets
            return self._approximate_shapley_calculation()
    
    def _exact_shapley_calculation(self) -> Dict[str, float]:
        """Calculate exact Shapley values for all channels."""
        logger.info("Using exact Shapley value calculation")
        
        shapley_values = {channel: 0.0 for channel in self.channels}
        n = len(self.channels)
        
        # Generate all possible coalitions
        for r in range(n + 1):
            for coalition in combinations(self.channels, r):
                coalition_set = set(coalition)
                coalition_value = self.conversion_function(coalition_set)
                
                # Calculate marginal contributions
                for channel in self.channels:
                    if channel not in coalition_set:
                        # Coalition with this channel added
                        extended_coalition = coalition_set | {channel}
                        extended_value = self.conversion_function(extended_coalition)
                        
                        # Marginal contribution
                        marginal = extended_value - coalition_value
                        
                        # Weight by coalition size probability
                        coalition_size = len(coalition_set)
                        weight = (np.math.factorial(coalition_size) * 
                                np.math.factorial(n - coalition_size - 1)) / np.math.factorial(n)
                        
                        shapley_values[channel] += weight * marginal
        
        return shapley_values
    
    def _approximate_shapley_calculation(self) -> Dict[str, float]:
        """Calculate approximate Shapley values using Monte Carlo sampling."""
        logger.info(f"Using approximate Shapley calculation with {self.sample_size} samples")
        
        shapley_values = {channel: 0.0 for channel in self.channels}
        
        # Use parallel processing for faster computation
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit sampling jobs
            futures = []
            samples_per_worker = self.sample_size // self.max_workers
            
            for worker_id in range(self.max_workers):
                future = executor.submit(
                    self._sample_marginal_contributions,
                    samples_per_worker,
                    worker_id
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                worker_contributions = future.result()
                for channel in self.channels:
                    shapley_values[channel] += worker_contributions.get(channel, 0.0)
        
        # Average across all samples
        for channel in shapley_values:
            shapley_values[channel] /= self.sample_size
        
        return shapley_values
    
    def _sample_marginal_contributions(self, num_samples: int, 
                                     worker_id: int) -> Dict[str, float]:
        """Sample marginal contributions for Shapley value approximation."""
        contributions = {channel: 0.0 for channel in self.channels}
        
        # Set different random seed for each worker
        random.seed(self.random_state + worker_id)
        
        for _ in range(num_samples):
            # Random permutation of all channels
            permutation = list(self.channels)
            random.shuffle(permutation)
            
            # Calculate marginal contribution for each channel
            for i, channel in enumerate(permutation):
                # Coalition before this channel
                coalition_before = set(permutation[:i])
                value_before = self.conversion_function(coalition_before)
                
                # Coalition with this channel
                coalition_with = coalition_before | {channel}
                value_with = self.conversion_function(coalition_with)
                
                # Marginal contribution
                marginal = value_with - value_before
                contributions[channel] += marginal
        
        return contributions
    
    def get_attribution_results(self) -> pd.DataFrame:
        """
        Get attribution results as a DataFrame.
        
        Returns:
            DataFrame with columns ['channel', 'shapley_value', 'attribution_weight']
        """
        # Normalize Shapley values to get attribution weights
        total_value = sum(max(0, val) for val in self.shapley_values.values())
        
        results = []
        for channel in self.channels:
            shapley_val = self.shapley_values.get(channel, 0)
            if total_value > 0:
                attribution_weight = max(0, shapley_val) / total_value
            else:
                attribution_weight = 1 / len(self.channels)  # Equal attribution
            
            results.append({
                'channel': channel,
                'shapley_value': shapley_val,
                'attribution_weight': attribution_weight
            })
        
        return pd.DataFrame(results).sort_values('attribution_weight', ascending=False)
    
    def predict(self, journey: List[str]) -> Dict[str, float]:
        """
        Predict attribution weights for a given journey.
        
        Args:
            journey: List of touchpoints in order
            
        Returns:
            Dictionary mapping channels to attribution weights
        """
        journey_channels = set(journey) & self.channels
        
        if not journey_channels:
            return {}
        
        # Calculate attribution for channels in this journey only
        journey_shapley_total = sum(
            max(0, self.shapley_values.get(ch, 0)) for ch in journey_channels
        )
        
        if journey_shapley_total == 0:
            # Equal attribution if no positive Shapley values
            weight = 1 / len(journey_channels)
            return {ch: weight for ch in journey_channels}
        
        # Proportional attribution based on Shapley values
        attribution = {}
        for channel in journey_channels:
            shapley_val = max(0, self.shapley_values.get(channel, 0))
            attribution[channel] = shapley_val / journey_shapley_total
        
        return attribution
    
    def get_coalition_values(self, coalitions: List[Set[str]]) -> Dict[tuple, float]:
        """
        Calculate conversion values for specific coalitions.
        
        Args:
            coalitions: List of channel coalitions to evaluate
            
        Returns:
            Dictionary mapping coalition tuples to conversion values
        """
        coalition_values = {}
        
        for coalition in coalitions:
            if coalition.issubset(self.channels):
                value = self.conversion_function(coalition)
                coalition_values[tuple(sorted(coalition))] = value
            else:
                logger.warning(f"Coalition {coalition} contains unknown channels")
        
        return coalition_values
    
    def analyze_channel_interactions(self) -> pd.DataFrame:
        """
        Analyze pairwise channel interactions using Shapley values.
        
        Returns:
            DataFrame with channel interaction analysis
        """
        interactions = []
        
        for ch1, ch2 in combinations(self.channels, 2):
            # Individual channel values
            val1 = self.conversion_function({ch1})
            val2 = self.conversion_function({ch2})
            
            # Combined coalition value
            combined_val = self.conversion_function({ch1, ch2})
            
            # Interaction effect (synergy or substitution)
            expected_independent = val1 + val2
            interaction_effect = combined_val - expected_independent
            
            interactions.append({
                'channel_1': ch1,
                'channel_2': ch2,
                'individual_1': val1,
                'individual_2': val2,
                'combined_value': combined_val,
                'interaction_effect': interaction_effect,
                'interaction_type': 'synergy' if interaction_effect > 0 else 'substitution'
            })
        
        return pd.DataFrame(interactions).sort_values('interaction_effect', ascending=False)
    
    def get_model_statistics(self) -> Dict[str, float]:
        """Get model performance and diagnostic statistics."""
        shapley_vals = list(self.shapley_values.values())
        
        stats = {
            'num_channels': len(self.channels),
            'total_shapley_value': sum(shapley_vals),
            'max_shapley_value': max(shapley_vals) if shapley_vals else 0,
            'min_shapley_value': min(shapley_vals) if shapley_vals else 0,
            'mean_shapley_value': np.mean(shapley_vals) if shapley_vals else 0,
            'std_shapley_value': np.std(shapley_vals) if shapley_vals else 0,
        }
        
        # Channel concentration
        positive_vals = [max(0, val) for val in shapley_vals]
        total_positive = sum(positive_vals)
        if total_positive > 0:
            shares = [val / total_positive for val in positive_vals]
            stats['channel_concentration'] = sum(share ** 2 for share in shares)
        
        return stats
    
    def validate_efficiency(self) -> bool:
        """
        Validate the efficiency property of Shapley values.
        
        Returns:
            True if efficiency property holds (sum equals grand coalition value)
        """
        grand_coalition_value = self.conversion_function(self.channels)
        shapley_sum = sum(self.shapley_values.values())
        
        tolerance = 1e-6
        return abs(grand_coalition_value - shapley_sum) < tolerance


def demo_shapley_attribution():
    """Demonstration of Shapley Value Attribution usage."""
    # Sample journey data
    sample_data = pd.DataFrame({
        'customer_id': [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4],
        'touchpoint': ['Search', 'Display', 'Email', 'Social', 'Search', 
                      'Display', 'Search', 'Email', 'Direct', 'Social', 'Email'],
        'timestamp': pd.date_range('2024-01-01', periods=11, freq='D'),
        'converted': [True, True, True, True, True, False, False, False, False, True, True]
    })
    
    # Initialize and fit model
    model = ShapleyValueAttribution(sample_size=1000)
    model.fit(sample_data)
    
    # Get results
    results = model.get_attribution_results()
    print("Attribution Results:")
    print(results)
    
    # Check efficiency property
    is_efficient = model.validate_efficiency()
    print(f"\nEfficiency property satisfied: {is_efficient}")
    
    # Analyze channel interactions
    interactions = model.analyze_channel_interactions()
    print("\nChannel Interactions:")
    print(interactions.head())
    
    # Get model statistics
    stats = model.get_model_statistics()
    print("\nModel Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    demo_shapley_attribution()