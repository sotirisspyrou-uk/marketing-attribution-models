"""
Markov Chain Attribution Model

Implements probabilistic attribution modeling using Markov chains to understand
customer journey transitions and calculate removal effects for each touchpoint.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import networkx as nx
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


class MarkovChainAttribution:
    """
    Markov Chain Attribution Model for multi-touch attribution analysis.
    
    Uses transition probabilities between touchpoints to calculate the removal
    effect and attribute conversions fairly across the customer journey.
    """
    
    def __init__(self, order: int = 1, null_state: str = "null", 
                 conversion_state: str = "conversion"):
        """
        Initialize Markov Chain Attribution model.
        
        Args:
            order: Order of the Markov chain (default: 1 for first-order)
            null_state: Name for the null/start state
            conversion_state: Name for the conversion/end state
        """
        self.order = order
        self.null_state = null_state
        self.conversion_state = conversion_state
        self.transition_matrix = {}
        self.channels = set()
        self.removal_effects = {}
        
    def fit(self, journey_data: pd.DataFrame) -> 'MarkovChainAttribution':
        """
        Fit the Markov chain model on customer journey data.
        
        Args:
            journey_data: DataFrame with columns ['customer_id', 'touchpoint', 
                         'timestamp', 'converted'] 
        
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Markov Chain Attribution model")
        
        # Validate input data
        required_cols = ['customer_id', 'touchpoint', 'timestamp', 'converted']
        if not all(col in journey_data.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Build customer journeys
        journeys = self._build_journeys(journey_data)
        
        # Extract unique channels
        self.channels = self._extract_channels(journeys)
        
        # Build transition matrix
        self.transition_matrix = self._build_transition_matrix(journeys)
        
        # Calculate removal effects
        self.removal_effects = self._calculate_removal_effects()
        
        logger.info(f"Model fitted with {len(self.channels)} channels")
        return self
    
    def _build_journeys(self, data: pd.DataFrame) -> List[List[str]]:
        """Build ordered touchpoint sequences for each customer."""
        journeys = []
        
        for customer_id in data['customer_id'].unique():
            customer_data = data[data['customer_id'] == customer_id].sort_values('timestamp')
            
            # Build journey path
            journey = [self.null_state]
            journey.extend(customer_data['touchpoint'].tolist())
            
            # Add conversion state if customer converted
            if customer_data['converted'].iloc[-1]:
                journey.append(self.conversion_state)
            
            journeys.append(journey)
            
        return journeys
    
    def _extract_channels(self, journeys: List[List[str]]) -> Set[str]:
        """Extract all unique channels from journey data."""
        channels = set()
        for journey in journeys:
            channels.update(journey)
        
        # Remove special states
        channels.discard(self.null_state)
        channels.discard(self.conversion_state)
        
        return channels
    
    def _build_transition_matrix(self, journeys: List[List[str]]) -> Dict[str, Dict[str, float]]:
        """Build transition probability matrix from journey sequences."""
        transitions = defaultdict(lambda: defaultdict(int))
        
        # Count transitions
        for journey in journeys:
            for i in range(len(journey) - 1):
                from_state = journey[i]
                to_state = journey[i + 1]
                transitions[from_state][to_state] += 1
        
        # Convert counts to probabilities
        transition_matrix = {}
        for from_state in transitions:
            total_transitions = sum(transitions[from_state].values())
            transition_matrix[from_state] = {
                to_state: count / total_transitions
                for to_state, count in transitions[from_state].items()
            }
        
        return transition_matrix
    
    def _calculate_removal_effects(self) -> Dict[str, float]:
        """Calculate removal effect for each channel using absorption probabilities."""
        removal_effects = {}
        
        # Original conversion probability
        original_prob = self._calculate_absorption_probability(exclude_channels=set())
        
        # Calculate removal effect for each channel
        for channel in self.channels:
            removed_prob = self._calculate_absorption_probability(exclude_channels={channel})
            removal_effect = original_prob - removed_prob
            removal_effects[channel] = max(0, removal_effect)  # Ensure non-negative
        
        # Normalize removal effects to sum to original probability
        total_removal = sum(removal_effects.values())
        if total_removal > 0:
            normalization_factor = original_prob / total_removal
            removal_effects = {
                channel: effect * normalization_factor
                for channel, effect in removal_effects.items()
            }
        
        return removal_effects
    
    def _calculate_absorption_probability(self, exclude_channels: Set[str]) -> float:
        """
        Calculate absorption probability to conversion state using matrix algebra.
        
        Args:
            exclude_channels: Set of channels to exclude from the transition matrix
            
        Returns:
            Probability of reaching conversion state from null state
        """
        # Create modified transition matrix excluding specified channels
        modified_matrix = self._create_modified_matrix(exclude_channels)
        
        if not modified_matrix:
            return 0.0
        
        # Get transient states (excluding conversion state)
        states = list(modified_matrix.keys())
        if self.conversion_state in states:
            states.remove(self.conversion_state)
        
        if not states:
            return 0.0
        
        # Build Q matrix (transient to transient transitions)
        Q = np.zeros((len(states), len(states)))
        for i, from_state in enumerate(states):
            for j, to_state in enumerate(states):
                Q[i, j] = modified_matrix.get(from_state, {}).get(to_state, 0)
        
        # Build R matrix (transient to absorption transitions)  
        R = np.zeros((len(states), 1))
        for i, from_state in enumerate(states):
            R[i, 0] = modified_matrix.get(from_state, {}).get(self.conversion_state, 0)
        
        try:
            # Calculate fundamental matrix N = (I - Q)^(-1)
            I = np.eye(len(states))
            N = np.linalg.inv(I - Q)
            
            # Calculate absorption probabilities B = N * R
            B = N @ R
            
            # Return absorption probability from null state
            if self.null_state in states:
                null_index = states.index(self.null_state)
                return B[null_index, 0]
            else:
                return 0.0
                
        except np.linalg.LinAlgError:
            logger.warning("Matrix inversion failed, returning 0 probability")
            return 0.0
    
    def _create_modified_matrix(self, exclude_channels: Set[str]) -> Dict[str, Dict[str, float]]:
        """Create transition matrix with specified channels removed."""
        modified_matrix = {}
        
        for from_state in self.transition_matrix:
            if from_state in exclude_channels:
                continue
                
            modified_matrix[from_state] = {}
            total_prob = 0
            
            # Copy transitions to non-excluded states
            for to_state, prob in self.transition_matrix[from_state].items():
                if to_state not in exclude_channels:
                    modified_matrix[from_state][to_state] = prob
                    total_prob += prob
            
            # Renormalize probabilities if needed
            if total_prob > 0 and total_prob != 1.0:
                for to_state in modified_matrix[from_state]:
                    modified_matrix[from_state][to_state] /= total_prob
        
        return modified_matrix
    
    def get_attribution_results(self) -> pd.DataFrame:
        """
        Get attribution results as a DataFrame.
        
        Returns:
            DataFrame with columns ['channel', 'attribution_weight', 'removal_effect']
        """
        results = []
        for channel in self.channels:
            results.append({
                'channel': channel,
                'attribution_weight': self.removal_effects.get(channel, 0),
                'removal_effect': self.removal_effects.get(channel, 0)
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
        
        # Calculate contribution of each channel in this journey
        attribution = {}
        total_removal = sum(self.removal_effects.get(ch, 0) for ch in journey_channels)
        
        if total_removal > 0:
            for channel in journey_channels:
                weight = self.removal_effects.get(channel, 0) / total_removal
                attribution[channel] = weight
        
        return attribution
    
    def visualize_transition_graph(self, min_probability: float = 0.01) -> nx.DiGraph:
        """
        Create a NetworkX graph of the transition matrix for visualization.
        
        Args:
            min_probability: Minimum transition probability to include in graph
            
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        
        # Add nodes
        all_states = set(self.transition_matrix.keys())
        for state_dict in self.transition_matrix.values():
            all_states.update(state_dict.keys())
        G.add_nodes_from(all_states)
        
        # Add edges with weights
        for from_state, transitions in self.transition_matrix.items():
            for to_state, prob in transitions.items():
                if prob >= min_probability:
                    G.add_edge(from_state, to_state, weight=prob, probability=prob)
        
        return G
    
    def get_model_statistics(self) -> Dict[str, float]:
        """Get model performance and diagnostic statistics."""
        stats = {
            'num_channels': len(self.channels),
            'num_transitions': len(self.transition_matrix),
            'total_removal_effect': sum(self.removal_effects.values()),
            'max_removal_effect': max(self.removal_effects.values()) if self.removal_effects else 0,
            'min_removal_effect': min(self.removal_effects.values()) if self.removal_effects else 0,
        }
        
        # Channel concentration (Herfindahl index)
        if self.removal_effects:
            total_effect = sum(self.removal_effects.values())
            if total_effect > 0:
                shares = [effect / total_effect for effect in self.removal_effects.values()]
                stats['channel_concentration'] = sum(share ** 2 for share in shares)
        
        return stats


def demo_markov_attribution():
    """Demonstration of Markov Chain Attribution usage."""
    # Sample journey data
    sample_data = pd.DataFrame({
        'customer_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        'touchpoint': ['Search', 'Display', 'Email', 'Social', 'Search', 
                      'Display', 'Search', 'Email', 'Direct'],
        'timestamp': pd.date_range('2024-01-01', periods=9, freq='D'),
        'converted': [True, True, True, True, True, False, False, False, False]
    })
    
    # Initialize and fit model
    model = MarkovChainAttribution()
    model.fit(sample_data)
    
    # Get results
    results = model.get_attribution_results()
    print("Attribution Results:")
    print(results)
    
    # Get model statistics
    stats = model.get_model_statistics()
    print("\nModel Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    demo_markov_attribution()