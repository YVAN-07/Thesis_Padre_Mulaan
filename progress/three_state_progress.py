"""
Three-State Progress Tracking for Robot Manipulation Tasks

This module tracks progress through three distinct robot states:
1. INITIAL STATE: Arm at resting position, gripper open, no interaction
2. PICKING STATE: Arm has approached and grasped an object
3. PLACING STATE: Arm is positioning or placing the grasped object

BENEFITS:
- Clear state discrimination using CLIP semantic understanding
- No training required (zero-shot learning)
- Works with any gripper/object configuration
- Suitable for pick-and-place tasks

MATHEMATICAL FOUNDATION:
Given similarity scores to 3 task states: s₁, s₂, s₃
Apply softmax with temperature τ (lower τ → sharper distribution):

    P_i = exp(s_i / τ) / Σⱼ exp(s_j / τ)

Then convert probabilities to single progress value (0-1):

    progress = Σᵢ(state_position_i × P_i)

where state_position = [0.0, 0.5, 1.0] for [initial, picking, placing]

TEMPERATURE TUNING:
- τ = 0.01: Ultra-sharp (binary-like, one state dominates)
- τ = 0.1:  Sharp (clear state differentiation, RECOMMENDED)
- τ = 0.5:  Moderate (smoother transitions)
- τ = 1.0:  Soft (nearly uniform distribution)
"""

import numpy as np


class ThreeStateProgressTracker:
    """
    Three-state progress tracking using Softmax probability distribution.
    
    Converts CLIP similarity differences between three robot states into
    clear probabilities for state discrimination.
    """
    
    # State position weights in final progress value (0.0 = initial, 1.0 = placing)
    STATE_POSITIONS = {
        "initial": 0.0,
        "picking": 0.5,
        "placing": 1.0
    }
    
    def __init__(self, temperature=0.1):
        """
        Initialize Three-State progress tracker.
        
        Args:
            temperature (float): Softmax temperature controlling probability sharpness.
                - Lower (0.01-0.1): Sharper distinctions between states
                - Higher (0.5-1.0): Smoother transitions
                - Recommended: 0.1 for clear state separation
        """
        self.temperature = temperature
        self.state_names = ["initial", "picking", "placing"]
        
        # Will be set by update() to track probabilities for logging
        self.last_probabilities = {name: 0.0 for name in self.state_names}
    
    def update(
            self,
            similarity_initial: float,
            similarity_picking: float,
            similarity_placing: float
    ) -> tuple:
        """
        Compute three-state progress from CLIP similarities.
        
        Args:
            similarity_initial (float): CLIP cosine similarity to initial state (0-1)
            similarity_picking (float): CLIP cosine similarity to picking state (0-1)
            similarity_placing (float): CLIP cosine similarity to placing state (0-1)
        
        Returns:
            tuple: (
                state_progress (float): Weighted sum of state probabilities (0-1)
                    0.0 = clearly in initial state
                    0.5 = transitioning or in picking state
                    1.0 = clearly in placing state
                probs (dict): Individual state probabilities {state_name: probability}
                    Sums to 1.0, enables detailed logging and debugging
            )
        
        EXAMPLE:
            If similarities are [0.3, 0.6, 0.4] to states [initial, picking, placing]:
            - Softmax with τ=0.1 converts to [0.05, 0.90, 0.05]
            - Progress = 0.0*0.05 + 0.5*0.90 + 1.0*0.05 = 0.50
            - System clearly identifies picking state
        """
        # Collect similarities in order matching state_names
        similarities = np.array([
            similarity_initial,
            similarity_picking,
            similarity_placing
        ], dtype=np.float32)

        # Apply softmax with temperature to amplify small differences
        # Use log-sum-exp trick for numerical stability
        scaled_sims = similarities / self.temperature
        max_scaled = np.max(scaled_sims)
        exp_sims = np.exp(scaled_sims - max_scaled)
        sum_exp = np.sum(exp_sims)
        
        # Guard against division by zero or NaN
        if sum_exp <= 0 or not np.isfinite(sum_exp):
            # Fallback: uniform distribution
            probabilities = np.ones(len(similarities)) / len(similarities)
        else:
            probabilities = exp_sims / sum_exp
            # Check for NaN and fallback if needed
            if not np.all(np.isfinite(probabilities)):
                probabilities = np.ones(len(similarities)) / len(similarities)
        
        # Store probabilities for logging
        self.last_probabilities = {
            name: float(prob) if np.isfinite(prob) else (1.0 / len(self.state_names))
            for name, prob in zip(self.state_names, probabilities)
        }
        
        # Compute single progress value as weighted sum of state probabilities
        state_progress = sum(
            self.STATE_POSITIONS[name] * prob
            for name, prob in self.last_probabilities.items()
        )
        
        # Return progress and detailed probabilities for logging
        return float(state_progress), self.last_probabilities
    
    def reset(self):
        """Reset tracker state for new episode."""
        self.last_probabilities = {name: 0.0 for name in self.state_names}
        # Clear any spatial bias left from previous episode
        if hasattr(self, "_last_spatial_bias"):
            try:
                del self._last_spatial_bias
            except Exception:
                pass
        if hasattr(self, "_last_bias_weight"):
            try:
                del self._last_bias_weight
            except Exception:
                pass

    def set_spatial_bias(self, bias, weight=0.0):
        """
        Optional: Set a spatial bias vector and weight that will be added to
        similarities before softmax. `bias` must be iterable of length 3.

        Args:
            bias: iterable of 3 floats corresponding to states [initial, picking, placing]
            weight: scalar weight applied to the bias before adding to similarities
        """
        try:
            b = np.asarray(bias, dtype=np.float32)
            if b.shape[0] == len(self.state_names):
                self._last_spatial_bias = b
                self._last_bias_weight = float(weight)
        except Exception:
            # ignore invalid bias
            pass


def softmax_with_temperature(similarities: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    """
    Apply Softmax with temperature to amplify small differences.
    
    Formula: P_i = exp(s_i / τ) / Σ(exp(s_j / τ))
    
    Args:
        similarities (np.ndarray): Similarity scores (shape: (N,))
        temperature (float): Softmax temperature
            - Lower values (0.01-0.1): Sharper, more binary-like distribution
            - Higher values (0.5-1.0): Smoother, more uniform distribution
    
    Returns:
        np.ndarray: Probability distribution (sums to 1.0, shape matches input)
    
    EXAMPLE:
        similarities = [0.45, 0.65, 0.50]  # Picking state is higher
        temp = 0.1
        → probabilities = [0.10, 0.80, 0.10]  # Clear picking identification
    """
    # Subtract max for numerical stability (prevents overflow)
    exp_sims = np.exp((similarities - np.max(similarities)) / temperature)
    return exp_sims / np.sum(exp_sims)
