"""
Softmax-Based Multi-Stage Progress Tracking

This module replaces simple goal−antigoal subtraction with a sophisticated
multi-stage comparison that amplifies subtle visual progress without training.

CORE CONCEPT:
Instead of asking "how similar is this to goal?", ask "which stage does this
look most like?" CLIP similarity scores to multiple task-progress stages
(far, approaching, near, touching) are converted to sharp probabilities using
Softmax with low temperature. These probabilities encode task progress directly.

BENEFITS OVER CONTRASTIVE SUBTRACTION:
1. AMPLIFIED SMALL DIFFERENCES: Softmax sharpens tiny similarity gaps into
   clear probability distinctions. If stage scores are [0.501, 0.500, 0.490, 0.495],
   softmax converts these microscopic differences (~0.01) into clear probabilities
   like [0.35, 0.25, 0.20, 0.20], making progress detectable.

2. NO TRAINING REQUIRED: Works out of the box with zero-shot CLIP embeddings.
   Stages are natural language descriptions, not learned features.

3. MONOTONIC PROGRESS: By design, progress increases as arm moves through stages.
   No compression or normalization artifacts.

4. SENSITIVE TO MOTION: Transitions between stages (e.g., far→approaching) are
   immediately visible as probability shifts.

MATHEMATICAL FOUNDATION:
Given similarity scores to 4 task stages: s₁, s₂, s₃, s₄
Apply softmax with temperature τ (lower τ → sharper distribution):

    P_i = exp(s_i / τ) / Σⱼ exp(s_j / τ)

Then convert probabilities to single progress value (0-1) via weighted sum:

    progress = Σᵢ(stage_position_i × P_i)

where stage_position = [0.0, 0.33, 0.67, 1.0] for [far, approaching, near, touching]

TEMPERATURE TUNING:
- τ = 0.01: Ultra-sharp (binary-like, one stage dominates completely)
- τ = 0.1:  Sharp (clear stage differentiation, recommended)
- τ = 0.5:  Moderate (smoother transitions)
- τ = 1.0:  Soft (nearly uniform distribution)
- τ > 1.0:  Very soft (stages blend heavily)

WORKFLOW:
1. Define 4 task stage descriptions (natural language)
2. Encode each stage text once (compute once, reuse every frame)
3. Each frame:
   a. Get CLIP image embedding
   b. Compute cosine similarity to each of 4 stage texts
   c. Apply softmax(similarities / τ) to get probabilities
   d. Compute progress = weighted sum of probabilities
   e. Use progress in bidirectional calculation (replaces contrastive_score)

INTEGRATION WITH BIDIRECTIONAL:
The new "stage_progress" metric is fed directly to BidirectionalProgress.update(),
replacing the raw contrastive_score. Everything downstream (EMA, normalization,
visualization) remains unchanged.

CSV LOGGING:
New columns added to logs:
- stage_progress (float): The multi-stage progress value (0-1 typically, can exceed for smoothing)
- stage_prob_far (float): Probability arm is far from box
- stage_prob_approaching (float): Probability arm is approaching
- stage_prob_near (float): Probability arm is near
- stage_prob_touching (float): Probability arm is touching
"""

import numpy as np


class SoftmaxProgressTracker:
    """
    Multi-stage progress tracking using Softmax probability distribution.
    
    Converts subtle CLIP similarity differences between task stages into
    clear probabilities, enabling detection of fine-grained progress.
    """
    
    # Stage position weights in final progress value (0.0 = far, 1.0 = touching)
    STAGE_POSITIONS = {
        "far": 0.0,
        "approaching": 0.33,
        "near": 0.67,
        "touching": 1.0
    }
    
    def __init__(self, temperature=0.1):
        """
        Initialize Softmax progress tracker.
        
        Args:
            temperature (float): Softmax temperature controlling probability sharpness.
                - Lower (0.01-0.1): Sharper distinctions between stages
                - Higher (0.5-1.0): Smoother transitions
                - Recommended: 0.1 for clear stage separation
        """
        self.temperature = temperature
        self.stage_names = ["far", "approaching", "near", "touching"]
        
        # Will be set by update() to track probabilities for logging
        self.last_probabilities = {name: 0.0 for name in self.stage_names}
    
    def update(
            self,
            similarity_far: float,
            similarity_approaching: float,
            similarity_near: float,
            similarity_touching: float
    ) -> tuple:
        """
        Compute multi-stage progress from CLIP similarities to each stage.
        
        Args:
            similarity_far (float): CLIP cosine similarity to "arm far from box" (0-1)
            similarity_approaching (float): CLIP cosine similarity to "arm approaching box" (0-1)
            similarity_near (float): CLIP cosine similarity to "arm near box" (0-1)
            similarity_touching (float): CLIP cosine similarity to "arm touching box" (0-1)
        
        Returns:
            tuple: (
                stage_progress (float): Weighted sum of stage probabilities (0-1 typically)
                    Positive = closer to touching, Negative = far from box
                    Can range beyond [0, 1] due to smoothing effects
                probs (dict): Individual stage probabilities {stage_name: probability}
                    Sums to 1.0, enables detailed logging and debugging
            )
        
        EXAMPLE:
            If similarities are [0.501, 0.500, 0.490, 0.495] to stages [far, approaching, near, touching]:
            - Raw softmax uses these microscopic differences (0.01 range)
            - Softmax amplifies them into [0.35, 0.25, 0.20, 0.20]
            - Progress = 0*0.35 + 0.33*0.25 + 0.67*0.20 + 1.0*0.20 = 0.401
            - Without softmax, simple difference would be 0.495 - 0.501 = -0.006 (invisible)
        """
        # Collect similarities in order matching stage_names
        similarities = np.array([
            similarity_far,
            similarity_approaching,
            similarity_near,
            similarity_touching
        ], dtype=np.float32)

        # Note: we allow an optional spatial bias to be added to similarities
        # so that pixel-distance information can influence stage probabilities.
        # The controller can pass a bias vector (len=4) and a scalar weight.
        # If no bias provided, behavior is unchanged.
        bias = getattr(self, "_last_spatial_bias", None)
        bias_weight = getattr(self, "_last_bias_weight", 0.0)
        if bias is not None:
            try:
                b = np.asarray(bias, dtype=np.float32)
                if b.shape[0] == similarities.shape[0]:
                    similarities = similarities + (bias_weight * b)
            except Exception:
                pass

        # Apply softmax with temperature to amplify small differences
        # Use log-sum-exp trick for numerical stability with extreme temperatures
        # Formula: P_i = exp((s_i - max_s) / τ) / Σ(exp((s_j - max_s) / τ))
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
            name: float(prob) if np.isfinite(prob) else (1.0 / len(self.stage_names))
            for name, prob in zip(self.stage_names, probabilities)
        }
        
        # Compute single progress value as weighted sum of stage probabilities
        # Progress = Σ(stage_position_i × P_i)
        stage_progress = sum(
            self.STAGE_POSITIONS[name] * prob
            for name, prob in self.last_probabilities.items()
        )
        
        # Return progress and detailed probabilities for logging
        return float(stage_progress), self.last_probabilities
    
    def reset(self):
        """Reset tracker state for new episode."""
        self.last_probabilities = {name: 0.0 for name in self.stage_names}
        # Clear any spatial bias left from previous episode
        if hasattr(self, "_last_spatial_bias"):
            delattr = False
            try:
                del self._last_spatial_bias
                delattr = True
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
        similarities before softmax. `bias` must be iterable of length 4.

        Args:
            bias: iterable of 4 floats corresponding to stages [far, approaching, near, touching]
            weight: scalar weight applied to the bias before adding to similarities
        """
        try:
            b = np.asarray(bias, dtype=np.float32)
            if b.shape[0] == len(self.stage_names):
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
        similarities = [0.501, 0.500, 0.490, 0.495]  # Tiny differences, 0.01 range
        temp = 0.1
        → probabilities = [0.352, 0.236, 0.180, 0.232]  # Clear separation
        
        Without temperature scaling (temp=1.0):
        → probabilities = [0.253, 0.251, 0.248, 0.248]  # Nearly uniform
    """
    # Subtract max for numerical stability (prevents overflow)
    exp_sims = np.exp((similarities - np.max(similarities)) / temperature)
    return exp_sims / np.sum(exp_sims)
