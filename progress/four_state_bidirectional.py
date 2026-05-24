"""
Four-State Bidirectional Progress Estimator for Pick-and-Place Tasks

This module tracks progress through four distinct task states using CLIP:
1. INITIAL: Arm at resting position, no motion
2. APPROACHING: Arm moving toward the target object
3. PICKING: Arm grasping the target object
4. DONE: Target object placed on the blue square surface

The system combines:
- CLIP semantic understanding (detecting arm/gripper states)
- Bidirectional progress (detecting direction of motion/progress)
- Task completion detection (red box on blue square)

MATHEMATICAL FOUNDATION:
Given similarity scores to 4 task states: s₁, s₂, s₃, s₄
Apply softmax with temperature τ (lower τ → sharper distribution):

    P_i = exp(s_i / τ) / Σⱼ exp(s_j / τ)

Convert probabilities to single progress value (0-1):

    progress = Σᵢ(state_position_i × P_i)

where state_position = [0.0, 0.33, 0.67, 1.0] for [initial, approaching, picking, done]
"""

import numpy as np


class FourStateProgressTracker:
    """
    Four-state progress tracking using Softmax probability distribution.
    
    Tracks progression through pick-and-place task states with clear
    semantic discrimination using CLIP.
    """
    
    # State position weights in final progress value (0.0 = initial, 1.0 = done)
    STATE_POSITIONS = {
        "initial": 0.0,
        "approaching": 0.33,
        "picking": 0.67,
        "done": 1.0
    }
    
    def __init__(self, temperature=0.1):
        """
        Initialize Four-State progress tracker.
        
        Args:
            temperature (float): Softmax temperature controlling probability sharpness.
                - Lower (0.01-0.1): Sharper distinctions between states
                - Higher (0.5-1.0): Smoother transitions
                - Recommended: 0.1 for clear state separation
        """
        self.temperature = temperature
        self.state_names = ["initial", "approaching", "picking", "done"]
        
        # Track probabilities for logging
        self.last_probabilities = {name: 0.0 for name in self.state_names}
        
        # Track state history for bidirectional analysis
        self.state_history = []
        self.last_progress = 0.0
    
    def update(
            self,
            similarity_initial: float,
            similarity_approaching: float,
            similarity_picking: float,
            similarity_done: float
    ) -> tuple:
        """
        Compute four-state progress from CLIP similarities.
        
        Args:
            similarity_initial (float): CLIP similarity to initial state (0-1)
            similarity_approaching (float): CLIP similarity to approaching state (0-1)
            similarity_picking (float): CLIP similarity to picking state (0-1)
            similarity_done (float): CLIP similarity to done state (0-1)
        
        Returns:
            tuple: (
                state_progress (float): Weighted sum of state probabilities (0-1)
                    0.0 = clearly in initial state
                    0.33 = transitioning to approaching
                    0.67 = in picking state
                    1.0 = task done (red box on blue square)
                probs (dict): Individual state probabilities {state_name: probability}
                    Sums to 1.0, enables detailed logging
                bidirectional_raw (float): Frame-to-frame progress change
                    Positive = moving toward completion
                    Negative = moving away from completion
            )
        """
        # Collect similarities in order matching state_names
        similarities = np.array([
            similarity_initial,
            similarity_approaching,
            similarity_picking,
            similarity_done
        ], dtype=np.float32)

        # Apply softmax with temperature to amplify small differences
        scaled_sims = similarities / self.temperature
        max_scaled = np.max(scaled_sims)
        exp_sims = np.exp(scaled_sims - max_scaled)
        sum_exp = np.sum(exp_sims)
        
        # Guard against division by zero or NaN
        if sum_exp <= 0 or not np.isfinite(sum_exp):
            probabilities = np.ones(len(similarities)) / len(similarities)
        else:
            probabilities = exp_sims / sum_exp
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
        
        # Compute bidirectional progress (frame-to-frame change)
        bidirectional_raw = state_progress - self.last_progress
        self.last_progress = state_progress
        
        # Track state history
        self.state_history.append({
            "progress": state_progress,
            "bidirectional": bidirectional_raw,
            "probabilities": self.last_probabilities.copy()
        })
        
        # Keep history bounded (last 100 frames)
        if len(self.state_history) > 100:
            self.state_history.pop(0)
        
        return float(state_progress), self.last_probabilities, float(bidirectional_raw)
    
    def reset(self):
        """Reset tracker state for new episode."""
        self.last_probabilities = {name: 0.0 for name in self.state_names}
        self.state_history = []
        self.last_progress = 0.0
    
    def get_current_state(self) -> str:
        """Return the most probable current state."""
        if not self.last_probabilities:
            return "unknown"
        return max(self.last_probabilities, key=self.last_probabilities.get)
    
    def get_confidence(self) -> float:
        """Return the confidence (max probability) of current state."""
        if not self.last_probabilities:
            return 0.0
        return max(self.last_probabilities.values())
    
    def is_task_complete(self, confidence_threshold=0.75) -> bool:
        """
        Check if task is complete (done state with high confidence).
        
        Args:
            confidence_threshold (float): Min probability for confident done detection (0-1)
        
        Returns:
            bool: True if in done state with confidence >= threshold
        """
        return (
            self.last_probabilities.get("done", 0.0) >= confidence_threshold
        )


class BidirectionalFourStateEstimator:
    """
    Combines four-state CLIP tracking with bidirectional progress estimation.
    
    This is the main interface for the complete system. It:
    1. Tracks four task states using CLIP similarities
    2. Computes bidirectional progress (direction and magnitude)
    3. Applies optional EMA smoothing
    4. Detects task completion
    5. Enables detailed logging and visualization
    """
    
    def __init__(self, temperature=0.1, ema_alpha=0.7):
        """
        Initialize bidirectional four-state estimator.
        
        Args:
            temperature (float): Softmax temperature for state discrimination (0.1 recommended)
            ema_alpha (float): EMA smoothing factor (0.7 = 70% new, 30% old)
                - Higher (0.8-0.9): More responsive to changes
                - Lower (0.3-0.5): More smoothing, less noise
        """
        self.tracker = FourStateProgressTracker(temperature=temperature)
        self.ema_alpha = ema_alpha
        self.bidirectional_ema = 0.0
    
    def update(
            self,
            similarity_initial: float,
            similarity_approaching: float,
            similarity_picking: float,
            similarity_done: float
    ) -> dict:
        """
        Update estimator with CLIP similarities and compute progress metrics.
        
        Args:
            similarity_initial (float): CLIP similarity to "arm at rest" (0-1)
            similarity_approaching (float): CLIP similarity to "arm approaching box" (0-1)
            similarity_picking (float): CLIP similarity to "arm picking box" (0-1)
            similarity_done (float): CLIP similarity to "red box on blue square" (0-1)
        
        Returns:
            dict: Complete progress metrics:
                {
                    "state_progress": float,           # Overall task progress (0-1)
                    "current_state": str,              # Most probable state name
                    "confidence": float,               # Confidence in current state
                    "probabilities": dict,             # All state probabilities
                    "similarities": dict,              # All raw similarities
                    "bidirectional_raw": float,        # Frame-to-frame change (raw)
                    "bidirectional_ema": float,        # Frame-to-frame change (smoothed)
                    "is_task_complete": bool,          # Task done with high confidence?
                    "completion_confidence": float     # How confident about done state
                }
        """
        # Update tracker
        state_progress, probs, bidir_raw = self.tracker.update(
            similarity_initial,
            similarity_approaching,
            similarity_picking,
            similarity_done
        )
        
        # Apply EMA smoothing to bidirectional signal
        self.bidirectional_ema = (
            self.ema_alpha * bidir_raw + (1.0 - self.ema_alpha) * self.bidirectional_ema
        )
        
        # Check task completion
        is_complete = self.tracker.is_task_complete(confidence_threshold=0.75)
        done_confidence = probs.get("done", 0.0)
        
        return {
            "state_progress": state_progress,
            "current_state": self.tracker.get_current_state(),
            "confidence": self.tracker.get_confidence(),
            "probabilities": probs,
            "similarities": {
                "initial": similarity_initial,
                "approaching": similarity_approaching,
                "picking": similarity_picking,
                "done": similarity_done
            },
            "bidirectional_raw": bidir_raw,
            "bidirectional_ema": self.bidirectional_ema,
            "is_task_complete": is_complete,
            "completion_confidence": done_confidence
        }
    
    def reset(self):
        """Reset for new episode."""
        self.tracker.reset()
        self.bidirectional_ema = 0.0


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
    """
    exp_sims = np.exp((similarities - np.max(similarities)) / temperature)
    return exp_sims / np.sum(exp_sims)
