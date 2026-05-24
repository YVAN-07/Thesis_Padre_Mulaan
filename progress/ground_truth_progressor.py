#ground_truth_progressor.py
"""
Ground Truth Progress based on Distance Metrics.

This module tracks actual physical distance to the target and provides:
  - GT_RAW: Frame-to-frame change in distance (DISTANCE_t - DISTANCE_(t-1))
  - GT_NORM: Normalized distance metric (0.0 = far, 1.0 = very close/touching)

Used for evaluation and validation of vision-based progress metrics.
"""

from progress.window_buffer import WindowBuffer


class GroundTruthProgress:
    """
    Ground truth progress based on Euclidean distance to target.
    
    INPUTS:
        distance: Euclidean distance from end effector to target (meters)
        min_dist: Minimum distance to treat as "far" (e.g., 0.0 meters)
        max_dist: Maximum distance to treat as "very close" (e.g., 0.5 meters)
    
    OUTPUTS:
        GT_RAW (float): Frame-to-frame change in distance
            Negative: Getting closer (good)
            Positive: Moving away (bad)
            Magnitude: Speed of distance change
        
        GT_NORM (float): Window-normalized distance (0.0 to 1.0)
            0.0 = Far away from target
            1.0 = Very close or touching (< 0.015m implies contact)
    
    PURPOSE:
        Validate correlation between vision-based metrics and actual
        physical progress. Not used in reward computation but essential
        for training analysis and algorithm validation.
    """

    def __init__(self, window: WindowBuffer, min_dist=0.0, max_dist=0.5):
        """
        Initialize ground truth progress tracker.
        
        Args:
            window: WindowBuffer for tracking distance values (size=10)
            min_dist: Minimum distance threshold (default 0.0 m)
            max_dist: Maximum distance threshold (default 0.5 m)
        """
        self.window = window
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.prev_dist = None

    def update(self, distance: float):
        """
        Update ground truth metrics based on current distance.
        
        Args:
            distance: Euclidean distance from end effector to target (meters)
        
        Returns:
            tuple: (gt_norm, gt_raw)
            
            gt_norm (float): Normalized distance (0.0 to 1.0)
                0.0 = far (≈ max_dist or beyond)
                1.0 = very close (≈ min_dist or closer)
            
            gt_raw (float): Raw distance change per frame (meters)
                Negative: Getting closer
                Positive: Moving away
        """
        # STEP 1: Calculate GT_RAW (frame-to-frame change in distance)
        if self.prev_dist is None:
            gt_raw = 0.0  # First frame
        else:
            gt_raw = self.prev_dist - distance  # Negative when getting closer
        
        # Save current distance for next frame
        self.prev_dist = distance
        
        # STEP 2: Add distance to rolling window for normalization context
        self.window.add(distance)
        
        # STEP 3: Normalize actual distance value to [0.0, 1.0] range
        # 1.0 = very close   (at or below min_dist)
        # 0.0 = very far     (at or above max_dist)
        dist_range = self.max_dist - self.min_dist
        
        if dist_range <= 0:
            # Invalid range, return neutral
            gt_norm = 0.5
        else:
            # Invert: closer distance → higher value (1.0 is good)
            # Formula: (max_dist - distance) / (max_dist - min_dist)
            gt_norm = (self.max_dist - distance) / dist_range
            # Clamp to [0.0, 1.0]
            gt_norm = max(0.0, min(1.0, gt_norm))
        
        return gt_norm, gt_raw
