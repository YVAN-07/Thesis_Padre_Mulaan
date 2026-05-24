# progress_logger.py
"""
CSV-only logger for autonomous episode recording.
Creates sequential CSV files (csv_001.csv, csv_002.csv, ...) for each episode.
One row per frame with direct disk writes and no buffering.
"""
import csv
import os
from datetime import datetime


class ProgressLogger:
    """
    Lightweight CSV logger for autonomous episode recording with FULL PRECISION PRESERVATION.
    Creates sequential CSV files for each episode without trial folders.
    
    CRITICAL: FULL FLOATING-POINT PRECISION PRESERVATION FOR ALL METRICS
    =====================================================================
    Promise:
        - All values written as raw Python float objects (IEEE 754)
        - NO formatting, rounding, or truncation before CSV write
        - Path: sensor.getValue() -> log() -> csv.writer -> disk (unchanged)
        - Very small deltas (e.g., 0.000000000001234) preserved exactly
        - csv.writer: 15-17 significant digits preserved for all floats
    
    CSV Format:
        Columns: timestep, STAGE_SIMILARITY_FAR, STAGE_SIMILARITY_APPROACHING,
                 STAGE_SIMILARITY_NEAR, STAGE_SIMILARITY_TOUCHING, STAGE_PROGRESS,
                 STAGE_PROB_FAR, STAGE_PROB_APPROACHING, STAGE_PROB_NEAR,
                 STAGE_PROB_TOUCHING, BIDIRECTIONAL_RAW, BIDIRECTIONAL_EMA,
                 BIDIRECTIONAL_NORM, BIDIRECTIONAL_VISUAL, DISTANCE, GT_RAW,
                 GT_NORM, EPISODE_PROGRESS
        Each row = one frame (32ms), all numeric values at full precision
    
    Behavior:
        - Creates sequential CSV files: csv_001.csv, csv_002.csv, etc.
        - Writes one row per frame directly to disk (no buffering)
        - Flushes after each write for real-time safety
        - Closes safely when episode ends
        - Zero in-memory buffering or tensor storage
    
    Verification:
        Open CSV in TEXT EDITOR (NOT Excel) to inspect raw float precision.
        - Tiny deltas appear as: 0.000000123456789 or 1.23e-10
        - Full precision: 0.7499999999999999 (not 0.75)
    """

    # Column names matching CSV header per metrics definition
    # Updated for four-state bidirectional system
    COLUMNS = [
        "timestep",
        "sim_initial",
        "sim_approaching",
        "sim_picking",
        "sim_done",
        "state_progress",
        "state_prob_initial",
        "state_prob_approaching",
        "state_prob_picking",
        "state_prob_done",
        "done_detected",
        "red_on_blue",
        "task_complete",
        "bidirectional_raw",
        "bidirectional_ema",
        "distance",
        "current_state",
        "state_confidence",
        "done_confidence",
        "episode_progress",
    ]

    def __init__(self, log_dir="logs", episode_number=1):
        """
        Initialize CSV logger for a new episode.
        
        Args:
            log_dir (str): Base directory to save CSV files
            episode_number (int): Sequential episode number (1, 2, 3, ...)
        """
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Create filename: csv_###.csv (zero-padded to 3 digits)
        filename = f"csv_{episode_number:03d}.csv"
        self.csv_path = os.path.join(log_dir, filename)
        
        # Open file for writing
        self.file = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.file)

        # Write CSV header
        self.writer.writerow(self.COLUMNS)
        self.file.flush()  # Ensure header is written immediately
        
        self.episode_number = episode_number
        self.t = 0  # Frame counter for this episode

    def log(
        self,
        sim_initial,
        sim_approaching,
        sim_picking,
        sim_done,
        state_progress,
        state_prob_initial,
        state_prob_approaching,
        state_prob_picking,
        state_prob_done,
        done_detected,
        red_on_blue,
        task_complete,
        bidirectional_raw,
        bidirectional_ema,
        distance,
        current_state,
        state_confidence,
        done_confidence,
        episode_progress,
    ):
        """
        Log one timestep of data directly to CSV file WITH FULL PRECISION.
        
        *** CRITICAL: PASS RAW FLOAT VALUES ONLY ***
        Do NOT pre-format, round, or truncate before calling this method.
        
        WRONG:  log(round(value, 3), ...)
        WRONG:  log(f"{value:.4f}", ...)
        RIGHT:  log(value, ...)  # Raw float, full IEEE 754 precision preserved
        
        Args:
            sim_initial (float): CLIP similarity to initial state (0-1)
            sim_approaching (float): CLIP similarity to approaching state (0-1)
            sim_picking (float): CLIP similarity to picking state (0-1)
            sim_done (float): CLIP similarity to done state (0-1)
                
            state_progress (float): Four-state progress metric (typically 0-1)
                Computed by converting state similarities to probabilities via Softmax,
                then taking weighted sum: progress = Σ(state_position_i × P_i)
                where state_position = [0.0, 0.33, 0.67, 1.0] for the 4 states
                
            state_prob_initial (float): Softmax probability of initial state (0-1)
            state_prob_approaching (float): Probability of approaching state (0-1)
            state_prob_picking (float): Probability of picking state (0-1)
            state_prob_done (float): Probability of done state (0-1)
                All 4 probabilities sum to 1.0
            
            done_detected (bool): Color detection found red box on blue square
            red_on_blue (bool): Is red currently on blue?
            task_complete (bool): Task complete (CLIP or color detection)?
            
            bidirectional_raw (float): *** CRITICAL ***
                Temporal delta in state_progress from previous frame
                MUST be raw float with full IEEE 754 precision
                Preserve tiny deltas: 0.000000000001234
                NEVER apply deadband, thresholding, or rounding
                
            bidirectional_ema (float): EMA smoothing of raw (raw, unformatted)
                
            distance (float): Euclidean distance to target in meters (raw float)
            current_state (str): Most probable state name
            state_confidence (float): Confidence in current state (0-1)
            done_confidence (int): Done detection confidence (frames on blue)
            
            episode_progress (float): Cumulative state progress since episode start
                Formula: state_progress - episode_start_state_progress
                Raw float with full IEEE 754 precision, no smoothing/normalization
        """
        # Write row directly to CSV preserving full floating-point precision
        # csv.writer.writerow() converts Python floats to strings
        # Conversion preserves: 15-17 significant digits (IEEE 754 standard)
        # Result: VERY SMALL DELTAS recorded exactly (e.g., 1.23e-10)
        row = [
            self.t,
            sim_initial,                    # Python float -> string (full precision)
            sim_approaching,                # Python float -> string (full precision)
            sim_picking,                    # Python float -> string (full precision)
            sim_done,                       # Python float -> string (full precision)
            state_progress,                 # Four-state progress (0-1), full precision
            state_prob_initial,             # Probability of initial state
            state_prob_approaching,         # Probability of approaching state
            state_prob_picking,             # Probability of picking state
            state_prob_done,                # Probability of done state
            1 if done_detected else 0,      # Boolean to int
            1 if red_on_blue else 0,        # Boolean to int
            1 if task_complete else 0,      # Boolean to int
            bidirectional_raw,              # CRITICAL: RAW float, 15-17 sig figs
            bidirectional_ema,              # Python float -> string (full precision)
            distance,                       # Python float -> string (full precision)
            current_state,                  # State name string
            state_confidence,               # Confidence in current state
            done_confidence,                # Frames on blue (int)
            episode_progress,               # Cumulative progress, full precision
        ]
        self.writer.writerow(row)
        self.file.flush()  # Flush immediately for real-time safety
        self.t += 1

    def close(self):
        """Close CSV file."""
        self.file.close()
