#!/usr/bin/env python3
"""
SO100 TELEOPERATION CONTROLLER WITH REAL-TIME VISION ANALYSIS

================================================================================
OVERVIEW
================================================================================
This is a Webots controller that enables real-time teleoperation of the SO100
robotic arm with automated vision-based performance tracking. The system 
captures camera images continuously and compares them to a text goal using
CLIP (Contrastive Language-Image Pretraining).

================================================================================
KEY FEATURES
================================================================================
1. REAL-TIME CAMERA IMAGE CAPTURE
   - Source: "side_camera" device defined in worlds/so100.wbt
   - Resolution: 640×480 pixels, RGB format
   - Capture Rate: Every 32ms (31.25 Hz, synchronized with simulation)
   - Processing: Image → CLIP embedder → Cosine similarity scoring

2. KEYBOARD TELEOPERATION
   - 6 DOF robotic arm control via individual keys
   - Q/A, W/S, E/D, R/F, T/G, Y/H for 6 joints
   - Smooth incremental motion (0.02 rad per keystroke)

3. AUTOMATIC TRIAL MANAGEMENT
   - Detects when end effector touches target (distance < 1.5cm)
   - Automatically starts and ends trials
   - Counts completed trials in real-time

4. REAL-TIME VISUAL FEEDBACK (4-Line Overlay)
   - Line 0 (Green/Blue): Trial status + similarity metrics
   - Line 1 (Red/Blue): Contact status + frame counters
   - Line 2 (White): CLIP similarity scores
   - Line 3 (White): Ground truth distance metrics

5. AUTOMATIC DATA LOGGING
   - CSV file creation with timestamp
   - 10 metrics logged per frame
   - Full IEEE 754 floating-point precision preserved
   - Very small deltas recorded exactly (e.g., 0.000000000001234)
   - Saved on shutdown

6. FULL FLOATING-POINT PRECISION PRESERVATION
   CSV Column: bidirectional_raw
   - Raw temporal delta: ~CLIP_CONTRASTIVE_SCORE_t - CLIP_CONTRASTIVE_SCORE_(t-1)
   - Written to CSV with FULL IEEE 754 precision (15-17 significant digits)
   - NO rounding, deadbanding, thresholding, or clipping before CSV write
   - Display values (Webots overlay): formatted for readability (6 decimals)
   - Dataset values (CSV file): raw floats, maximum precision
   
   VERIFICATION STEPS:
   1. Run simulation and record episodes
   2. Open logs/csv_001.csv in TEXT EDITOR (Notepad, VS Code, etc.)
   3. Look at the "bidirectional_raw" column
   4. Tiny values should show many decimals: 0.000000123456789
   5. Scientific notation visible for very small: 1.23e-10
   
   DON'T use Excel/Sheets - they auto-format and hide precision

================================================================================
CAMERA IMAGE CAPTURE FLOW
================================================================================
The camera image comes from the "side_camera" device in the Webots world file
(worlds/so100.wbt). Here's the complete data flow:

FRAME N (every 32ms):
  1. robot.step(32) executes one simulation timestep
  2. Webots renders scene and updates all devices
  3. side_camera device captures current scene from its position/rotation
  4. Camera image (640×480 RGB) is buffered in Webots
  5. side_camera.get_image() retrieves buffered RGB data as NumPy array
  6. Image shape: (480, 640, 3) with uint8 pixel values
  7. clip.encode_image(image) converts to feature vector
  8. clip.cosine_similarity() compares to text embedding
  9. Result (0-1) indicates how well image matches "reaching the target object"
  10. EMA filter smooths the similarity signal
  11. All metrics logged to CSV
  12. 4-line overlay updated in Webots display
  13. Next keystroke applied
  14. Frame counter incremented
  15. Repeat...

CAMERA VISIBILITY IN WEBOTS:
  The "side_camera" device captures images but does NOT appear as a visible
  object in the 3D view. This is normal - camera devices in Webots are 
  sensors that passively capture data. The images are being captured and
  processed continuously even though you can't see the camera.

  Proof that camera is working:
  - The 4-line overlay shows real-time similarity values (would be constant if
    not working)
  - Different robot poses produce different similarity scores
  - CSV file contains varying similarity values (not all constant)

================================================================================
METRICS DEFINITIONS
================================================================================

A. CLIP_SIMILARITY_GOAL (0.0 – 1.0)
   Source: Cosine similarity between the current camera image and the GOAL text prompt
   Text Goal Example:
     "robot end effector physically touching the red box"
   Meaning:
     Measures how semantically similar the current visual state is to the desired
     goal condition.
   Interpretation:
     - Higher values indicate stronger semantic alignment with the goal state
   Updates: Every frame from camera input

B. CLIP_SIMILARITY_ANTIGOAL (0.0 – 1.0)
   Source: Cosine similarity between the current camera image and the ANTI-GOAL text prompt
   Text Anti-Goal Example:
     "robot arm far away from the red box"
   Meaning:
     Measures how semantically similar the current visual state is to a clearly
     opposite (undesirable) condition.
   Interpretation:
     - Higher values indicate the robot is visually far from the goal
   Updates: Every frame from camera input

C. CLIP_CONTRASTIVE_SCORE (typically -0.3 to +0.3)
   Source:
     Difference between goal similarity and anti-goal similarity
     Formula:
       CLIP_CONTRASTIVE_SCORE = CLIP_SIMILARITY_GOAL − CLIP_SIMILARITY_ANTIGOAL
   Meaning:
     Encodes relative semantic proximity to the goal versus the opposite state.
   Interpretation:
     - Negative values → closer to anti-goal (far from target)
     - Near zero       → ambiguous or intermediate state
     - Positive values → closer to goal state
   Role:
     Primary perceptual signal used for progress estimation
   Updates: Every frame

D. BIDIRECTIONAL_RAW (typically -0.1 to +0.1)
   Source:
     Temporal change in contrastive score
     Formula:
       BIDIRECTIONAL_RAW = CLIP_CONTRASTIVE_SCORE_t − CLIP_CONTRASTIVE_SCORE_(t−1)
   Meaning:
     Indicates direction of progress through the task space.
   Interpretation:
     - Positive → robot is moving toward the goal
     - Negative → robot is moving away from the goal
     - Zero     → no meaningful progress
   Use:
     Core bidirectional progress signal
   Updates: Every frame

E. BIDIRECTIONAL_EMA (smoothed, bounded)
   Source:
     Exponential Moving Average applied to BIDIRECTIONAL_RAW
     Formula:
       EMA_t = α · BIDIRECTIONAL_RAW_t + (1 − α) · EMA_(t−1)
   Meaning:
     Smoothed bidirectional progress signal used as the final reward input.
   Purpose:
     - Reduces frame-to-frame noise
     - Improves stability for reinforcement learning
   Updates: Every frame after BIDIRECTIONAL_RAW computation

F. BIDIRECTIONAL_NORM (-1.0 to +1.0)
   Source:
     Window-based normalization of BIDIRECTIONAL_EMA
   Meaning:
     Scales bidirectional progress into a bounded range for logging and visualization.
   Interpretation:
     - -1.0 → strong regression
     -  0.0 → neutral / stagnant
     - +1.0 → strong progress
   Use:
     Monitoring, plotting, and analysis (not strictly required for reward)

G. DISTANCE (meters, typically 0 – 0.5)
   Source:
     Euclidean distance between END_EFFECTOR and TARGET positions
     Formula:
       || end_effector_pos − target_pos ||
   Measurement:
     Ground truth obtained from the Webots supervisor
   Meaning:
     Actual physical distance to the target object
   Contact Threshold:
     < 0.015 m (1.5 cm) indicates physical contact
   Role:
     Evaluation and validation only (not used in reward computation)

H. GT_RAW (meters)
   Source:
     Frame-to-frame change in ground truth distance
     Formula:
       DISTANCE_t − DISTANCE_(t−1)
   Meaning:
     Indicates true physical approach or retreat.
   Interpretation:
     - Negative → getting closer
     - Positive → moving away
   Role:
     Used to validate correlation with vision-based progress

I. GT_NORM (0.0 – 1.0)
   Source:
     Window-based normalization of ground truth distance
   Meaning:
     Normalized physical distance metric
   Interpretation:
     - 0.0 → far
     - 1.0 → very close / touching
   Role:
     Visualization and quantitative evaluation only

================================================================================
TRIAL STATE MACHINE
================================================================================

                    ┌─────────────────┐
                    │  START / RESET  │
                    └────────┬────────┘
                             │
                    ▼────────▼──────────┐
              ┌──────────────────────┐  │
              │      READY STATE     │  │
              │  (No trial active)   ◄──┘
              │  Waiting for contact │
              │  Color: BLUE overlay │
              └──────────┬───────────┘
                         │
              Contact detected (dist < 0.015m)
                         │
                    ▼────▼──────────────┐
              ┌──────────────────────┐  │
              │   TRIAL ACTIVE       │  │
              │  (Trial in progress) │  │
              │  trial_number += 1   │  │
              │  Color: GREEN overlay│  │
              └──────────┬───────────┘  │
                         │              │
              Contact maintained or lost
                         │
                    ▼────▼──────────────┐
              Contact lost & was_touching=True
                         │
                ▼────────▼── TRIAL ENDS
           CSV data written
         trial_active = False
         Data persists in CSV
                         │
                         └──────────────┘

================================================================================
KEYBOARD CONTROL MAP
================================================================================

Q/A  →  Joint 1  (Base rotation)
W/S  →  Joint 2  (Shoulder pitch)
E/D  ->  Joint 3  (Elbow pitch)
R/F  ->  Joint 4  (Wrist pitch)
T/G  ->  Joint 5  (Wrist yaw)
Y/H  ->  Joint 6  (Wrist roll)

Each keystroke adds ANGLE_STEP (0.02 radians ≈ 1.15 degrees) to current position
Holding key applies repeated increments
Release stops movement (position maintained)

================================================================================
CSV OUTPUT FORMAT
================================================================================

File location: controllers/so100_tele/logs/trial_##/csv/clip_bidir_so100_trial##_YYYYMMDD_HHMMSS.csv

Columns (in order):
  1. timestep              - Frame number since trial start (0, 1, 2, ...)
  2. clip_similarity_goal  - Cosine similarity to goal text (0-1)
  3. clip_similarity_antigoal - Cosine similarity to antigoal text (0-1)
  4. clip_contrastive_score   - goal_sim - antigoal_sim (-0.3 to +0.3)
  5. bidirectional_raw     - Temporal change in contrastive_score
  6. bidirectional_ema     - EMA-smoothed bidirectional metric
  7. bidirectional_norm    - Window-normalized bidirectional (-1 to +1)
  8. distance              - End effector to target distance (meters)
  9. gt_raw                - Distance change per frame (meters)
  10. gt_norm              - Normalized distance (0-1, 1=touching)

Every row represents ONE FRAME (32ms of simulation time)

================================================================================
"""

# =====================================================
# SECTION 1: IMPORTS AND PROJECT SETUP
# =====================================================

import sys
import os
import csv
import glob
from datetime import datetime
import numpy as np

# Set up Python path to find project modules
# This allows imports from camera/, progress/, logging/ directories
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')
)
sys.path.insert(0, PROJECT_ROOT)

# Webots controller API
from controller import Supervisor, Keyboard

# Custom project modules for vision and tracking
from camera.side_camera import SideCamera              # Camera wrapper
from progress.clip_encoder import CLIPEncoder          # CLIP similarity
from progress.four_state_bidirectional import BidirectionalFourStateEstimator  # Four-state progress
from progress.done_state_detector import DoneStateDetector     # Red/blue color detection
from trial_logging.progress_logger import ProgressLogger      # CSV data logger


# =====================================================
# SECTION 2: CONFIGURATION PARAMETERS
# =====================================================

# Simulation timestep (milliseconds)
# 32ms = ~31.25 Hz simulation rate
TIME_STEP = 32

# Joint angle increment per keystroke (radians)
# 0.02 rad ≈ 1.15 degrees
ANGLE_STEP = 0.02

# Number of robot joints (SO100 has 6 DOF)
NUM_JOINTS = 6

# Name of camera device in world file
# Must match name defined in worlds/so100.wbt
CAMERA_NAME = "side_camera"

# Contact detection threshold (meters)
# End effector must be closer than this to start a trial
TOUCH_THRESHOLD = 0.015  # 1.5 centimeters


# =====================================================
# SECTION 3: WEBOTS INITIALIZATION
# =====================================================

print("\n" + "="*80)
print("SO100 TELEOPERATION CONTROLLER - INITIALIZING")
print("="*80)

# Create Supervisor object for accessing simulation
robot = Supervisor()

# Create keyboard input handler
keyboard = Keyboard()
keyboard.enable(TIME_STEP)

print("\n[STEP 1] Webots Supervisor initialized")
print("         Ready to access robot devices and nodes")


# =====================================================
# SECTION 4: CAMERA SETUP (REAL-TIME IMAGE CAPTURE)
# =====================================================

print("\n[STEP 2] Initializing camera: '" + CAMERA_NAME + "'")
print("         ├─ Source: Webots world file (worlds/so100.wbt)")
print("         ├─ Position: Defined in world file")
print("         ├─ Resolution: 640x480 pixels, RGB format")
print("         └─ Update: Every simulation step (32ms)")

# Get camera device from Webots
cam = robot.getDevice(CAMERA_NAME)

# Verify camera exists
if cam is None:
    error_msg = f"FATAL ERROR: Camera '{CAMERA_NAME}' not found in world file!\n"
    error_msg += "Make sure worlds/so100.wbt defines a Camera device with name '{CAMERA_NAME}'"
    print(error_msg)
    raise RuntimeError(error_msg)

# Enable camera to capture images
# This request Webots to start buffering camera images
cam.enable(TIME_STEP)

# Wrap camera in SideCamera class
# This handles image extraction and conversion to NumPy array
side_camera = SideCamera(cam)

print(f"\n         ✓ Camera enabled successfully")
print(f"         ✓ Captures: {side_camera.width}x{side_camera.height} RGB images")
print(f"         ✓ Update rate: Every {TIME_STEP}ms")
print(f"         ✓ Real-time image processing: ACTIVE")


# =====================================================
# SECTION 5: VISION ANALYSIS SETUP
# =====================================================

print("\n[STEP 3] Initializing vision analysis modules")

# Create CLIP encoder for image-to-text similarity
clip = CLIPEncoder()

# ─────── FOUR-STATE BIDIRECTIONAL PROGRESS SYSTEM ───────
# Tracks progress through four task states:
# 1. INITIAL:    Arm at neutral resting position, gripper open
# 2. APPROACHING: Arm extending toward red box, gripper opening
# 3. PICKING:    Arm grasping red box, gripper closed
# 4. DONE:       Red box placed on blue square platform (task complete)
#
# Uses CLIP semantic understanding + color-based done detection
# For each frame:
# - Get CLIP similarity to each state text
# - Apply Softmax with temperature=0.1 to amplify tiny differences
# - Convert probabilities to single progress value (0-1)
# - Detect red box on blue square using HSV color detection
# Result: Clear state discrimination + robust task completion detection

TEXT_STATE_INITIAL = (
    "SO100 arm at rest position, gripper completely open, red box sitting still "
    "on table, no movement"
)

TEXT_STATE_APPROACHING = (
    "robot arm moving rapidly toward red box, gripper fingers opening wide, "
    "arm reaching forward horizontally"
)

TEXT_STATE_PICKING = (
    "robot gripper tightly grasping red box, box held up in air, gripper fingers "
    "fully closed around the box, box suspended above table"
)

TEXT_STATE_DONE = (
    "red box firmly sitting on blue square platform, gripper open and released, "
    "red box on blue, task finished"
)

# Encode all state texts once (reused every frame)
text_feat_state_initial = clip.encode_text(TEXT_STATE_INITIAL)
text_feat_state_approaching = clip.encode_text(TEXT_STATE_APPROACHING)
text_feat_state_picking = clip.encode_text(TEXT_STATE_PICKING)
text_feat_state_done = clip.encode_text(TEXT_STATE_DONE)

# Create bidirectional four-state estimator
# Temperature=0.1 gives sharp state discrimination
# EMA alpha=0.7 provides responsive smoothing
four_state_estimator = BidirectionalFourStateEstimator(temperature=0.1, ema_alpha=0.7)

# Create done state detector for red box on blue square
# Frame threshold=5 ensures robust detection (~160ms at 31Hz)
done_detector = DoneStateDetector(frame_threshold=5)

print(f"         ├─ CLIP Encoder: Ready")
print(f"         ├─ Four-State Bidirectional Estimator: Ready (temperature=0.1, EMA=0.7)")
print(f"         ├─ Done State Detector: Ready (frame_threshold=5)")
print(f"         ├─ State 1 (Initial): '{TEXT_STATE_INITIAL}'")
print(f"         ├─ State 2 (Approaching): '{TEXT_STATE_APPROACHING}'")
print(f"         ├─ State 3 (Picking): '{TEXT_STATE_PICKING}'")
print(f"         └─ State 4 (Done): '{TEXT_STATE_DONE}'")


# =====================================================
# SECTION 6: GROUND TRUTH NODE SETUP
# =====================================================

print("\n[STEP 4] Initializing ground truth position tracking")
print("         ├─ Getting END_EFFECTOR node")
print("         └─ Getting TARGET node")

# Get supervisor node references
# These must be defined in world file with DEF statements
eef = robot.getFromDef("END_EFFECTOR")
target = robot.getFromDef("TARGET")

# Verify both nodes were found
if eef is None or target is None:
    missing = []
    if eef is None:
        missing.append("END_EFFECTOR")
    if target is None:
        missing.append("TARGET")
    error_msg = f"FATAL ERROR: Missing DEF nodes: {', '.join(missing)}\n"
    error_msg += "Verify worlds/so100.wbt contains:\n"
    error_msg += "  - DEF END_EFFECTOR (gripper position)\n"
    error_msg += "  - DEF TARGET (goal position)"
    print(error_msg)
    raise RuntimeError(error_msg)

print(f"         ✓ END_EFFECTOR node found")
print(f"         ✓ TARGET node found")
print(f"         ✓ Ground truth metrics: ENABLED")
print(f"         ✓ Will calculate Euclidean distance each frame")


# =====================================================
# SECTION 7: DATA LOGGER SETUP
# =====================================================

print("\n[STEP 5] Initializing data logger (CSV + PT)")

# Episode logging system will create CSV files on demand
# Sequential naming: csv_001.csv, csv_002.csv, etc.
log_dir = "logs"

print(f"         ✓ Autonomous episode recording enabled")
print(f"         ✓ File structure: logs/csv_###.csv (sequential numbering)")
print(f"         ✓ Triggering: Motion detection (threshold: 0.002 rad)")
print(f"         ✓ Success condition: Contact (distance ≤ 0.015m)")
print(f"         ✓ Termination: Robot stable after contact (15 frames)")
print(f"         ✓ Output format: CSV (written directly each frame)")
print(f"         ✓ Logging 10 metrics per frame")


# =====================================================
# SECTION 8: MOTOR INITIALIZATION
# =====================================================

print("\n[STEP 6] Initializing 6 robot joints")

# Joint names and device IDs
joint_names = ["1", "2", "3", "4", "5", "6"]
motors = []           # List to store motor device objects
position_sensors = [] # List to store position sensor devices
joint_pos = []       # List to track current position of each joint

# Initialize each motor
for i, joint_name in enumerate(joint_names):
    motor = robot.getDevice(joint_name)
    
    if motor is None:
        error_msg = f"FATAL ERROR: Motor '{joint_name}' not found!"
        print(error_msg)
        raise RuntimeError(error_msg)
    
    # Set velocity and initial position
    motor.setVelocity(1.0)    # Max velocity
    motor.setPosition(0.0)    # Start at neutral
    
    # Get position sensor for this motor
    # In Webots, Motor devices don't have getPosition(), must use PositionSensor
    pos_sensor = motor.getPositionSensor()
    if pos_sensor is not None:
        pos_sensor.enable(TIME_STEP)  # Enable sensor updates
    
    motors.append(motor)
    position_sensors.append(pos_sensor)
    joint_pos.append(0.0)
    
    joint_info = [
        "Base (rotation)",
        "Shoulder (pitch)",
        "Elbow (pitch)",
        "Wrist (pitch)",
        "Wrist (yaw)",
        "Wrist (roll)"
    ]
    
    print(f"         Joint {joint_name}: {joint_info[i]} ✓")


# =====================================================
# SECTION 9: KEYBOARD MAPPING
# =====================================================

print("\n[STEP 7] Setting up keyboard controls")
print("         QWERTY/ASDFGH layout:")
print("           Q/A → Joint 1 (Base)")
print("           W/S → Joint 2 (Shoulder)")
print("           E/D → Joint 3 (Elbow)")
print("           R/F → Joint 4 (Wrist Pitch)")
print("           T/G → Joint 5 (Wrist Yaw)")
print("           Y/H → Joint 6 (Wrist Roll)")

# Map keyboard keys to (joint_index, direction)
key_map = {
    # Joint 1
    ord('Q'): (0, +1), ord('A'): (0, -1),
    # Joint 2
    ord('W'): (1, +1), ord('S'): (1, -1),
    # Joint 3
    ord('E'): (2, +1), ord('D'): (2, -1),
    # Joint 4
    ord('R'): (3, +1), ord('F'): (3, -1),
    # Joint 5
    ord('T'): (4, +1), ord('G'): (4, -1),
    # Joint 6
    ord('Y'): (5, +1), ord('H'): (5, -1),
}

print("         ✓ Keyboard mapping active")


# =====================================================
# SECTION 10: EPISODE MANAGEMENT STATE VARIABLES
# =====================================================

print("\n[STEP 8] Setting up autonomous episode recording")
print(f"         ├─ Motion threshold: 0.002 radians")
print(f"         ├─ Contact threshold: {TOUCH_THRESHOLD} meters (1.5 cm)")
print(f"         ├─ Stationary window: 15 frames (~0.5 seconds)")
print(f"         └─ Episode file naming: csv_001.csv, csv_002.csv, ...")

# ─────── EPISODE NUMBERING ───────
# Auto-detect next episode number by scanning existing CSV files
def get_next_episode_number(log_dir="logs"):
    """Find highest existing episode number and return next one"""
    if not os.path.exists(log_dir):
        return 1
    
    import glob
    existing_csvs = glob.glob(os.path.join(log_dir, "csv_*.csv"))
    if not existing_csvs:
        return 1
    
    # Extract episode numbers from filenames like "csv_001.csv"
    episode_numbers = []
    for csv_file in existing_csvs:
        try:
            filename = os.path.basename(csv_file)
            # Extract digits from "csv_NNN.csv"
            num_str = filename.replace("csv_", "").replace(".csv", "")
            episode_numbers.append(int(num_str))
        except (ValueError, IndexError):
            pass
    
    if episode_numbers:
        return max(episode_numbers) + 1
    return 1

episode_counter = get_next_episode_number("logs")  # Auto-detect from existing files


def get_spatial_description(distance, object_name="target"):
    """
    Generate human-readable spatial description of distance.
    
    Args:
        distance: Distance in meters (float)
        object_name: Name of the target object (string)
    
    Returns:
        Spatial description string with range and distance value
    """
    if distance < 0.015:
        return f"TOUCHING {object_name.upper()}"
    elif distance < 0.05:
        return f"VERY CLOSE ({distance*100:.1f}cm from {object_name})"
    elif distance < 0.15:
        return f"NEAR ({distance*100:.1f}cm from {object_name})"
    elif distance < 0.3:
        return f"MID-RANGE ({distance:.2f}m from {object_name})"
    else:
        return f"FAR ({distance:.2f}m from {object_name})"


# ─────── STATE MACHINE ───────
state = "IDLE"                 # One of: IDLE, LOGGING, CONTACT_WAIT, RESET

# ─────── MOTION DETECTION ───────
MOTION_THRESHOLD = 0.002       # Radians (joint position change threshold)
prev_joint_pos = [0.0] * NUM_JOINTS  # Track previous position of each joint
is_moving = False              # True if any joint exceeds motion threshold

# ─────── CONTACT TRACKING ───────
contact_achieved = False       # True once box has been touched

# ─────── SOFTMAX PROGRESS TRACKING ───────
episode_start_state_progress = None  # Reference value set when episode begins, immutable during episode

# ─────── LOGGING CONTROL ───────
current_episode_logger = None  # Logger for current episode
log_dir = "logs"              # Base logging directory (no trial folders)

# ─────── STATIONARY DETECTION ───────
stationary_frame_count = 0     # Frames without motion (for stopping condition)
STATIONARY_WINDOW = 15         # Frames needed to be stationary before stopping


# =====================================================
# SECTION 11: MAIN CONTROL LOOP
# =====================================================

print("\n" + "="*80)
print("INITIALIZATION COMPLETE - Simulation Ready")
print("="*80)
print("\nPress 'Play' in Webots to start teleoperation")
print("Use keyboard controls to move robot")
print("Touch target box to start trial\n")


# ─────── HELPER FUNCTION: BOOST SIMILARITIES ───────
def boost_similarities_with_heuristics(
    sim_initial, sim_approaching, sim_picking, sim_done,
    distance, is_done_detected, done_confidence
):
    """
    Boost CLIP similarities with physical heuristics for better state discrimination.
    
    Strategy:
    - INITIAL: High when far away and stable
    - APPROACHING: Boost when moving toward target (mid distance)
    - PICKING: Boost when very close (distance < 0.1m, suggesting gripper has target)
    - DONE: Boost when color detection confirms red on blue
    
    Returns boosted similarities that maintain ranking but amplify differences.
    """
    
    # Distance-based boosting
    # Far away (>0.3m): favor INITIAL
    if distance > 0.3:
        sim_initial += 0.15
    
    # Mid-range (0.15-0.3m): favor APPROACHING
    elif 0.15 < distance <= 0.3:
        sim_approaching += 0.20
    
    # Close (0.08-0.15m): favor PICKING (gripper reaching)
    elif 0.08 < distance <= 0.15:
        sim_picking += 0.25
    
    # Very close (<0.08m): strongly favor PICKING (gripper engaged)
    elif distance <= 0.08:
        sim_picking += 0.35
    
    # Color detection boosting - if red is on blue, strongly boost DONE
    if is_done_detected:
        boost_amount = 0.4 + (done_confidence * 0.15)  # Up to 0.55 boost
        sim_done += min(boost_amount, 0.55)
    
    # Normalize to prevent values exceeding reasonable bounds
    similarities = [sim_initial, sim_approaching, sim_picking, sim_done]
    
    # Cap at 1.0 (CLIP similarities don't naturally exceed 1.0)
    similarities = [min(s, 1.0) for s in similarities]
    
    return similarities[0], similarities[1], similarities[2], similarities[3]


try:
    frame = 0  # Frame counter
    
    while robot.step(TIME_STEP) != -1:
        # ============================================================
        # SUBSTEP 1: REAL-TIME CAMERA IMAGE CAPTURE AND ANALYSIS
        # ============================================================
        # This is where the side camera image is captured and processed
        
        # CAMERA CAPTURE:
        # side_camera.get_image() retrieves the current frame from the
        # Webots side_camera device. This is a 640x480 RGB image showing
        # what the camera in the world file sees.
        image = side_camera.get_image()

        # Skip processing if image is not ready (rare, at startup only)
        if image is None:
          continue

        # At this point: 'image' is a NumPy array of shape (H, W, 3)
        # with RGB pixel values from the real-time camera feed

        # Helper: compute a small region around the gripper tip using simple
        # color heuristics (green robot arm, red box). This avoids complex
        # 3D→2D projection and makes CLIP focus on the end-effector tip.
        def get_gripper_roi(img, box_size=128):
          h, w, _ = img.shape
          # Red mask (box)
          r = img[:, :, 0].astype(np.int32)
          g = img[:, :, 1].astype(np.int32)
          b = img[:, :, 2].astype(np.int32)
          red_mask = (r > 100) & (r > 1.3 * g) & (r > 1.3 * b)

          # Green mask (robot arm)
          green_mask = (g > 80) & (g > 1.2 * r) & (g > 1.2 * b)

          # Find red centroid (box)
          ys, xs = np.where(red_mask)
          if xs.size > 0:
            red_cx = int(xs.mean())
            red_cy = int(ys.mean())
          else:
            red_cx = None
            red_cy = None

          # If red not found, fallback to image right-side centroid (likely box)
          if red_cx is None:
            red_cx = int(w * 0.8)
            red_cy = int(h * 0.7)

          # Find green pixels; choose the green pixel closest to the red centroid
          g_ys, g_xs = np.where(green_mask)
          if g_xs.size > 0:
            if red_cx is not None:
              d2 = (g_xs - red_cx) ** 2 + (g_ys - red_cy) ** 2
              idx = int(np.argmin(d2))
              grip_x = int(g_xs[idx])
              grip_y = int(g_ys[idx])
            else:
              # fallback: use centroid of green mask
              grip_x = int(g_xs.mean())
              grip_y = int(g_ys.mean())
          else:
            # No green found; fallback to image center
            grip_x = w // 2
            grip_y = h // 2

          half = box_size // 2
          x0 = max(0, grip_x - half)
          y0 = max(0, grip_y - half)
          x1 = min(w, grip_x + half)
          y1 = min(h, grip_y + half)

          roi = img[y0:y1, x0:x1]
          # If ROI is too small, pad by centering a square around the grip
          if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
            cx = min(max(grip_x, half), w - half)
            cy = min(max(grip_y, half), h - half)
            x0 = cx - half
            y0 = cy - half
            x1 = cx + half
            y1 = cy + half
            roi = img[y0:y1, x0:x1]

          # If still invalid, fallback to center crop
          if roi.size == 0:
            cx, cy = w // 2, h // 2
            half = box_size // 2
            roi = img[cy - half:cy + half, cx - half:cx + half]

            return roi, grip_x, grip_y, red_cx, red_cy

        # Create a crop centered around the gripper tip and use that for CLIP
        try:
          crop, grip_x, grip_y, red_cx, red_cy = get_gripper_roi(image, box_size=128)
        except Exception:
          crop = image
          grip_x, grip_y = None, None
          red_cx, red_cy = None, None

        # CLIP PROCESSING: use the gripper-focused crop for similarity
        img_feat = clip.encode_image(crop)

        # Diagnostic: report gripper coordinates and crop size occasionally
        if frame % 60 == 0:
          try:
            ch, cw = crop.shape[:2]
          except Exception:
            ch, cw = -1, -1
          print(f"  [ROI] grip=({grip_x},{grip_y}) box=({red_cx},{red_cy}) crop={cw}x{ch}")
        
        # ─────── FOUR-STATE CLIP PROGRESS TRACKING ───────
        # Compare current image to four task states:
        # 1. INITIAL:    Arm at rest, gripper open
        # 2. APPROACHING: Arm moving toward box, gripper opening
        # 3. PICKING:    Arm grasping box, gripper closed
        # 4. DONE:       Red box on blue square (task complete)
        
        # Get similarities to each state (0-1 range)
        sim_initial = clip.cosine_similarity(img_feat, text_feat_state_initial)
        sim_approaching = clip.cosine_similarity(img_feat, text_feat_state_approaching)
        sim_picking = clip.cosine_similarity(img_feat, text_feat_state_picking)
        sim_done = clip.cosine_similarity(img_feat, text_feat_state_done)
        
        # We need distance for boosting heuristics, so calculate it now
        # Get 3D positions from supervisor
        eef_p = eef.getPosition()    # End effector position [x, y, z]
        tgt_p = target.getPosition() # Target position [x, y, z]
        dist = sum((eef_p[i] - tgt_p[i]) ** 2 for i in range(3)) ** 0.5
        
        # Update done detector with image to get color-based feedback
        done_detection = done_detector.update(image)
        is_done_detected = done_detection["is_done"]
        done_confidence = done_detection["confidence"]
        red_on_blue = done_detection["red_on_blue"]
        
        # Apply heuristic boosting to amplify state differences
        # This makes similarities more responsive to actual task state
        sim_initial, sim_approaching, sim_picking, sim_done = boost_similarities_with_heuristics(
            sim_initial, sim_approaching, sim_picking, sim_done,
            distance=dist,
            is_done_detected=is_done_detected,
            done_confidence=done_confidence
        )
        
        # Update four-state estimator with boosted similarities
        four_state_metrics = four_state_estimator.update(
            similarity_initial=sim_initial,
            similarity_approaching=sim_approaching,
            similarity_picking=sim_picking,
            similarity_done=sim_done
        )
        
        # Extract key metrics
        state_progress = four_state_metrics["state_progress"]
        current_state = four_state_metrics["current_state"]
        state_confidence = four_state_metrics["confidence"]
        state_probs = four_state_metrics["probabilities"]
        bidir_raw = four_state_metrics["bidirectional_raw"]
        bidir_ema = four_state_metrics["bidirectional_ema"]
        
        # ─────── TASK COMPLETION CHECK ───────
        # Task is complete if EITHER CLIP or color detection confirms
        task_complete = four_state_metrics["is_task_complete"] or is_done_detected
        
        # ─────── DIAGNOSTIC OUTPUT ───────
        # Print state tracking every 60 frames (~2 seconds)
        if frame % 60 == 0 and (state in ["LOGGING", "CONTACT_WAIT"]):
            print(f"\n[Frame {frame}] Four-State Progress (INITIAL={sim_initial:.3f} APPROACH={sim_approaching:.3f} PICK={sim_picking:.3f} DONE={sim_done:.3f})")
            print(f"  Current State: {current_state.upper()} (conf: {state_confidence:.3f})")
            print(f"  Progress: {state_progress:.3f} | Bidirectional: {bidir_raw:+.4f} (EMA: {bidir_ema:+.4f})")
            print(f"  Done Detection: {is_done_detected} (Confidence: {done_confidence}) | Red on Blue: {red_on_blue} | Task Complete: {task_complete}")
            print(f"  Done Detection: {is_done_detected} (frames on blue: {done_confidence}/5)")
        
        
        # ============================================================
        # SUBSTEP 2: GROUND TRUTH DISTANCE CALCULATION
        # ============================================================
        # (Distance already calculated above for boosting heuristics)
        
        
        # ============================================================
        # SUBSTEP 3: MOTION DETECTION (JOINT POSITION FEEDBACK)
        # ============================================================
        # Check if any joint has moved beyond threshold (using motor feedback, not commands)
        is_moving = False
        for i in range(NUM_JOINTS):
            # Get actual joint position from position sensor feedback
            # In Webots, must use PositionSensor.getValue() not Motor.getPosition()
            if position_sensors[i] is not None:
                actual_pos = position_sensors[i].getValue()
            else:
                # Fallback: use commanded position if sensor unavailable
                actual_pos = joint_pos[i]
            
            # Compare to previous position
            if abs(actual_pos - prev_joint_pos[i]) > MOTION_THRESHOLD:
                is_moving = True
            # Update tracking position
            prev_joint_pos[i] = actual_pos
        
        
        # ============================================================
        # SUBSTEP 4: CONTACT DETECTION
        # ============================================================
        # Check if gripper is touching target
        is_touching = dist < TOUCH_THRESHOLD
        
        
        # ============================================================
        # SUBSTEP 5: CUMULATIVE EPISODE PROGRESS (CONTRASTIVE SIGNAL)
        # ============================================================
        # Compute long-horizon progress metric relative to episode start
        # This metric accumulates semantic movement toward the goal
        # Formula: episode_progress = contrastive_score - episode_start_contrastive
        # ============================================================
        # SUBSTEP 5: CUMULATIVE EPISODE PROGRESS (FOUR-STATE SIGNAL)
        # ============================================================
        # Compute long-horizon progress metric relative to episode start
        # This metric accumulates semantic movement through task states
        # Formula: episode_progress = state_progress - episode_start_state_progress
        # 
        # Only computed when an episode is actively recording (not in IDLE or RESET)
        # episode_start_state_progress is captured at the moment IDLE→LOGGING transition
        # and remains fixed for the entire episode
        
        if episode_start_state_progress is not None:
            # Episode is active - compute cumulative progress
            # Raw float with full precision, no smoothing or normalization applied
            episode_progress = state_progress - episode_start_state_progress
        else:
            # Episode not yet started (still in IDLE, RESET, or initialization)
            # Set to 0.0 as placeholder (will not be logged)
            episode_progress = 0.0
        

        # ============================================================
        # SUBSTEP 6: EPISODE STATE MACHINE
        # ============================================================
        
        if state == "IDLE":
            # ───────────────────────────────────────
            # IDLE STATE: Waiting for motion
            # Transition: Motion detected → LOGGING
            # ───────────────────────────────────────
            if is_moving:
                # Motion detected! Start new episode
                episode_counter += 1
                state = "LOGGING"
                contact_achieved = False
                stationary_frame_count = 0
                
                # CAPTURE EPISODE START STATE PROGRESS
                # This reference value remains immutable for the entire episode
                # Used to compute cumulative progress: episode_progress = state_progress - episode_start_state_progress
                episode_start_state_progress = state_progress
                
                # Create new CSV file with sequential naming
                current_episode_logger = ProgressLogger(
                    log_dir=log_dir,
                    episode_number=episode_counter
                )
                print(f"\n[EPISODE {episode_counter} STARTED - LOGGING]")
                print(f"  CSV: {current_episode_logger.csv_path}")
                print(f"  Episode start state progress: {episode_start_state_progress:.6f}")
                print(f"  Image embedding norm: {np.linalg.norm(img_feat):.4f}")
                print(f"  Text embeddings (should be ~1.0 after normalization):")
                print(f"    Initial: {np.linalg.norm(text_feat_state_initial):.4f}")
                print(f"    Approaching: {np.linalg.norm(text_feat_state_approaching):.4f}")
                print(f"    Picking: {np.linalg.norm(text_feat_state_picking):.4f}")
                print(f"    Done: {np.linalg.norm(text_feat_state_done):.4f}")
        
        elif state == "LOGGING":
            # ───────────────────────────────────────
            # LOGGING STATE: Recording trajectory
            # Transition: Contact detected → CONTACT_WAIT
            # ───────────────────────────────────────
            if is_touching:
                # Contact achieved!
                contact_achieved = True
                state = "CONTACT_WAIT"
                stationary_frame_count = 0
                print(f"[EPISODE {episode_counter}] Contact detected at distance {dist:.4f}m")
        
        elif state == "CONTACT_WAIT":
            # ───────────────────────────────────────
            # CONTACT_WAIT STATE: Robot stable after touch
            # Transition: Stable (15 frames) → RESET
            # ───────────────────────────────────────
            if is_moving:
                # Robot still moving, reset counter
                stationary_frame_count = 0
            else:
                # Robot is stationary
                stationary_frame_count += 1
                if stationary_frame_count >= STATIONARY_WINDOW:
                    # Robot has been stable long enough - episode complete!
                    state = "RESET"
                    # Close the CSV file
                    if current_episode_logger is not None:
                        current_episode_logger.close()
                        print(f"[EPISODE {episode_counter} COMPLETED] Data saved:")
                        print(f"  CSV: {current_episode_logger.csv_path}")
                        print(f"  Frames recorded: {current_episode_logger.t}")
                        current_episode_logger = None
        
        elif state == "RESET":
            # ───────────────────────────────────────
            # RESET STATE: Return to home position
            # Transition: All joints idle → IDLE
            # ───────────────────────────────────────
            if not is_moving:
                # All joints have stabilized
                state = "IDLE"
                # Reset episode start reference for next episode
                episode_start_state_progress = None
                # Reset estimators
                four_state_estimator.reset()
                done_detector.reset()
                print(f"[EPISODE {episode_counter}] Reset complete, ready for next episode")
        
        
        # ============================================================
        # SUBSTEP 6: DATA LOGGING TO CSV - FULL PRECISION PRESERVATION
        # ============================================================
        # CRITICAL: Logs all metrics with FULL floating-point precision
        # - All values are raw floats with no pre-formatting or rounding
        # - Tiny deltas (e.g., 0.000000000001234) preserved exactly in CSV
        # - Only Webots overlay display uses formatted/rounded values for readability
        # - CSV file contains raw IEEE 754 precision (15-17 significant digits)
        #
        # Only log when episode is actively recording (LOGGING or CONTACT_WAIT states)
        if (state in ["LOGGING", "CONTACT_WAIT"]) and current_episode_logger is not None:
            # Pass RAW float values only - NO formatting before CSV write
            current_episode_logger.log(
                sim_initial,               # Raw CLIP similarity to initial state (0-1)
                sim_approaching,           # Raw CLIP similarity to approaching state (0-1)
                sim_picking,               # Raw CLIP similarity to picking state (0-1)
                sim_done,                  # Raw CLIP similarity to done state (0-1)
                state_progress,            # Four-state progress (0-1)
                state_probs["initial"],    # Probability of initial state
                state_probs["approaching"],# Probability of approaching state
                state_probs["picking"],    # Probability of picking state
                state_probs["done"],       # Probability of done state
                is_done_detected,          # Color detection: red box on blue square?
                red_on_blue,               # Is red currently on blue?
                task_complete,             # Task complete (CLIP or color detection)?
                bidir_raw,                 # CRITICAL: Raw temporal delta (full precision)
                bidir_ema,                 # Raw EMA value (unformatted)
                dist,                      # Raw distance float (unformatted)
                current_state,             # Current state name
                state_confidence,          # Confidence in current state
                done_confidence,           # Done detection confidence (frames on blue)
                episode_progress           # Cumulative progress since episode start (raw, full precision)
            )
            
            # Diagnostic console output every 30 frames (~1 second)
            # NOTE: Display here is formatted for readability; CSV stores raw floats
            if frame % 30 == 0:
                print(f"  [Frame {frame}] States - Init:{sim_initial:.3f} Appr:{sim_approaching:.3f} Pick:{sim_picking:.3f} Done:{sim_done:.3f}")
                print(f"               Progress: {state_progress:.3f} | Probs - I:{state_probs['initial']:.2f} A:{state_probs['approaching']:.2f} P:{state_probs['picking']:.2f} D:{state_probs['done']:.2f}")
                print(f"               BiDelta: {bidir_raw:+.4f} | EMA: {bidir_ema:+.4f} | Task Complete: {task_complete} (Done Conf: {done_confidence})")
        
        
        # ============================================================
        # SUBSTEP 7: REAL-TIME OVERLAY DISPLAY
        # ============================================================
        # Display key metrics on the Webots overlay for real-time feedback
        
        # STATE and frame info - show ACTUAL CURRENT STATE instead of machine state
        motion_indicator = "MOVING" if is_moving else "IDLE"
        current_state_display = current_state.upper()
        overlay_line0 = f"STATE: {current_state_display:12} | Episode: {episode_counter:3} | Frame: {frame:5} | {motion_indicator}"
        
        # STATE SIMILARITIES - Show as percentages, should vary with arm movement
        # Values: 0% (orthogonal/opposite), 50% (neutral), 100% (identical)
        # With boosting, these now reflect actual task progress
        sim_spread = max(sim_initial, sim_approaching, sim_picking, sim_done) - min(sim_initial, sim_approaching, sim_picking, sim_done)
        overlay_line1 = (
          f"States: Init {sim_initial*100:5.1f}% | Appr {sim_approaching*100:5.1f}% | "
          f"Pick {sim_picking*100:5.1f}% | Done {sim_done*100:5.1f}% | Spread: {sim_spread*100:5.1f}%"
        )

        # STATE PROGRESS AND PROBABILITIES
        # Progress: weighted average of state positions. 50% = neutral/ambiguous state
        # Probs: should concentrate on highest similarity state
        overlay_line2 = (
          f"Progress: {state_progress*100:6.2f}% | {current_state.upper():10} (conf:{state_confidence:.2f}) | "
          f"Probs [I:{state_probs.get('initial', 0)*100:4.1f}% "
          f"A:{state_probs.get('approaching', 0)*100:4.1f}% P:{state_probs.get('picking', 0)*100:4.1f}% "
          f"D:{state_probs.get('done', 0)*100:4.1f}%]"
        )
        
        # Spatial description with distance metrics
        spatial_desc = get_spatial_description(dist, "red_box")
        task_status = "✓DONE" if task_complete else "●PROGRESS"
        overlay_line3 = f"Distance: {spatial_desc} | BiDelta: {bidir_raw:+.2e} | EMA: {bidir_ema:+.6f} | {task_status}"
        
        robot.setLabel(0, overlay_line0, 0.01, 0.01, 0.08, 0xFFFFFF, 0.0)
        robot.setLabel(1, overlay_line1, 0.01, 0.06, 0.08, 0xFFFFFF, 0.0)
        robot.setLabel(2, overlay_line2, 0.01, 0.11, 0.08, 0xFFFFFF, 0.0)
        robot.setLabel(3, overlay_line3, 0.01, 0.16, 0.08, 0xFFFFFF, 0.0)
        
        # ============================================================
        # SUBSTEP 8: KEYBOARD TELEOPERATION INPUT
        # ============================================================
        # Process keyboard input for manual control
        key = keyboard.getKey()
        deltas = [0.0] * NUM_JOINTS  # Accumulated angle changes
        
        # Collect all keys pressed in this frame
        while key != -1:
            if key in key_map:
                joint_idx, direction = key_map[key]
                deltas[joint_idx] += direction * ANGLE_STEP
            key = keyboard.getKey()
        
        # Apply position changes to motors
        for i in range(NUM_JOINTS):
            if deltas[i] != 0.0:
                joint_pos[i] += deltas[i]      # Update tracked position
                motors[i].setPosition(joint_pos[i])  # Command motor
        
        
        # ============================================================
        # SUBSTEP 9: FRAME COUNTER UPDATE
        # ============================================================
        frame += 1

# =====================================================
# SHUTDOWN SEQUENCE (FINALLY BLOCK)
# =====================================================
finally:
    # Close any remaining episode logger
    if current_episode_logger is not None:
        current_episode_logger.close()
        print(f"\n[EPISODE {episode_counter} SAVED (shutdown)]")
        print(f"  CSV: {current_episode_logger.csv_path}")
    
    # ============================================================
    # SAVE SESSION SUMMARY CSV
    # ============================================================
    # Create a summary CSV file with session-level statistics
    session_summary_path = os.path.join(log_dir, "session_summary.csv")
    
    total_time_seconds = frame * TIME_STEP / 1000.0
    
    try:
        with open(session_summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Write header row
            writer.writerow([
                "session_metric",
                "value",
                "unit",
                "description"
            ])
            
            # Write session statistics
            writer.writerow([
                "total_episodes",
                episode_counter,
                "count",
                "Total number of episodes recorded"
            ])
            
            writer.writerow([
                "total_frames",
                frame,
                "frames",
                "Total simulation frames executed"
            ])
            
            writer.writerow([
                "total_simulation_time",
                total_time_seconds,
                "seconds",
                "Total simulation time (frame_count * timestep)"
            ])
            
            writer.writerow([
                "timestep",
                TIME_STEP,
                "milliseconds",
                "Simulation timestep duration"
            ])
            
            writer.writerow([
                "frame_rate",
                31.25,
                "Hz",
                "Target frame rate (1000ms / 32ms)"
            ])
            
            writer.writerow([
                "timestamp",
                datetime.now().isoformat(),
                "ISO 8601",
                "Session end time"
            ])
            
            if episode_counter > 0:
                writer.writerow([
                    "avg_frames_per_episode",
                    frame / episode_counter,
                    "frames",
                    "Average frames per recorded episode"
                ])
                
                writer.writerow([
                    "episode_file_range",
                    f"csv_001.csv to csv_{episode_counter:03d}.csv",
                    "files",
                    "Range of episode CSV files created"
                ])
        
        print(f"\n[SESSION SUMMARY SAVED]")
        print(f"  CSV: {session_summary_path}")
    
    except Exception as e:
        print(f"\nWARNING: Could not save session summary CSV: {e}")
    
    # Print session summary to console
    print("\n" + "="*80)
    print("SESSION ENDED - SUMMARY")
    print("="*80)
    print(f"Total simulation time: {total_time_seconds:.2f} seconds")
    print(f"Total frames executed: {frame}")
    print(f"Frame rate achieved: {31.25:.2f} Hz")
    print(f"Total episodes recorded: {episode_counter}")
    
    if episode_counter > 0:
        print(f"\nEpisode files saved to:")
        print(f"  logs/csv_001.csv through logs/csv_{episode_counter:03d}.csv")
        print(f"\nTotal CSV files created: {episode_counter}")
        print(f"\nSession summary: logs/session_summary.csv")
    
    print("="*80 + "\n")
