"""
Done State Detection: Red Box on Blue Square

This module detects task completion by identifying when the red box
is placed on the blue square platform using color-based segmentation.

DETECTION METHOD:
1. Extract red box location using HSV color detection
2. Extract blue square location using HSV color detection
3. Check if red box center is within blue square bounds
4. Verify sustained placement (multiple frames) for robust detection

This works independently from CLIP and provides a ground-truth signal
for validating the "done" state detection.
"""

import numpy as np
import cv2


class DoneStateDetector:
    """
    Detects task completion by identifying red box placement on blue square.
    
    Uses HSV color space for robust detection across lighting variations.
    """
    
    # HSV color ranges (in OpenCV: H=0-180, S/V=0-255)
    # These ranges are tuned for the specific colors in the Webots simulation
    
    # RED BOX detection - expanded range to catch all red hues
    RED_LOWER1 = np.array([0, 80, 80])        # Lower hue range for red (more permissive)
    RED_UPPER1 = np.array([15, 255, 255])
    RED_LOWER2 = np.array([165, 80, 80])      # Upper hue range for red (wraps around)
    RED_UPPER2 = np.array([180, 255, 255])
    
    # BLUE SQUARE detection - expanded range for better detection
    BLUE_LOWER = np.array([90, 80, 80])       # Wider blue hue range
    BLUE_UPPER = np.array([135, 255, 255])
    
    def __init__(self, frame_threshold=3, position_tolerance=0.25):
        """
        Initialize done state detector.
        
        Args:
            frame_threshold (int): Number of consecutive frames needed to confirm placement.
                Default 3 frames ≈ 96ms at 31 Hz, less stringent for immediate feedback
            position_tolerance (float): Fractional tolerance for position check.
                Default 0.25 = ±25% of blue square dimensions (more forgiving)
        """
        self.frame_threshold = frame_threshold
        self.position_tolerance = position_tolerance
        self.placement_frame_count = 0
        
        # Track detection state
        self.last_red_box = None      # (x, y, w, h)
        self.last_blue_square = None  # (x, y, w, h)
        self.is_done = False
        
        # Debug info
        self.last_red_area = 0
        self.last_blue_area = 0
    
    def detect_red_box(self, hsv_image: np.ndarray) -> tuple:
        """
        Detect red box in HSV image using color range masking.
        
        Args:
            hsv_image: Image in HSV color space
        
        Returns:
            tuple: (x, y, w, h) of bounding box, or None if not found
        """
        try:
            # Create mask for red color (two ranges due to HSV wrap-around)
            mask1 = cv2.inRange(hsv_image, self.RED_LOWER1, self.RED_UPPER1)
            mask2 = cv2.inRange(hsv_image, self.RED_LOWER2, self.RED_UPPER2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # Apply morphological operations to clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Filter contours by area and aspect ratio
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Lower threshold to catch red box even when partially obscured
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect = float(w) / h if h > 0 else 0
                    # Red box should be roughly square-ish (0.3-3.0 aspect ratio)
                    if 0.3 <= aspect <= 3.0:
                        valid_contours.append((contour, area, (x, y, w, h)))
            
            if not valid_contours:
                return None
            
            # Get largest valid contour
            largest = max(valid_contours, key=lambda x: x[1])
            x, y, w, h = largest[2]
            self.last_red_area = largest[1]
            
            return (x, y, w, h)
        
        except Exception as e:
            print(f"[DoneDetector] Error detecting red box: {e}")
            return None
    
    def detect_blue_square(self, hsv_image: np.ndarray) -> tuple:
        """
        Detect blue square in HSV image using color range masking.
        
        Args:
            hsv_image: Image in HSV color space
        
        Returns:
            tuple: (x, y, w, h) of bounding box, or None if not found
        """
        try:
            # Create mask for blue color
            blue_mask = cv2.inRange(hsv_image, self.BLUE_LOWER, self.BLUE_UPPER)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Filter contours by area (blue square should be fairly large)
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 300:  # Lowered threshold for better detection
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect = float(w) / h if h > 0 else 0
                    # Blue square should be roughly square-ish (0.5-2.0 aspect ratio)
                    if 0.5 <= aspect <= 2.0:
                        valid_contours.append((contour, area, (x, y, w, h)))
            
            if not valid_contours:
                return None
            
            # Get largest valid contour (blue square)
            largest = max(valid_contours, key=lambda x: x[1])
            x, y, w, h = largest[2]
            self.last_blue_area = largest[1]
            
            return (x, y, w, h)
        
        except Exception as e:
            print(f"[DoneDetector] Error detecting blue square: {e}")
            return None
    
    def is_red_on_blue(self, red_box: tuple, blue_square: tuple) -> bool:
        """
        Check if red box center is within blue square bounds.
        
        Args:
            red_box (tuple): (x, y, w, h) of red box bounding box
            blue_square (tuple): (x, y, w, h) of blue square bounding box
        
        Returns:
            bool: True if red box is positioned on/inside blue square
        """
        if red_box is None or blue_square is None:
            return False
        
        # Get centers
        red_x, red_y, red_w, red_h = red_box
        red_cx = red_x + red_w / 2
        red_cy = red_y + red_h / 2
        
        blue_x, blue_y, blue_w, blue_h = blue_square
        blue_cx = blue_x + blue_w / 2
        blue_cy = blue_y + blue_h / 2
        
        # Define tolerance (allow red box to be slightly outside due to perspective)
        # This includes red partially outside the blue square
        tol_x = blue_w * self.position_tolerance
        tol_y = blue_h * self.position_tolerance
        
        # Check if red center is within expanded blue square bounds
        # The expanded bounds account for the red box size and perspective
        within_x = (blue_x - tol_x) <= red_cx <= (blue_x + blue_w + tol_x)
        within_y = (blue_y - tol_y) <= red_cy <= (blue_y + blue_h + tol_y)
        
        # Also check if red box significantly overlaps with blue square
        # Calculate intersection area
        overlap_x = max(0, min(red_x + red_w, blue_x + blue_w) - max(red_x, blue_x))
        overlap_y = max(0, min(red_y + red_h, blue_y + blue_h) - max(red_y, blue_y))
        overlap_area = overlap_x * overlap_y
        
        # Red box is considered "on blue" if center is close AND there's significant overlap
        center_close = within_x and within_y
        has_overlap = overlap_area > (red_w * red_h * 0.2)  # At least 20% overlap
        
        return center_close or has_overlap
    
    def update(self, image: np.ndarray) -> dict:
        """
        Update done state detector with new frame.
        
        Args:
            image (np.ndarray): RGB image from camera (H, W, 3)
        
        Returns:
            dict: Detection results:
                {
                    "is_done": bool,              # Task complete with confidence?
                    "red_box": tuple or None,     # (x, y, w, h) of red box
                    "blue_square": tuple or None, # (x, y, w, h) of blue square
                    "red_on_blue": bool,          # Is red currently on blue?
                    "confidence": int,            # Frames held on blue (0-frame_threshold)
                    "red_visible": bool,          # Red box detected in frame?
                    "blue_visible": bool          # Blue square detected in frame?
                }
        """
        if image is None or image.size == 0:
            return {
                "is_done": False,
                "red_box": None,
                "blue_square": None,
                "red_on_blue": False,
                "confidence": 0,
                "red_visible": False,
                "blue_visible": False
            }
        
        try:
            # Convert to HSV
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Detect objects
            red_box = self.detect_red_box(hsv_image)
            blue_square = self.detect_blue_square(hsv_image)
            
            self.last_red_box = red_box
            self.last_blue_square = blue_square
            
            # Check placement
            red_on_blue = self.is_red_on_blue(red_box, blue_square)
            
            # Update placement counter
            if red_on_blue:
                self.placement_frame_count += 1
            else:
                self.placement_frame_count = 0
            
            # Task is done if placement sustained for frame_threshold frames
            self.is_done = self.placement_frame_count >= self.frame_threshold
            
            return {
                "is_done": self.is_done,
                "red_box": red_box,
                "blue_square": blue_square,
                "red_on_blue": red_on_blue,
                "confidence": min(self.placement_frame_count, self.frame_threshold),
                "red_visible": red_box is not None,
                "blue_visible": blue_square is not None
            }
        
        except Exception as e:
            print(f"[DoneDetector] Error in update: {e}")
            return {
                "is_done": False,
                "red_box": None,
                "blue_square": None,
                "red_on_blue": False,
                "confidence": 0,
                "red_visible": False,
                "blue_visible": False
            }
    
    def reset(self):
        """Reset detector state for new episode."""
        self.placement_frame_count = 0
        self.last_red_box = None
        self.last_blue_square = None
        self.is_done = False
        self.last_red_area = 0
        self.last_blue_area = 0
    
    def get_debug_info(self) -> str:
        """Return debug information about current detection state."""
        return (
            f"Done={self.is_done}, "
            f"RedBox={self.last_red_box}, "
            f"BlueSquare={self.last_blue_square}, "
            f"FrameCount={self.placement_frame_count}/{self.frame_threshold}, "
            f"RedArea={self.last_red_area}, BlueArea={self.last_blue_area}"
        )


def visualize_detection(image: np.ndarray, detector: DoneStateDetector) -> np.ndarray:
    """
    Draw detection results on image for visualization.
    
    Args:
        image (np.ndarray): RGB image
        detector (DoneStateDetector): Detector instance with last results
    
    Returns:
        np.ndarray: Image with drawn bounding boxes and status
    """
    result_image = image.copy()
    
    # Draw blue square
    if detector.last_blue_square:
        x, y, w, h = detector.last_blue_square
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(result_image, "BLUE", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw red box
    if detector.last_red_box:
        x, y, w, h = detector.last_red_box
        color = (0, 255, 0) if detector.is_done else (255, 0, 0)  # Green if done, red if not
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result_image, "RED", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw status text
    status = "TASK COMPLETE!" if detector.is_done else f"Frames on blue: {detector.placement_frame_count}"
    color = (0, 255, 0) if detector.is_done else (255, 255, 255)
    cv2.putText(result_image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return result_image
