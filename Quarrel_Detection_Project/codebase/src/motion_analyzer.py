"""
Motion Analysis Module for Quarrel Detection
Analyzes motion intensity, interpersonal proximity, and rapid movements
Complements CNN-based detection with traditional CV techniques
"""

import cv2
import numpy as np
from collections import deque

class MotionAnalyzer:
    """
    Analyzes motion patterns to detect aggressive behavior.
    Combines frame differencing with background subtraction.
    """
    
    def __init__(self, 
                 motion_threshold=25,
                 min_contour_area=500,
                 proximity_threshold=100,
                 history_size=30):
        """
        Initialize motion analyzer.
        
        Args:
            motion_threshold: Threshold for motion detection
            min_contour_area: Minimum contour area to consider
            proximity_threshold: Distance threshold for proximity detection (pixels)
            history_size: Number of frames for motion history
        """
        self.motion_threshold = motion_threshold
        self.min_contour_area = min_contour_area
        self.proximity_threshold = proximity_threshold
        
        # Background subtractor (MOG2)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, 
            varThreshold=40, 
            detectShadows=True
        )
        
        # Previous frame for frame differencing
        self.prev_frame = None
        
        # Motion history
        self.motion_history = deque(maxlen=history_size)
        self.motion_intensity_history = deque(maxlen=10)
        
    def calculate_motion_intensity(self, frame):
        """
        Calculate motion intensity using frame differencing.
        
        Args:
            frame: Current frame (BGR)
            
        Returns:
            Motion intensity score (0-1)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return 0.0
        
        # Frame difference
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Calculate motion intensity as percentage of changed pixels
        motion_pixels = np.sum(thresh > 0)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        motion_intensity = motion_pixels / total_pixels
        
        self.prev_frame = gray
        
        return motion_intensity
    
    def detect_motion_regions(self, frame):
        """
        Detect motion regions using background subtraction.
        
        Args:
            frame: Current frame (BGR)
            
        Returns:
            List of motion contours and foreground mask
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (value 127 in MOG2)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
        
        return valid_contours, fg_mask
    
    def calculate_interpersonal_proximity(self, person_boxes):
        """
        Calculate minimum distance between detected people.
        Close proximity may indicate confrontation.
        
        Args:
            person_boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
            
        Returns:
            Minimum distance between people (pixels), proximity score (0-1)
        """
        if len(person_boxes) < 2:
            return float('inf'), 0.0
        
        # Calculate centers of bounding boxes
        centers = []
        for box in person_boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append((cx, cy))
        
        # Find minimum distance between any two people
        min_distance = float('inf')
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                              (centers[i][1] - centers[j][1])**2)
                min_distance = min(min_distance, dist)
        
        # Convert to proximity score (closer = higher score)
        if min_distance < self.proximity_threshold:
            proximity_score = 1.0 - (min_distance / self.proximity_threshold)
        else:
            proximity_score = 0.0
        
        return min_distance, proximity_score
    
    def detect_rapid_movements(self, current_intensity):
        """
        Detect sudden spikes in motion (rapid movements).
        
        Args:
            current_intensity: Current motion intensity
            
        Returns:
            Boolean indicating rapid movement, spike score
        """
        self.motion_intensity_history.append(current_intensity)
        
        if len(self.motion_intensity_history) < 3:
            return False, 0.0
        
        # Calculate moving average
        avg_intensity = np.mean(list(self.motion_intensity_history)[:-1])
        
        # Check for spike (current > 2x average)
        if current_intensity > 2 * avg_intensity and current_intensity > 0.05:
            spike_score = min(current_intensity / avg_intensity / 2, 1.0)
            return True, spike_score
        
        return False, 0.0
    
    def calculate_motion_score(self, frame, person_boxes):
        """
        Calculate comprehensive motion-based quarrel score.
        Combines multiple motion features.
        
        Args:
            frame: Current frame
            person_boxes: Detected person bounding boxes
            
        Returns:
            Dictionary with motion analysis results
        """
        # 1. Motion intensity (25% weight)
        motion_intensity = self.calculate_motion_intensity(frame)
        
        # 2. Motion regions
        motion_contours, fg_mask = self.detect_motion_regions(frame)
        motion_area_score = min(len(motion_contours) / 10.0, 1.0)  # Normalize
        
        # 3. Interpersonal proximity (20% weight)
        min_distance, proximity_score = self.calculate_interpersonal_proximity(person_boxes)
        
        # 4. Rapid movements (15% weight)
        is_rapid, spike_score = self.detect_rapid_movements(motion_intensity)
        
        # 5. Person count factor (25% weight)
        person_count = len(person_boxes)
        count_score = 0.0
        if person_count >= 2:
            count_score = min(person_count / 4.0, 1.0)  # 2-4 people = score increases
        
        # Weighted combination
        weights = {
            'count': 0.25,
            'motion': 0.25,
            'proximity': 0.20,
            'rapid': 0.15,
            'motion_area': 0.15
        }
        
        total_score = (
            weights['count'] * count_score +
            weights['motion'] * motion_intensity * 10 +  # Scale up motion
            weights['proximity'] * proximity_score +
            weights['rapid'] * spike_score +
            weights['motion_area'] * motion_area_score
        )
        
        total_score = min(total_score, 1.0)  # Cap at 1.0
        
        return {
            'total_score': total_score,
            'motion_intensity': motion_intensity,
            'motion_area_score': motion_area_score,
            'proximity_score': proximity_score,
            'min_distance': min_distance,
            'spike_score': spike_score,
            'is_rapid': is_rapid,
            'person_count': person_count,
            'count_score': count_score,
            'motion_contours': len(motion_contours),
            'fg_mask': fg_mask
        }
    
    def reset(self):
        """Reset motion analyzer state"""
        self.prev_frame = None
        self.motion_history.clear()
        self.motion_intensity_history.clear()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, 
            varThreshold=40, 
            detectShadows=True
        )


# Example usage
if __name__ == "__main__":
    print("Motion Analyzer - Test mode")
    print("This module is meant to be imported by detection.py")
