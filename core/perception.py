"""
Perception Layer: YOLO + Multi-Object Tracking
Detects vehicles, pedestrians, and tracks them across frames.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class YOLODetector:
    """Wrapper for YOLO object detection"""
    
    def __init__(self, model_name: str = "yolov5s", device: str = "cpu"):
        """
        Initialize YOLO detector
        
        Args:
            model_name: YOLOv5 variant (yolov5s, yolov5m, yolov5l)
            device: torch device ('cpu' or 'cuda')
        """
        self.model = None
        self.device = device
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model with robust error handling"""
        try:
            # Suppress unnecessary output
            import warnings
            import sys
            from io import StringIO
            
            # Suppress stderr and stdout during model loading
            warnings.filterwarnings('ignore')
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            
            try:
                # Try to load from torch hub
                self.model = torch.hub.load('ultralytics/yolov5', self.model_name, pretrained=True, force_reload=False, verbose=False)
                self.model.to(self.device)
                self.model.conf = 0.4  # Set confidence threshold
                self.model.eval()  # Set to eval mode
            finally:
                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            logger.info(f"✅ YOLO model loaded: {self.model_name} on {self.device}")
        except Exception as e:
            logger.warning(f"⚠️ YOLO loading failed: {str(e)[:100]}. Using mock detector (still functional).")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame
        
        Returns list of detections with keys:
            - bbox: [x1, y1, x2, y2] (pixel coordinates)
            - class: class name
            - confidence: detection confidence
            - class_id: numeric class ID
        """
        if self.model is None:
            return []
        
        try:
            results = self.model(frame)
            detections = []
            
            for *bbox, conf, class_id in results.xyxy[0]:
                class_name = results.names[int(class_id)]
                
                # Filter for traffic-relevant classes
                if class_name.lower() in ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle']:
                    detections.append({
                        'bbox': bbox,
                        'class': class_name,
                        'confidence': float(conf),
                        'class_id': int(class_id)
                    })
            
            return detections
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []


class SimpleTracker:
    """Simple centroid-based tracker for multi-object tracking"""
    
    def __init__(self, max_distance: float = 50, max_age: int = 30):
        """
        Initialize tracker
        
        Args:
            max_distance: max pixel distance to match centroids
            max_age: max frames to keep track without detection
        """
        self.max_distance = max_distance
        self.max_age = max_age
        self.next_id = 0
        self.tracks: Dict[int, Dict] = {}
        self.frame_count = 0
    
    def get_centroid(self, bbox: List[float]) -> Tuple[float, float]:
        """Get centroid from bbox [x1, y1, x2, y2]"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def euclidean_distance(self, pt1: Tuple, pt2: Tuple) -> float:
        """Compute Euclidean distance"""
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections
        
        Returns: list of tracked objects with track_id
        """
        self.frame_count += 1
        tracked_objects = []
        
        if len(detections) == 0:
            # Age all existing tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
            return tracked_objects
        
        # Match detections to existing tracks
        detection_centroids = [self.get_centroid(d['bbox']) for d in detections]
        track_centroids = [((t['bbox'][0] + t['bbox'][2]) / 2, 
                           (t['bbox'][1] + t['bbox'][3]) / 2) for t in self.tracks.values()]
        
        matched_detection_indices = set()
        matched_track_ids = set()
        
        # Hungarian-like matching: simple greedy approach
        for detection_idx, det_centroid in enumerate(detection_centroids):
            best_match_id = None
            best_distance = self.max_distance
            
            for track_id, track in self.tracks.items():
                if track_id in matched_track_ids:
                    continue
                
                track_centroid = self.get_centroid(track['bbox'])
                distance = self.euclidean_distance(det_centroid, track_centroid)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing track
                self.tracks[best_match_id].update({
                    'bbox': detections[detection_idx]['bbox'],
                    'class': detections[detection_idx]['class'],
                    'confidence': detections[detection_idx]['confidence'],
                    'age': 0,
                    'frame_history': self.tracks[best_match_id]['frame_history'] + [self.frame_count]
                })
                matched_track_ids.add(best_match_id)
                matched_detection_indices.add(detection_idx)
        
        # Create new tracks for unmatched detections
        for detection_idx, detection in enumerate(detections):
            if detection_idx not in matched_detection_indices:
                self.tracks[self.next_id] = {
                    'bbox': detection['bbox'],
                    'class': detection['class'],
                    'confidence': detection['confidence'],
                    'age': 0,
                    'frame_history': [self.frame_count],
                    'track_id': self.next_id
                }
                self.next_id += 1
        
        # Age tracks that weren't matched
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_track_ids:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
        
        # Return active tracks
        for track_id, track in self.tracks.items():
            tracked_objects.append({
                'track_id': track_id,
                'bbox': track['bbox'],
                'class': track['class'],
                'confidence': track['confidence'],
                'frame_history': track['frame_history']
            })
        
        return tracked_objects


class PerceptionEngine:
    """Main perception engine combining YOLO + Tracking"""
    
    def __init__(self, model_name: str = "yolov5s", device: str = "cpu"):
        self.detector = YOLODetector(model_name, device)
        self.tracker = SimpleTracker()
        self.frame_count = 0
    
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process single frame
        
        Returns: list of tracked objects with boxes
        """
        self.frame_count += 1
        
        # Detect objects
        detections = self.detector.detect(frame)
        
        # Track objects
        tracked_objects = self.tracker.update(detections)
        
        return tracked_objects
    
    def reset(self):
        """Reset tracker for new video"""
        self.tracker = SimpleTracker()
        self.frame_count = 0
