"""
Feature Extraction Layer: Compute motion, distance, speed, and interaction features
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

VEHICLE_CLASSES = {'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'bike'}
PEDESTRIAN_CLASSES = {'person'}


class TrajectoryManager:
    """Manages trajectories and computes trajectory-based features"""
    
    def __init__(self, window_size: int = 30, fps: float = 25.0):
        """
        Args:
            window_size: number of frames in analysis window (1-3 seconds)
            fps: frames per second of video
        """
        self.window_size = window_size
        self.fps = fps
        self.trajectories: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.class_map: Dict[int, str] = {}
        self.frame_count = 0
    
    def update(self, tracked_objects: List[Dict]):
        """Update trajectories with new frame data"""
        self.frame_count += 1
        
        for obj in tracked_objects:
            track_id = obj['track_id']
            x1, y1, x2, y2 = obj['bbox']
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            self.trajectories[track_id].append(centroid)
            self.class_map[track_id] = obj['class']
            
            # Keep only last window_size frames
            if len(self.trajectories[track_id]) > self.window_size:
                self.trajectories[track_id] = self.trajectories[track_id][-self.window_size:]
    
    def get_speed(self, track_id: int) -> float:
        """Estimate speed in pixels/second from trajectory"""
        traj = self.trajectories.get(track_id, [])
        if len(traj) < 2:
            return 0.0
        
        # Use last 5 frames for robustness
        recent_frames = min(5, len(traj))
        p1 = traj[-recent_frames]
        p2 = traj[-1]
        
        distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        time_frames = recent_frames - 1
        time_seconds = time_frames / self.fps
        
        if time_seconds > 0:
            speed = distance_pixels / time_seconds
        else:
            speed = 0.0
        
        return speed
    
    def get_displacement(self, track_id: int) -> Tuple[float, float]:
        """Get displacement vector in last frame"""
        traj = self.trajectories.get(track_id, [])
        if len(traj) < 2:
            return (0.0, 0.0)
        
        p1 = traj[-2]
        p2 = traj[-1]
        return (p2[0] - p1[0], p2[1] - p1[1])
    
    def get_direction_angle(self, track_id: int) -> float:
        """Get movement direction in degrees (0-360)"""
        dx, dy = self.get_displacement(track_id)
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return angle % 360
    
    def is_accelerating(self, track_id: int, window: int = 10) -> bool:
        """Check if object is accelerating"""
        traj = self.trajectories.get(track_id, [])
        if len(traj) < window + 1:
            return False
        
        # Compare speeds in two halves of window
        first_half = traj[:window // 2]
        second_half = traj[-window // 2:]
        
        if len(first_half) < 2 or len(second_half) < 2:
            return False
        
        speed1 = np.sqrt((first_half[-1][0] - first_half[0][0])**2 + 
                        (first_half[-1][1] - first_half[0][1])**2)
        speed2 = np.sqrt((second_half[-1][0] - second_half[0][0])**2 + 
                        (second_half[-1][1] - second_half[0][1])**2)
        
        return speed2 > speed1 * 1.2  # 20% threshold


class FeatureExtractor:
    """Extracts features from tracked objects for analysis"""
    
    def __init__(self, window_size: int = 30, fps: float = 25.0):
        self.trajectory_mgr = TrajectoryManager(window_size, fps)
        self.fps = fps
        self.frame_data = []
    
    def process_frame(self, tracked_objects: List[Dict], frame_idx: int):
        """Process frame and accumulate data"""
        self.trajectory_mgr.update(tracked_objects)
        
        frame_info = {
            'frame_idx': frame_idx,
            'objects': []
        }
        
        for obj in tracked_objects:
            track_id = obj['track_id']
            obj_info = {
                'track_id': track_id,
                'class': obj['class'],
                'bbox': obj['bbox'],
                'speed': self.trajectory_mgr.get_speed(track_id),
                'direction': self.trajectory_mgr.get_direction_angle(track_id),
                'displacement': self.trajectory_mgr.get_displacement(track_id),
                'is_accelerating': self.trajectory_mgr.is_accelerating(track_id)
            }
            frame_info['objects'].append(obj_info)
        
        self.frame_data.append(frame_info)
    
    def get_pairwise_distance(self, bbox1: List, bbox2: List) -> float:
        """Get minimum distance between two bboxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Closest point on bbox1 to bbox2
        closest_x = np.clip(x2_min, x1_min, x1_max)
        closest_y = np.clip(y2_min, y1_min, y1_max)
        
        distance = np.sqrt((closest_x - x2_min)**2 + (closest_y - y2_min)**2)
        return distance
    
    def compute_interaction_features(self, frame_idx: int = -1) -> Dict:
        """
        Compute interaction features between tracked objects
        
        Returns dict with interaction metrics
        """
        if frame_idx < 0:
            frame_idx = len(self.frame_data) + frame_idx
        
        if frame_idx < 0 or frame_idx >= len(self.frame_data):
            return {}
        
        frame_info = self.frame_data[frame_idx]
        objects = frame_info['objects']
        
        features = {
            'min_vehicle_pedestrian_distance': float('inf'),
            'min_vehicle_vehicle_distance': float('inf'),
            'max_closing_speed': 0.0,
            'min_ttc': float('inf'),  # Time-To-Collision
            'high_speed_count': 0,
            'low_distance_count': 0,
            'interaction_count': 0
        }
        
        vehicles = [o for o in objects if o['class'].lower() in VEHICLE_CLASSES]
        pedestrians = [o for o in objects if o['class'].lower() in PEDESTRIAN_CLASSES]
        
        # Vehicle-Pedestrian interactions
        for veh in vehicles:
            for ped in pedestrians:
                distance = self.get_pairwise_distance(veh['bbox'], ped['bbox'])
                features['min_vehicle_pedestrian_distance'] = min(
                    features['min_vehicle_pedestrian_distance'], distance
                )
                
                if distance < 100:  # Close interaction
                    features['interaction_count'] += 1
                    features['low_distance_count'] += 1
                    
                    # Compute closing speed
                    veh_speed = veh['speed']
                    closing_speed = veh_speed  # Simplified: assume ped stationary
                    features['max_closing_speed'] = max(features['max_closing_speed'], closing_speed)
                    
                    # Compute TTC (very simplified)
                    if closing_speed > 0.5:  # Moving
                        ttc = distance / (closing_speed / self.fps)  # frames to collision
                        features['min_ttc'] = min(features['min_ttc'], ttc)
        
        # Vehicle-Vehicle interactions
        for i, v1 in enumerate(vehicles):
            for v2 in vehicles[i+1:]:
                distance = self.get_pairwise_distance(v1['bbox'], v2['bbox'])
                features['min_vehicle_vehicle_distance'] = min(
                    features['min_vehicle_vehicle_distance'], distance
                )
                
                if distance < 150:  # Close interaction
                    features['interaction_count'] += 1
                    
                    # Closing speed between vehicles
                    closing_speed = abs(v1['speed'] - v2['speed'])
                    features['max_closing_speed'] = max(features['max_closing_speed'], closing_speed)
        
        # Count high-speed vehicles
        for obj in vehicles:
            if obj['speed'] > 100:  # pixels/sec threshold
                features['high_speed_count'] += 1
        
        # Convert inf to reasonable values
        if features['min_vehicle_pedestrian_distance'] == float('inf'):
            features['min_vehicle_pedestrian_distance'] = 1000
        if features['min_vehicle_vehicle_distance'] == float('inf'):
            features['min_vehicle_vehicle_distance'] = 1000
        if features['min_ttc'] == float('inf'):
            features['min_ttc'] = 1000
        
        return features
    
    def get_window_features(self, window_size: int = 30) -> np.ndarray:
        """
        Get aggregated features for a time window
        
        Returns: feature vector (1D numpy array)
        """
        if len(self.frame_data) < window_size:
            return np.zeros(16)  # Default feature count
        
        window_data = self.frame_data[-window_size:]
        features = []
        
        # Aggregate object counts
        total_vehicles = 0
        total_pedestrians = 0
        avg_speed = 0
        max_speed = 0
        
        for frame_info in window_data:
            for obj in frame_info['objects']:
                if obj['class'].lower() in VEHICLE_CLASSES:
                    total_vehicles += 1
                elif obj['class'].lower() in PEDESTRIAN_CLASSES:
                    total_pedestrians += 1
                
                avg_speed += obj['speed']
                max_speed = max(max_speed, obj['speed'])
        
        frame_count = len(window_data)
        avg_speed = avg_speed / max(1, frame_count * 10)  # Normalize
        
        # Get last frame interactions
        interactions = self.compute_interaction_features(frame_idx=-1)
        
        # Combine into feature vector
        feature_vector = np.array([
            total_vehicles / max(1, frame_count),  # Avg vehicles per frame
            total_pedestrians / max(1, frame_count),  # Avg pedestrians per frame
            avg_speed / 100,  # Normalized avg speed
            max_speed / 200,  # Normalized max speed
            interactions['min_vehicle_pedestrian_distance'] / 500,  # Normalized distance
            interactions['min_vehicle_vehicle_distance'] / 500,
            interactions['max_closing_speed'] / 100,  # Normalized closing speed
            interactions['min_ttc'] / 100,  # Normalized TTC
            interactions['high_speed_count'] / max(1, frame_count),
            interactions['low_distance_count'] / max(1, frame_count),
            interactions['interaction_count'] / max(1, frame_count),
            1 if total_pedestrians > 0 else 0,  # Pedestrian present
            1 if total_vehicles > 0 and total_pedestrians > 0 else 0,  # Mixed traffic
            max_speed > 100,  # High speed threshold
            interactions['min_vehicle_pedestrian_distance'] < 100,  # Danger distance
            interactions['max_closing_speed'] > 50  # High closing speed
        ], dtype=np.float32)
        
        return feature_vector
    
    def reset(self):
        """Reset for new video"""
        self.trajectory_mgr.trajectories.clear()
        self.trajectory_mgr.class_map.clear()
        self.trajectory_mgr.frame_count = 0
        self.frame_data.clear()
