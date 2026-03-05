"""
Main Pipeline: Orchestrates all components for end-to-end inference
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

from core.perception import PerceptionEngine
from core.features import FeatureExtractor
from models.temporal import TemporalAnomalyDetector
from core.decision import HybridSafetyDecider, SafetyLevel

logger = logging.getLogger(__name__)


class SafetySentinelPipeline:
    """Complete safety detection pipeline"""
    
    def __init__(self, 
                 yolo_model: str = "yolov5s",
                 temporal_model_path: Optional[str] = None,
                 device: str = "cpu",
                 fps: float = 25.0,
                 window_size: int = 30):
        """
        Initialize pipeline
        
        Args:
            yolo_model: YOLOv5 variant
            temporal_model_path: path to pretrained temporal model
            device: compute device ('cpu' or 'cuda')
            fps: video frame rate
            window_size: analysis window in frames
        """
        self.device = device
        self.fps = fps
        self.window_size = window_size
        
        # Performance tracking
        self.processing_times = {
            'perception': [],
            'features': [],
            'temporal': [],
            'decision': [],
            'total': []
        }
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize components
        logger.info("🔧 Initializing perception engine...")
        self.perception_engine = PerceptionEngine(yolo_model, device)
        
        logger.info("📊 Initializing feature extractor...")
        self.feature_extractor = FeatureExtractor(window_size, fps)
        
        logger.info("🧠 Initializing temporal model...")
        self.temporal_detector = TemporalAnomalyDetector(
            input_size=16,
            hidden_size=64,
            device=device,
            model_path=temporal_model_path
        )
        
        logger.info("⚖️ Initializing decision engine...")
        self.decision_engine = HybridSafetyDecider()
        
        self.frame_count = 0
        self.events = []  # List of detected events
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info("✅ Pipeline initialization complete")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process single frame through pipeline with performance tracking
        
        Returns: frame result dict with detection and decision
        """
        start_time = time.time()
        
        with self._lock:
            self.frame_count += 1
        
        try:
            # Stage 1: Perception - detect and track
            perception_start = time.time()
            tracked_objects = self.perception_engine.process_frame(frame)
            perception_time = time.time() - perception_start
            
            # Stage 2: Features - extract motion and interaction features
            features_start = time.time()
            self.feature_extractor.process_frame(tracked_objects, self.frame_count)
            features_time = time.time() - features_start
            
            # Stage 3: Deep temporal model - compute anomaly score
            temporal_start = time.time()
            window_features = self.feature_extractor.get_window_features(self.window_size)
            self.temporal_detector.add_features(window_features)
            
            deep_anomaly_score, embedding = self.temporal_detector.detect(min_seq_len=10)
            temporal_time = time.time() - temporal_start
            
            # Stage 4: Decision - combine rules + deep + MCDM
            decision_start = time.time()
            interaction_features = self.feature_extractor.compute_interaction_features()
            
            # Add derived features
            interaction_features['max_speed'] = max(
                [obj.get('speed', 0) for obj in self.feature_extractor.frame_data[-1]['objects']],
                default=0
            ) if self.feature_extractor.frame_data else 0
            
            has_veh = any(obj['class'].lower() in ['car', 'truck', 'bus'] 
                         for obj in self.feature_extractor.frame_data[-1]['objects']) \
                     if self.feature_extractor.frame_data else False
            has_ped = any(obj['class'].lower() == 'person' 
                         for obj in self.feature_extractor.frame_data[-1]['objects']) \
                     if self.feature_extractor.frame_data else False
            
            interaction_features['has_vehicles'] = has_veh
            interaction_features['has_pedestrians'] = has_ped
            
            safety_level, risk_score, decision_info = self.decision_engine.decide(
                interaction_features,
                deep_anomaly_score,
                embedding
            )
            decision_time = time.time() - decision_start
            
            total_time = time.time() - start_time
            
            # Track performance
            self.processing_times['perception'].append(perception_time)
            self.processing_times['features'].append(features_time)
            self.processing_times['temporal'].append(temporal_time)
            self.processing_times['decision'].append(decision_time)
            self.processing_times['total'].append(total_time)
            
            frame_result = {
                'frame_idx': self.frame_count,
                'timestamp': self.frame_count / self.fps,
                'tracked_objects': tracked_objects,
                'safety_level': safety_level,
                'risk_score': risk_score,
                'deep_anomaly_score': deep_anomaly_score,
                'decision_info': decision_info,
                'interaction_features': interaction_features,
                'processing_times': {
                    'perception': perception_time,
                    'features': features_time,
                    'temporal': temporal_time,
                    'decision': decision_time,
                    'total': total_time
                }
            }
            
            # Record events
            if safety_level in [SafetyLevel.WARNING, SafetyLevel.CRITICAL]:
                with self._lock:
                    event = {
                        'frame_idx': self.frame_count,
                        'timestamp': self.frame_count / self.fps,
                        'level': safety_level.name,
                        'risk_score': risk_score,
                        'details': decision_info
                    }
                    self.events.append(event)
                
                logger.warning(f"⚠️ Frame {self.frame_count}: {safety_level.name} event detected "
                              f"(risk={risk_score:.2f})")
            
            return frame_result
            
        except Exception as e:
            logger.error(f"❌ Error processing frame {self.frame_count}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return error result
            return {
                'frame_idx': self.frame_count,
                'timestamp': self.frame_count / self.fps,
                'tracked_objects': [],
                'safety_level': SafetyLevel.SAFE,
                'risk_score': 0.0,
                'deep_anomaly_score': 0.0,
                'decision_info': {'error': str(e)},
                'interaction_features': {},
                'processing_times': {'total': time.time() - start_time},
                'error': True
            }
    
    def process_video(self, video_path: str, max_frames: Optional[int] = None) -> List[Dict]:
        """
        Process entire video file
        
        Args:
            video_path: path to video file
            max_frames: limit number of frames (for testing)
        
        Returns: list of frame results
        """
        logger.info(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []
        
        self.reset()
        results = []
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_idx >= max_frames:
                    break
                
                # Resize for faster processing
                frame_resized = cv2.resize(frame, (640, 480))
                
                result = self.process_frame(frame_resized)
                results.append(result)
                frame_idx += 1
                
                if frame_idx % 30 == 0:
                    logger.info(f"Processed {frame_idx} frames...")
        
        finally:
            cap.release()
        
        logger.info(f"Completed video processing: {frame_idx} frames, {len(self.events)} events")
        return results
    
    def get_annotated_frame(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw detection and safety information on frame
        
        Returns: annotated frame
        """
        annotated = frame.copy()
        
        # Safety level color
        safety_level = result['safety_level']
        if safety_level == SafetyLevel.CRITICAL:
            color = (0, 0, 255)  # Red
            text_color = (255, 255, 255)
        elif safety_level == SafetyLevel.WARNING:
            color = (0, 165, 255)  # Orange
            text_color = (255, 255, 255)
        else:
            color = (0, 255, 0)  # Green
            text_color = (255, 255, 255)
        
        # Draw tracked objects
        for obj in result['tracked_objects']:
            x1, y1, x2, y2 = [int(v) for v in obj['bbox']]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID:{obj['track_id']} {obj['class'][:4]}"
            cv2.putText(annotated, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw safety status
        risk_score = result['risk_score']
        status_text = f"{safety_level.name} - Risk: {risk_score:.2f}"
        
        cv2.rectangle(annotated, (10, 10), (400, 50), (0, 0, 0), -1)
        cv2.putText(annotated, status_text, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw frame counter
        frame_text = f"Frame: {result['frame_idx']} | Time: {result['timestamp']:.2f}s"
        cv2.putText(annotated, frame_text, (10, annotated.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def save_annotated_video(self, video_path: str, output_path: str, 
                           max_frames: Optional[int] = None):
        """
        Process video and save annotated output
        """
        logger.info(f"Processing and saving: {output_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.reset()
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_idx >= max_frames:
                    break
                
                # Process with smaller size for speed
                frame_small = cv2.resize(frame, (640, 480))
                result = self.process_frame(frame_small)
                
                # Annotate
                annotated = self.get_annotated_frame(frame_small, result)
                
                # Resize back and save
                annotated_full = cv2.resize(annotated, (width, height))
                out.write(annotated_full)
                
                frame_idx += 1
                if frame_idx % 30 == 0:
                    logger.info(f"Saved {frame_idx} frames...")
        
        finally:
            cap.release()
            out.release()
        
        logger.info(f"Saved annotated video: {output_path}")
        return True
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for the pipeline"""
        stats = {}
        for stage, times in self.processing_times.items():
            if times:
                stats[stage] = {
                    'avg_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'total_time': np.sum(times),
                    'frames_processed': len(times)
                }
            else:
                stats[stage] = {
                    'avg_time': 0,
                    'min_time': 0,
                    'max_time': 0,
                    'total_time': 0,
                    'frames_processed': 0
                }
        
        # Calculate FPS
        if stats['total']['frames_processed'] > 0:
            total_processing_time = stats['total']['total_time']
            stats['overall_fps'] = stats['total']['frames_processed'] / total_processing_time
        else:
            stats['overall_fps'] = 0
            
        return stats
    
    def get_events(self) -> List[Dict]:
        """Get detected Warning/Critical events"""
        return self.events
    
    def reset(self):
        """Reset pipeline for new video"""
        logger.info("🔄 Resetting pipeline state...")
        
        self.perception_engine.reset()
        self.feature_extractor.reset()
        self.temporal_detector.reset()
        self.decision_engine.reset()
        
        with self._lock:
            self.frame_count = 0
            self.events = []
        
        # Clear performance stats
        for key in self.processing_times:
            self.processing_times[key].clear()
        
        logger.info("✅ Pipeline reset complete")
