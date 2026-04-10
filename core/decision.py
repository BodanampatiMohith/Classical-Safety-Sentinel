"""
Classical Rules + MCDM Decision Layer
Combines deep model output with rule-based logic for final safety decision
"""

import numpy as np
from typing import Dict, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety state classification"""
    SAFE = 0
    WARNING = 1
    CRITICAL = 2


class ClassicalRuleEngine:
    """Rule-based safety assessment"""
    
    def __init__(self):
        """Initialize rule thresholds"""
        # Distance thresholds (pixels)
        self.critical_distance_veh_ped = 100  # Vehicle-pedestrian critical
        self.warning_distance_veh_ped = 180
        
        self.critical_distance_veh_veh = 120
        self.warning_distance_veh_veh = 220
        
        # Speed thresholds (pixels/second)
        self.critical_speed = 110
        self.warning_speed = 70
        
        # TTC threshold (frames at current speed)
        self.critical_ttc = 20  # ~0.8 sec at 25fps
        self.warning_ttc = 40  # ~1.6 sec
        
        # Closing speed threshold
        self.critical_closing_speed = 60
        self.warning_closing_speed = 30
    
    def evaluate_rules(self, features: Dict) -> Dict[str, bool]:
        """
        Evaluate safety rules on features
        
        Returns: dict of rule violations
        """
        violations = {
            'critical_veh_ped_distance': False,
            'warning_veh_ped_distance': False,
            'critical_veh_veh_distance': False,
            'warning_veh_veh_distance': False,
            'high_speed': False,
            'very_high_speed': False,
            'moderate_closing_speed': False,
            'high_closing_speed': False,
            'warning_ttc': False,
            'low_ttc': False,
            'mixed_traffic_close': False,
            'pedestrian_present': False
        }
        
        # Rule 1: Vehicle-pedestrian distance
        veh_ped_dist = features.get('min_vehicle_pedestrian_distance', 1000)
        if veh_ped_dist < self.critical_distance_veh_ped:
            violations['critical_veh_ped_distance'] = True
        elif veh_ped_dist < self.warning_distance_veh_ped:
            violations['warning_veh_ped_distance'] = True
        
        # Rule 2: Vehicle-vehicle distance
        veh_veh_dist = features.get('min_vehicle_vehicle_distance', 1000)
        if veh_veh_dist < self.critical_distance_veh_veh:
            violations['critical_veh_veh_distance'] = True
        elif veh_veh_dist < self.warning_distance_veh_veh:
            violations['warning_veh_veh_distance'] = True
        
        # Rule 3: Speed
        max_speed = features.get('max_speed', 0)
        if max_speed > self.critical_speed:
            violations['very_high_speed'] = True
        elif max_speed > self.warning_speed:
            violations['high_speed'] = True
        
        # Rule 4: Closing speed
        closing_speed = features.get('max_closing_speed', 0)
        if closing_speed > self.critical_closing_speed:
            violations['high_closing_speed'] = True
        elif closing_speed > self.warning_closing_speed:
            violations['moderate_closing_speed'] = True
        
        # Rule 5: Time-to-collision
        ttc = features.get('min_ttc', 1000)
        if ttc < self.critical_ttc:
            violations['low_ttc'] = True
        elif ttc < self.warning_ttc:
            violations['warning_ttc'] = True
        
        # Rule 6: Mixed traffic (vehicles + pedestrians close)
        has_ped = features.get('has_pedestrians', False)
        has_veh = features.get('has_vehicles', False)
        close_interaction = veh_ped_dist < 200
        
        if has_ped and has_veh and close_interaction:
            violations['mixed_traffic_close'] = True
        
        if has_ped:
            violations['pedestrian_present'] = True
        
        return violations
    
    def count_violations(self, violations: Dict[str, bool]) -> Tuple[int, int]:
        """
        Count critical and warning violations
        
        Returns: (critical_count, warning_count)
        """
        critical_rules = [
            'critical_veh_ped_distance',
            'critical_veh_veh_distance',
            'very_high_speed',
            'high_closing_speed',
            'low_ttc'
        ]
        
        warning_rules = [
            'warning_veh_ped_distance',
            'warning_veh_veh_distance',
            'high_speed',
            'mixed_traffic_close',
            'warning_ttc',
            'moderate_closing_speed'
        ]
        
        critical_count = sum(1 for rule in critical_rules if violations.get(rule, False))
        warning_count = sum(1 for rule in warning_rules if violations.get(rule, False))
        
        return critical_count, warning_count


class MCDMDecisionEngine:
    """Multi-Criteria Decision Making engine"""
    
    def __init__(self):
        """Initialize MCDM weights and thresholds"""
        # Weights for different criteria
        self.weights = {
            'deep_anomaly': 0.25,  # Deep learning anomaly score
            'rule_violations': 0.35,  # Number of rule violations
            'distance': 0.20,  # Minimum distance to danger
            'speed': 0.10,  # Speed factor
            'pedestrian_risk': 0.10  # Pedestrian interaction risk
        }
        
        # Safety state thresholds
        self.warning_threshold = 0.32
        self.critical_threshold = 0.58
    
    def compute_risk_score(self, 
                          deep_anomaly_score: float,
                          rule_violations: Dict[str, bool],
                          features: Dict,
                          embedding: np.ndarray = None) -> float:
        """
        Compute unified risk score combining deep + rules
        
        Args:
            deep_anomaly_score: 0-1 anomaly score from LSTM
            rule_violations: dict of rule violations
            features: interaction features dict
            embedding: optional embedding from deep model
        
        Returns: risk score 0-1
        """
        critical_count, warning_count = self._count_violations(rule_violations)
        
        # Score component 1: Deep anomaly score (0-1)
        score_deep = deep_anomaly_score
        
        # Score component 2: Rule violations (0-1)
        max_critical = 5
        violation_score = min(1.0, (critical_count * 2 + warning_count) / max_critical)
        
        # Score component 3: Distance risk (0-1)
        min_dist = min(
            features.get('min_vehicle_pedestrian_distance', 1000),
            features.get('min_vehicle_vehicle_distance', 1000)
        )
        distance_score = max(0, 1.0 - (min_dist / 250))
        
        # Score component 4: Speed risk (0-1)
        max_speed = features.get('max_speed', 0)
        speed_score = min(1.0, max_speed / 200)
        
        # Score component 5: Pedestrian interaction risk (0-1)
        ped_risk = (
            1.0 if rule_violations.get('critical_veh_ped_distance', False) else
            0.8 if rule_violations.get('mixed_traffic_close', False) else
            0.6 if rule_violations.get('pedestrian_present', False) else
            0.0
        )
        
        # Combine with weights
        final_score = (
            self.weights['deep_anomaly'] * score_deep +
            self.weights['rule_violations'] * violation_score +
            self.weights['distance'] * distance_score +
            self.weights['speed'] * speed_score +
            self.weights['pedestrian_risk'] * ped_risk
        )
        
        # Boost critical scenarios
        if critical_count >= 2 or (
            rule_violations.get('critical_veh_ped_distance', False) and
            rule_violations.get('high_speed', False)
        ) or (
            rule_violations.get('critical_veh_veh_distance', False) and
            rule_violations.get('very_high_speed', False)
        ):
            final_score = min(1.0, final_score * 1.3)
        
        return final_score
    
    def classify(self, risk_score: float) -> SafetyLevel:
        """Classify risk score to safety level"""
        if risk_score >= self.critical_threshold:
            return SafetyLevel.CRITICAL
        elif risk_score >= self.warning_threshold:
            return SafetyLevel.WARNING
        else:
            return SafetyLevel.SAFE
    
    def _count_violations(self, violations: Dict[str, bool]) -> Tuple[int, int]:
        """Count violations"""
        critical_rules = [
            'critical_veh_ped_distance',
            'critical_veh_veh_distance',
            'very_high_speed',
            'high_closing_speed',
            'low_ttc'
        ]
        
        warning_rules = [
            'warning_veh_ped_distance',
            'warning_veh_veh_distance',
            'high_speed',
            'mixed_traffic_close',
            'warning_ttc',
            'moderate_closing_speed'
        ]
        
        critical_count = sum(1 for rule in critical_rules if violations.get(rule, False))
        warning_count = sum(1 for rule in warning_rules if violations.get(rule, False))
        
        return critical_count, warning_count


class HybridSafetyDecider:
    """Main safety decision engine combining all components"""
    
    def __init__(self):
        self.rule_engine = ClassicalRuleEngine()
        self.mcdm_engine = MCDMDecisionEngine()
        self.detection_history = []  # For smoothing decisions
        self.history_window = 5
    
    def decide(self,
              features: Dict,
              deep_anomaly_score: float = 0.0,
              embedding: np.ndarray = None,
              use_smoothing: bool = True) -> Tuple[SafetyLevel, float, Dict]:
        """
        Make final safety decision
        
        Args:
            features: interaction features from extractor
            deep_anomaly_score: deep learning anomaly likelihood
            embedding: learned embedding (optional)
            use_smoothing: apply temporal smoothing
        
        Returns:
            safety_level, risk_score, decision_info
        """
        # Evaluate rules
        violations = self.rule_engine.evaluate_rules(features)
        
        # Compute risk score
        risk_score = self.mcdm_engine.compute_risk_score(
            deep_anomaly_score,
            violations,
            features,
            embedding
        )
        
        # Classify
        safety_level = self.mcdm_engine.classify(risk_score)
        
        # Apply temporal smoothing
        if use_smoothing:
            self.detection_history.append((safety_level, risk_score))
            if len(self.detection_history) > self.history_window:
                self.detection_history = self.detection_history[-self.history_window:]
            
            # Smooth: take most common state in window
            counts = {}
            for level, score in self.detection_history:
                counts[level] = counts.get(level, 0) + 1
            
            safety_level = max(counts, key=counts.get)
        
        # Prepare decision info
        decision_info = {
            'safety_level': safety_level.name,
            'risk_score': float(risk_score),
            'violations': violations,
            'deep_score': float(deep_anomaly_score),
            'features_summary': {
                'min_veh_ped_dist': features.get('min_vehicle_pedestrian_distance', 0),
                'min_veh_veh_dist': features.get('min_vehicle_vehicle_distance', 0),
                'max_speed': features.get('max_speed', 0),
                'max_closing_speed': features.get('max_closing_speed', 0),
                'min_ttc': features.get('min_ttc', 0)
            }
        }
        
        return safety_level, risk_score, decision_info
    
    def reset(self):
        """Reset history"""
        self.detection_history = []
