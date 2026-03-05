# 🧮 Safety Sentinel - Hybrid Fusion Algorithm Documentation

**Version:** 1.1.0 | **Date:** February 2026

## 📋 Table of Contents

1. [Hybrid Decision Engine Overview](#hybrid-decision-engine-overview)
2. [Deep Learning Path (LSTM)](#deep-learning-path-lstm)
3. [Classical Rule Engine](#classical-rule-engine)
4. [Fusion & MCDM](#fusion--mcdm)
5. [Safety Classification](#safety-classification)
6. [Worked Examples](#worked-examples)
7. [Implementation Details](#implementation-details)

---

## 🎯 Hybrid Decision Engine Overview

The Safety Sentinel system makes safety decisions by **fusing two independent analysis paths**:

```
Video Frame
    │
    ├─►[ DEEP LEARNING PATH ]        ├─►[ CLASSICAL RULES PATH ]
    │   LSTM Temporal Anomaly        │   Deterministic Thresholds
    │   - Learns normal patterns     │   - Vehicle-pedestrian distance
    │   - Detects deviations        │   - Vehicle-vehicle distance
    │   → Anomaly Score (0-1)        │   - Speeds and closing rates
    │                                │   - Time-to-collision (TTC)
    │                                │   → Violation Count
    │                                │
    ├────────────────┬───────────────┤
    │                │               │
    │            MCDM FUSION         │
    │     (Multi-Criteria Decision)  │
    │                │               │
    └────────────────┴───────────────┘
                     │
            ┌────────▼─────────┐
            │   Risk Score     │
            │      (0-1)       │
            └────────┬─────────┘
                     │
        Thresholded Safety Level:
        SAFE (< 0.4) | WARNING (0.4-0.7) | CRITICAL (≥ 0.7)
```

**Why Hybrid?**
- **Deep Learning**: Captures complex patterns that rules miss
- **Classical Rules**: Explainability, fast, proven domain knowledge
- **Together**: Robust, interpretable, safety-compliant

---

## 🧠 Deep Learning Path: LSTM Anomaly Detection

### 1. LSTM Model Architecture

```
Input Features (Dimension: 16)
    ↓ [embedding]
┌─────────────────────┐
│ Sequence of 10-30   │
│ Feature Vectors     │
│ Shape: (seq_len,16) │
└──────────┬──────────┘
           ↓
      ┌─────────────────────────────────────┐
      │  LSTM Layer 1                       │
      │  - Hidden Size: 64                  │
      │  - Dropout: 0.2                     │
      │  - Bidirectional: False             │
      └──────────────┬──────────────────────┘
                     ↓
                ┌─────────────────────────────────────┐
                │  LSTM Layer 2                       │
                │  - Hidden Size: 64                  │
                │  - Dropout: 0.2                     │
                └──────────────┬──────────────────────┘
                               ↓
                        Last Hidden State (64-dim)
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
        ┌────────▼────────┐        ┌────────▼────────┐
        │ FC Layers       │        │ Embedding Layer │
        │ 64→32→16→1      │        │ 64→8            │
        │ (ReLU + Dropout)│        │                 │
        │ + Sigmoid       │        │                 │
        └────────┬────────┘        └────────┬────────┘
                 │                          │
        ┌────────▼──────────┐      ┌────────▼──────────┐
        │ Anomaly Score     │      │ Embedding Vector  │
        │ AnomalyScr (0-1)  │      │ (8-dimensional)   │
        └───────────────────┘      └───────────────────┘
```

### 2. Input Features (16-dimensional)

The LSTM ingests **16 safety-relevant features** extracted from tracked objects:

```python
features = [
    # Spatial Distances
    min_vehicle_pedestrian_distance,          # [0]  Min V-P distance (pixels)
    avg_vehicle_pedestrian_distance,          # [1]  Avg V-P distance
    min_vehicle_vehicle_distance,             # [2]  Min V-V distance
    avg_vehicle_vehicle_distance,             # [3]  Avg V-V distance
    
    # Temporal/Velocity
    max_speed,                                # [4]  Max object speed (px/s)
    avg_speed,                                # [5]  Avg speed
    max_closing_speed,                        # [6]  Max relative speed (closing)
    avg_closing_speed,                        # [7]  Avg closing speed
    
    # Interaction Features
    min_ttc,                                  # [8]  Min time-to-collision (frames)
    avg_ttc,                                  # [9]  Avg TTC
    num_crossing_events,                      # [10] Path crossing count
    num_conflict_pairs,                       # [11] Object pairs in conflict
    
    # Scene Context
    pedestrian_count,                         # [12] Active pedestrians
    vehicle_count,                            # [13] Active vehicles
    relative_velocity_variance,               # [14] Velocity consistency
    scene_complexity,                         # [15] Number of dynamic objects
]
```

### 3. Forward Pass Algorithm

```python
def lstm_forward_pass(feature_sequence):
    """
    Input:
        feature_sequence: Shape (batch_size=1, seq_len=10-30, features=16)
    
    Process:
        1. LSTM processes entire sequence
        2. Extract last hidden state
        3. Pass through anomaly detection head
        4. Extract embedding
    
    Output:
        anomaly_score: Float in [0, 1]
        embedding: Vector in R^8
    """
    
    # LSTM forward
    lstm_output, (h_n, c_n) = lstm_layer(feature_sequence)
    # lstm_output shape: (batch, seq_len, hidden=64)
    # h_n shape: (num_layers=2, batch, hidden=64)
    
    # Use last hidden state from last layer
    last_hidden = lstm_output[:, -1, :]  # (batch, 64)
    
    # Anomaly score head: 64 → 32 → 16 → 1 (sigmoid)
    x = relu(linear_64_32(last_hidden))
    x = dropout(x, p=0.2)
    x = relu(linear_32_16(x))
    anomaly_score = sigmoid(linear_16_1(x))  # (batch, 1) ∈ [0,1]
    
    # Embedding head: 64 → 8
    embedding = embedding_fc(last_hidden)  # (batch, 8)
    
    return anomaly_score.squeeze(), embedding
```

### 4. Anomaly Score Interpretation

| Score Range | Classification | Interpretation |
|-------------|-----------------|---|
| 0.0 - 0.3  | **Low** | Normal traffic, no anomaly detected |
| 0.3 - 0.6  | **Medium** | Unusual pattern, monitor closely |
| 0.6 - 0.85 | **High** | Strong anomaly, likely near-miss |
| 0.85 - 1.0 | **Critical** | Severe anomaly, immediate action |

**Note:** Thresholds are configurable and should be calibrated on domain data.

---

## 📋 Classical Rule Engine

### 1. Configuration Thresholds

```python
# Distance thresholds (pixels)
CRITICAL_DISTANCE_VEH_PED = 80      # Vehicle-pedestrian collision imminent
WARNING_DISTANCE_VEH_PED = 150      # Close approach, risky
CRITICAL_DISTANCE_VEH_VEH = 150     # Vehicle collision risk
WARNING_DISTANCE_VEH_VEH = 300      # Vehicles too close

# Speed thresholds (pixels/second at 25 FPS)
# Note: ~1 pixel at 640px width = ~0.15%  of image width
# For typical highway: 1000px @ 30mph = 33 px/s
CRITICAL_SPEED = 150                 # Extremely high speed
WARNING_SPEED = 100                  # High speed

# Time-to-Collision (TTC) thresholds (frames @ 25 FPS)
CRITICAL_TTC = 15                    # ~0.6 seconds
WARNING_TTC = 30                     # ~1.2 seconds

# Closing speed thresholds (px/s)
CRITICAL_CLOSING_SPEED = 100         # Rapid approach
WARNING_CLOSING_SPEED = 50           # Moderate approach
```

### 2. Rule Evaluation Algorithm

```python
def evaluate_rules(features: Dict) -> Dict[str, bool]:
    """
    Evaluate safety rules on current features
    
    Returns: Dictionary of boolean rule violations
    """
    
    violations = {}
    
    # RULE 1: Vehicle-Pedestrian Distance
    veh_ped_dist = features.get('min_vehicle_pedestrian_distance', 1000)
    violations['critical_veh_ped_distance'] = veh_ped_dist < CRITICAL_DISTANCE_VEH_PED
    violations['warning_veh_ped_distance'] = veh_ped_dist < WARNING_DISTANCE_VEH_PED
    
    # RULE 2: Vehicle-Vehicle Distance
    veh_veh_dist = features.get('min_vehicle_vehicle_distance', 1000)
    violations['critical_veh_veh_distance'] = veh_veh_dist < CRITICAL_DISTANCE_VEH_VEH
    violations['warning_veh_veh_distance'] = veh_veh_dist < WARNING_DISTANCE_VEH_VEH
    
    # RULE 3: Speed
    max_speed = features.get('max_speed', 0)
    violations['very_high_speed'] = max_speed > CRITICAL_SPEED
    violations['high_speed'] = max_speed > WARNING_SPEED
    
    # RULE 4: Closing Speed
    closing_speed = features.get('max_closing_speed', 0)
    violations['high_closing_speed'] = closing_speed > CRITICAL_CLOSING_SPEED
    
    # RULE 5: Time-to-Collision
    min_ttc = features.get('min_ttc', 1000)  # in frames
    violations['low_ttc'] = min_ttc < CRITICAL_TTC
    
    # RULE 6: Mixed traffic (vehicles + pedestrians in close proximity)
    has_pedestrians = features.get('has_pedestrians', False)
    has_vehicles = features.get('has_vehicles', False)
    close_interaction = veh_ped_dist < 200
    violations['mixed_traffic_close'] = has_pedestrians and has_vehicles and close_interaction
    
    # RULE 7: Pedestrian presence
    violations['pedestrian_present'] = has_pedestrians
    
    return violations
```

### 3. Violation Counting & Weighting

```python
def count_violations(violations: Dict[str, bool]) -> Tuple[int, int]:
    """
    Count critical and warning violations
    Returns: (critical_count, warning_count)
    """
    
    # Rules weighted as CRITICAL (higher severity)
    critical_rules = [
        'critical_veh_ped_distance',
        'critical_veh_veh_distance',
        'very_high_speed',
        'high_closing_speed',
        'low_ttc'
    ]
    
    # Rules weighted as WARNING (lower severity)
    warning_rules = [
        'warning_veh_ped_distance',
        'warning_veh_veh_distance',
        'high_speed',
        'mixed_traffic_close'
    ]
    
    critical_count = sum(1 for rule in critical_rules 
                        if violations.get(rule, False))
    warning_count = sum(1 for rule in warning_rules 
                       if violations.get(rule, False))
    
    return critical_count, warning_count


def compute_rule_score(violations: Dict, 
                      critical_count: int,
                      warning_count: int) -> float:
    """
    Convert violations to normalized score (0-1)
    
    Formula:
        rule_score = min(1.0, (2*critical_count + warning_count) / max_expected)
    """
    
    max_expected_violations = 5  # Tunable parameter
    
    rule_score = (2.0 * critical_count + warning_count) / max_expected_violations
    rule_score = min(1.0, rule_score)  # Clamp to [0, 1]
    
    return rule_score
```

---

## 🔗 Fusion & MCDM (Multi-Criteria Decision Making)

### 1. Component Scoring

**Deep Learning Score:**
```python
deep_score = anomaly_score  # Directly from LSTM (0-1)
```

**Classical Rule Score:**
```python
# Based on violation counts
critical_count, warning_count = count_violations(violations)
classical_score = (2.0 * critical_count + warning_count) / 5.0
classical_score = min(1.0, classical_score)
```

**Distance Risk Score:**
```python
distance_score = 0.0
if 'min_vehicle_pedestrian_distance' in features:
    dist = features['min_vehicle_pedestrian_distance']
    distance_score = max(0, 1.0 - (dist / 200))  # Normalized: 0 at 200px, 1 at 0px
```

**Speed Risk Score:**
```python
speed_score = 0.0
if 'max_speed' in features:
    speed = features['max_speed']
    speed_score = min(1.0, speed / 200)  # Normalized: 0 at 0 px/s, 1 at 200 px/s
```

**Pedestrian Interaction Risk:**
```python
ped_risk = 1.0 if violations.get('pedestrian_present', False) else 0.0
```

### 2. Weighted Fusion Formula

Let:
- $D$ = deep anomaly score (0-1)
- $R$ = classical rule score (0-1)
- $w_D = 0.6$ = deep weight
- $w_R = 0.4$ = classical weight
- Additional factors: distance, speed, pedestrian

**Base Risk Score:**
$$S_{base} = w_D \cdot D + w_R \cdot R$$

**Alternative (with more factors):**
$$S_{weighted} = 0.25 \cdot D + 0.35 \cdot R + 0.20 \cdot D_{risk} + 0.10 \cdot S_{speed} + 0.10 \cdot P_{risk}$$

Where weights sum to 1.0 for proper normalization.

### 3. Boost Logic (Penalty/Bonus)

```python
def apply_boost_logic(base_score: float, 
                     violations: Dict,
                     critical_count: int) -> float:
    """
    Apply multiplicative boosts for high-risk scenarios
    """
    
    score = base_score
    
    # BOOST 1: Multiple critical violations detected
    if critical_count >= 2:
        score = min(1.0, score * 1.3)
    
    # BOOST 2: Vehicle-pedestrian critical distance + high speed
    if (violations.get('critical_veh_ped_distance', False) and 
        violations.get('very_high_speed', False)):
        score = min(1.0, score * 1.2)
    
    # BOOST 3: Low TTC + pedestrian interaction
    if (violations.get('low_ttc', False) and 
        violations.get('pedestrian_present', False)):
        score = min(1.0, score * 1.25)
    
    # Final clamp
    score = max(0.0, min(1.0, score))
    
    return score


def compute_final_risk_score(deep_score: float,
                            classical_score: float,
                            violations: Dict,
                            critical_count: int) -> float:
    """
    Compute final risk score with all components
    """
    
    # Base weighted fusion
    base_score = 0.6 * deep_score + 0.4 * classical_score
    
    # Apply boost logic
    final_score = apply_boost_logic(base_score, violations, critical_count)
    
    return final_score
```

### 4. MCDM Algorithm Overview

```
┌─────────────────────────────────────────┐
│    Input: Features + LSTM Output        │
│    - 16D feature vector                 │
│    - Deep anomaly score                 │
│    - Embedding (8D)                     │
└──────────────────┬──────────────────────┘
                   │
         ┌─────────▼─────────┐
         │ Extract Components │
         │ - Deep Score      │
         │ - Rule Violations │
         │ - Counts          │
         └────────┬──────────┘
                  │
      ┌───────────▼───────────┐
      │ Compute Sub-Scores    │
      │ - Deep (0.60 weight)  │
      │ - Classical (0.40)    │
      │ - Distance            │
      │ - Speed               │
      │ - Pedestrian Interaction
      └────────┬──────────────┘
               │
        ┌──────▼─────────┐
        │  Weighted Sum  │
        │  Base Score    │
        └────────┬───────┘
                 │
        ┌────────▼────────┐
        │  Boost Logic    │
        │  (if needed)    │
        └────────┬────────┘
                 │
        ┌────────▼──────────┐
        │  Final Risk Score │
        │  ∈ [0, 1]         │
        └────────┬──────────┘
                 │
         ┌───────▼────────────┐
         │ Return with factors │
         │ for explainability  │
         └────────────────────┘
```

---

## 🚨 Safety Classification

### Thresholds & Mapping

```python
def classify_safety_level(risk_score: float) -> SafetyLevel:
    """
    Maps continuous risk score to discrete safety level
    """
    
    if risk_score >= 0.7:
        return SafetyLevel.CRITICAL
    elif risk_score >= 0.4:
        return SafetyLevel.WARNING
    else:
        return SafetyLevel.SAFE
```

### Safety Level Definitions

| Level | Score | Meaning | Action |
|-------|-------|---------|--------|
| **SAFE** | < 0.4 | No immediate danger | Continue monitoring |
| **WARNING** | 0.4 - 0.7 | Potential risk detected | Alert operator |
| **CRITICAL** | ≥ 0.7 | High-risk near-miss | Immediate intervention |

### Decision Output Structure

```python
@dataclass
class SafetyDecision:
    safety_level: SafetyLevel                    # SAFE/WARNING/CRITICAL
    risk_score: float                            # 0-1 quantitative score
    
    # Component contributions (for transparency)
    decision_factors: Dict = {
        'deep_anomaly_score': 0.70,              # From LSTM
        'classical_rule_score': 0.75,            # From rules
        'critical_violations_count': 2,
        'warning_violations_count': 1,
        'distance_risk': 0.85,
        'speed_risk': 0.60,
        'pedestrian_interaction_risk': 0.90,
        'ttc_risk': 0.88
    }
    
    # Explainability
    dominant_cause: str                          # e.g., "low_ttc_pedestrian_conflict"
    contributing_factors: List[str]              # ["critical_veh_ped_distance", ...]
    timestamp: float                             # Frame time
    objects_involved: List[Dict]                 # Tracked object details
```

---

## 📊 Worked Examples

### Example 1: Safe Intersection Crossing

**Input Features:**
```python
features = {
    'min_vehicle_pedestrian_distance': 500,      # Far apart
    'avg_vehicle_pedestrian_distance': 600,
    'min_vehicle_vehicle_distance': 800,
    'max_speed': 40,                             # Moderate speed
    'avg_speed': 38,
    'max_closing_speed': 10,
    'min_ttc': 500,                              # No collision risk
    'avg_ttc': 450,
    'pedestrian_count': 1,
    'vehicle_count': 1,
}
```

**Rule Evaluation:**
```python
violations = {
    'critical_veh_ped_distance': False,          # 500 >= 80 ✓
    'warning_veh_ped_distance': False,           # 500 >= 150 ✓
    'critical_veh_veh_distance': False,          # 800 >= 150 ✓
    'warning_veh_veh_distance': False,           # 800 >= 300 ✓
    'very_high_speed': False,                    # 40 <= 150 ✓
    'high_speed': False,                         # 40 <= 100 ✓
    'high_closing_speed': False,                 # 10 <= 100 ✓
    'low_ttc': False,                            # 500 >= 15 ✓
    'mixed_traffic_close': False,                # 500 >= 200 ✓
    'pedestrian_present': True
}

critical_count = 0
warning_count = 0
rule_score = 0.0
```

**Deep Learning:**
```python
anomaly_score = 0.15  # Low anomaly (normal pattern)
```

**Fusion:**
```python
base_score = 0.6 × 0.15 + 0.4 × 0.0 = 0.09
final_score = apply_boost(0.09, violations, 0) = 0.09
```

**Decision:**
```
risk_score = 0.09
safety_level = SAFE (0.09 < 0.4) ✓
```

---

### Example 2: Warning Level Near-Miss

**Input Features:**
```python
features = {
    'min_vehicle_pedestrian_distance': 140,      # Close!
    'avg_vehicle_pedestrian_distance': 200,
    'max_speed': 85,                             # Moderate speed
    'max_closing_speed': 60,                     # Approaching
    'min_ttc': 28,                               # ~1.1 seconds
}
```

**Rule Evaluation:**
```python
violations = {
    'warning_veh_ped_distance': True,            # 140 < 150 ✗
    'high_closing_speed': False,                 # 60 <= 100 ✓
    'pedestrian_present': True,
    # ... others False
}

critical_count = 0
warning_count = 1          # One warning violation
rule_score = (2×0 + 1) / 5 = 0.2
```

**Deep Learning:**
```python
anomaly_score = 0.45  # Moderate anomaly detected
```

**Fusion:**
```python
base_score = 0.6 × 0.45 + 0.4 × 0.2 = 0.27 + 0.08 = 0.35
final_score = apply_boost(0.35, violations, 0) = 0.35
```

**Decision:**
```
risk_score = 0.35
safety_level = SAFE (0.35 < 0.4)  [Borderline!]
```

*Note: This demonstrates the "warning threshold creep" - recommend adjusting to 0.35-0.45 range based on datasets.*

---

### Example 3: Critical Collision Risk

**Input Features:**
```python
features = {
    'min_vehicle_pedestrian_distance': 50,       # VERY CLOSE
    'max_speed': 160,                            # High speed
    'max_closing_speed': 150,                    # Rapid approach
    'min_ttc': 10,                               # ~0.4 seconds!
    'pedestrian_count': 1,
    'vehicle_count': 1,
}
```

**Rule Evaluation:**
```python
violations = {
    'critical_veh_ped_distance': True,           # 50 < 80 ✗
    'very_high_speed': True,                     # 160 > 150 ✗
    'high_closing_speed': True,                  # 150 > 100 ✗
    'low_ttc': True,                             # 10 < 15 ✗
    'pedestrian_present': True,
    # additional violations...
}

critical_count = 4                               # Multiple critical!
warning_count = 1
rule_score = (2×4 + 1) / 5 = 1.8 → clamped to 1.0
```

**Deep Learning:**
```python
anomaly_score = 0.92  # Strong anomaly (highly unusual pattern)
```

**Fusion:**
```python
base_score = 0.6 × 0.92 + 0.4 × 1.0 = 0.552 + 0.4 = 0.952

# Apply boosts
critical_count = 4 >= 2  → Boost ×1.3
final_score = min(1.0, 0.952 × 1.3) = 1.0
```

**Decision:**
```
risk_score = 1.0
safety_level = CRITICAL (1.0 >= 0.7) ✓✓✓
dominant_cause = "critical_veh_ped_distance + high_speed + low_ttc"
```

---

## 🔧 Implementation Details

### Python Classes

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

class SafetyLevel(Enum):
    SAFE = 0
    WARNING = 1
    CRITICAL = 2

@dataclass
class DecisionInfo:
    deep_anomaly_score: float
    rule_violation_score: float
    violation_count: Tuple[int, int]  # (critical, warning)
    contributing_factors: List[str]
    dominant_cause: str

def compute_hybrid_decision(
    interaction_features: Dict,
    deep_anomaly_score: float,
    embedding: np.ndarray,
    rule_violation_weights: Dict[str, float] = None
) -> Tuple[SafetyLevel, float, DecisionInfo]:
    """
    Main decision function combining all pathways
    """
    # Evaluation logic here...
    pass
```

### Configurable Parameters

```python
# In config.yaml or environment variables
WEIGHTS:
  DEEP: 0.6
  CLASSICAL: 0.4

THRESHOLDS:
  SAFE_CRITICAL: 0.4
  CRITICAL_CRITICAL: 0.7
  
  DISTANCE_VEH_PED:
    CRITICAL: 80
    WARNING: 150
  
  DISTANCE_VEH_VEH:
    CRITICAL: 150
    WARNING: 300
  
  SPEED:
    CRITICAL: 150
    WARNING: 100
  
  TTC:
    CRITICAL: 15
    WARNING: 30

BOOST_FACTORS:
  MULTI_VIOLATION: 1.3
  PED_SPEED_COMBO: 1.2
  TTC_PED_COMBO: 1.25
```

---

## 📈 Validation & Calibration

### Metrics

```python
# Per-decision metrics
precision = TP / (TP + FP)   # Of detected near-misses, how many were true?
recall = TP / (TP + FN)      # Of actual near-misses, how many detected?
f1_score = 2 * (precision * recall) / (precision + recall)

# Across entire test set
mean_average_precision (mAP)
area_under_roc_curve (AUC)
```

### Calibration Process

1. **Collect Ground Truth:** Annotate ~500 video clips with near-miss labels
2. **Train LSTM:** 70% train, 15% validation, 15% test
3. **Threshold Search:** Find optimal risk score thresholds via ROC analysis
4. **Rule Weight Tuning:** Adjust classical rule weights for best F1 score
5. **Cross-Validation:** Validate on unseen intersections/conditions

---

## 📚 References

- **LSTM Fundamentals**: Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"
- **Anomaly Detection**: Goldstein & Uchida (2016) - Review of unsupervised methods
- **Time-to-Collision**: Hayward et al. (2015) - Surrogate Safety Measures & Crash Prediction
- **Multi-Criteria Decision Making**: Hwang & Yoon (1981) - MCDM methods

---

**Last Updated:** February 22, 2026 | **Maintainer:** Safety Sentinel Team
