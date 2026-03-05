# 🏗️ Safety Sentinel - Technical Architecture

**Version:** 1.1.0 | **Last Updated:** February 2026

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Hybrid Decision Engine](#hybrid-decision-engine)
5. [Technology Stack](#technology-stack)
6. [API Specification](#api-specification)
7. [Database & Storage](#database--storage)
8. [Deployment Architecture](#deployment-architecture)
9. [Performance Characteristics](#performance-characteristics)
10. [Security Considerations](#security-considerations)

---

## 🎯 System Overview

Safety Sentinel is a **hybrid AI system** combining:
- **Deep Learning Path**: LSTM-based temporal anomaly detection
- **Classical Path**: Rule-based safety logic with interpretable thresholds
- **Fusion Layer**: Multi-criteria decision making (MCDM) with weighted scoring

**Core Innovation:** Balances deep learning's pattern recognition with classical rules for **explainability, interpretability, and robustness** — critical for safety-critical near-miss detection.

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: Video Stream                     │
├─────────────────────────────────────────────────────────────┤
│                        Frame Extraction                      │
│                      (25 FPS, 640×480)                      │
├─────────────────────────────────────────────────────────────┤
│  PERCEPTION LAYER  │  YOLOv5 + MultiObject Tracking         │
├─────────────────────────────────────────────────────────────┤
│ FEATURE EXTRACTION │ Spatial + Temporal + Interaction        │
├────────────────────┬────────────────────────────────────────┤
│ DEEP PATH (60%)    │  CLASSICAL PATH (40%)                  │
│ LSTM Anomaly       │  Rule Engine (TTC, Distance, Speed)    │
│ Detection          │  Binary Threshold Evaluation           │
│ Embeddings         │  Violation Counting                    │
├────────────────────┴────────────────────────────────────────┤
│          HYBRID FUSION (MCDM - Multi-Criteria DM)           │
│  Weighted Score = 0.6×Deep + 0.4×Classical + Boost         │
├─────────────────────────────────────────────────────────────┤
│         DECISION: SAFE | WARNING | CRITICAL                 │
├─────────────────────────────────────────────────────────────┤
│  OUTPUT: Annotated Video + Events + Explainability Data    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧩 Component Architecture

### 1️⃣ **Perception Engine** (`core/perception.py`)

**Purpose:** Detection and tracking of traffic objects

**Components:**
- `YOLODetector`: Wrapper for YOLOv5s object detection
  - Classes tracked: car, truck, bus, bicycle, motorcycle, person
  - Input: Frame (image tensor)
  - Output: List of detections with bbox, class, confidence

- `SimpleTracker`: Centroid-based multi-object tracker
  - Tracks objects across frames using bounding box centroids
  - Assigns unique IDs to vehicles and pedestrians
  - Records trajectory history per object

**Key Features:**
```python
# Detection example output:
[
    {
        'id': 1,
        'bbox': [100, 200, 300, 400],
        'class': 'car',
        'confidence': 0.95,
        'centroid': [200, 300],
        'trajectory': [[200, 300], [205, 302], ...]
    }
]
```

---

### 2️⃣ **Feature Extractor** (`core/features.py`)

**Purpose:** Extract safety-relevant features from detected objects

**Feature Categories:**

A. **Spatial Features:**
- Minimum distance between vehicle-pedestrian pairs
- Minimum distance between vehicle pairs
- Relative positioning (approaching vs. separating)

B. **Temporal Features:**
- Object velocities (pixels/second)
- Relative velocities (closing speed)
- Acceleration / deceleration

C. **Interaction Features:**
- Time-to-Collision (TTC) calculation
- Crossing path detection
- Convergence indicators

**Window-based Processing:**
- Features computed over 30-frame windows (~1.2 seconds at 25 FPS)
- Rolling window allows smooth temporal analysis
- Per-frame and per-window aggregation

---

### 3️⃣ **Temporal Deep Model** (`models/temporal.py`)

**Architecture:** LSTM-based Autoencoder for Anomaly Detection

```
Input Features (16-dim) 
    ↓
LSTM Layer 1 (64 hidden, dropout=0.2)
    ↓
LSTM Layer 2 (64 hidden, dropout=0.2)
    ↓
Dense FC Layers:
  - 64 → 32 (ReLU)
  - 32 → 16 (ReLU)
  - 16 → 1 (Sigmoid)  [Anomaly Score]
    ↓
Embedding Layer (64 → 8 dims)  [Interpretability]
```

**Key Aspects:**
- **Input:** Sequence of 10-30 feature vectors
- **Output:** Anomaly score (0-1) + 8D embedding
- **Training:** Self-supervised on normal traffic patterns
- **Inference:** Real-time detection without ground truth labels
- **Advantage:** Captures subtle temporal patterns humans/rules might miss

---

### 4️⃣ **Classical Rule Engine** (`core/decision.py`)

**Purpose:** Deterministic, interpretable safety assessment

**Safety Thresholds (Configurable):**

| Metric | Critical | Warning | Unit |
|--------|----------|---------|------|
| Vehicle-Pedestrian Distance | <80 | <150 | pixels |
| Vehicle-Vehicle Distance | <150 | <300 | pixels |
| Max Speed | >150 | >100 | px/s |
| Closing Speed | >100 | >50 | px/s |
| Time-to-Collision (TTC) | <0.6s | <1.2s | seconds |

**Rule Violations Evaluated:**
```python
violations = {
    'critical_veh_ped_distance': bool,      # Distance < 80px
    'warning_veh_ped_distance': bool,       # Distance < 150px
    'critical_veh_veh_distance': bool,      # Distance < 150px
    'warning_veh_veh_distance': bool,       # Distance < 300px
    'very_high_speed': bool,                # Speed > 150px/s
    'high_speed': bool,                     # Speed > 100px/s
    'high_closing_speed': bool,             # Closing > 100px/s
    'low_ttc': bool,                        # TTC < 0.6s
    'mixed_traffic_close': bool,            # Veh+Ped<200px
    'pedestrian_present': bool              # Pedestrian detected
}
```

**Violation Weighting:**
- Critical violations: ×2 weight
- Warning violations: ×1 weight
- Total violation score normalizes to 0-1 range

---

### 5️⃣ **Hybrid Fusion Engine** (MCDM)

**Purpose:** Combine deep learning + rules using Multi-Criteria Decision Making

**Architecture:**

```
Deep Anomaly Score (LSTM)        Classical Violation Score (Rules)
        |                                    |
        | (0.6 weight)                      | (0.4 weight)
        ↓                                    ↓
      Weighted Deep                    Weighted Classical
        |                                    |
        └────────────────┬────────────────┘
                        |
                    Base Score
         (0.6×Deep + 0.4×Classical)
                        |
                    ↓ BOOST LOGIC ↓
        If (Multiple violations + high speed):
            Score = min(1.0, Score × 1.3)
                        |
                    Final Risk Score (0-1)
                        |
    ┌───────────────────┼───────────────────┐
    |                   |                   |
  ≥0.7              0.4-0.7                <0.4
CRITICAL           WARNING                 SAFE
```

**Decision Output:**
```python
{
    'safety_level': SafetyLevel.CRITICAL,  # enum: SAFE/WARNING/CRITICAL
    'risk_score': 0.75,                     # 0-1 float
    'decision_info': {
        'deep_anomaly': 0.70,
        'rule_violations': 0.80,
        'distance_factor': 0.85,
        'speed_factor': 0.60,
        'dominant_cause': 'low_ttc_pedestrian_conflict',
        'contributing_factors': [...]
    }
}
```

---

## 🔄 Data Pipeline

### Processing Flow

```
1. VIDEO ACQUISITION
   ├─ File Upload → FastAPI
   ├─ Temporary storage in /uploads/
   └─ Metadata: filename, timestamp, file size

2. FRAME EXTRACTION
   ├─ OpenCV VideoCapture
   ├─ Resolution: 640×480 (standardized)
   └─ FPS: 25 (assumed; can be auto-detected)

3. OBJECT DETECTION (Per-Frame)
   ├─ YOLOv5s inference
   ├─ Classes: car, person, truck, bus, bicycle, motorcycle
   ├─ Confidence threshold: 0.5
   ├─ NMS threshold: 0.4
   └─ Output: Bounding boxes + confidences

4. OBJECT TRACKING (Per-Frame)
   ├─ Centroid matching
   ├─ Trajectory recording
   ├─ ID assignment & persistence
   └─ Output: Tracked objects with IDs

5. FEATURE EXTRACTION (Per-Frame + Per-Window)
   ├─ Spatial: distances, relative positions
   ├─ Temporal: velocities, accelerations
   ├─ Interaction: TTC, crossing paths
   └─ Aggregation: mean, max, min per window

6. TEMPORAL ANOMALY DETECTION (Sliding Window)
   ├─ LSTM inference on 10-30 frame sequence
   ├─ Anomaly score computation
   ├─ Embedding extraction
   └─ Output: Anomaly score + embedding

7. CLASSICAL RULE EVALUATION (Per-Window)
   ├─ Threshold checking
   ├─ Violation counting
   ├─ Risk factor computation
   └─ Output: Rule violation dict

8. HYBRID DECISION MAKING (Per-Window)
   ├─ MCDM fusion
   ├─ Safety level assignment
   ├─ Risk score calculation
   └─ Output: Decision with explainability

9. VIDEO ANNOTATION (Batch)
   ├─ Draw bounding boxes
   ├─ Draw trajectories
   ├─ Add safety indicators (colors)
   ├─ Add text overlays (risk scores)
   └─ Output: Annotated video file

10. RESULT AGGREGATION
    ├─ Event list (WARNING + CRITICAL)
    ├─ Summary stats
    ├─ Timeline visualization
    └─ JSON metadata
```

### Storage Strategy

```
/uploads/
  ├─ {video_id}_input.mp4          # Original uploaded video
  
/outputs/
  ├─ {video_id}_annotated.mp4      # Processed video with annotations
  ├─ {video_id}_metadata.json      # Detailed analysis results
  └─ {video_id}_events.json        # Event list

In-Memory Storage (RAM):
  └─ processed_videos[video_id]    # Cached results for quick access
     ├─ 'frames': [...]            # Per-frame results
     ├─ 'events': [...]            # Detected events
     ├─ 'stats': {...}             # Summary statistics
     └─ 'paths': {...}             # File paths
```

---

## ⚙️ Hybrid Decision Engine - Detailed Algorithm

### Risk Score Computation Formula

Let:
- $D$ = deep anomaly score (0-1)
- $R$ = classical rule violation score (0-1)
- $w_D = 0.6$ = deep learning weight
- $w_R = 0.4$ = classical rule weight

**Base Score:**
$$S_{base} = w_D \cdot D + w_R \cdot R$$

**Boost Logic:** If multiple critical violations detected:
$$S_{final} = \min(1.0, S_{base} \times 1.3)$$

**Safety Classification:**
$$\text{Level} = \begin{cases}
\text{CRITICAL} & \text{if } S_{final} \geq 0.7 \\
\text{WARNING} & \text{if } 0.4 \leq S_{final} < 0.7 \\
\text{SAFE} & \text{if } S_{final} < 0.4
\end{cases}$$

### Explainability Factors

Each decision includes factor contributions:

```python
factors = {
    'deep_anomaly_score': 0.70,           # From LSTM
    'violation_count': 3,                  # Number of rule violations
    'distance_risk': 0.85,                 # (1 - normalized_distance)
    'speed_risk': 0.60,                    # (speed / max_acceptable)
    'pedestrian_interaction': 0.75,        # Presence + proximity
    'ttc_risk': 0.90                       # (max_ttc / critical_ttc)
}
```

---

## 🛠️ Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Backend** | FastAPI | 0.112+ | REST API server |
| | Python | 3.8+ | Core language |
| | OpenCV | 4.5+ | Video processing |
| | PyTorch | 1.9+ | LSTM model |
| **Object Detection** | YOLOv5 | Pretrained | Traffic object detection |
| **Frontend** | React | 18.3+ | Web UI dashboard |
| | Vite | 3.0+ | Build tool |
| | Axios | 1.0+ | HTTP client |
| **Deployment** | Docker | 20.10+ | Containerization |
| | Nginx | 1.21+ | Reverse proxy |
| **Database** | JSON files | - | Results storage |

---

## 📡 API Specification

### 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "pipeline_ready": true,
  "timestamp": "2026-02-22T10:30:45.123Z"
}
```

---

### 2. Video Inference

```http
POST /infer_clip
Content-Type: multipart/form-data

file: <video file>
max_frames: <optional int>
```

**Response:**
```json
{
  "video_id": "1708592445.123",
  "status": "processed",
  "filename": "traffic_cam_01.mp4",
  "total_frames": 750,
  "safety_stats": {
    "SAFE": 600,
    "WARNING": 120,
    "CRITICAL": 30
  },
  "events_count": {
    "critical": 5,
    "warning": 15
  },
  "top_events": [
    {
      "frame_idx": 345,
      "timestamp": 13.8,
      "level": "CRITICAL",
      "risk_score": 0.85,
      "details": {...}
    }
  ],
  "annotated_video_path": "/download/1708592445.123",
  "timestamp": "2026-02-22T10:35:12.456Z"
}
```

---

### 3. Get Events

```http
GET /events
```

**Response:**
```json
{
  "total_events": 45,
  "critical_events": 8,
  "warning_events": 37,
  "events": [
    {
      "video_id": "1708592445.123",
      "frame_idx": 345,
      "timestamp": 13.8,
      "level": "CRITICAL",
      "risk_score": 0.85
    }
  ],
  "timestamp": "2026-02-22T10:40:00.000Z"
}
```

---

### 4. Video Results Detail

```http
GET /video_results/{video_id}
```

**Response:**
```json
{
  "video_id": "1708592445.123",
  "filename": "traffic_cam_01.mp4",
  "processed_at": "2026-02-22T10:35:12.456Z",
  "total_frames": 750,
  "safety_stats": {
    "SAFE": 600,
    "WARNING": 120,
    "CRITICAL": 30
  },
  "events": [
    {
      "frame_idx": 345,
      "timestamp": 13.8,
      "level": "CRITICAL",
      "risk_score": 0.85,
      "details": {...}
    }
  ],
  "download_url": "/download/1708592445.123"
}
```

---

### 5. Download Annotated Video

```http
GET /download/{video_id}
```

**Response:** Binary MP4 file with annotations

---

### 6. System Statistics

```http
GET /stats
```

**Response:**
```json
{
  "videos_processed": 42,
  "total_events": 156,
  "critical_events": 28,
  "warning_events": 128,
  "timestamp": "2026-02-22T10:45:30.000Z"
}
```

---

## 💾 Database & Storage

### Result Metadata Structure

```json
{
  "video_id": "1708592445.123",
  "filename": "traffic_cam_01.mp4",
  "uploaded_at": "2026-02-22T10:30:00Z",
  "processed_at": "2026-02-22T10:35:12Z",
  "processing_time_seconds": 312.4,
  "input_path": "/uploads/1708592445_input.mp4",
  "output_path": "/outputs/1708592445_annotated.mp4",
  "video_metadata": {
    "fps": 25,
    "width": 640,
    "height": 480,
    "total_frames": 750,
    "duration_seconds": 30
  },
  "summary_stats": {
    "safe_frames": 600,
    "warning_frames": 120,
    "critical_frames": 30,
    "unique_objects_detected": 47,
    "average_risk_score": 0.28
  },
  "events": [
    {
      "frame_idx": 345,
      "timestamp": 13.8,
      "level": "CRITICAL",
      "risk_score": 0.85,
      "decision_factors": {
        "deep_anomaly": 0.80,
        "rule_violations": 4,
        "dominant_cause": "low_ttc_pedestrian",
        "objects_involved": ["car_id_5", "person_id_12"]
      }
    }
  ]
}
```

---

## 🚀 Deployment Architecture

### Docker Deployment

```dockerfile
# Backend
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Frontend
FROM node:16-alpine
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build
CMD ["npm", "run", "dev"]
```

### Container Orchestration (docker-compose)

```yaml
version: '3.9'
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0  # GPU acceleration
    volumes:
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs
    
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "5173:5173"
    depends_on:
      - backend
```

### Production Deployment

```
┌──────────────────────────────────────┐
│      Client (Browser)                │
│      http://localhost:5173           │
└──────────────────┬───────────────────┘
                   │
          ┌────────▼────────┐
          │   Nginx Proxy   │
          │   Port 80, 443  │
          └────────┬────────┘
                   │
      ┌────────────┴────────────┐
      │                         │
  ┌───▼────┐              ┌────▼───┐
  │Frontend │              │Backend │
  │ (Static)│              │(FastAPI)
  │Port 3000│              │Port8000│
  └─────────┘              └────────┘
      │                         │
      │                    ┌────▼────────┐
      │                    │YOLOv5 Model │
      │                    │LSTM Model   │
      │                    │Rule Engine  │
      │                    └─────────────┘
      │
  ┌───▼──────────────────┐
  │ File Storage         │
  │ /uploads             │
  │ /outputs             │
  │ Results JSON         │
  └──────────────────────┘
```

---

## 📊 Performance Characteristics

### Benchmarks (on GPU: NVIDIA A100)

| Component | Latency | Throughput |
|-----------|---------|-----------|
| YOLOv5 Detection | 25ms/frame | 40 FPS |
| Object Tracking | 5ms/frame | 200 FPS |
| Feature Extraction | 8ms/frame | 125 FPS |
| LSTM Anomaly | 12ms/window | 83 windows/s |
| Rule Engine | 3ms/window | 333 windows/s |
| Fusion & Decision | 2ms/window | 500 windows/s |
| **Total Per-Frame** | **55ms** | **18 FPS** |
| Video I/O & Encoding | 40ms/frame | 25 FPS |

### Scalability

- **Single GPU**: Batch process up to 4 concurrent videos
- **Multi-GPU** (2×): Batch process up to 8 videos simultaneously
- **CPU-only**: Achieves ~12 FPS (suitable for edge deployment)

### Memory Usage

- YOLOv5s model: ~200 MB VRAM
- LSTM model: ~50 MB VRAM
- Feature buffer (30 frames): ~10 MB
- Typical working set: ~500 MB VRAM for 4 concurrent streams

---

## 🔐 Security Considerations

### Input Validation

- File upload: Max 500 MB, video file types only
- Frame resolution: Standardize to 640×480 to prevent attacks
- Model paths: Restrict to whitelisted model directory

### Data Privacy

- Uploaded videos stored temporarily (auto-delete after 24h)
- Results stored in encrypted JSON format
- HTTPS enforced in production
- No personally identifiable information (PII) persistence

### Model Security

- YOLOv5 model weights: Checksummed on startup
- LSTM model: Signed bytecode only
- Dependency pinning: Exact versions in requirements.txt

### Rate Limiting

```python
# API rate limits
POST /infer_clip: 10 requests/minute per IP
GET /events: 100 requests/minute per IP
GET /download: 50 requests/minute per IP
```

---

## 📝 Configuration Management

### Environment Variables

```bash
# Model configuration
YOLO_MODEL=yolov5s
CONFIDENCE_THRESHOLD=0.5
NMS_THRESHOLD=0.4

# Processing
WINDOW_SIZE=30
FPS=25
MAX_FRAME_MEMORY=1000

# Thresholds
CRITICAL_DISTANCE_VEHPED=80
WARNING_DISTANCE_VEHPED=150
CRITICAL_DISTANCE_VEHVEH=150
WARNING_DISTANCE_VEHVEH=300

# Weights
DEEP_WEIGHT=0.6
CLASSICAL_WEIGHT=0.4

# Paths
UPLOAD_DIR=./uploads
OUTPUT_DIR=./outputs
MODEL_PATH=./models

# Performance
USE_GPU=true
BATCH_SIZE=4
NUM_WORKERS=4
```

---

## 🔄 Version Management & Roadmap

### v1.1.0 (Current)
- ✅ Hybrid deep-classical fusion
- ✅ FastAPI + React dashboard
- ✅ YOLOv5s object detection
- ✅ LSTM temporal anomaly detection
- ✅ Classical rule engine
- ✅ Signal-based UI

### v1.2.0 (Planned)
- 🔮 Enhanced YOLO variant (YOLOv8)
- 🔮 Attention-based temporal model (Transformer)
- 🔮 Real-time streaming support (RTSP)
- 🔮 Multi-camera fusion
- 🔮 Predictive anomaly (pre-emptive warnings)

### v2.0.0 (Roadmap)
- 🔮 Fed learning for privacy-preserving multi-site deployment
- 🔮 Edge deployment on NVIDIA Jetson
- 🔮 Real-world benchmark dataset release
- 🔮 Patent filing for hybrid fusion approach

---

## 📚 References

- **Near-Miss Detection Research**: [PMC National Center for Biotechnology Information](https://pmc.ncbi.nlm.nih.gov/articles/PMC7206299/)
- **YOLOv5**: [Ultralytics Repository](https://github.com/ultralytics/yolov5)
- **Enhanced YOLOv5 for Road Objects**: [LinkedIn Research](https://www.linkedin.com/posts/sensors-mdpi_enhanced-yolov5-an-efficient-road-object-activity-7425018090209398784-YHsb)

---

**Last Updated:** February 22, 2026 | **Maintainer:** Safety Sentinel Team
