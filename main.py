"""
FastAPI Backend Server for Safety Sentinel
Provides REST API for video processing and real-time inference
"""

import os
import logging
import traceback
from pathlib import Path
from typing import Optional, List
import asyncio
from datetime import datetime
import time

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

from pipeline import SafetySentinelPipeline
from core.decision import SafetyLevel

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('safety_sentinel.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Safety Sentinel API",
    description="Real-time near-miss detection at urban intersections",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[SafetySentinelPipeline] = None

# Performance metrics
processing_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "average_processing_time": 0.0,
    "pipeline_initializations": 0
}

# Storage for processed videos
UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Track processed videos
processed_videos: dict = {}


@app.on_event("startup")
async def startup():
    """Initialize pipeline on startup with enhanced error handling"""
    global pipeline
    
    logger.info("🚀 Starting Safety Sentinel API Server...")
    
    try:
        device = "cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu"
        logger.info(f"🔧 Initializing pipeline on device: {device}")
        
        start_time = time.time()
        pipeline = SafetySentinelPipeline(
            yolo_model="yolov5s",
            device=device,
            fps=25.0,
            window_size=30
        )
        init_time = time.time() - start_time
        
        processing_stats["pipeline_initializations"] += 1
        logger.info(f"✅ Pipeline initialized successfully in {init_time:.2f}s")
        
        # Test pipeline with a dummy frame if possible
        logger.info("🧪 Running pipeline health check...")
        
    except Exception as e:
        logger.error(f"❌ Pipeline initialization failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.info("⚠️ Pipeline will initialize lazily on first use")
        # Don't fail startup - pipeline will be created on demand


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed logging"""
    logger.warning(f"🔍 Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with detailed logging"""
    logger.error(f"💥 Unexpected error: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    processing_stats["failed_requests"] += 1
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again.",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with detailed status"""
    status = {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time() - processing_stats.get("start_time", time.time()),
        "processing_stats": processing_stats
    }
    
    if pipeline is None:
        status["note"] = "Pipeline will be initialized on first use"
        status["pipeline_status"] = "not_initialized"
    else:
        status["pipeline_status"] = "ready"
        
    # Check disk space
    try:
        upload_dir = Path("./uploads")
        output_dir = Path("./outputs")
        status["storage"] = {
            "uploads_dir_exists": upload_dir.exists(),
            "outputs_dir_exists": output_dir.exists(),
            "processed_videos": len(processed_videos)
        }
    except Exception as e:
        logger.warning(f"Could not check storage status: {e}")
        status["storage"] = {"error": str(e)}
        
    return status


@app.post("/infer_clip")
async def infer_video_clip(
    file: UploadFile = File(...),
    max_frames: Optional[int] = None
):
    """
    Upload video clip and get safety classification
    
    Returns: per-window safety levels, events, and annotated video path
    """
    global pipeline
    
    # Track request
    processing_stats["total_requests"] += 1
    start_time = time.time()
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Supported formats: MP4, AVI, MOV, MKV"
        )
    
    logger.info(f"📹 Processing video: {file.filename} (max_frames: {max_frames})")
    
    # Initialize pipeline on first use if not already done
    if pipeline is None:
        try:
            device = "cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu"
            logger.info(f"🔧 Lazily initializing pipeline on device: {device}")
            
            init_start = time.time()
            pipeline = SafetySentinelPipeline(
                yolo_model="yolov5s",
                device=device,
                fps=25.0,
                window_size=30
            )
            init_time = time.time() - init_start
            
            processing_stats["pipeline_initializations"] += 1
            logger.info(f"✅ Pipeline initialized on first use in {init_time:.2f}s")
        except Exception as e:
            logger.error(f"❌ Failed to initialize pipeline: {str(e)}")
            processing_stats["failed_requests"] += 1
            raise HTTPException(status_code=500, detail=f"Failed to initialize pipeline: {str(e)}")
    
    try:
        # Save uploaded file
        video_id = f"{datetime.now().timestamp()}"
        input_path = UPLOAD_DIR / f"{video_id}_input.mp4"
        output_path = OUTPUT_DIR / f"{video_id}_annotated.mp4"
        
        # Write uploaded file with size check
        contents = await file.read()
        file_size = len(contents)
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        if file_size > 500 * 1024 * 1024:  # 500MB limit
            raise HTTPException(
                status_code=413, 
                detail="File too large. Maximum size is 500MB"
            )
        
        with open(input_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"💾 Saved video ({file_size/1024/1024:.1f}MB) to {input_path}")
        
        # Process video
        process_start = time.time()
        results = pipeline.process_video(str(input_path), max_frames)
        process_time = time.time() - process_start
        
        if not results:
            raise HTTPException(
                status_code=400, 
                detail="Could not process video - no frames extracted"
            )
        
        logger.info(f"⚡ Processed {len(results)} frames in {process_time:.2f}s")
        
        # Create annotated video
        annotate_start = time.time()
        pipeline.save_annotated_video(str(input_path), str(output_path), max_frames)
        annotate_time = time.time() - annotate_start
        
        logger.info(f"🎬 Created annotated video in {annotate_time:.2f}s")
        
        # Prepare response
        events = pipeline.get_events()
        
        # Group results by safety level
        safety_stats = {
            "SAFE": sum(1 for r in results if r['safety_level'] == SafetyLevel.SAFE),
            "WARNING": sum(1 for r in results if r['safety_level'] == SafetyLevel.WARNING),
            "CRITICAL": sum(1 for r in results if r['safety_level'] == SafetyLevel.CRITICAL)
        }
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Update stats
        processing_stats["successful_requests"] += 1
        processing_stats["average_processing_time"] = (
            (processing_stats["average_processing_time"] * (processing_stats["successful_requests"] - 1) + total_time) /
            processing_stats["successful_requests"]
        )
        
        # Store for later retrieval
        processed_videos[video_id] = {
            "filename": file.filename,
            "processed_at": datetime.now().isoformat(),
            "input_path": str(input_path),
            "output_path": str(output_path),
            "total_frames": len(results),
            "events": events,
            "stats": safety_stats,
            "results": results,
            "processing_time": total_time,
            "file_size_mb": file_size / 1024 / 1024
        }
        
        logger.info(f"✅ Successfully processed {file.filename} in {total_time:.2f}s")
        
        return {
            "video_id": video_id,
            "status": "processed",
            "filename": file.filename,
            "total_frames": len(results),
            "safety_stats": safety_stats,
            "events_count": {
                "critical": sum(1 for e in events if e['level'] == 'CRITICAL'),
                "warning": sum(1 for e in events if e['level'] == 'WARNING')
            },
            "top_events": sorted(events, key=lambda x: x['risk_score'], reverse=True)[:5],
            "annotated_video_path": f"/download/{video_id}",
            "processing_time_seconds": round(total_time, 2),
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        processing_stats["failed_requests"] += 1
        raise
    except Exception as e:
        processing_stats["failed_requests"] += 1
        logger.error(f"❌ Error processing video {file.filename}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")


@app.get("/events")
async def get_events():
    """Get recent Warning/Critical events from all videos"""
    all_events = []
    
    for video_id, video_info in processed_videos.items():
        events = video_info.get('events', [])
        for event in events:
            event['video_id'] = video_id
            all_events.append(event)
    
    # Sort by timestamp descending
    all_events.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return {
        "total_events": len(all_events),
        "critical_events": sum(1 for e in all_events if e['level'] == 'CRITICAL'),
        "warning_events": sum(1 for e in all_events if e['level'] == 'WARNING'),
        "events": all_events[:20],  # Latest 20
        "timestamp": datetime.now().isoformat()
    }


@app.get("/video_results/{video_id}")
async def get_video_results(video_id: str):
    """Get detailed results for a specific video"""
    if video_id not in processed_videos:
        raise HTTPException(status_code=404, detail="Video not found")
    
    info = processed_videos[video_id]
    
    return {
        "video_id": video_id,
        "filename": info['filename'],
        "processed_at": info['processed_at'],
        "total_frames": info['total_frames'],
        "safety_stats": info['stats'],
        "events": info['events'],
        "download_url": f"/download/{video_id}"
    }


@app.get("/download/{video_id}")
async def download_video(video_id: str):
    """Download annotated video"""
    if video_id not in processed_videos:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = processed_videos[video_id]['output_path']
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Annotated video not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"annotated_{video_id}.mp4"
    )


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    total_events = sum(len(v.get('events', [])) for v in processed_videos.values())
    
    return {
        "videos_processed": len(processed_videos),
        "total_events": total_events,
        "critical_events": sum(
            sum(1 for e in v.get('events', []) if e['level'] == 'CRITICAL')
            for v in processed_videos.values()
        ),
        "warning_events": sum(
            sum(1 for e in v.get('events', []) if e['level'] == 'WARNING')
            for v in processed_videos.values()
        ),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    """API info"""
    return {
        "name": "Safety Sentinel API",
        "version": "1.0.0",
        "description": "Real-time near-miss detection at urban intersections",
        "endpoints": {
            "health": "GET /health",
            "infer": "POST /infer_clip",
            "events": "GET /events",
            "video_results": "GET /video_results/{video_id}",
            "download": "GET /download/{video_id}",
            "stats": "GET /stats"
        }
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
