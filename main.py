import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

from database import Database
# Import processing modules
from video_processor import VideoProcessor

# Load environment variables from .env file
load_dotenv()

# =========================
# CONFIGURATION FROM ENVIRONMENT VARIABLES
# =========================
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:admin123%20@localhost:27017/video_detection_db?authSource=admin")
DATABASE_NAME = os.getenv("DATABASE_NAME", "video_detection_db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/detected")
TEMP_DIR = os.getenv("TEMP_DIR", "temp")

# Ensure directories exist
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Global instances
db = None
video_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # =========================
    # STARTUP PHASE
    # =========================
    global db, video_processor

    # Initialize MongoDB connection
    client = AsyncIOMotorClient(MONGODB_URL)
    db = Database(client[DATABASE_NAME])

    # Initialize VideoProcessor with unified DataLoader for models
    # This ensures all models are loaded from the models/ directory
    video_processor = VideoProcessor(db, OUTPUT_DIR, TEMP_DIR)

    # Create database indexes for better query performance
    try:
        await db.create_indexes()
        print("✅ Database indexes created")
    except Exception as e:
        print(f"⚠️  Warning: Could not create indexes: {e}")

    print("✅ Connected to MongoDB and initialized processors")

    yield

    # =========================
    # SHUTDOWN PHASE
    # =========================
    if db:
        await db.close()
    print("👋 Shutting down...")


app = FastAPI(
    title="Video Person Detection & Recognition API",
    lifespan=lifespan
)


# =========================
# PYDANTIC MODELS (Request/Response Schemas)
# =========================

class VideoURLRequest(BaseModel):
    """Request model for processing video from URL"""
    url: HttpUrl
    video_name: Optional[str] = None


class FeatureVectorRequest(BaseModel):
    """Request model for adding known person to database"""
    person_id: str
    feature_vector: List[float]
    name: Optional[str] = None


@app.get("/")
async def root():
    return {
        "message": "Video Person Detection & Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "POST /process/upload": "Upload and process video file",
            "POST /process/url": "Process video from URL",
            "GET /results/{job_id}": "Get processing results",
            "GET /detections/{video_id}": "Get all detections for a video",
            "POST /features/add": "Add known person feature vector",
            "GET /features/list": "List all known persons"
        }
    }


@app.post("/process/upload")
async def process_uploaded_video():
    """Upload and process video file"""
    try:
        # Use hardcoded file path
        video_path = "temp/bali.mp4"

        # Process video
        result = await video_processor.process_video_file(video_path, "bali.mp4")

        return JSONResponse({
            "status": "success",
            "job_id": result["job_id"],
            "video_id": result["video_id"],
            "message": "Video processing completed",
            "results": result
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/process/url")
async def process_video_from_url(request: VideoURLRequest):
    """Process video from URL"""
    try:
        video_name = request.video_name or f"video_{datetime.now().timestamp()}"
        result = await video_processor.process_video_from_url(str(request.url), video_name)

        return JSONResponse({
            "status": "success",
            "job_id": result["job_id"],
            "video_id": result["video_id"],
            "message": "Video processing completed",
            "results": result
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get processing results by job ID"""
    results = await db.get_processing_results(job_id)
    if not results:
        raise HTTPException(status_code=404, detail="Job not found")
    return results


@app.get("/detections/{video_id}")
async def get_detections(video_id: str, skip: int = 0, limit: int = 100):
    """Get all detections for a specific video"""
    detections = await db.get_detections_by_video(video_id, skip, limit)
    return {
        "video_id": video_id,
        "total": len(detections),
        "detections": detections
    }


@app.post("/features/add")
async def add_known_person(request: FeatureVectorRequest):
    """Add a known person's feature vector to database"""
    try:
        result = await db.add_known_person(
            request.person_id,
            request.feature_vector,
            request.name
        )
        return {
            "status": "success",
            "message": "Known person added successfully",
            "person_id": request.person_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add person: {str(e)}")


@app.get("/features/list")
async def list_known_persons():
    """List all known persons in database"""
    persons = await db.get_all_known_persons()
    return {
        "total": len(persons),
        "persons": persons
    }


@app.delete("/detections/{video_id}")
async def delete_video_detections(video_id: str):
    """Delete all detections for a video"""
    deleted_count = await db.delete_detections_by_video(video_id)
    return {
        "status": "success",
        "deleted_count": deleted_count
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected" if db else "disconnected"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)