from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class VideoUpload(BaseModel):
    filename: str
    content_type: Optional[str] = None

class DetectionResult(BaseModel):
    timestamp: float
    bbox: List[int]
    confidence: float
    class_id: Optional[int] = None
    class_name: Optional[str] = None

class ProcessingStatus(BaseModel):
    job_id: str
    video_id: str
    status: str
    progress: float
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None
