# database.py
from datetime import datetime, timezone
from typing import Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

class Database:
    """MongoDB database operations"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.detections = db.detections
        self.known_persons = db.known_persons
        self.processing_results = db.processing_results
        
    async def close(self):
        """Close database connection"""
        if self.db.client:
            self.db.client.close()
    
    async def save_detection(self, detection: Dict) -> str:
        """Save a single detection to database"""
        result = await self.detections.insert_one(detection)
        return str(result.inserted_id)
    
    async def get_detection(self, detection_id: str) -> Optional[Dict]:
        """Get a single detection by ID"""
        return await self.detections.find_one({"detection_id": detection_id})
    
    async def get_detections_by_video(self, video_id: str, skip: int = 0, limit: int = 100) -> List[Dict]:
        """Get all detections for a specific video"""
        cursor = self.detections.find({"video_id": video_id}).skip(skip).limit(limit)
        detections = await cursor.to_list(length=limit)
        return detections
    
    async def delete_detections_by_video(self, video_id: str) -> int:
        """Delete all detections for a video"""
        result = await self.detections.delete_many({"video_id": video_id})
        return result.deleted_count
    
    async def add_known_person(
        self,
        person_id: str,
        feature_vector: List[float],
        name: Optional[str] = None,
    ) -> str:
        """Add a feature vector to a known person, creating the person if needed."""
        now = datetime.now(timezone.utc).isoformat()
        person = {
            "person_id": person_id,
            "name": name or "Unknown",
            "updated_at": now,
        }

        await self.known_persons.update_one(
            {"person_id": person_id},
            {
                "$set": person,
                "$setOnInsert": {"added_at": now},
                "$addToSet": {"feature_vectors": feature_vector},
            },
            upsert=True,
        )
        return person_id
    
    async def get_known_person(self, person_id: str) -> Optional[Dict]:
        """Get a known person by ID"""
        return await self.known_persons.find_one({"person_id": person_id})
    
    async def get_all_known_persons(self) -> List[Dict]:
        """Get all known persons"""
        cursor = self.known_persons.find({})
        persons = await cursor.to_list(length=1000)
        return persons
    
    async def save_processing_result(self, result: Dict) -> str:
        """Save video processing result"""
        doc = await self.processing_results.insert_one(result)
        return str(doc.inserted_id)
    
    async def get_processing_results(self, job_id: str) -> Optional[Dict]:
        """Get processing results by job ID"""
        return await self.processing_results.find_one({"job_id": job_id})
    
    async def create_indexes(self):
        """Create database indexes for better query performance"""
        try:
            await self.detections.create_index("detection_id", unique=True)
            await self.detections.create_index("video_id")
            await self.detections.create_index("is_known")
            await self.detections.create_index("matched_person_id")
            await self.detections.create_index([("video_id", 1), ("timestamp", 1)])
            
            await self.known_persons.create_index("person_id", unique=True)
            
            await self.processing_results.create_index("job_id", unique=True)
            await self.processing_results.create_index("video_id")
            
            print("✅ Database indexes created")
        except Exception as e:
            print(f"⚠️  Index creation warning: {e}")
