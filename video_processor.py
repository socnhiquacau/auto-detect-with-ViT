import cv2
import numpy as np
from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import requests
from typing import List, Dict, Tuple
import os
import uuid
from datetime import datetime
from image_enhancement import ImageEnhancer
from feature_extractor import FeatureExtractor

class VideoProcessor:
    def __init__(self, db, output_dir: str, temp_dir: str, yolo_path: str = None, vit_path: str = None):
        self.db = db
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        
        # Initialize YOLO model for person detection
        yolo_model_path = yolo_path or os.getenv('YOLO_MODEL_PATH', 'models/yolov8_person_detection.pt')
        print(f"🎯 Loading YOLO model from: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        print("✅ YOLO model loaded")
        
        # Initialize ViT feature extractor
        vit_model_path = vit_path or os.getenv('VIT_MODEL_PATH', 'models/dino_vits16_epoch100.pth')
        self.feature_extractor = FeatureExtractor(vit_model_path)
        print("✅ VIT model loaded")

        # Initialize image enhancer
        self.image_enhancer = ImageEnhancer()
        
        # Detection confidence threshold
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', 0.75))
        
        # Similarity thresholds
        self.known_person_threshold = float(os.getenv('KNOWN_PERSON_THRESHOLD', 0.95))
        
    async def process_video_from_url(self, url: str, video_name: str) -> Dict:
        """Download and process video from URL"""
        video_path = os.path.join(self.temp_dir, f"{video_name}.mp4")
        
        # Download video
        response = requests.get(url, stream=True)
        with open(video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return await self.process_video_file(video_path, video_name)
    
    async def process_video_file(self, video_path: str, video_name: str) -> Dict:
        """Main video processing pipeline"""
        job_id = str(uuid.uuid4())  # Tu tao job ID
        video_id = str(uuid.uuid4()) #  Tu tao video id
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame extraction rate (2 FPS)
        frame_interval = int(fps / 2) if fps > 2 else 1
        
        detections = []
        frame_count = 0
        extracted_count = 0
        
        print(f"🎬 Processing video: {video_name}")
        print(f"   FPS: {fps}, Total frames: {total_frames}")
        print(f"   Extracting every {frame_interval} frames (2 FPS)")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frames at 2 FPS
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                
                # Step 3: Enhance image quality
                enhanced_frame = self.image_enhancer.enhance(frame)
                
                # Step 4: Detect persons with YOLO
                results = self.yolo_model(enhanced_frame, conf=self.confidence_threshold)
                
                # Step 5: Process detections
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Filter only 'person' class (class 0 in COCO)
                        if int(box.cls[0]) == 0:
                            detection = await self._process_detection(
                                enhanced_frame,
                                box,
                                video_id,
                                video_name,
                                timestamp,
                                frame_count,
                                extracted_count
                            )
                            detections.append(detection)
                            extracted_count += 1
                
                if extracted_count % 10 == 0:
                    print(f"   Processed {extracted_count} detections...")
            
            frame_count += 1
        
        cap.release()
        
        # Save processing results
        result = {
            "job_id": job_id,
            "video_id": video_id,
            "video_name": video_name,
            "processed_at": datetime.now().isoformat(),
            "total_frames": total_frames,
            "processed_frames": frame_count,
            "extracted_frames": extracted_count,
            "total_detections": len(detections),
            "known_persons": sum(1 for d in detections if d.get("is_known")),
            "unknown_persons": sum(1 for d in detections if not d.get("is_known")),
            "detections": detections
        }
        
        await self.db.save_processing_result(result)
        
        print(f"✅ Processing complete: {len(detections)} detections")
        
        return result
    
    async def _process_detection(
        self,
        frame: np.ndarray,
        box,
        video_id: str,
        video_name: str,
        timestamp: float,
        frame_number: int,
        detection_index: int
    ) -> Dict:
        """Process a single person detection"""
        
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        
        # Crop person from frame
        person_img = frame[y1:y2, x1:x2]
        
        # Step 6: Extract feature vector using ViT
        feature_vector = self.feature_extractor.extract(person_img)
        d = feature_vector.shape
        
        # Step 7: Compare with known persons
        match_result = await self._find_matching_person(feature_vector)
        
        # Save detected frame
        detection_id = str(uuid.uuid4())
        frame_filename = f"{video_id}_{detection_index}_{detection_id}.jpg"
        frame_path = os.path.join(self.output_dir, frame_filename)
        cv2.imwrite(frame_path, person_img)
        
        # Prepare detection result
        detection = {
            "detection_id": detection_id,
            "video_id": video_id,
            "video_name": video_name,
            "frame_number": frame_number,
            "timestamp": timestamp,
            "bounding_box": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1
            },
            "confidence": confidence,
            "feature_vector": feature_vector.tolist(),
            "saved_frame_path": frame_path,
            "is_known": match_result["is_known"],
            "matched_person_id": match_result.get("person_id"),
            "matched_person_name": match_result.get("person_name"),
            "similarity_score": match_result.get("similarity_score"),
            "detected_at": datetime.now().isoformat()
        }
        
        # Step 8: Save to MongoDB
        await self.db.save_detection(detection)
        
        return detection
    
    async def _find_matching_person(self, feature_vector: np.ndarray) -> Dict:
        """Compare feature vector with known persons in database"""
        known_persons = await self.db.get_all_known_persons()
        
        if not known_persons:
            return {"is_known": False}
        
        best_match = None
        best_similarity = -1
        
        for person in known_persons:
            stored_vector = np.array(person["feature_vector"])
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(feature_vector, stored_vector)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person
        
        # Check if similarity exceeds threshold
        if best_similarity >= self.known_person_threshold and best_match is not None:
            return {
                "is_known": True,
                "person_id": best_match["person_id"],
                "person_name": best_match.get("name", "Unknown"),
                "similarity_score": float(best_similarity)
            }
        
        return {"is_known": False}
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        vec1_normalized = vec1 / norm1
        vec2_normalized = vec2 / norm2
        
        # Calculate cosine similarity
        similarity = np.dot(vec1_normalized, vec2_normalized)
        return float(similarity)
    
    @staticmethod
    def _euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Euclidean distance between two vectors"""
        return np.linalg.norm(vec1 - vec2)