import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import requests
from typing import List, Dict, Tuple
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv

from image_enhancement import ImageEnhancer
from common import DataLoader, CONFIDENCE_THRESHOLD, KNOWN_PERSON_THRESHOLD, FRAME_EXTRACTION_FPS

# Load environment variables
load_dotenv()


class VideoProcessor:
    """
    Main video processing pipeline for person detection and recognition.

    Pipeline steps:
    1. Load video frames at specified FPS rate
    2. Enhance image quality
    3. Detect persons using YOLO model
    4. Extract feature vectors using ViT model
    5. Match features against known persons in database
    6. Save detection results with metadata
    """

    def __init__(self, db, output_dir: str, temp_dir: str, yolo_path: str = None, vit_path: str = None):
        """
        Initialize VideoProcessor with database connection and model paths.

        Args:
            db: Database instance for storing detection results
            output_dir: Directory for saving detected person crops
            temp_dir: Temporary directory for downloading/processing videos
            yolo_path: Optional path to YOLO model (overrides env var)
            vit_path: Optional path to ViT model (overrides env var)
        """
        self.db = db
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        
        # Use unified DataLoader for consistent model initialization from models/ directory
        self.data_loader = DataLoader(yolo_path=yolo_path, vit_path=vit_path)
        self.yolo_model = self.data_loader.get_yolo_model()
        self.feature_extractor = self.data_loader.get_feature_extractor()

        # Initialize image quality enhancement
        self.image_enhancer = ImageEnhancer()
        
        # Thresholds for detection and person matching (loaded from env or defaults)
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.known_person_threshold = KNOWN_PERSON_THRESHOLD
        self.frame_extraction_fps = FRAME_EXTRACTION_FPS

    async def process_video_from_url(self, url: str, video_name: str) -> Dict:
        """
        Download video from URL and process it through the pipeline.

        Args:
            url: URL to video file
            video_name: Name to identify the video in processing logs

        Returns:
            Dictionary with processing results and detection information
        """
        video_path = os.path.join(self.temp_dir, f"{video_name}.mp4")
        
        # Download video file in chunks to handle large files
        print(f"📥 Downloading video from URL: {url}")
        response = requests.get(url, stream=True)
        with open(video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Video downloaded to: {video_path}")

        return await self.process_video_file(video_path, video_name)
    
    async def process_video_file(self, video_path: str, video_name: str) -> Dict:
        """
        Main video processing pipeline:
        1. Extract frames at target FPS rate
        2. Enhance image quality
        3. Detect persons with YOLO
        4. Extract feature vectors with ViT
        5. Match against known persons
        6. Save detections to database

        Args:
            video_path: Path to video file
            video_name: Identifier for the video

        Returns:
            Dictionary with processing results containing:
            - job_id: unique job identifier
            - video_id: unique video identifier
            - total_detections: number of persons detected
            - known_persons: count of identified known persons
            - unknown_persons: count of unidentified persons
        """
        job_id = str(uuid.uuid4())  # Unique job identifier for tracking this processing
        video_id = str(uuid.uuid4())  # Unique video identifier for database storage

        # Open video file and extract metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame extraction interval: extract frames at self.frame_extraction_fps rate
        # Example: if video is 30fps and we want 2fps extraction, interval = 30/2 = 15
        frame_interval = int(fps / self.frame_extraction_fps) if fps > self.frame_extraction_fps else 1

        detections = []
        frame_count = 0
        extracted_count = 0
        
        print(f"🎬 Processing video: {video_name}")
        print(f"   FPS: {fps}, Total frames: {total_frames}")
        print(f"   Extracting every {frame_interval} frames (~{self.frame_extraction_fps} FPS)")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frames at target rate (2 FPS default)
            # This reduces computation while capturing enough motion information
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                
                # Step 1: Enhance image quality to improve detection accuracy
                # This may include contrast adjustment, noise reduction, etc.
                enhanced_frame = self.image_enhancer.enhance(frame)
                
                # Step 2: Detect all persons in frame using YOLO
                # Returns bounding boxes with confidence scores
                results = self.yolo_model(enhanced_frame, conf=self.confidence_threshold)
                
                # Step 3: Process each detected person
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Filter only 'person' class (class 0 in COCO dataset)
                        # COCO classes: 0=person, 1=bicycle, 2=car, etc.
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
                
                # Log progress every 10 detections
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
        """
        Process a single person detection:
        1. Extract bounding box coordinates
        2. Crop person image from frame
        3. Extract feature vector using ViT model
        4. Match features against known persons in database
        5. Save detection crop and metadata to database

        Args:
            frame: OpenCV BGR frame from video
            box: YOLO detection box object with coordinates and confidence
            video_id: Unique video identifier
            video_name: Human-readable video name
            timestamp: Timestamp in video when detection occurred
            frame_number: Frame number in original video
            detection_index: Index of this detection in extraction batch

        Returns:
            Dictionary with detection metadata and similarity scores
        """

        # Extract bounding box coordinates from YOLO detection
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        
        # Crop person region from frame for feature extraction
        person_img = frame[y1:y2, x1:x2]
        
        # Step 1: Extract feature vector using ViT model
        # This produces a normalized 384-dimensional feature vector (or based on model output)
        # representing the visual characteristics of the detected person
        feature_vector = self.feature_extractor.extract(person_img)

        # Step 2: Compare with known persons in database to find best match
        # Uses cosine similarity to measure distance between feature vectors
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
        """
        Find the best matching known person by comparing feature vectors.

        Algorithm:
        1. Retrieve all known persons from database
        2. For each known person, calculate cosine similarity between their stored feature vector
           and the detected person's feature vector
        3. Track the person with highest similarity score
        4. If best similarity exceeds threshold, return match; otherwise return unknown

        Cosine Similarity: measures the angle between two vectors. Values range from -1 to 1,
        where 1 means identical vectors, 0 means orthogonal, and -1 means opposite directions.
        For normalized feature vectors (L2-norm = 1), cosine similarity equals dot product.

        Args:
            feature_vector: Normalized feature vector from ViT model

        Returns:
            Dictionary with match results:
            - is_known: bool indicating if match was found above threshold
            - person_id, person_name, similarity_score: populated if is_known=True
        """
        known_persons = await self.db.get_all_known_persons()
        
        if not known_persons:
            return {"is_known": False}
        
        best_match = None
        best_similarity = -1
        
        # Compare against all known persons to find best match
        for person in known_persons:
            stored_vector = np.array(person["feature_vector"])
            
            # Calculate cosine similarity between detected and stored vectors
            similarity = self._cosine_similarity(feature_vector, stored_vector)
            
            # Update best match if this person has higher similarity
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person
        
        # Verify that best match exceeds the recognition threshold
        # This prevents false positives from low-quality detections
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
        """
        Calculate cosine similarity between two vectors.

        Formula: similarity = (vec1 · vec2) / (||vec1|| * ||vec2||)

        Where:
        - vec1 · vec2 is the dot product (sum of element-wise products)
        - ||vec|| is the Euclidean norm (L2-norm) of the vector

        Properties:
        - Range: [-1, 1] where 1 = identical vectors
        - For normalized vectors (norm=1), similarity equals dot product
        - Robust to vector magnitude differences

        Args:
            vec1: First feature vector
            vec2: Second feature vector

        Returns:
            Cosine similarity score in range [-1, 1]
        """
        # Calculate L2-norm (Euclidean length) of each vector
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Handle edge case: if either vector is zero, they're orthogonal
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Normalize vectors to unit length
        vec1_normalized = vec1 / norm1
        vec2_normalized = vec2 / norm2
        
        # Calculate dot product of normalized vectors (= cosine similarity)
        similarity = np.dot(vec1_normalized, vec2_normalized)
        return float(similarity)
    
    @staticmethod
    def _euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Euclidean distance between two vectors"""
        return np.linalg.norm(vec1 - vec2)