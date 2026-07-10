import traceback

import cv2
import numpy as np
import torch
import requests
from typing import Dict
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

    def __init__(
        self,
        db,
        output_dir: str,
        temp_dir: str,
        yolo_path: str = None,
        vit_path: str = None,
        classifier_path: str = None,
    ):
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
        self.data_loader = DataLoader(
            yolo_path=yolo_path,
            vit_path=vit_path,
            classifier_path=classifier_path,
        )
        self.yolo_model = self.data_loader.get_yolo_model()
        self.feature_extractor = self.data_loader.get_feature_extractor()
        self.known_unknown_classifier = None

        # Initialize image quality enhancement
        self.image_enhancer = ImageEnhancer()
        
        # Thresholds for detection and person matching (loaded from env or defaults)
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.known_person_threshold = KNOWN_PERSON_THRESHOLD
        self.frame_extraction_fps = FRAME_EXTRACTION_FPS

    def _get_known_unknown_classifier(self):
        if self.known_unknown_classifier is None:
            self.known_unknown_classifier = self.data_loader.get_known_unknown_classifier()
        return self.known_unknown_classifier

    @staticmethod
    def _build_track_url(frame_filename: str) -> str:
        return f"/static/detected/{frame_filename}"

    async def process_video_from_url(
        self,
        url: str,
        video_name: str,
        use_classification_model: bool = False,
    ) -> Dict:
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

        return await self.process_video_file(
            video_path,
            video_name,
            use_classification_model=use_classification_model,
        )
    
    async def process_video_file(
        self,
        video_path: str,
        video_name: str,
        use_classification_model: bool = False,
    ) -> Dict:
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
                                extracted_count,
                                use_classification_model=use_classification_model,
                            )
                            if detection is not None:
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
            "recognition_mode": "classification" if use_classification_model else "similarity",
            "use_classification_model": use_classification_model,
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
            detection_index: int,
            use_classification_model: bool = False,
    ) -> Dict:

        # ========== 1. Extract bbox ==========
        xyxy = box.xyxy[0]
        coords = xyxy.detach().cpu().numpy() if hasattr(xyxy, "cpu") else np.array(xyxy)
        x1, y1, x2, y2 = map(int, coords)

        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None  # ❗ bỏ detection lỗi

        confidence = float(box.conf[0].item())

        # ========== 2. Crop person ==========
        person_img = frame[y1:y2, x1:x2]
        if person_img.size == 0:
            return None

        feat = None

        if use_classification_model:
            classifier = self._get_known_unknown_classifier()
            match_result = classifier.predict(person_img)
        else:
            # ========== 3. Extract feature (SAFE) ==========
            with torch.no_grad():
                feat = self.feature_extractor.extract(person_img)

            if feat is None:
                return None

            if isinstance(feat, torch.Tensor):
                feat = feat.detach().cpu().numpy()
            elif isinstance(feat, list):
                feat = np.array(feat, dtype=np.float32)
            else:
                feat = np.asarray(feat, dtype=np.float32)

            feat = feat.flatten()
            if feat.size == 0:
                return None

            # ========== 4. Matching ==========
            match_result = await self._find_matching_person(feat)

        # ========== 5. Save image ==========
        detection_id = str(uuid.uuid4())
        frame_filename = f"{video_id}_{detection_index}_{detection_id}.jpg"
        frame_path = os.path.join(self.output_dir, frame_filename)
        cv2.imwrite(frame_path, person_img)
        track_url = self._build_track_url(frame_filename)

        detection = {
            "detection_id": detection_id,
            "video_id": video_id,
            "video_name": video_name,
            "frame_number": frame_number,
            "timestamp": timestamp,
            "bounding_box": {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1
            },
            "confidence": confidence,
            "feature_vector": feat.tolist() if feat is not None else None,
            "saved_frame_path": frame_path,
            "track_url": track_url,
            "recognition_mode": "classification" if use_classification_model else "similarity",
            "is_known": match_result.get("is_known", False),
            "display_label": "quen" if match_result.get("is_known", False) else "lạ",
            "classification_label": match_result.get("predicted_class"),
            "classification_confidence": match_result.get("confidence"),
            "matched_person_id": match_result.get("person_id"),
            "matched_person_name": match_result.get("person_name"),
            "similarity_score": match_result.get("similarity_score"),
            "detected_at": datetime.now().isoformat()
        }

        if use_classification_model:
            detection["matched_person_id"] = None
            detection["matched_person_name"] = None
            detection["similarity_score"] = None

        try:
            await self.db.save_detection(detection)
        except Exception as e:
            print(f"⚠️ DB save failed: {e}")

        return detection

    async def _find_matching_person(self, feature_vector: np.ndarray) -> Dict:
        known_persons = await self.db.get_all_known_persons()

        if not known_persons or feature_vector.size == 0:
            return {"is_known": False}

        best_match = None
        best_similarity = -1.0

        for person in known_persons:
            feature_list = person.get("feature_vectors")
            if feature_list is None and person.get("feature_vector") is not None:
                feature_list = [person["feature_vector"]]
            if not feature_list:
                continue

            for stored_vec in feature_list:
                stored_vec = np.array(stored_vec, dtype=np.float32)
                if stored_vec.shape != feature_vector.shape:
                    continue

                similarity = self._cosine_similarity(feature_vector, stored_vec)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person

        if best_match and best_similarity >= self.known_person_threshold:
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
