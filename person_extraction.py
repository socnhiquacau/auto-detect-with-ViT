"""
Person Extraction from Videos
==============================
Functionality: Detect and extract person shapes from videos
- Read videos from uploads folder
- Detect people in motion or activity
- Use last.pt model to detect persons
- Extract frames containing persons and save with standardized format
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime
from ultralytics import YOLO
import hashlib


class PersonExtractor:
    """
    Class specialized in extracting person shapes from videos

    Pipeline:
    1. Load video from uploads folder
    2. Detect motion (motion detection)
    3. Detect persons using YOLO model (last.pt)
    4. Extract and save frames containing persons
    """

    def __init__(
        self,
        model_path: str = "models/last.pt",
        uploads_folder: str = "uploads",
        output_folder: str = "extracted_persons",
        confidence_threshold: float = 0.5,
        motion_threshold: int = 25
    ):
        """
        Initialize PersonExtractor

        Args:
            model_path: Path to YOLO model (last.pt)
            uploads_folder: Folder containing videos to process
            output_folder: Folder to save extracted person images
            confidence_threshold: Confidence threshold for detection
            motion_threshold: Motion detection threshold (0-255)
        """
        self.model_path = model_path
        self.uploads_folder = uploads_folder
        self.output_folder = output_folder
        self.confidence_threshold = confidence_threshold
        self.motion_threshold = motion_threshold

        # Load YOLO model
        print(f"📦 Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        print("✅ Model loaded successfully")

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )

    def detect_motion(self, frame: np.ndarray) -> bool:
        """
        Detect motion in frame

        Args:
            frame: Current frame

        Returns:
            True if motion detected, False otherwise
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Count pixels with motion
        motion_pixels = np.sum(fg_mask > self.motion_threshold)
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        motion_ratio = motion_pixels / total_pixels

        # Consider as activity if > 1% of pixels have motion
        return motion_ratio > 0.01

    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in frame using YOLO

        Args:
            frame: Frame to detect

        Returns:
            List of detections with bbox and confidence
        """
        # Run YOLO inference
        results = self.model(
            frame, conf=self.confidence_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Only take class 0 (person)
                if int(box.cls[0]) == 0:
                    # Get bbox coordinates
                    xyxy = box.xyxy[0]
                    coords = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.array(xyxy)
                    x1, y1, x2, y2 = map(int, coords)

                    # Get confidence score
                    confidence = float(box.conf[0].item())

                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence
                    })

        return detections

    def compute_image_hash(self, image: np.ndarray) -> str:
        """
        Compute perceptual hash of an image for duplicate detection

        Args:
            image: Image to hash

        Returns:
            Hash string representing the image
        """
        # Resize to small size for comparison
        small = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)

        # Convert to grayscale if needed
        if len(small.shape) == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Compute average hash
        avg = small.mean()
        diff = small > avg

        # Convert to hash string
        hash_str = hashlib.md5(diff.tobytes()).hexdigest()
        return hash_str

    def extract_person_crop(
        self,
        frame: np.ndarray,
        bbox: List[int]
    ) -> np.ndarray:
        """
        Crop person shape from frame according to bbox

        Args:
            frame: Original frame
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Cropped image containing person
        """
        x1, y1, x2, y2 = bbox

        # Ensure bbox is within frame
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Crop image
        person_img = frame[y1:y2, x1:x2]

        return person_img

    def _save_person_detection(
        self,
        frame: np.ndarray,
        detection: Dict,
        video_id: str,
        video_output_folder: str,
        frame_id: int,
        img_id: int,
        saved_hashes: Set[str]
    ) -> Optional[Dict]:
        """
        Save a single person detection to disk

        Args:
            frame: Original frame
            detection: Detection dict with bbox and confidence
            video_id: Video identifier
            video_output_folder: Output folder path
            frame_id: Current frame number
            img_id: Image identifier
            saved_hashes: Set of already saved image hashes

        Returns:
            Dictionary with saved image info or None if invalid or duplicate
        """
        bbox = detection['bbox']
        confidence = detection['confidence']

        # Crop person image
        person_img = self.extract_person_crop(frame, bbox)

        # Check valid size
        if person_img.size == 0 or person_img.shape[0] < 10 or person_img.shape[1] < 10:
            return None

        # Check for duplicate images
        img_hash = self.compute_image_hash(person_img)
        if img_hash in saved_hashes:
            return None  # Skip duplicate

        # Add hash to set
        saved_hashes.add(img_hash)

        # Create filename according to format: <video_id>_<frame_id>_<img_id>.png
        filename = f"{video_id}_{frame_id:06d}_{img_id:04d}.png"
        save_path = os.path.join(video_output_folder, filename)

        # Save image
        cv2.imwrite(save_path, person_img)

        return {
            'filename': filename,
            'path': save_path,
            'frame_id': frame_id,
            'img_id': img_id,
            'confidence': confidence,
            'bbox': bbox
        }

    def _process_frame_detections(
        self,
        frame: np.ndarray,
        video_id: str,
        video_output_folder: str,
        frame_id: int,
        img_id: int,
        saved_images: List[Dict],
        saved_hashes: Set[str]
    ) -> int:
        """
        Process all person detections in a frame

        Args:
            frame: Frame to process
            video_id: Video identifier
            video_output_folder: Output folder path
            frame_id: Current frame number
            img_id: Starting image identifier
            saved_images: List to append saved images to
            saved_hashes: Set of already saved image hashes

        Returns:
            Updated img_id counter
        """
        detections = self.detect_persons(frame)

        if len(detections) == 0:
            return img_id

        # Extract and save each detected person
        for detection in detections:
            saved_info = self._save_person_detection(
                frame, detection, video_id, video_output_folder, frame_id, img_id, saved_hashes
            )

            if saved_info:
                saved_images.append(saved_info)
                img_id += 1

        return img_id

    def process_video(self, video_path: str) -> Optional[Dict]:
        """
        Process a video: detect and extract person shapes

        Args:
            video_path: Path to video file

        Returns:
            Dictionary containing processing information or None if error
        """
        # Create video_id from filename
        video_name = Path(video_path).stem
        video_id = video_name  # Or can use uuid

        print(f"\n{'='*60}")
        print(f"🎬 Processing video: {video_name}")
        print(f"{'='*60}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None

        # Get video information
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        print("📊 Video info:")
        print(f"   - FPS: {fps:.2f}")
        print(f"   - Total frames: {total_frames}")
        print(f"   - Duration: {duration:.2f}s")

        # Create separate folder for this video
        video_output_folder = os.path.join(self.output_folder, video_id)
        os.makedirs(video_output_folder, exist_ok=True)

        frame_id = 0
        img_id = 0
        saved_images = []
        saved_hashes = set()  # Track image hashes to detect duplicates
        motion_frames = 0
        person_detections = 0
        duplicate_count = 0  # Track skipped duplicates

        # Reset background subtractor for new video
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Check for motion
            has_motion = self.detect_motion(frame)

            if has_motion:
                motion_frames += 1

                # Count detections before processing
                detections_before = len(self.detect_persons(frame))

                # Process detections in this frame
                prev_img_id = img_id
                img_id = self._process_frame_detections(
                    frame, video_id, video_output_folder, frame_id, img_id, saved_images, saved_hashes
                )

                # Track duplicates
                detections_saved = img_id - prev_img_id
                duplicate_count += detections_before - detections_saved

                # Track frames with actual person detections
                if img_id > prev_img_id:
                    person_detections += 1

                # Display progress
                if img_id % 10 == 0 and img_id > 0:
                    progress = (frame_id / total_frames) * 100
                    print(f"   📸 Saved {img_id} images... ({progress:.1f}%)")

            frame_id += 1

        cap.release()

        # Summary of results
        result = {
            'video_id': video_id,
            'video_name': video_name,
            'video_path': video_path,
            'output_folder': video_output_folder,
            'total_frames': total_frames,
            'motion_frames': motion_frames,
            'person_detections': person_detections,
            'total_saved_images': len(saved_images),
            'duplicate_count': duplicate_count,
            'saved_images': saved_images,
            'processed_at': datetime.now().isoformat()
        }

        print("\n✅ Processing complete!")
        print(f"   - Total frames: {total_frames}")
        print(f"   - Frames with motion: {motion_frames}")
        print(f"   - Frames with persons: {person_detections}")
        print(f"   - Total saved images: {len(saved_images)}")
        print(f"   - Duplicates skipped: {duplicate_count}")
        print(f"   - Output folder: {video_output_folder}")

        return result

    def process_all_videos(self) -> List[Dict]:
        """
        Process all videos in uploads folder

        Returns:
            List of processing results for each video
        """
        # Find all video files in uploads folder
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        video_files = []

        for ext in video_extensions:
            video_files.extend(Path(self.uploads_folder).glob(f"*{ext}"))

        if len(video_files) == 0:
            print(f"⚠️  No video files found in: {self.uploads_folder}")
            return []

        print(f"\n🎯 Found {len(video_files)} video(s) to process")

        results = []
        for video_path in video_files:
            result = self.process_video(str(video_path))
            if result:
                results.append(result)

        # Overall summary
        total_images = sum(r['total_saved_images'] for r in results)
        print(f"\n{'='*60}")
        print("🎉 All videos processed!")
        print(f"{'='*60}")
        print(f"   - Total videos: {len(results)}")
        print(f"   - Total extracted images: {total_images}")
        print(f"   - Output folder: {self.output_folder}")

        return results


def main():
    """
    Main function to run the program
    """
    print("\n🚀 Person Extraction from Videos")
    print("="*60)

    # Initialize extractor
    extractor = PersonExtractor(
        model_path="models/last.pt",
        uploads_folder="uploads",
        output_folder="extracted_persons",
        confidence_threshold=0.5,
        motion_threshold=25
    )

    # Process all videos
    results = extractor.process_all_videos()

    # Save results to JSON file (optional)
    import json
    output_json = os.path.join(
        extractor.output_folder, "extraction_results.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {output_json}")


if __name__ == "__main__":
    main()
