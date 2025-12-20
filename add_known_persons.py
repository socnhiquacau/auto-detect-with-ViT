"""
Add known persons to database from images.

This script supports two folder structures:
1. Flat: known_persons/person_name.jpg (one image per person)
2. Nested: known_persons/person_name/photo1.jpg, photo2.jpg, ... (multiple images per person)

The script:
- Automatically detects folder structure
- Extracts feature vectors using ViT model
- Averages multiple features for same person (nested structure)
- Stores results in MongoDB

Usage: python add_known_persons.py <folder_path>
Example: python add_known_persons.py known_persons/
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import cv2
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Import DataLoader for unified model loading
from common import DataLoader, YOLO_MODEL_PATH, VIT_MODEL_PATH

# Load environment variables
load_dotenv()

# =========================
# CONFIGURATION FROM ENVIRONMENT
# =========================
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:admin123@localhost:27017/video_detection_db?authSource=admin")
DATABASE_NAME = os.getenv("DATABASE_NAME", "video_detection_db")

# Supported image extensions for person images
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


class KnownPersonsManager:
    """
    Manager for adding known persons to the database from image files.

    Features:
    - Supports flat and nested folder structures
    - Extracts feature vectors using ViT model
    - Averages multiple features per person (for nested structure)
    - Stores normalized features in MongoDB
    """

    def __init__(self, db, feature_extractor):
        """
        Initialize KnownPersonsManager.

        Args:
            db: Database instance for MongoDB operations
            feature_extractor: FeatureExtractor instance for extracting vectors
        """
        self.db = db
        self.feature_extractor = feature_extractor
        self.known_persons = db.known_persons

    async def add_from_folder(self, folder_path: str, structure: str = "auto"):
        """
        Add known persons from folder.

        Args:
            folder_path: Path to folder containing person images
            structure: "flat", "nested", or "auto" (auto-detect)
        """
        folder = Path(folder_path)

        if not folder.exists():
            print(f"❌ Folder not found: {folder_path}")
            return

        # Auto-detect folder structure if not specified
        if structure == "auto":
            structure = self._detect_structure(folder)
            print(f"📁 Detected structure: {structure}")

        if structure == "flat":
            await self._process_flat_structure(folder)
        elif structure == "nested":
            await self._process_nested_structure(folder)
        else:
            print(f"❌ Unknown structure: {structure}")

    def _detect_structure(self, folder: Path) -> str:
        """
        Auto-detect folder structure by checking for subdirectories vs image files.

        Returns:
            "nested" if subdirectories exist, "flat" otherwise
        """
        subdirs = [d for d in folder.iterdir() if d.is_dir()]
        image_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

        if subdirs and not image_files:
            return "nested"
        elif image_files and not subdirs:
            return "flat"
        elif subdirs and image_files:
            # Mixed structure, prefer nested
            return "nested"
        else:
            return "flat"

    async def _process_flat_structure(self, folder: Path):
        """
        Process flat structure: known_persons/person_name.jpg
        Each image = one person's feature vector
        """
        print("\n📂 Processing flat structure...")
        print(f"   Folder: {folder}")

        image_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

        if not image_files:
            print("❌ No image files found!")
            return

        print(f"   Found {len(image_files)} images")

        added = 0
        for img_file in image_files:
            # Person name from filename (without extension)
            person_name = img_file.stem
            person_id = self._sanitize_id(person_name)

            success = await self._process_person_image(img_file, person_id, person_name)
            if success:
                added += 1

        print(f"\n✅ Added {added}/{len(image_files)} persons to database")

    async def _process_nested_structure(self, folder: Path):
        """
        Process nested structure: known_persons/person_name/photo1.jpg, photo2.jpg

        Algorithm:
        1. For each person subdirectory:
        2. Extract feature vectors from all images
        3. Average the feature vectors to get a single representative vector
        4. Store in database

        Averaging multiple features improves robustness by capturing variations
        in lighting, pose, and expression across multiple photos.
        """
        print("\n📂 Processing nested structure...")
        print(f"   Folder: {folder}")

        person_folders = [d for d in folder.iterdir() if d.is_dir()]

        if not person_folders:
            print("❌ No person folders found!")
            return

        print(f"   Found {len(person_folders)} person folders")

        added = 0
        for person_folder in person_folders:
            person_name = person_folder.name
            person_id = self._sanitize_id(person_name)

            # Get all images for this person
            image_files = [f for f in person_folder.iterdir()
                           if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

            if not image_files:
                print(f"   ⚠️  No images found for: {person_name}")
                continue

            print(f"\n👤 Processing: {person_name} ({len(image_files)} images)")

            # Extract features from all images
            feature_vectors = []
            for img_file in image_files:
                try:
                    feature = self._extract_feature_from_file(img_file)
                    if feature is not None:
                        feature_vectors.append(feature)
                        print(f"      ✅ {img_file.name}")
                except Exception as e:
                    print(f"      ❌ {img_file.name}: {e}")

            if not feature_vectors:
                print(f"   ❌ No valid features extracted for: {person_name}")
                continue

            # Average all feature vectors
            avg_feature = np.mean(feature_vectors, axis=0)

            # Normalize
            norm = np.linalg.norm(avg_feature)
            if norm > 0:
                avg_feature = avg_feature / norm

            # Save to database
            success = await self._save_person(person_id, person_name, avg_feature)
            if success:
                added += 1
                print(f"   ✅ Saved with averaged features from {len(feature_vectors)} images")

        print(f"\n✅ Added {added}/{len(person_folders)} persons to database")

    async def _process_person_image(self, img_path: Path, person_id: str, person_name: str) -> bool:
        """
        Process a single person image for flat structure.

        Args:
            img_path: Path to image file
            person_id: Unique person identifier (sanitized from name)
            person_name: Human-readable person name

        Returns:
            True if successfully processed and saved, False otherwise
        """
        print(f"\n👤 Processing: {person_name}")
        print(f"   Image: {img_path.name}")

        try:
            # Extract feature vector using ViT model
            feature_vector = self._extract_feature_from_file(img_path)

            if feature_vector is None:
                print(f"   ❌ Failed to extract features")
                return False

            # Save person with extracted feature to database
            success = await self._save_person(person_id, person_name, feature_vector)

            if success:
                print(f"   ✅ Added to database")
                return True
            else:
                print(f"   ❌ Failed to save to database")
                return False

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

    def _extract_feature_from_file(self, img_path: Path) -> np.ndarray:
        """
        Extract feature vector from image file using ViT model.

        Args:
            img_path: Path to image file

        Returns:
            Normalized feature vector as numpy array, or None if extraction fails
        """
        # Read image using OpenCV (BGR format)
        image = cv2.imread(str(img_path))

        if image is None:
            raise Exception(f"Cannot read image: {img_path}")

        # Extract feature vector using FeatureExtractor (handles normalization)
        feature_vector = self.feature_extractor.extract(image)

        return feature_vector

    async def _save_person(self, person_id: str, person_name: str, feature_vector: np.ndarray) -> bool:
        """
        Save person with feature vector to MongoDB database.

        Uses upsert logic: creates new record if person_id doesn't exist,
        or updates existing record (for re-processing).

        Args:
            person_id: Unique person identifier (used as primary key)
            person_name: Human-readable person name
            feature_vector: Normalized feature vector (L2-norm = 1)

        Returns:
            True if successful, False if database error
        """
        try:
            person_data = {
                "person_id": person_id,
                "name": person_name,
                "feature_vector": feature_vector.tolist(),  # Convert numpy array to list for JSON storage
                "added_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }

            # Upsert: insert if new, update if exists
            # This allows re-processing person images without duplicates
            await self.known_persons.update_one(
                {"person_id": person_id},
                {"$set": person_data},
                upsert=True
            )

            return True

        except Exception as e:
            print(f"   Database error: {e}")
            return False

    @staticmethod
    def _sanitize_id(name: str) -> str:
        """Convert name to valid person_id"""
        # Remove special characters, replace spaces with underscore
        import re
        sanitized = re.sub(r'[^\w\s-]', '', name.lower())
        sanitized = re.sub(r'[\s-]+', '_', sanitized)
        return sanitized


async def list_known_persons():
    """List all known persons in database"""
    print("\n📋 Known Persons in Database:")
    print("=" * 60)

    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]
    collection = db.known_persons

    cursor = collection.find({})
    persons = await cursor.to_list(length=1000)

    if not persons:
        print("   (No known persons found)")
    else:
        for i, person in enumerate(persons, 1):
            print(f"\n{i}. {person['name']}")
            print(f"   ID: {person['person_id']}")
            print(f"   Feature dim: {len(person['feature_vector'])}")
            print(f"   Added: {person['added_at']}")

    print(f"\nTotal: {len(persons)} persons")

    client.close()


async def delete_person(person_id: str):
    """Delete a person from database"""
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]
    collection = db.known_persons

    result = await collection.delete_one({"person_id": person_id})

    if result.deleted_count > 0:
        print(f"✅ Deleted person: {person_id}")
    else:
        print(f"❌ Person not found: {person_id}")

    client.close()


async def main():
    """
    Main entry point for adding known persons to database.

    Supports operations:
    - add <folder>: Add persons from folder (auto-detect structure)
    - list: List all known persons
    - delete <person_id>: Delete a person by ID
    """
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python add_known_persons.py add <folder_path>       - Add from folder")
        print("  python add_known_persons.py list                    - List all persons")
        print("  python add_known_persons.py delete <person_id>      - Delete person")
        return

    operation = sys.argv[1].lower()

    try:
        if operation == "add" and len(sys.argv) >= 3:
            folder_path = sys.argv[2]
            await add_persons_from_folder(folder_path)

        elif operation == "list":
            await list_known_persons()

        elif operation == "delete" and len(sys.argv) >= 3:
            person_id = sys.argv[2]
            await delete_person(person_id)

        else:
            print(f"❌ Unknown operation: {operation}")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


async def add_persons_from_folder(folder_path: str):
    """
    Add known persons from folder to database.

    Steps:
    1. Connect to MongoDB
    2. Initialize FeatureExtractor using DataLoader
    3. Load images from folder
    4. Extract features
    5. Save to database

    Args:
        folder_path: Path to folder containing person images
    """
    print(f"\n🚀 Adding persons from: {folder_path}")

    # Connect to MongoDB
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]

    # Initialize DataLoader for unified model loading
    print("📦 Initializing models...")
    data_loader = DataLoader()
    feature_extractor = data_loader.get_feature_extractor()

    # Initialize manager
    manager = KnownPersonsManager(db, feature_extractor)

    # Process folder
    await manager.add_from_folder(folder_path, structure="auto")

    client.close()

    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())