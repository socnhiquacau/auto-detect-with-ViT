"""
Add known persons to database from images
Supports two folder structures:
1. Flat: known_persons/person_name.jpg
2. Nested: known_persons/person_name/photo1.jpg, photo2.jpg, ...

Usage: python add_known_persons.py <folder_path>
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

# Import your feature extractor
from feature_extractor import FeatureExtractor

# Load environment
load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:admin123@localhost:27017/video_detection_db?authSource=admin")
DATABASE_NAME = "video_detection_db"
VIT_MODEL_PATH = os.getenv("VIT_MODEL_PATH", "models/vit_dino_feature_extractor.pth")

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


class KnownPersonsManager:
    """Manager for adding known persons from images"""

    def __init__(self, db, feature_extractor):
        self.db = db
        self.feature_extractor = feature_extractor
        self.known_persons = db.known_persons

    async def add_from_folder(self, folder_path: str, structure: str = "auto"):
        """
        Add known persons from folder

        Args:
            folder_path: Path to folder containing images
            structure: "flat", "nested", or "auto" (auto-detect)
        """
        folder = Path(folder_path)

        if not folder.exists():
            print(f"❌ Folder not found: {folder_path}")
            return

        # Auto-detect structure
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
        """Auto-detect folder structure"""
        subdirs = [d for d in folder.iterdir() if d.is_dir()]
        image_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

        if subdirs and not image_files:
            return "nested"
        elif image_files and not subdirs:
            return "flat"
        elif subdirs and image_files:
            # Mixed, prefer nested
            return "nested"
        else:
            return "flat"

    async def _process_flat_structure(self, folder: Path):
        """
        Process flat structure: known_persons/person_name.jpg
        Each image = one person
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
        Multiple images per person, average their features
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
        """Process a single person image"""
        print(f"\n👤 Processing: {person_name}")
        print(f"   Image: {img_path.name}")

        try:
            # Extract feature
            feature_vector = self._extract_feature_from_file(img_path)

            if feature_vector is None:
                print(f"   ❌ Failed to extract features")
                return False

            # Save to database
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
        """Extract feature vector from image file"""
        # Read image
        image = cv2.imread(str(img_path))

        if image is None:
            raise Exception(f"Cannot read image: {img_path}")

        # Extract feature using ViT
        feature_vector = self.feature_extractor.extract(image)

        return feature_vector

    async def _save_person(self, person_id: str, person_name: str, feature_vector: np.ndarray) -> bool:
        """Save person to database"""
        try:
            person_data = {
                "person_id": person_id,
                "name": person_name,
                "feature_vector": feature_vector.tolist(),
                "added_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }

            # Upsert (insert or update)
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
    """Main function"""

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Add persons from folder:")
        print("    python add_known_persons.py <folder_path>")
        print("")
        print("  List known persons:")
        print("    python add_known_persons.py --list")
        print("")
        print("  Delete person:")
        print("    python add_known_persons.py --delete <person_id>")
        print("")
        print("Folder structures supported:")
        print("  1. Flat: known_persons/person_name.jpg")
        print("  2. Nested: known_persons/person_name/photo1.jpg, photo2.jpg")
        return

    command = sys.argv[1]

    if command == "--list":
        await list_known_persons()
        return

    if command == "--delete":
        if len(sys.argv) < 3:
            print("❌ Usage: python add_known_persons.py --delete <person_id>")
            return
        await delete_person(sys.argv[2])
        return

    # Add persons from folder
    folder_path = command

    print("=" * 60)
    print("Adding Known Persons to Database")
    print("=" * 60)

    # Initialize feature extractor
    print(f"\n📦 Loading ViT model...")
    feature_extractor = FeatureExtractor(VIT_MODEL_PATH)
    print("✅ Model loaded")

    # Connect to database
    print(f"\n🔗 Connecting to MongoDB...")
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]
    print("✅ Connected")

    # Process folder
    manager = KnownPersonsManager(db, feature_extractor)
    await manager.add_from_folder(folder_path)

    # Show summary
    await list_known_persons()

    client.close()

    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())