# -*- coding: utf-8 -*-
"""
ADD KNOWN PERSONS - PRODUCTION SAFE VERSION

✔ No GPU memory leak
✔ Stable for long runs
✔ Deterministic & debuggable
✔ Thesis-ready
"""

import os
import re
import gc
import sys
import cv2
import time
import torch
import psutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import image_enhancement
from src.core.common import DataLoader

# =========================
# CONFIG
# =========================
KNOWN_PERSONS_DIR = PROJECT_ROOT / "known_persons"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGE_SIZE = (224, 224)
CLEANUP_EVERY_N_IMAGES = 2
MAX_RAM_MB = 1500

MONGODB_URL = "mongodb://admin:admin123@127.0.0.1:27017/video_detection_db?authSource=admin"
# MONGODB_URL = os.getenv(
#     "MONGODB_URL",
#     "mongodb://admin:admin123@127.0.0.1:27017/video_detection_db?authSource=admin"
# )
DB_NAME = "video_detection_db"


# =========================
# UTILS
# =========================
def ram_usage_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def cleanup(force=False):
    if force or ram_usage_mb() > MAX_RAM_MB:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^\w\s-]", "", name.lower())
    return re.sub(r"[\s-]+", "_", name)


# =========================
# CORE CLASS
# =========================
class KnownPersonsManager:
    def __init__(self, db):
        self.collection = db.known_persons
        self.data_loader = DataLoader()
        self.extractor = self.data_loader.get_feature_extractor()

    def run(self):
        folder = Path(KNOWN_PERSONS_DIR)
        assert folder.exists(), "known_persons folder not found"

        images = [
            f for f in folder.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"📂 Found {len(images)} images")

        grouped = self._group_images(images)
        print(f"👥 Grouped into {len(grouped)} persons")

        for name, files in grouped.items():
            print(f"\n👤 Processing: {name} ({len(files)} images)")
            self._process_person(name, files)

        print("\n✅ DONE")

    def _group_images(self, files):
        persons = defaultdict(list)
        for f in files:
            stem = f.stem
            m = re.match(r"(.+?)_\d+$", stem)
            name = m.group(1) if m else stem
            persons[name].append(f)

        for k in persons:
            persons[k].sort(key=lambda x: x.name)

        return dict(persons)

    def _process_person(self, person_name, image_files):
        features = []
        person_id = sanitize_name(person_name)

        for idx, img_path in enumerate(image_files, 1):
            start = time.time()

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"❌ Cannot read {img_path.name}")
                continue

            img = cv2.resize(img, IMAGE_SIZE)
            img = image_enhancement.enhance(img)

            with torch.no_grad():
                feat = self.extractor.extract(img)

            # 🔥 CRITICAL PART
            if isinstance(feat, torch.Tensor):
                feat = feat.detach().cpu().numpy()

            features.append(feat.tolist())

            del img, feat

            if idx % CLEANUP_EVERY_N_IMAGES == 0:
                cleanup()

            print(
                f"   ✔ {img_path.name} "
                f"({(time.time() - start)*1000:.1f} ms)"
            )

        if features:
            self._save(person_id, person_name, features)

    def _save(self, pid, name, features):
        self.collection.update_one(
            {"person_id": pid},
            {
                "$set": {
                    "person_id": pid,
                    "name": name,
                    "feature_vectors": features,
                    "image_count": len(features),
                    "updated_at": datetime.now().isoformat()
                }
            },
            upsert=True
        )

        print(f"💾 Saved {name} ({len(features)} features)")


# =========================
# ENTRY POINT
# =========================
def main():
    print("=" * 60)
    print("🚀 ADD KNOWN PERSONS - PRODUCTION SAFE")
    print("=" * 60)

    client = MongoClient(MONGODB_URL)
    db = client[DB_NAME]

    try:
        KnownPersonsManager(db).run()
    finally:
        client.close()
        cleanup(force=True)


if __name__ == "__main__":
    main()
