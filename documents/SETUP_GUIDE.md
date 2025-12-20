# Setup & Best Practices Guide

## Quick Start

### 1. Setup Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```dotenv
# Database
MONGODB_URL=mongodb://admin:admin123%20@localhost:27017/video_detection_db?authSource=admin
DATABASE_NAME=video_detection_db

# Directories
UPLOAD_DIR=uploads
OUTPUT_DIR=output/detected
TEMP_DIR=temp

# Models (MUST be in models/ directory)
YOLO_MODEL_PATH=models/yolov8_person_detection.pt
VIT_MODEL_PATH=models/vit_final_model.pth

# Detection Thresholds
CONFIDENCE_THRESHOLD=0.75
KNOWN_PERSON_THRESHOLD=0.95
FRAME_EXTRACTION_FPS=2

# Server
HOST=0.0.0.0
PORT=8000
RELOAD=true
```

### 2. Verify Models Directory

All models MUST be in the `models/` directory:

```
models/
├── yolov8_person_detection.pt
├── vit_final_model.pth
├── dino_vits16_epoch100.pth
└── (other models)
```

To check available models:

```python
from common import find_model_file

# Find specific model
path = find_model_file("vit_final_model.pth")

# Find first available model
path = find_model_file()
```

### 3. Start the API

```bash
python main.py
```

API will be available at: `http://localhost:8000`

---

## Working with Known Persons

### Add from Flat Structure

```bash
# Directory structure:
# known_persons/
#   person1.jpg
#   person2.png
#   person3.jpg

python add_known_persons.py add known_persons/
```

### Add from Nested Structure

```bash
# Directory structure:
# known_persons/
#   person1/
#     photo1.jpg
#     photo2.jpg
#     photo3.jpg
#   person2/
#     photo1.jpg
#     photo2.jpg

python add_known_persons.py add known_persons/
```

### List Known Persons

```bash
python add_known_persons.py list
```

Output:
```
📋 Known Persons in Database:
============================================================

1. John Doe
   ID: john_doe
   Feature dim: 384
   Added: 2024-12-20T10:30:45.123456

2. Jane Smith
   ID: jane_smith
   Feature dim: 384
   Added: 2024-12-20T10:31:20.654321

Total: 2 persons
```

### Delete a Person

```bash
python add_known_persons.py delete john_doe
```

---

## Configuration Tuning

### Confidence Threshold (YOLO Detection)

Controls which detections to process. Lower = more detections (including false positives).

```
Range: 0.0 - 1.0
Default: 0.75

Scenarios:
- Strict (high precision):     0.85 - 0.95
- Balanced (default):           0.70 - 0.80
- Permissive (high recall):     0.40 - 0.60
```

**Recommendation**: Start with 0.75, adjust based on false positive/negative rates.

### Known Person Threshold (Matching)

Controls how similar a detection must be to match a known person. Higher = stricter matching.

```
Range: 0.0 - 1.0 (cosine similarity)
Default: 0.95

Scenarios:
- Strict (few false matches):    0.90 - 0.98
- Balanced (default):             0.80 - 0.90
- Permissive (catch all):         0.60 - 0.80
```

**Recommendation**: Start with 0.95, lower if missing known persons, raise if getting false matches.

### Frame Extraction FPS

How many frames per second to extract from video (affects processing speed).

```
Range: 1 - 30+
Default: 2

Scenarios:
- Fast (low detail):     1 FPS
- Balanced (default):    2 FPS
- Detailed (slow):       5 FPS
- High detail:           10+ FPS
```

**Recommendation**: 2 FPS for real-time processing, 5+ FPS for analysis mode.

---

## Database Management

### Check MongoDB Connection

```python
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio

async def check_db():
    client = AsyncIOMotorClient("mongodb://admin:admin123@localhost:27017/")
    try:
        await client.admin.command('ping')
        print("✅ MongoDB is running")
    except Exception as e:
        print(f"❌ MongoDB error: {e}")

asyncio.run(check_db())
```

### View Database Collections

```bash
# Using MongoDB CLI
mongo -u admin -p "admin123 " video_detection_db

# Commands:
db.known_persons.find().pretty()
db.detections.count()
db.processing_results.find({status: "completed"}).count()
```

### Backup Database

```bash
# Backup
mongodump --uri="mongodb://admin:admin123@localhost:27017/" -d video_detection_db -o ./backup/

# Restore
mongorestore --uri="mongodb://admin:admin123@localhost:27017/" ./backup/video_detection_db/
```

---

## GPU Configuration

### Enable CUDA (NVIDIA GPU)

```bash
# Check available GPUs
python -c "import torch; print(torch.cuda.device_count())"

# In .env
CUDA_VISIBLE_DEVICES=0

# Or for multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2
```

### Monitor GPU Usage

```bash
# NVIDIA GPUs
nvidia-smi

# Watch GPU
watch -n 1 nvidia-smi
```

### Run on CPU Only

```bash
# In .env
CUDA_VISIBLE_DEVICES=
```

Or in Python:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

---

## Performance Optimization

### 1. Model Loading Optimization

Models are lazy-loaded (only when first used):

```python
# First call: model loads
loader = DataLoader()
feature_extractor = loader.get_feature_extractor()  # ~2-3s

# Subsequent calls: use cached model
feature_extractor = loader.get_feature_extractor()  # ~0s
```

### 2. Batch Feature Extraction

Extract features from multiple images at once (more efficient):

```python
from feature_extractor import FeatureExtractor
import cv2

extractor = FeatureExtractor()

# Read multiple images
images = [cv2.imread(f"person{i}.jpg") for i in range(10)]

# Extract batch features (more efficient than individual)
features = extractor.extract_batch(images)  # Shape: (10, 384)
```

### 3. Frame Rate Optimization

Reduce processing load by adjusting frame extraction:

```
Real-time processing: FRAME_EXTRACTION_FPS = 1-2
Analysis mode: FRAME_EXTRACTION_FPS = 2-5
Detailed analysis: FRAME_EXTRACTION_FPS = 5-10
```

### 4. Database Query Optimization

Indexes are created automatically. For large datasets:

```python
# Check index usage
db.detections.aggregate([{"$indexStats": {}}])

# Create custom index if needed
db.detections.createIndex({"video_id": 1, "is_known": 1})
```

---

## Troubleshooting

### Model Not Found Error

```
FileNotFoundError: Model ... not found inside models/ directory
```

**Solution:**
1. Verify model file exists in `models/` directory
2. Check model filename in `.env`
3. Try with full path from `models/`:
   ```python
   from common import find_model_file
   find_model_file("vit_final_model.pth")
   ```

### MongoDB Connection Failed

```
ServerSelectionTimeoutError: No servers found yet
```

**Solution:**
1. Start MongoDB: `docker-compose up -d mongodb` or `mongod`
2. Check connection string in `.env`
3. Verify credentials are correct
4. Test connection: `python -m pip install pymongo; python -c "from pymongo import MongoClient; MongoClient('mongodb://admin:admin123@localhost:27017/')"`

### Out of Memory (CUDA)

```
RuntimeError: CUDA out of memory
```

**Solution:**
1. Reduce `FRAME_EXTRACTION_FPS` to process fewer frames
2. Set `CUDA_VISIBLE_DEVICES` to use only available GPU memory
3. Use CPU instead: `CUDA_VISIBLE_DEVICES=`
4. Process videos in smaller batches

### Slow Processing

**Optimization steps:**
1. Increase `CONFIDENCE_THRESHOLD` to skip low-confidence detections
2. Decrease `FRAME_EXTRACTION_FPS` to process fewer frames
3. Enable GPU with `CUDA_VISIBLE_DEVICES=0`
4. Use smaller image size if possible

---

## Testing

### Test Model Loading

```bash
python -c "from common import find_model_file; print(find_model_file())"
```

### Test Feature Extraction

```bash
python -c "from common import DataLoader; d = DataLoader(); e = d.get_feature_extractor(); print(e.get_feature_dimension())"
```

### Test API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Get features list
curl http://localhost:8000/features/list

# Add a known person
curl -X POST http://localhost:8000/features/add \
  -H "Content-Type: application/json" \
  -d '{
    "person_id": "test_person",
    "name": "Test Person",
    "feature_vector": [0.1, 0.2, 0.3, ...]
  }'
```

---

## Development Tips

### Adding New Configuration Variables

1. Add to `.env.example`:
```dotenv
NEW_VARIABLE=default_value  # Description
```

2. Load in `common.py`:
```python
NEW_VARIABLE = os.getenv("NEW_VARIABLE", "default_value")
```

3. Use in other modules:
```python
from common import NEW_VARIABLE
```

### Adding New Preprocessing Methods

All preprocessing is centralized in `common.py`:

```python
def preprocess_custom(input_data, device="cpu"):
    """
    Custom preprocessing method.
    
    Args:
        input_data: Input data
        device: torch device
        
    Returns:
        Preprocessed tensor
    """
    # Implementation
    pass
```

Access via:
```python
from common import DataLoader
loader = DataLoader()
# Use static method
tensor = loader.preprocess_custom(data)
```

### Extending DataLoader

Add new model types:

```python
class DataLoader:
    def get_segmentation_model(self, model_name=None):
        """Get segmentation model for instance segmentation."""
        if self._seg_model is None:
            model_path = model_name or os.getenv("SEG_MODEL_PATH")
            self._seg_model = load_model_from_models(model_path, self.device)
        return self._seg_model
```

---

## Production Deployment

### Checklist

- [ ] All models in `models/` directory
- [ ] `.env` file configured with production settings
- [ ] MongoDB running and accessible
- [ ] All required directories created (uploads, output, temp)
- [ ] API tested with sample data
- [ ] Known persons loaded into database
- [ ] GPU verified working (if applicable)
- [ ] Logs configured for monitoring
- [ ] Database backups configured
- [ ] HTTPS enabled for API (if public)

### Production Settings

```dotenv
# Strict thresholds
CONFIDENCE_THRESHOLD=0.85
KNOWN_PERSON_THRESHOLD=0.95

# Optimize for speed
FRAME_EXTRACTION_FPS=1

# Use GPU
CUDA_VISIBLE_DEVICES=0

# Production server
HOST=0.0.0.0
PORT=8000
RELOAD=false
```

### Docker Deployment

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f api
```

---

## Documentation & Comments

All code includes comprehensive comments explaining:
- **Complex algorithms**: Feature matching, frame extraction, averaging
- **Configuration variables**: Range, purpose, impact
- **Data structures**: Input/output formats
- **Error handling**: Edge cases and recovery

See `REFACTOR_SUMMARY.md` for detailed refactoring documentation.

