# 🎯 DHT - Person Detection & Recognition System

A complete AI-powered system for detecting and recognizing people in images and videos using YOLOv8 and ReID models.

## ✨ Features

- 🔍 **Person Detection** - YOLOv8 for accurate person detection
- 👤 **Person Recognition** - ReID model for identity matching
- 🎬 **Video Processing** - Track persons across video frames
- 📸 **Image Processing** - Single image person recognition
- 👥 **Gallery Management** - Manage known persons database
- 🎨 **Multiple Interfaces** - Web UI, REST API, and CLI

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup folders (auto-created on first run)
mkdir -p models uploads output/detected temp known_gallery
```

### 2. Add Models

Place these files in `models/` folder:

- `yolov8_person_detection.pt` - YOLO detection model
- `best_model_state_dict.pth` - ReID recognition model

### 3. Choose Your Interface

#### Option A: Streamlit Web UI ⭐ **Recommended for beginners**

```bash
# Run UI
./run_ui.sh
# Or: streamlit run streamlit_app.py

# Open browser: http://localhost:8501
```

**Features:**

- 🎨 Beautiful web interface
- 📊 Interactive charts and visualizations
- 👥 Visual gallery management
- 📸 Drag & drop image/video upload
- 💾 Download results as CSV

[📖 Full Guide](STREAMLIT_UI_GUIDE.md)

#### Option B: FastAPI REST API ⭐ **For production**

```bash
# Run API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Open API docs: http://localhost:8000/docs
```

**Endpoints:**

- `POST /process/upload` - Process video file
- `POST /process/url` - Process video from URL
- `GET /results/{job_id}` - Get processing results
- `POST /features/add` - Add known person
- `GET /features/list` - List known persons

[📖 Full Guide](SOURCE_ANALYSIS.md)

#### Option C: Python Service ⭐ **For scripting**

```bash
# Use service.py directly
python service.py
```

[📖 Full Guide](SERVICE_GUIDE.md)

## 📁 Project Structure

```
DHT/
├── streamlit_app.py          # Streamlit Web UI ⭐
├── main.py                   # FastAPI REST API
├── service.py                # ReID Pipeline Service
├── video_processor.py        # Video processing core
├── feature_extractor.py      # Feature extraction
├── database.py               # MongoDB operations
├── common.py                 # Shared utilities
├── image_enhancement.py      # Image preprocessing
│
├── models/                   # AI Models
│   ├── yolov8_person_detection.pt
│   └── best_model_state_dict.pth
│
├── known_gallery/            # Known persons database
│   ├── person1/
│   │   ├── photo1.jpg
│   │   └── photo2.jpg
│   └── person2/
│       └── img1.jpg
│
├── uploads/                  # Uploaded files
├── output/detected/          # Detection results
├── temp/                     # Temporary files
│
└── Documentation/
    ├── STREAMLIT_UI_GUIDE.md      # UI documentation
    ├── SERVICE_GUIDE.md           # Service documentation
    ├── SOURCE_ANALYSIS.md         # Code analysis
    └── STREAMLIT_STATUS.md        # Setup status
```

## 🎓 Usage Guide

### Setup Known Persons Gallery

1. **Create person folders:**

```bash
mkdir -p known_gallery/person_name
```

2. **Add images:** (5-10 images per person recommended)

```bash
cp photos/*.jpg known_gallery/person_name/
```

3. **Build gallery:**
   - **Streamlit UI:** Go to "Manage Gallery" → "Build Gallery"
   - **Python:**
   ```python
   from service import ReIDPipelineService
   service = ReIDPipelineService()
   service.build_known_gallery("known_gallery", "known_gallery/gallery_embeddings.npz")
   ```

### Process Images

**Streamlit UI:**

1. Go to "Process Image" page
2. Upload image
3. Click "Detect & Recognize"
4. View results with charts

**API:**

```bash
curl -X POST http://localhost:8000/process/upload \
  -F "file=@image.jpg"
```

**Python:**

```python
from service import ReIDPipelineService
service = ReIDPipelineService()
service.load_known_gallery("known_gallery/gallery_embeddings.npz")
result = service.process_query_image("image.jpg", topk=5)
```

### Process Videos

**Streamlit UI:**

1. Go to "Process Video" page
2. Upload video
3. Wait for processing
4. Download CSV results

**API:**

```bash
curl -X POST http://localhost:8000/process/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/video.mp4", "video_name": "test"}'
```

**Python:**

```python
result = service.process_video("video.mp4", topk=5)
formatted = service.format_video_result(result)
print(formatted)
```

## 🎯 Comparison of Interfaces

| Feature              | Streamlit UI  | FastAPI          | CLI Demo      |
| -------------------- | ------------- | ---------------- | ------------- |
| **Ease of Use**      | ⭐⭐⭐⭐⭐    | ⭐⭐⭐           | ⭐⭐⭐⭐      |
| **Visual Interface** | ✅ Yes        | ❌ No            | ❌ No         |
| **Charts & Viz**     | ✅ Built-in   | ❌ Need frontend | ❌ Text only  |
| **API Integration**  | ❌ No         | ✅ REST API      | ❌ No         |
| **Database**         | ❌ File-based | ✅ MongoDB       | ❌ File-based |
| **Batch Processing** | ❌ Manual     | ✅ Yes           | ✅ Yes        |
| **Production Ready** | ⭐⭐⭐        | ⭐⭐⭐⭐⭐       | ⭐⭐          |

**Use Cases:**

- **Streamlit UI:** Demos, testing, internal tools, non-technical users
- **FastAPI:** Production backend, mobile apps, integrations
- **CLI Demo:** Quick testing, automation scripts, debugging

## 🔧 Configuration

### Environment Variables (.env)

```bash
# MongoDB (for FastAPI)
MONGODB_URL=mongodb://admin:admin123@127.0.0.1:27017/video_detection_db?authSource=admin
DATABASE_NAME=video_detection_db

# Model Paths
YOLO_MODEL_PATH=models/yolov8_person_detection.pt
VIT_MODEL_PATH=models/best_model_state_dict.pth

# Detection Settings
CONFIDENCE_THRESHOLD=0.75       # YOLO confidence (0-1)
KNOWN_PERSON_THRESHOLD=0.95     # Recognition threshold (0-1)
FRAME_EXTRACTION_FPS=2          # Frames per second to process

# Directories
UPLOAD_DIR=uploads
OUTPUT_DIR=output/detected
TEMP_DIR=temp
```

## 📊 Technical Details

### Models

1. **YOLOv8** - Person Detection
   - Architecture: CSPDarknet + PANet
   - Input: Any resolution (auto-resize)
   - Output: Bounding boxes with confidence scores
   - Speed: ~30 FPS on GPU, ~5 FPS on CPU

2. **ReID Model** - Person Recognition
   - Architecture: MobileNetV2 + ViT + ETFFM fusion
   - Input: 256x128 person crops
   - Output: 512-dim embeddings
   - Similarity: Cosine distance

### Pipeline

```
Input (Image/Video)
    ↓
Frame Extraction (2 FPS for videos)
    ↓
Image Enhancement (Gamma correction)
    ↓
YOLO Detection (Confidence > 0.75)
    ↓
Person Crops
    ↓
ReID Feature Extraction (512-dim)
    ↓
Cosine Similarity Matching (Threshold > 0.95)
    ↓
Results (with person IDs and confidence)
```

## 📚 Documentation

- **[STREAMLIT_UI_GUIDE.md](STREAMLIT_UI_GUIDE.md)** - Complete Streamlit UI guide
- **[SERVICE_GUIDE.md](SERVICE_GUIDE.md)** - Service.py documentation
- **[SOURCE_ANALYSIS.md](SOURCE_ANALYSIS.md)** - Full source code analysis
- **[STREAMLIT_STATUS.md](STREAMLIT_STATUS.md)** - Current setup status

## 🐛 Troubleshooting

### Models not found

```bash
# Check models folder
ls -la models/

# Models should be ~100MB (ReID) and ~20MB (YOLO)
```

### MongoDB connection error (FastAPI only)

```bash
# Install MongoDB
brew install mongodb-community  # macOS
# Or use Docker
docker run -d -p 27017:27017 -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=admin123 mongo
```

### Gallery not building

```bash
# Check images exist
ls -la known_gallery/*/

# Rebuild via Streamlit UI: Manage Gallery > Build Gallery
# Or use Python:
python -c "from service import ReIDPipelineService; s=ReIDPipelineService(); s.build_known_gallery('known_gallery', 'known_gallery/gallery_embeddings.npz')"
```

### Slow processing

```bash
# Use CPU mode
device = "cpu"  # In Settings or code

# Or reduce video FPS
FRAME_EXTRACTION_FPS=1  # Process 1 frame per second
```

## 🚀 Development

### Run Tests

```bash
# Test service
python service.py

# Or test via Streamlit UI
./run_ui.sh
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📦 Dependencies

Core:

- torch, torchvision - Deep learning
- ultralytics - YOLOv8
- timm - ReID models
- opencv-python - Image/video processing

API:

- fastapi, uvicorn - REST API
- motor, pymongo - MongoDB

UI:

- streamlit - Web interface
- plotly - Interactive charts
- pandas - Data handling

## 🎉 Credits

- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
- DINO ViT: [Facebook Research](https://github.com/facebookresearch/dino)
- Streamlit: [Streamlit.io](https://streamlit.io)

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Status:** ✅ Production Ready  
**Version:** 1.0.0  
**Last Updated:** March 8, 2026

For questions or issues, please check the documentation files above.

**Happy Coding!** 🚀
