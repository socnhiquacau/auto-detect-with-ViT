# 🎯 DHT - Person Detection & Recognition System

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)
![Status](https://img.shields.io/badge/status-production-brightgreen.svg)

**A complete AI-powered system for detecting and recognizing people in images and videos**

[Quick Start](#-quick-start) • [Features](#-features) • [Documentation](#-documentation) • [Demo](#-demo)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage-guide)
- [Interfaces](#-interfaces)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Technical Details](#-technical-details)
- [Documentation](#-documentation)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## 🌟 Overview

DHT is a production-ready person detection and re-identification system powered by state-of-the-art deep learning models:

- **YOLOv8** for fast and accurate person detection
- **ReID Model** (MobileNetV2 + ViT) for robust person recognition
- Multiple interfaces: Web UI, REST API, Python SDK
- Real-time video processing with tracking
- Easy-to-use gallery management

### Why DHT?

- ✅ **Zero Manual Setup** - Upload images directly via UI
- ✅ **Production Ready** - RESTful API with MongoDB backend
- ✅ **Easy to Use** - Beautiful Streamlit web interface
- ✅ **High Accuracy** - Advanced dual-branch ReID model
- ✅ **Flexible** - Use as API, UI, or Python library
- ✅ **Well Documented** - Comprehensive guides and examples

---

## ✨ Features

### Core Capabilities

- 🔍 **Person Detection** - YOLOv8 for accurate person detection in images & videos
- 👤 **Person Recognition** - ReID model with 512-dim embeddings for identity matching
- 🎬 **Video Processing** - Track persons across frames with temporal coherence
- 📸 **Image Processing** - Single image person recognition with Top-K results
- 📊 **Visualization** - Interactive charts, progress bars, and statistics
- 💾 **Export Results** - Download results as CSV or JSON

### Gallery Management

- 🆕 **Create New Person** - Upload multiple images at once
- 📸 **Add More Images** - Update existing persons with new photos
- 🔨 **Build Gallery** - Automatic feature extraction and indexing
- 👁️ **Preview Images** - View all gallery images with thumbnails
- 📊 **Statistics** - Track number of persons and images

### Three Interfaces

| Interface           | Best For       | Key Features                        |
| ------------------- | -------------- | ----------------------------------- |
| **🎨 Streamlit UI** | Demos, Testing | Visual interface, drag-drop, charts |
| **📡 FastAPI**      | Production     | REST API, MongoDB, scalable         |
| **💻 Python SDK**   | Scripting      | Direct access, batch processing     |

---

## 🚀 Quick Start

### Prerequisites

```bash
# System Requirements
Python 3.9+
CUDA 11.8+ (optional, for GPU)
~500 MB disk space for models
```

### 1. Clone & Install

```bash
# Clone repository
git clone <repo-url>
cd DHT

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Models

Download and place in `models/` folder:

- `yolov8_person_detection.pt` (~25 MB)
- `best_model_state_dict.pth` (~88 MB)

### 3. Launch

```bash
# Streamlit UI (Recommended)
./run_ui.sh
# Opens: http://localhost:8501

# Or FastAPI
uvicorn main:app --reload
# Opens: http://localhost:8000/docs
```

### 4. Add Known Persons

**Via UI (Easy):**

1. Go to "👥 Manage Gallery" page
2. Click **"🆕 Create New Person"** tab
3. Enter name: `john_doe`
4. Upload 5-10 photos (Ctrl+Click for multiple)
5. Click **"🚀 Create & Save"**
6. Click **"🔨 Build Gallery"** to finalize

**Via Manual (Advanced):**

```bash
mkdir -p known_gallery/john_doe
cp photos/*.jpg known_gallery/john_doe/
# Then build via UI or Python
```

### 5. Process & Recognize

**Images:**

1. Go to "📸 Process Image"
2. Upload image
3. View results with confidence scores

**Videos:**

1. Go to "🎬 Process Video"
2. Upload video file
3. Wait for processing
4. Download CSV results

---

## 📦 Installation

### Option A: Pip Install (Recommended)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import streamlit; import ultralytics; print('✅ All dependencies installed')"
```

### Option B: Docker (Coming Soon)

```bash
docker-compose up -d
```

### Dependencies Overview

```yaml
Core:
  - torch>=2.0.0 # Deep learning framework
  - torchvision # Vision utilities
  - ultralytics # YOLOv8
  - timm # Vision models (ViT)
  - opencv-python # Image/video processing

API:
  - fastapi # REST API framework
  - uvicorn # ASGI server
  - motor # Async MongoDB driver
  - pymongo # MongoDB sync driver

UI:
  - streamlit>=1.28.0 # Web interface
  - plotly # Interactive charts
  - pandas # Data manipulation
  - pillow # Image handling
```

---

## 📖 Usage Guide

### 🎨 Streamlit UI (Recommended for Beginners)

#### Launch UI

```bash
./run_ui.sh
# Or: streamlit run streamlit_app.py
```

#### Features

- **🏠 Home** - Overview and quick stats
- **📸 Process Image** - Upload image, detect & recognize persons
- **🎬 Process Video** - Upload video, get tracking results
- **👥 Manage Gallery** - Create persons, upload images, build embeddings
- **⚙️ Settings** - Configure models, thresholds, device

#### Gallery Management Workflow

**Create New Person:**

```
1. Navigate to 👥 Manage Gallery
2. Select tab: 🆕 Create New Person
3. Enter name: "cristiano_ronaldo"
4. Click "Select images" button
5. Choose 5-10 photos (Ctrl/Cmd + Click)
6. Click 🚀 Create & Save
7. Click 🔨 Build Gallery
```

**Add More Images:**

```
1. Tab: 📸 Add Images to Existing Person
2. Select person from dropdown
3. Upload more photos
4. Click 💾 Save Images
5. Click 🔨 Build Gallery to update
```

#### Processing Workflow

**Image Recognition:**

```
1. Go to 📸 Process Image
2. Upload image file
3. Select Top K (number of matches)
4. Click 🔍 Detect & Recognize
5. View results:
   - Left: Uploaded image
   - Right: Top matches with confidence
   - Click "🔍 View Full Size" to enlarge
6. Scroll down for similarity chart
```

**Video Processing:**

```
1. Go to 🎬 Process Video
2. Upload .mp4/.avi/.mov file
3. Select Top K results
4. Click 🎬 Process Video
5. Wait for processing (shows progress)
6. View summary statistics
7. Browse detection table
8. Download results as CSV
```

[📖 Full UI Guide](Docs/STREAMLIT_UI_GUIDE.md)

---

### 📡 FastAPI REST API (Production)

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

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/DHT.git
cd DHT

# Create branch
git checkout -b feature/your-feature

# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Make changes and test
python service.py
./run_ui.sh

# Format code
black .
flake8 .

# Commit and push
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature

# Create Pull Request
```

### Run Tests

```bash
# Test service
python service.py

# Test UI
./run_ui.sh

# Test API
uvicorn main:app --reload
curl http://localhost:8000/health
```

### Debug Mode

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Run your code
from service import ReIDPipelineService
service = ReIDPipelineService()
```

---

## 📦 Dependencies

### Core Libraries

```yaml
Deep Learning:
  - torch>=2.0.0 # PyTorch framework
  - torchvision # Vision utilities
  - ultralytics # YOLOv8
  - timm # Vision Transformers

Image/Video:
  - opencv-python # Image/video processing
  - pillow # Image handling
  - numpy # Array operations

Web Framework:
  - fastapi # REST API
  - uvicorn # ASGI server
  - streamlit>=1.28.0 # Web UI

Database:
  - motor # Async MongoDB
  - pymongo # Sync MongoDB

Visualization:
  - plotly # Interactive charts
  - pandas # Data manipulation
  - matplotlib # Plotting

Utilities:
  - python-dotenv # Environment variables
  - requests # HTTP client
  - tqdm # Progress bars
```

### Installation

```bash
# All dependencies
pip install -r requirements.txt

# Core only (minimal)
pip install -r requirements-core.txt

# With GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## 🎉 Credits & Acknowledgments

### Technologies

- **YOLOv8** - [Ultralytics](https://github.com/ultralytics/ultralytics)
- **DINO ViT** - [Facebook Research](https://github.com/facebookresearch/dino)
- **Streamlit** - [Streamlit.io](https://streamlit.io)
- **FastAPI** - [FastAPI](https://fastapi.tiangolo.com)
- **PyTorch** - [PyTorch](https://pytorch.org)

### Research Papers

```bibtex
@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics}
}

@article{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and others},
  journal={ICCV},
  year={2021}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

**For educational and research purposes.**

---

## 🤝 Contributing

We welcome contributions! Here's how:

### Guidelines

- ✅ Follow PEP 8 style guide
- ✅ Add docstrings to functions
- ✅ Update documentation
- ✅ Test your changes
- ✅ Write clear commit messages

### Areas for Contribution

- 🔨 Add unit tests
- 📝 Improve documentation
- 🚀 Optimize performance
- 🎨 Enhance UI/UX
- 🐛 Fix bugs
- ✨ Add new features

### Steps

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## 📞 Contact & Support

- 📧 **Email:** support@example.com
- 🐛 **Issues:** [GitHub Issues](https://github.com/yourusername/DHT/issues)
- 📖 **Docs:** [Full Documentation](Docs/COMPLETE_DOCUMENTATION.md)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/yourusername/DHT/discussions)

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

---

<div align="center">

### 🎯 DHT - Person Detection & Recognition System

**Made with ❤️ by DHT Team**

![Status](https://img.shields.io/badge/status-production_ready-brightgreen.svg)
![Update](https://img.shields.io/badge/last_update-March_2026-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)

---

**⭐ Star this repo if you find it useful!**

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🐛 Report Bug](https://github.com/yourusername/DHT/issues) • [✨ Request Feature](https://github.com/yourusername/DHT/issues)

---

**Status:** ✅ Production Ready  
**Version:** 1.0.0  
**Last Updated:** March 8, 2026

</div>
