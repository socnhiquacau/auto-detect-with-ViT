# DHT Documentation

Chào mừng đến với tài liệu hệ thống Video Person Detection & Recognition!

## 📚 Danh sách tài liệu

### Hướng dẫn chính

1. **[README.md](../README.md)** - Hướng dẫn tổng quan hệ thống
   - Quick start guide
   - Cài đặt và cấu hình
   - API endpoints
   - Troubleshooting

### Hướng dẫn chi tiết từng module

2. **[PERSON_EXTRACTION_GUIDE.md](PERSON_EXTRACTION_GUIDE.md)** - Person Extraction Module
   - Trích xuất hình ảnh người từ video
   - Motion detection & person detection
   - Duplicate removal
   - Batch processing
   - API reference chi tiết
   - Ví dụ nâng cao

3. **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** *(Coming soon)*
   - FastAPI endpoints reference
   - Request/Response schemas
   - Authentication & authorization
   - Rate limiting

4. **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** *(Coming soon)*
   - Production deployment
   - Docker setup
   - Nginx configuration
   - Monitoring & logging

## 🗂️ Cấu trúc dự án

```
DHT/
├── docs/                              # 📁 Documentation
│   ├── README.md                      # Index tài liệu
│   ├── PERSON_EXTRACTION_GUIDE.md     # Person extraction guide
│   └── ...
│
├── main.py                            # FastAPI application
├── video_processor.py                 # Video processing pipeline
├── person_extraction.py               # Person extraction module
├── feature_extractor.py               # Feature extraction
├── feature_comparison.py              # Person matching
├── image_enhancement.py               # Image preprocessing
├── database.py                        # MongoDB operations
├── common.py                          # Shared utilities
│
├── models/                            # AI Models
│   ├── last.pt                        # YOLO person detection
│   └── yolov8n.pt                     # Backup model
│
├── uploads/                           # Input videos
├── extracted_persons/                 # Person extraction output
├── temp/                              # Temporary files
└── output/                            # Detection results
```

## 🎯 Quick Links

### Bắt đầu nhanh
- [Quick Start Guide](../README.md#-quick-start)
- [Installation](../README.md#-cài-đặt)
- [Configuration](../README.md#-cấu-hình)

### API Usage
- [Upload & Process Video](../README.md#2-upload-video)
- [Get Results](../README.md#4-lấy-kết-quả)
- [Manage Known Persons](../README.md#7-quản-lý-database-người)

### Modules
- [Person Extraction](PERSON_EXTRACTION_GUIDE.md)
- [Video Processing](../README.md#-giới-thiệu)
- [Feature Extraction](../README.md#-giới-thiệu)

## 🔍 Tìm kiếm theo use case

### Use Case: Xử lý video từ API
👉 [API Usage Guide](../README.md#-sử-dụng-api)

### Use Case: Trích xuất dataset người từ video
👉 [Person Extraction Guide](PERSON_EXTRACTION_GUIDE.md)

### Use Case: Nhận diện người trong video
👉 [Main README - Video Processing](../README.md#-workflow-xử-lý)

### Use Case: Thêm người vào database
👉 [Add Known Persons](../README.md#7-quản-lý-database-người)

## 💡 Code Examples

### Example 1: Process video qua API

```python
import requests

# Upload video
with open('video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/process/upload',
        files={'file': f}
    )
    
result = response.json()
print(f"Job ID: {result['job_id']}")
```

### Example 2: Extract persons từ video

```python
from person_extraction import PersonExtractor

# Initialize
extractor = PersonExtractor(
    model_path="models/last.pt",
    confidence_threshold=0.5
)

# Process video
result = extractor.process_video("uploads/surveillance.mp4")
print(f"Extracted {result['total_saved_images']} persons")
```

### Example 3: Thêm known person vào database

```python
import requests

data = {
    "person_id": "john_doe_001",
    "name": "John Doe",
    "feature_vector": [0.123, 0.456, ...]  # 512-dim vector
}

response = requests.post(
    'http://localhost:8000/features/add',
    json=data
)
```

## 🛠️ Development

### Setup Development Environment

```bash
# 1. Clone repo
git clone <repo-url> DHT
cd DHT

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup MongoDB
mongosh
use admin
db.createUser({
  user: "admin",
  pwd: "admin123",
  roles: ["root"]
})

# 4. Create .env
cat > .env << EOF
MONGODB_URL=mongodb://admin:admin123@127.0.0.1:27017/video_detection_db?authSource=admin
DATABASE_NAME=video_detection_db
EOF

# 5. Run
python main.py
```

### Running Tests *(Coming soon)*

```bash
pytest tests/
```

### Contributing *(Coming soon)*

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📖 Glossary

**YOLO** - You Only Look Once, object detection algorithm  
**ViT** - Vision Transformer, deep learning model for image features  
**MOG2** - Mixture of Gaussians v2, background subtraction algorithm  
**Perceptual Hash** - Hash that identifies similar images  
**Feature Vector** - Numerical representation of image/face  
**Confidence Threshold** - Minimum confidence score for detection  

## 🔗 External Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

## 📝 License

*(Add your license information here)*

## 🤝 Support

Nếu gặp vấn đề:
1. Kiểm tra [Troubleshooting](../README.md#-xử-lý-lỗi-thường-gặp)
2. Review documentation tương ứng
3. Check error logs

---

**Last updated:** January 11, 2026
