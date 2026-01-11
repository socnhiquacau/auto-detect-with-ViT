# Person Extraction Guide

## 📋 Mục lục

1. [Giới thiệu](#giới-thiệu)
2. [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
3. [Cài đặt và Cấu hình](#cài-đặt-và-cấu-hình)
4. [Sử dụng](#sử-dụng)
5. [API Reference](#api-reference)
6. [Ví dụ nâng cao](#ví-dụ-nâng-cao)
7. [Tối ưu hóa](#tối-ưu-hóa)
8. [Xử lý lỗi](#xử-lý-lỗi)

---

## Giới thiệu

**PersonExtractor** là module chuyên dụng để phát hiện và trích xuất hình ảnh người từ video. Module này sử dụng kết hợp motion detection và YOLO để tối ưu hiệu suất xử lý.

### Tính năng chính

✅ **Motion Detection** - Chỉ xử lý frames có chuyển động  
✅ **Person Detection** - Phát hiện người với YOLO (model `last.pt`)  
✅ **Duplicate Removal** - Tự động loại bỏ ảnh trùng lặp với perceptual hashing  
✅ **Batch Processing** - Xử lý nhiều video tự động  
✅ **Structured Output** - Tổ chức output theo từng video với naming convention chuẩn  

### Use Cases

- 🎥 Trích xuất dataset người từ video surveillance
- 👥 Tạo dataset cho training face recognition
- 📊 Phân tích người trong video (counting, tracking)
- 🔍 Tìm kiếm frames chứa người trong video dài

---

## Kiến trúc hệ thống

### Pipeline xử lý

```
Video Input (uploads/)
    ↓
Frame-by-Frame Reading
    ↓
Motion Detection (MOG2)
    ↓ (chỉ xử lý nếu có motion)
Person Detection (YOLO)
    ↓
Bounding Box Extraction
    ↓
Perceptual Hashing
    ↓ (skip nếu duplicate)
Save to Disk (extracted_persons/)
```

### Components

#### 1. Motion Detector (MOG2)
- **Mục đích**: Giảm số lượng frames cần xử lý bằng YOLO
- **Thuật toán**: Background Subtraction (MOG2)
- **Threshold**: 1% pixels có chuyển động

```python
self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,        # Số frames để học background
    varThreshold=16,    # Ngưỡng phát hiện foreground
    detectShadows=True  # Phát hiện bóng đổ
)
```

#### 2. Person Detector (YOLO)
- **Model**: `models/last.pt` (custom trained hoặc YOLOv8)
- **Class**: 0 (person only)
- **Confidence**: Configurable (default: 0.5)

#### 3. Duplicate Detector (Perceptual Hash)
- **Thuật toán**: Average Hash (aHash)
- **Size**: 16x16 pixels
- **Hash**: MD5 của binary array

```python
# Resize → Grayscale → Compare with average → Hash
small = cv2.resize(image, (16, 16))
avg = small.mean()
diff = small > avg
hash_str = hashlib.md5(diff.tobytes()).hexdigest()
```

### Output Structure

```
extracted_persons/
├── video1/
│   ├── video1_000100_0001.png  # frame_100, person_1
│   ├── video1_000100_0002.png  # frame_100, person_2
│   ├── video1_000245_0001.png  # frame_245, person_1
│   └── ...
├── video2/
│   └── ...
└── extraction_results.json     # Metadata
```

**Naming Convention**: `{video_id}_{frame_id:06d}_{img_id:04d}.png`

---

## Cài đặt và Cấu hình

### Requirements

```bash
pip install opencv-python
pip install ultralytics
pip install numpy
```

### Cấu trúc thư mục

```
DHT/
├── person_extraction.py
├── models/
│   └── last.pt              # YOLO model (BẮT BUỘC)
├── uploads/                  # Input videos (tạo tự động)
└── extracted_persons/        # Output images (tạo tự động)
```

### Configuration Parameters

| Parameter | Type | Default | Mô tả |
|-----------|------|---------|-------|
| `model_path` | str | `"models/last.pt"` | Đường dẫn YOLO model |
| `uploads_folder` | str | `"uploads"` | Folder chứa videos input |
| `output_folder` | str | `"extracted_persons"` | Folder lưu ảnh output |
| `confidence_threshold` | float | `0.5` | Ngưỡng confidence YOLO (0-1) |
| `motion_threshold` | int | `25` | Ngưỡng motion detection (0-255) |

---

## Sử dụng

### 1. Quick Start

**Chuẩn bị:**
```bash
# 1. Đặt video vào uploads/
cp your_video.mp4 uploads/

# 2. Đảm bảo có model
ls models/last.pt

# 3. Chạy
python person_extraction.py
```

**Output:**
```
🚀 Person Extraction from Videos
============================================================
🎯 Found 1 video(s) to process

============================================================
🎬 Processing video: your_video
============================================================
📊 Video info:
   - FPS: 30.00
   - Total frames: 1500
   - Duration: 50.00s
   📸 Saved 10 images... (20.0%)
   📸 Saved 20 images... (40.0%)
   ...
✅ Processing complete!
   - Total frames: 1500
   - Frames with motion: 450
   - Frames with persons: 120
   - Total saved images: 85
   - Duplicates skipped: 35
   - Output folder: extracted_persons/your_video
```

### 2. Python API Usage

#### Basic Usage

```python
from person_extraction import PersonExtractor

# Initialize
extractor = PersonExtractor(
    model_path="models/last.pt",
    uploads_folder="uploads",
    output_folder="extracted_persons",
    confidence_threshold=0.5
)

# Process all videos
results = extractor.process_all_videos()

# Check results
for result in results:
    print(f"Video: {result['video_name']}")
    print(f"Images: {result['total_saved_images']}")
```

#### Process Single Video

```python
# Process một video cụ thể
result = extractor.process_video("uploads/my_video.mp4")

if result:
    print(f"Saved {result['total_saved_images']} images")
    print(f"Output: {result['output_folder']}")
```

#### Custom Configuration

```python
# Tăng sensitivity
extractor = PersonExtractor(
    confidence_threshold=0.3,  # Nhận nhiều detections hơn
    motion_threshold=15        # Nhạy với motion hơn
)

# Giảm false positives
extractor = PersonExtractor(
    confidence_threshold=0.7,  # Chỉ lấy detections chắc chắn
    motion_threshold=40        # Chỉ xử lý motion rõ ràng
)
```

---

## API Reference

### Class: `PersonExtractor`

#### `__init__()`

```python
def __init__(
    self,
    model_path: str = "models/last.pt",
    uploads_folder: str = "uploads",
    output_folder: str = "extracted_persons",
    confidence_threshold: float = 0.5,
    motion_threshold: int = 25
)
```

**Parameters:**
- `model_path`: Path to YOLO model file
- `uploads_folder`: Folder containing input videos
- `output_folder`: Folder to save extracted images
- `confidence_threshold`: Minimum confidence for person detection (0.0-1.0)
- `motion_threshold`: Pixel intensity threshold for motion (0-255)

**Example:**
```python
extractor = PersonExtractor(
    model_path="models/custom_yolo.pt",
    confidence_threshold=0.6
)
```

---

#### `process_video()`

```python
def process_video(self, video_path: str) -> Optional[Dict]
```

Xử lý một video và trích xuất tất cả hình ảnh người.

**Parameters:**
- `video_path` (str): Path to video file

**Returns:**
- `Dict` hoặc `None`: Dictionary chứa thông tin xử lý

**Return Dictionary Structure:**
```python
{
    'video_id': str,              # Video identifier
    'video_name': str,            # Video filename (no extension)
    'video_path': str,            # Full path to video
    'output_folder': str,         # Output directory path
    'total_frames': int,          # Total frames in video
    'motion_frames': int,         # Frames with detected motion
    'person_detections': int,     # Frames with persons detected
    'total_saved_images': int,    # Number of saved images
    'duplicate_count': int,       # Number of duplicates skipped
    'saved_images': List[Dict],   # List of saved image info
    'processed_at': str           # ISO timestamp
}
```

**Example:**
```python
result = extractor.process_video("uploads/surveillance_cam1.mp4")

if result:
    print(f"Processed: {result['video_name']}")
    print(f"Extracted: {result['total_saved_images']} images")
    print(f"Location: {result['output_folder']}")
else:
    print("Failed to process video")
```

---

#### `process_all_videos()`

```python
def process_all_videos(self) -> List[Dict]
```

Tự động xử lý tất cả videos trong `uploads_folder`.

**Supported Formats:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`

**Returns:**
- `List[Dict]`: List of results từ `process_video()` cho mỗi video

**Example:**
```python
results = extractor.process_all_videos()

for result in results:
    print(f"\nVideo: {result['video_name']}")
    print(f"  - Total frames: {result['total_frames']}")
    print(f"  - Motion frames: {result['motion_frames']}")
    print(f"  - Saved images: {result['total_saved_images']}")
    print(f"  - Duplicates: {result['duplicate_count']}")
```

---

#### `detect_motion()`

```python
def detect_motion(self, frame: np.ndarray) -> bool
```

Phát hiện chuyển động trong frame sử dụng Background Subtraction.

**Parameters:**
- `frame` (np.ndarray): Frame để kiểm tra

**Returns:**
- `bool`: `True` nếu có motion, `False` nếu không

**Logic:**
- Tính motion_ratio = motion_pixels / total_pixels
- Return `True` nếu motion_ratio > 0.01 (1%)

**Example:**
```python
cap = cv2.VideoCapture("video.mp4")
ret, frame = cap.read()

if extractor.detect_motion(frame):
    print("Motion detected!")
```

---

#### `detect_persons()`

```python
def detect_persons(self, frame: np.ndarray) -> List[Dict]
```

Phát hiện người trong frame sử dụng YOLO model.

**Parameters:**
- `frame` (np.ndarray): Frame để phát hiện

**Returns:**
- `List[Dict]`: List of detections

**Detection Dictionary:**
```python
{
    'bbox': [x1, y1, x2, y2],  # Bounding box coordinates
    'confidence': float         # Confidence score (0-1)
}
```

**Example:**
```python
detections = extractor.detect_persons(frame)

for det in detections:
    x1, y1, x2, y2 = det['bbox']
    conf = det['confidence']
    print(f"Person at ({x1},{y1})-({x2},{y2}) conf={conf:.2f}")
```

---

#### `extract_person_crop()`

```python
def extract_person_crop(
    self,
    frame: np.ndarray,
    bbox: List[int]
) -> np.ndarray
```

Crop hình ảnh người từ frame theo bounding box.

**Parameters:**
- `frame` (np.ndarray): Frame gốc
- `bbox` (List[int]): Bounding box [x1, y1, x2, y2]

**Returns:**
- `np.ndarray`: Cropped image

**Features:**
- Tự động clamp bbox vào kích thước frame
- Đảm bảo coordinates không âm

**Example:**
```python
bbox = [100, 50, 200, 300]  # x1, y1, x2, y2
person_img = extractor.extract_person_crop(frame, bbox)

# Save cropped image
cv2.imwrite("person.png", person_img)
```

---

#### `compute_image_hash()`

```python
def compute_image_hash(self, image: np.ndarray) -> str
```

Tính perceptual hash của ảnh để phát hiện duplicates.

**Parameters:**
- `image` (np.ndarray): Image để hash

**Returns:**
- `str`: Hash string (MD5 hex)

**Algorithm:**
1. Resize về 16x16 pixels
2. Convert sang grayscale
3. So sánh với average brightness
4. Hash binary array với MD5

**Example:**
```python
hash1 = extractor.compute_image_hash(image1)
hash2 = extractor.compute_image_hash(image2)

if hash1 == hash2:
    print("Images are duplicates!")
```

---

## Ví dụ nâng cao

### 1. Custom Processing Pipeline

```python
import cv2
from person_extraction import PersonExtractor

class CustomExtractor(PersonExtractor):
    """Extend PersonExtractor với custom logic"""
    
    def detect_persons(self, frame):
        """Override để thêm filtering"""
        detections = super().detect_persons(frame)
        
        # Filter: chỉ lấy detections có bbox đủ lớn
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1
            
            # Bỏ qua bbox quá nhỏ (< 50x100)
            if width >= 50 and height >= 100:
                filtered.append(det)
        
        return filtered

# Sử dụng
extractor = CustomExtractor()
results = extractor.process_all_videos()
```

### 2. Real-time Processing Monitor

```python
import time
from person_extraction import PersonExtractor

class MonitoredExtractor(PersonExtractor):
    """Extractor với real-time monitoring"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = None
        
    def process_video(self, video_path):
        self.start_time = time.time()
        result = super().process_video(video_path)
        
        if result:
            elapsed = time.time() - self.start_time
            fps_processed = result['total_frames'] / elapsed
            
            print(f"\n⏱️  Performance:")
            print(f"   - Processing time: {elapsed:.2f}s")
            print(f"   - Speed: {fps_processed:.2f} fps")
            print(f"   - Images/second: {result['total_saved_images']/elapsed:.2f}")
        
        return result

# Sử dụng
extractor = MonitoredExtractor()
extractor.process_video("uploads/long_video.mp4")
```

### 3. Batch Processing với Progress Bar

```python
from person_extraction import PersonExtractor
from pathlib import Path
from tqdm import tqdm

extractor = PersonExtractor()

# Lấy danh sách videos
video_files = list(Path("uploads").glob("*.mp4"))

# Process với progress bar
results = []
for video_path in tqdm(video_files, desc="Processing videos"):
    result = extractor.process_video(str(video_path))
    if result:
        results.append(result)

# Summary
total_images = sum(r['total_saved_images'] for r in results)
print(f"\n✅ Extracted {total_images} images from {len(results)} videos")
```

### 4. Filter by Time Range

```python
def process_video_time_range(extractor, video_path, start_sec, end_sec):
    """Process chỉ một đoạn của video"""
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    # Set vị trí start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    saved_images = []
    frame_id = start_frame
    
    while frame_id < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        if extractor.detect_motion(frame):
            detections = extractor.detect_persons(frame)
            for det in detections:
                person_img = extractor.extract_person_crop(frame, det['bbox'])
                # Save logic...
                
        frame_id += 1
    
    cap.release()
    return saved_images

# Sử dụng: chỉ process từ giây 60-120
extractor = PersonExtractor()
images = process_video_time_range(
    extractor, 
    "uploads/video.mp4", 
    start_sec=60, 
    end_sec=120
)
```

### 5. Multi-threaded Processing

```python
from concurrent.futures import ThreadPoolExecutor
from person_extraction import PersonExtractor
from pathlib import Path

def process_single_video(video_path):
    """Process một video - thread-safe"""
    extractor = PersonExtractor()  # Mỗi thread có extractor riêng
    return extractor.process_video(str(video_path))

# Lấy danh sách videos
video_files = list(Path("uploads").glob("*.mp4"))

# Process song song với 4 workers
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_single_video, video_files))

# Filter None results
results = [r for r in results if r is not None]

print(f"✅ Processed {len(results)} videos in parallel")
```

---

## Tối ưu hóa

### 1. Performance Tuning

#### Giảm số frames xử lý

```python
# Option 1: Tăng motion_threshold
extractor = PersonExtractor(motion_threshold=50)  # Chỉ detect motion mạnh

# Option 2: Skip frames
class SkipFrameExtractor(PersonExtractor):
    def __init__(self, *args, skip_frames=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_frames = skip_frames
        self.frame_counter = 0
    
    def detect_motion(self, frame):
        self.frame_counter += 1
        if self.frame_counter % self.skip_frames != 0:
            return False  # Skip frame
        return super().detect_motion(frame)

# Chỉ xử lý 1/3 frames
extractor = SkipFrameExtractor(skip_frames=3)
```

#### GPU Acceleration

```python
# Đảm bảo YOLO chạy trên GPU
extractor = PersonExtractor(model_path="models/last.pt")

# Check GPU
import torch
if torch.cuda.is_available():
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  Running on CPU")
```

### 2. Memory Optimization

```python
# Xóa background subtractor sau mỗi video để free memory
class MemoryEfficientExtractor(PersonExtractor):
    def process_video(self, video_path):
        result = super().process_video(video_path)
        
        # Reset để free memory
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        
        return result
```

### 3. Disk I/O Optimization

```python
# Batch write để giảm I/O operations
import numpy as np

class BatchWriteExtractor(PersonExtractor):
    def __init__(self, *args, batch_size=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.batch_buffer = []
    
    def _save_person_detection(self, *args, **kwargs):
        # Buffer images thay vì write ngay
        result = super()._save_person_detection(*args, **kwargs)
        
        if result:
            self.batch_buffer.append(result)
            
            # Flush khi đủ batch
            if len(self.batch_buffer) >= self.batch_size:
                self.flush_batch()
        
        return result
    
    def flush_batch(self):
        # Write batch
        print(f"💾 Writing batch of {len(self.batch_buffer)} images...")
        self.batch_buffer = []
```

---

## Xử lý lỗi

### 1. Video không mở được

**Lỗi:**
```
❌ Cannot open video: uploads/corrupted.mp4
```

**Nguyên nhân:**
- File video bị corrupt
- Codec không được hỗ trợ
- File path sai

**Giải pháp:**
```bash
# Kiểm tra file tồn tại
ls -lh uploads/video.mp4

# Test với ffmpeg
ffmpeg -i uploads/video.mp4 -f null -

# Convert sang format tương thích
ffmpeg -i input.mov -c:v libx264 -c:a aac output.mp4
```

### 2. Model không tìm thấy

**Lỗi:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/last.pt'
```

**Giải pháp:**
```bash
# Kiểm tra model exists
ls models/last.pt

# Tải model nếu thiếu (YOLOv8)
# from ultralytics import YOLO
# model = YOLO("yolov8n.pt")  # Tải pretrained
# model.save("models/last.pt")
```

### 3. CUDA Out of Memory

**Lỗi:**
```
RuntimeError: CUDA out of memory
```

**Giải pháp:**
```python
# Option 1: Force CPU
import torch
torch.cuda.is_available = lambda: False

# Option 2: Reduce batch size (nếu customize inference)
# Option 3: Process smaller video segments
```

### 4. Không phát hiện được người

**Vấn đề:** `total_saved_images: 0`

**Debug:**
```python
# Test detection trên single frame
import cv2

cap = cv2.VideoCapture("uploads/video.mp4")
ret, frame = cap.read()

# Test motion detection
has_motion = extractor.detect_motion(frame)
print(f"Motion detected: {has_motion}")

# Test person detection
detections = extractor.detect_persons(frame)
print(f"Persons detected: {len(detections)}")

# Thử giảm threshold
extractor.confidence_threshold = 0.3
detections = extractor.detect_persons(frame)
print(f"With lower threshold: {len(detections)}")
```

### 5. Too many duplicates

**Vấn đề:** `duplicate_count` rất cao

**Giải pháp:**
```python
# Tăng độ nhạy của hash bằng cách resize lớn hơn
class SensitiveHashExtractor(PersonExtractor):
    def compute_image_hash(self, image):
        # Resize 32x32 thay vì 16x16
        small = cv2.resize(image, (32, 32))
        
        if len(small.shape) == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        avg = small.mean()
        diff = small > avg
        
        import hashlib
        return hashlib.md5(diff.tobytes()).hexdigest()
```

### 6. Slow processing

**Vấn đề:** Processing quá chậm

**Checklist:**
```python
# 1. Check GPU usage
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# 2. Monitor CPU usage
# htop or Task Manager

# 3. Reduce resolution
class LowResExtractor(PersonExtractor):
    def process_video(self, video_path):
        # Override để resize frames
        pass  # Implement resize logic

# 4. Skip more frames
extractor = PersonExtractor(motion_threshold=50)  # Process ít frames hơn
```

---

## Best Practices

### 1. Naming và Organization

```python
# Tốt: Video ID có timestamp
video_id = f"{video_name}_{int(time.time())}"

# Tốt: Metadata file kèm images
import json
metadata = {
    'extraction_date': datetime.now().isoformat(),
    'model_version': 'last.pt',
    'settings': {
        'confidence_threshold': 0.5,
        'motion_threshold': 25
    }
}
with open(f"{output_folder}/metadata.json", 'w') as f:
    json.dump(metadata, f)
```

### 2. Error Handling

```python
# Wrap trong try-except
def safe_process_video(extractor, video_path):
    try:
        result = extractor.process_video(video_path)
        return result
    except Exception as e:
        print(f"❌ Error processing {video_path}: {e}")
        return None

# Log errors
import logging
logging.basicConfig(filename='extraction.log', level=logging.ERROR)

try:
    extractor.process_all_videos()
except Exception as e:
    logging.error(f"Processing failed: {e}", exc_info=True)
```

### 3. Resource Cleanup

```python
# Đảm bảo release resources
import atexit

def cleanup():
    print("🧹 Cleaning up...")
    # Close open files, connections, etc.

atexit.register(cleanup)
```

---

## FAQ

**Q: Model last.pt là gì?**  
A: Đây là custom YOLO model được train để detect persons. Có thể thay bằng YOLOv8 pretrained hoặc custom model.

**Q: Tại sao cần motion detection?**  
A: Để giảm số frames cần xử lý bằng YOLO (chỉ xử lý frames có chuyển động), tăng tốc xử lý đáng kể.

**Q: Làm sao để detect faces thay vì persons?**  
A: Thay YOLO person detection bằng face detection model (MTCNN, RetinaFace, etc.).

**Q: Có thể xử lý real-time từ webcam?**  
A: Có, thay `cv2.VideoCapture(file)` bằng `cv2.VideoCapture(0)` cho webcam.

**Q: Perceptual hash có thể bỏ qua không?**  
A: Có thể comment out hash check, nhưng sẽ có nhiều duplicates trong output.

---

## Changelog

### Version 1.0 (Current)
- ✅ Motion detection với MOG2
- ✅ Person detection với YOLO
- ✅ Duplicate removal với perceptual hash
- ✅ Batch processing
- ✅ Structured output với naming convention

### Roadmap
- [ ] Face detection và cropping
- [ ] Real-time webcam processing
- [ ] GPU optimization
- [ ] Web UI for monitoring
- [ ] Export to dataset formats (COCO, YOLO, etc.)

---

## Support

Nếu gặp vấn đề:
1. Kiểm tra [Xử lý lỗi](#xử-lý-lỗi)
2. Review [Best Practices](#best-practices)
3. Check logs và error messages

---

**Happy extracting! 🎬👥**
