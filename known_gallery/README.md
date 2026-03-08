# Known Gallery - Person Images

## Cấu trúc thư mục

```
known_gallery/
├── person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
├── person2/
│   ├── photo1.jpg
│   └── photo2.jpg
└── ...
```

## Hướng dẫn sử dụng

### 1. Thêm ảnh người cần nhận dạng

- Tạo một folder cho mỗi người (tên folder = ID người)
- Thêm nhiều ảnh của người đó vào folder
- Ảnh nên:
  - Chứa rõ khuôn mặt/toàn thân
  - Chất lượng tốt, không bị mờ
  - Nhiều góc độ khác nhau
  - Format: `.jpg`, `.png`, `.jpeg`

### 2. Build gallery embeddings

Chạy service.py lần đầu:

```bash
python service.py
```

Hoặc trong code Python:

```python
from service import ReIDPipelineService

service = ReIDPipelineService()
service.build_known_gallery("known_gallery", "known_gallery/gallery_embeddings.npz")
```

### 3. Sử dụng

Sau khi build gallery, bạn có thể:

**Process một ảnh:**

```python
service.load_known_gallery("known_gallery/gallery_embeddings.npz")
result = service.process_query_image("test_image.jpg", topk=5)
print(result)
```

**Process video:**

```python
service.load_known_gallery("known_gallery/gallery_embeddings.npz")
result = service.process_video("video.mp4", topk=5)
formatted = service.format_video_result(result)
print(formatted)
```

## Output Format

Gallery embeddings được lưu trong file `.npz`:

- `embeddings`: Feature vectors (N x D)
- `person_ids`: Person IDs (N)
- `image_paths`: Đường dẫn ảnh gốc (N)

## Notes

- Model sử dụng: ReIDModel (MobileNetV2 + ViT + ETFFM fusion)
- Feature dimension: 512-dim
- Similarity threshold mặc định: 0.8
- Detection: YOLOv8

## Khác biệt với main.py

`service.py` sử dụng **ReID model phức tạp hơn** với:

- ✅ Dual-branch architecture (CNN + Transformer)
- ✅ ETFFM feature fusion
- ✅ Person tracking trong video
- ✅ Quality-based crop selection
- ✅ Gallery management (.npz format)

`main.py` sử dụng **DINO ViT đơn giản hơn** với:

- ✅ Single ViT model
- ✅ MongoDB storage
- ✅ REST API endpoints
- ✅ FastAPI web service
