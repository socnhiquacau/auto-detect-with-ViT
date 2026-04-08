# Source Documentation (Streamlit-Only)

## 1. Muc tieu
Tai lieu nay mo ta codebase hien tai sau khi loai bo FastAPI/API va phan DB lien quan.

Phu hop cho:
- onboarding thanh vien moi
- tra cuu nhanh luong xu ly UI
- bao tri pipeline ReID

## 2. Cau truc thu muc

```text
DHT/
|-- streamlit_app.py
|-- run_ui.sh
|-- source_documentation.md
|-- src/
|   |-- __init__.py
|   |-- core/
|   |   |-- __init__.py
|   |   `-- common.py
|   `-- pipeline/
|       |-- __init__.py
|       |-- image_enhancement.py
|       |-- feature_extractor.py
|       |-- feature_comparison.py
|       `-- service.py
|-- scripts/
|   |-- check_model.py
|   `-- make_mevid_subset_fixed.py
|-- docs/
|-- models/
|-- known_gallery/
|-- result_frame/
|-- output/
|-- temp/
`-- uploads/
```

## 3. Entry points

### 3.1 Streamlit UI
- File: `streamlit_app.py`
- Chuc nang:
- Giao dien thao tac cho xu ly anh/video
- Quan ly thu vien `known_gallery`
- Su dung `ReIDPipelineService` tu `src/pipeline/service.py`

### 3.2 Python service (standalone)
- File: `src/pipeline/service.py`
- Chuc nang:
- Pipeline nhan dien end-to-end
- Track person trong video
- Chon frame chat luong tot nhat cho moi track
- Embed va compare voi gallery

## 4. Module core (`src/core`)

### 4.1 `common.py`
- Vai tro: Utility + data/model loader dung chung.
- Noi dung chinh:
- Config preprocess (`TARGET_SIZE`, `MEAN`, `STD`)
- `find_model_file()` tim model trong `models/`
- `load_model_from_models()` load model tu local file
- `DataLoader`:
- `get_yolo_model()`
- `get_feature_extractor()`
- `preprocess_image()/preprocess_pil()/preprocess_cv2()`

## 5. Module pipeline (`src/pipeline`)

### 5.1 `image_enhancement.py`
- Class `ImageEnhancer` cho enhancement anh.
- Cung cap wrapper module-level `enhance(image)`.

### 5.2 `feature_extractor.py`
- Class `FeatureExtractor`.
- Load model thong qua `src.core.common.load_model_from_models`.
- Tra feature vector da normalize (L2).

### 5.3 `feature_comparison.py`
- Tool test nhanh so sanh 2 anh bang cosine similarity.

### 5.4 `service.py`
- Module chinh cho nhan dien va tracking.
- Classes:
- `TrackCrop`
- `ETFFM_Lite`
- `ReIDModel`
- `KnownGalleryManager`
- `ReIDPipelineService`

- Ham quan trong cua `ReIDPipelineService`:
- tracking: `track_persons()`, `_track_persons_ultralytics()`, `_track_persons_iou_fallback()`
- scoring: `_quality_score()`, `select_best_frames()`
- embedding: `embed_image()`, `embed_crop()`
- gallery: `load_known_gallery()`, `build_known_gallery()`, `compare_with_gallery()`
- API su dung truc tiep tu UI: `process_query_image()`, `process_video()`, `format_video_result()`

## 6. Scripts (`scripts/`)

### 6.1 `check_model.py`
- Script inspect noi dung checkpoint model `.pth`.

### 6.2 `make_mevid_subset_fixed.py`
- Tao subset MEVID theo rule open-set.

## 7. Luong du lieu UI

### 7.1 Luong `streamlit_app.py` + `ReIDPipelineService`
1. Khoi tao service.
2. Load gallery embeddings (`known_gallery/gallery_embeddings.npz`).
3. Xu ly anh/video tu giao dien.
4. Hien thi top-k, thong ke, va anh track.
5. Build/reload gallery tu `known_gallery/`.

## 8. Thu muc du lieu runtime
- `models/`: trong so model (`.pt`, `.pth`)
- `known_gallery/`: gallery theo person + `gallery_embeddings.npz`
- `uploads/`: tep upload
- `temp/`: tep tam
- `result_frame/`: anh best-frame moi track
- `output/`: output phu tro

## 9. Bien moi truong quan trong
- `YOLO_MODEL_PATH`
- `VIT_MODEL_PATH`

## 10. Cach chay nhanh

### 10.1 UI
```bash
streamlit run streamlit_app.py
```

### 10.2 Service demo
```bash
python -m src.pipeline.service
```

### 10.3 Scripts
```bash
python scripts/check_model.py
python scripts/make_mevid_subset_fixed.py
```

