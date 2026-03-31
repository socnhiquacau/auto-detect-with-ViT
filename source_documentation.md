# Source Documentation (Refactored)

## 1. Muc tieu
Tai lieu nay mo ta codebase hien tai sau khi tach module theo `src/core`, `src/pipeline`, `scripts`.

Phu hop cho:
- onboarding thanh vien moi
- tra cuu nhanh luong xu ly
- bao tri va mo rong tinh nang

## 2. Cau truc thu muc

```text
DHT/
|-- main.py
|-- streamlit_app.py
|-- run_ui.sh
|-- source_documentation.md
|-- src/
|   |-- __init__.py
|   |-- core/
|   |   |-- __init__.py
|   |   |-- common.py
|   |   |-- database.py
|   |   `-- models.py
|   `-- pipeline/
|       |-- __init__.py
|       |-- image_enhancement.py
|       |-- feature_extractor.py
|       |-- feature_comparison.py
|       |-- video_processor.py
|       `-- service.py
|-- scripts/
|   |-- add_known_persons.py
|   |-- check_model.py
|   `-- make_mevid_subset_fixed.py
|-- models/
|-- known_gallery/
|-- output/
|-- result_frame/
|-- temp/
`-- uploads/
```

## 3. Entry points

### 3.1 FastAPI API
- File: `main.py`
- Chuc nang:
- Khoi tao ket noi MongoDB
- Tao `VideoProcessor`
- Expose endpoint xu ly video va tra ket qua
- Endpoint chinh:
- `POST /process/upload`
- `POST /process/url`
- `GET /results/{job_id}`
- `GET /detections/{video_id}`
- `POST /features/add`
- `GET /features/list`
- `DELETE /detections/{video_id}`
- `GET /health`

### 3.2 Streamlit UI
- File: `streamlit_app.py`
- Chuc nang:
- Giao dien thao tac cho xu ly anh/video
- Quan ly thu vien `known_gallery`
- Su dung `ReIDPipelineService` tu `src/pipeline/service.py`

### 3.3 Python service (standalone)
- File: `src/pipeline/service.py`
- Chuc nang:
- Pipeline nhan dien end-to-end
- Track person trong video
- Chon frame chat luong tot nhat cho moi track
- Embed va compare voi gallery

## 4. Module core (`src/core`)

### 4.1 `common.py`
- Vai tro: Utility + data/model loader dung chung toan project.
- Noi dung chinh:
- Config preprocess (`TARGET_SIZE`, `MEAN`, `STD`)
- `find_model_file()` tim model trong `models/`
- `load_model_from_models()` load model tu local file
- `DataLoader`:
- `get_yolo_model()`
- `get_feature_extractor()`
- `preprocess_image()/preprocess_pil()/preprocess_cv2()`
- Luu y:
- Tu dong resolve `PROJECT_ROOT` va load `.env` tai root
- Path model resolve theo root project

### 4.2 `database.py`
- Vai tro: abstraction cho MongoDB async.
- Class: `Database`
- Ham chinh:
- luu/lay detections
- luu/lay known persons
- luu/lay processing results
- tao index (`create_indexes`)

### 4.3 `models.py`
- Vai tro: pydantic schemas co ban.
- Classes:
- `VideoUpload`
- `DetectionResult`
- `ProcessingStatus`

## 5. Module pipeline (`src/pipeline`)

### 5.1 `image_enhancement.py`
- Class `ImageEnhancer` cho enhancement anh.
- Cung cap wrapper module-level: `enhance(image)` de giu tuong thich code cu.

### 5.2 `feature_extractor.py`
- Class `FeatureExtractor`
- Load model thong qua `src.core.common.load_model_from_models`
- Tra feature vector da normalize (L2)
- Ho tro `extract()`, `extract_batch()`, `get_feature_dimension()`

### 5.3 `feature_comparison.py`
- Tool test nhanh so sanh 2 anh bang cosine similarity.
- Phuc vu debug/validation model.

### 5.4 `video_processor.py`
- Class `VideoProcessor` (pipeline cho FastAPI route).
- Luong chinh:
- Doc video theo FPS cau hinh
- Enhance frame
- Detect person bang YOLO
- Extract feature bang ViT
- Match voi known persons trong DB
- Luu detection + processing result vao MongoDB

### 5.5 `service.py`
- Module nang cao cho nhan dien va tracking.
- Classes:
- `TrackCrop`
- `ETFFM_Lite`
- `ReIDModel`
- `KnownGalleryManager`
- `ReIDPipelineService`
- Ham/chuc nang quan trong cua `ReIDPipelineService`:
- tracking: `track_persons()`, `_track_persons_ultralytics()`, `_track_persons_iou_fallback()`
- scoring: `_quality_score()`, `select_best_frames()`
- embedding: `embed_image()`, `embed_crop()`
- gallery: `load_known_gallery()`, `build_known_gallery()`, `compare_with_gallery()`
- API su dung truc tiep: `process_query_image()`, `process_video()`, `format_video_result()`

## 6. Scripts (`scripts/`)

### 6.1 `add_known_persons.py`
- Them anh trong `known_persons/` vao DB known persons.
- Gom anh theo person, extract feature, upsert vao MongoDB.

### 6.2 `check_model.py`
- Script inspect noi dung checkpoint model `.pth`.
- Dung de xac dinh file dang full model hay state_dict.

### 6.3 `make_mevid_subset_fixed.py`
- Tao subset MEVID theo rule open-set.
- Sinh lai annotation files (`track_test_info`, `query_IDX`, `gallery_IDX`, ...).

## 7. Luong du lieu

### 7.1 Luong API (`main.py` + `VideoProcessor`)
1. Nhan request upload/url.
2. Mo video va cat frame theo `FRAME_EXTRACTION_FPS`.
3. Detect person + crop.
4. Extract feature va match voi DB.
5. Luu ket qua vao `detections` va `processing_results`.

### 7.2 Luong UI (`streamlit_app.py` + `ReIDPipelineService`)
1. Khoi tao service + load gallery embeddings.
2. Xu ly anh/video tu giao dien.
3. Hien thi top-k, thong ke, va anh crop theo track.
4. Build/reload gallery tu `known_gallery/`.

## 8. Thu muc du lieu runtime
- `models/`: trong so model (`.pt`, `.pth`)
- `known_gallery/`: gallery theo person + `gallery_embeddings.npz`
- `uploads/`: tep upload
- `temp/`: tep tam
- `output/detected/`: crops/detections tu API pipeline
- `result_frame/`: anh best-frame moi track (service pipeline)

## 9. Bien moi truong quan trong
- `DATABASE_NAME`
- `UPLOAD_DIR`
- `OUTPUT_DIR`
- `TEMP_DIR`
- `CONFIDENCE_THRESHOLD`
- `KNOWN_PERSON_THRESHOLD`
- `FRAME_EXTRACTION_FPS`
- `YOLO_MODEL_PATH`
- `VIT_MODEL_PATH`

## 10. Cach chay nhanh

### 10.1 API
```bash
uvicorn main:app --reload
```

### 10.2 UI
```bash
streamlit run streamlit_app.py
```

### 10.3 Service demo
```bash
python -m src.pipeline.service
```

### 10.4 Scripts
```bash
python scripts/check_model.py
python scripts/add_known_persons.py
python scripts/make_mevid_subset_fixed.py
```

## 11. Ghi chu bao tri
- Khi them module moi trong `src/`, uu tien import theo package (`from src...`) thay vi import file root.
- Khi doi path model/gallery, uu tien truyen path tu root project.
- Tranh import vong giua `common.py` va `feature_extractor.py`; hien tai da giai quyet bang lazy import trong `DataLoader`.
