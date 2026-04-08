# DHT - Person Detection and ReID (Streamlit-Only)

This repository now runs in Streamlit-only mode.
FastAPI, MongoDB pipeline, and API endpoints were removed from the codebase.

## 1. What this project does

- Detect and track people in videos using YOLO.
- Re-identify people with a ReID model.
- Match query image/video crops against `known_gallery`.
- Manage gallery and run inference from `streamlit_app.py`.

Main runtime path:

- `streamlit_app.py` -> `src/pipeline/service.py` (`ReIDPipelineService`)

## 2. Project structure

```text
DHT/
|-- streamlit_app.py
|-- run_ui.sh
|-- source_documentation.md
|-- src/
|   |-- core/
|   |   `-- common.py
|   `-- pipeline/
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

## 3. Requirements

- Python 3.9+
- Optional CUDA GPU
- Model files in `models/`:
- `yolov8_person_detection.pt`
- `best_model_state_dict.pth`
- Optional for classification mode:
- `classification.pth`

Install:

```bash
pip install -r requirements.txt
```

## 4. Run Streamlit

```bash
streamlit run streamlit_app.py
```

Or:

```bash
./run_ui.sh
```

Open:

- `http://localhost:8501`

## 5. Gallery workflow

1. Open the `Manage Gallery` page in Streamlit.
2. Add person folders/images under `known_gallery/` (or upload in UI).
3. Click `Build Gallery` to create `known_gallery/gallery_embeddings.npz`.
4. Use `Process Image` or `Process Video`.

## 6. Notes

- The `Settings` page currently shows some config widgets that are display-only.
- Video processing in Streamlit uses source video frames directly (no fixed 2 FPS downsampling in this path).
- Track crops are written to `result_frame/`.

## 7. Documentation

- Streamlit-service method mapping:
- `docs/STREAMLIT_SERVICE_METHODS.md`
- Streamlit UI guide:
- `docs/STREAMLIT_UI_GUIDE.md`
- Source overview:
- `source_documentation.md`

