# Streamlit UI Guide

## 1. Overview

Tai lieu nay mo ta day du giao dien Streamlit cua he thong phat hien va nhan dang nguoi.

Muc tieu:
- giup nguoi moi co the chay va su dung app nhanh
- giup dev hieu ro luong du lieu va trang thai UI
- ghi lai dung hanh vi hien tai cua `streamlit_app.py`

File giao dien chinh:
- `streamlit_app.py`

Cong nghe chinh:
- `streamlit`
- `plotly`
- `pandas`
- `PIL`
- `opencv-python`
- `src.pipeline.service.ReIDPipelineService`

## 2. How To Run

### Cach 1
```bash
streamlit run streamlit_app.py
```

### Cach 2
```bash
./run_ui.sh
```

UI mac dinh mo tai:
- `http://localhost:8501`

## 3. Preconditions

De UI hoat dong on dinh, can cac thanh phan sau:
- thu muc `models/`
- file `models/yolov8_person_detection.pt`
- file `models/best_model_state_dict.pth`
- thu muc `known_gallery/`

Neu muon nhan dang nguoi da biet, can them:
- file `known_gallery/gallery_embeddings.npz`

Neu file embedding chua ton tai, UI van mo duoc, nhung page xu ly anh/video se khong co gallery de so sanh.

## 4. High-Level Architecture

UI gom 5 trang:
- `Trang chu`
- `Xu ly anh`
- `Xu ly video`
- `Quan ly thu vien`
- `Cai dat`

Luong tong quat:
1. UI khoi tao service khi can.
2. Service load model YOLO va ReID.
3. UI load gallery embeddings neu co.
4. User upload anh/video.
5. Service xu ly input va tra ket qua.
6. UI hien thi ket qua bang card, bang du lieu va bieu do.

## 5. Session State

UI dung `st.session_state` de giu trang thai giua cac lan rerun.

Bien chinh:
- `service`
- `gallery_loaded`
- `last_result`
- `last_video_upload_signature`

Y nghia:
- `service`: luu instance `ReIDPipelineService`, tranh khoi tao lai lien tuc
- `gallery_loaded`: danh dau gallery embedding da duoc load vao service
- `last_result`: bien du phong cho ket qua cu
- `last_video_upload_signature`: dung de xac dinh video moi, tu do don `result_frame/`

## 6. Layout And Styling

UI su dung:
- `st.set_page_config(layout="wide")`
- sidebar de dieu huong
- CSS custom qua `st.markdown(..., unsafe_allow_html=True)`

Phong cach hien tai:
- tieu de gradient
- nut action full-width
- card metric nen sang
- hop thong bao mau xanh, vang, do
- hieu ung hover nhe tren anh

## 7. Sidebar

Sidebar gom 3 nhom:

### Dieu huong
User chon 1 trong 5 trang qua `st.radio`.

### Trang thai he thong
Hien thi:
- trang thai model YOLO
- trang thai model ReID
- trang thai gallery da load hay chua

### Thong tin
Mo ta ngan ve cong nghe:
- YOLOv8
- ReID
- Streamlit

## 8. Home Page

Nhan dien bang nhan:
- `Trang chu`

Noi dung chinh:
- tieu de he thong
- mo ta tong quan
- 4 metric nhanh:
- so nguoi da biet
- tong so anh trong gallery
- trang thai model
- trang thai san sang

Phan gioi thieu tinh nang:
- xu ly anh
- xu ly video
- quan ly thu vien
- tuy chon nang cao

Phan huong dan bat dau:
- thiet lap thu vien nguoi da biet
- xu ly anh
- xu ly video

Day la trang onboarding, khong thuc hien xu ly du lieu.

## 9. Image Processing Page

Nhan dien bang nhan:
- `Xu ly anh`

### Muc tieu
Cho phep upload 1 anh, detect nguoi trong anh, sau do so khop voi gallery da biet.

### Luong xu ly
1. Khoi tao `ReIDPipelineService` neu `session_state.service` dang `None`.
2. Nap `known_gallery/gallery_embeddings.npz` neu chua load.
3. User upload anh `.jpg`, `.jpeg`, `.png`.
4. User chon `topk`.
5. UI luu anh tam vao file `.jpg`.
6. Goi `service.process_query_image(tmp_path, topk=topk)`.
7. Hien thi ket qua.
8. Xoa file tam trong `finally`.

### Thanh phan giao dien
- `file_uploader` de nhan file anh
- `slider` chon so ket qua khop, tu 1 den 10
- nut `Phat hien & Nhan dang`

### Ket qua hien thi
Cot trai:
- preview anh upload

Cot phai:
- danh sach top-k ket qua
- `person_id`
- similarity score
- progress bar
- thumbnail anh tham chieu
- expander xem anh lon

Duoi cung:
- bieu do cot similarity theo `person_id`

### Hanh vi khi chua co gallery
Neu `known_gallery/gallery_embeddings.npz` chua ton tai:
- UI hien warning
- user duoc huong dan sang trang `Quan ly thu vien`

### Hanh vi loi
UI bat cac truong hop:
- khoi tao service that bai
- load gallery that bai
- xu ly anh loi
- anh khong detect duoc nguoi
- service khong tra ve ket qua hop le

## 10. Video Processing Page

Nhan dien bang nhan:
- `Xu ly video`

### Muc tieu
Xu ly 1 video, track nguoi, chon best frame moi track, sau do nhan dang theo 1 trong 2 che do:
- similarity mode
- classification mode

### Luong xu ly
1. Khoi tao `ReIDPipelineService` neu chua co.
2. Load gallery neu file embedding ton tai.
3. User upload video `.mp4`, `.avi`, `.mov`.
4. UI tao `signature` tu:
- ten file
- do dai file
- md5
5. Neu la video moi, UI don sach `result_frame/`.
6. User chon che do xu ly.
7. UI luu video vao file tam `.mp4`.
8. Goi `service.process_video(...)`.
9. Goi `service.format_video_result(...)`.
10. Hien thi ket qua chi tiet.
11. Xoa file tam.

### Dieu khien giao dien
- `checkbox`: bat/tat `Dung model phan loai`
- `slider topk`: chi hien trong similarity mode
- nut `Xu ly Video`

### Che do similarity
Hanh vi:
- moi track duoc embed
- so sanh voi gallery
- lay top-k ket qua

UI hien:
- tong so track
- so nguoi da biet
- so nguoi chua biet
- bang chi tiet
- bieu do tong quan
- pie chart phan bo ket luan
- expander cho tung track
- anh crop tu video
- top-k anh tham chieu
- bieu do similarity rieng cho tung track

### Che do classification
Hanh vi:
- dung model classifier quen/la
- `topk` bi tat va co dinh = 1
- khong hien bang so sanh giong ai theo gallery

UI hien:
- ket luan `Nguoi quen` hoac `Nguoi la`
- do tin cay phan loai
- bieu do confidence theo track

### Gia tri detect hien tai
UI hien caption:
- `confidence=0.85`
- `IoU=0.70`
- `min_box_area_ratio=0.0200`

Luu y:
- cac gia tri nay dang duoc su dung co dinh cho page video hien tai

### Thu muc output
Khi xu ly video, app su dung:
- `result_frame/` de luu best crop theo track

Khi user upload video moi:
- UI tu dong don noi dung cu trong `result_frame/`

## 11. Gallery Management Page

Nhan dien bang nhan:
- `Quan ly thu vien`

### Muc tieu
Quan ly du lieu nguoi da biet trong `known_gallery/` va build embedding gallery.

### Dashboard thong ke
Hien thi:
- tong so person folder
- tong so anh gallery
- trang thai da build embedding hay chua

### Tab 1: Tao nhan vat moi
User nhap:
- ten nhan vat
- nhieu anh cung luc

UI thuc hien:
1. lam sach ten bang cach thay space thanh `_`
2. tao thu muc `known_gallery/<ten>`
3. luu tung anh theo format:
- `<ten>_01.ext`
- `<ten>_02.ext`
4. hien thong bao thanh cong
5. `st.rerun()`

### Tab 2: Them anh vao nhan vat co san
User chon:
- 1 nhan vat trong dropdown
- 1 hoac nhieu anh moi

UI thuc hien:
1. doc so anh hien tai
2. dat ten tiep tuc theo so thu tu
3. luu them vao cung folder
4. hien thong bao thanh cong
5. `st.rerun()`

### Preview gallery
Moi person duoc hien trong 1 expander.

Thong tin hien thi:
- so anh hien co
- toi da 10 thumbnail dau tien

### Build gallery
Nut:
- `Xay dung Thu vien`

UI thuc hien:
1. khoi tao `ReIDPipelineService`
2. goi `service.build_known_gallery("known_gallery", gallery_path)`
3. tao file `known_gallery/gallery_embeddings.npz`
4. reset co `gallery_loaded`
5. `st.rerun()`

### Reload gallery
Nut:
- `Tai lai Thu vien`

UI thuc hien:
- dat `gallery_loaded = False`
- dat `service = None`

Muc dich:
- buoc page anh/video khoi tao lai service va load lai gallery o lan dung tiep theo

### Truong hop thu muc gallery chua ton tai
Neu `known_gallery/` khong co:
- UI tu dong tao thu muc
- rerun lai trang

## 12. Settings Page

Nhan dien bang nhan:
- `Cai dat`

### Thanh phan hien thi
- duong dan YOLO model
- duong dan ReID model
- slider threshold nhan dang
- selectbox thiet bi: `auto`, `cpu`, `cuda`
- thong tin ton tai cac thu muc
- version thu vien chinh

### Luu y quan trong
Trang `Cai dat` hien tai chu yeu la giao dien tham khao va kiem tra.

Gia tri nguoi dung thay doi tai day:
- chua duoc persist
- chua duoc truyen nguoc vao `ReIDPipelineService` dang chay
- chua anh huong truc tiep den page `Xu ly anh` hay `Xu ly video`

Nut `Dat lai Tat ca` chi reset:
- `service`
- `gallery_loaded`
- `last_result`

Sau do UI rerun lai.

## 13. Footer

Cuoi trang co footer HTML tinh:
- ten he thong
- cong nghe Streamlit
- nam `2026`

## 14. Streamlit Utilities In Code

### `build_fixed_preview(...)`
Muc dich:
- tao preview co kich thuoc co dinh
- giu ty le anh
- them background de card dong deu hon

Dung cho:
- anh track tu video
- anh tham chieu trong gallery match

### `clear_result_frame_dir(...)`
Muc dich:
- xoa toan bo anh/thu muc cu trong `result_frame/`
- tranh nham ket qua video cu voi video moi

## 15. Real User Workflows

### Workflow A: Build gallery tu dau
1. Mo `Quan ly thu vien`
2. Tao nhan vat moi
3. Upload nhieu anh
4. Lap lai cho cac nhan vat khac
5. Bam `Xay dung Thu vien`
6. Mo `Xu ly anh` hoac `Xu ly video`

### Workflow B: Nhan dang trong anh
1. Dam bao da co `gallery_embeddings.npz`
2. Mo `Xu ly anh`
3. Upload anh
4. Chon `topk`
5. Bam `Phat hien & Nhan dang`
6. Xem top-k va bieu do similarity

### Workflow C: Nhan dang trong video
1. Dam bao da co gallery
2. Mo `Xu ly video`
3. Upload video
4. Chon similarity mode hoac classification mode
5. Bam `Xu ly Video`
6. Xem summary, bang ket qua, tung track va anh crop

## 16. Dependencies Between UI And Backend

Page `Xu ly anh` va `Xu ly video` phu thuoc:
- `src.pipeline.service.ReIDPipelineService`

Page `Quan ly thu vien` phu thuoc:
- thao tac file tren `known_gallery/`
- `ReIDPipelineService.build_known_gallery()`

UI khong goi truc tiep `VideoProcessor`.

`VideoProcessor` chi duoc dung boi FastAPI trong `main.py`.

## 17. Known Limitations

### 17.1 Settings chua la runtime settings that su
Page `Cai dat` hien thi cau hinh nhung chua gan vao pipeline dang chay.

### 17.2 Chua co co che download ket qua tu UI
UI hien thi bang va bieu do, nhung chua co nut xuat file CSV/JSON trong code hien tai.

### 17.3 Gallery page luu file truc tiep
Page `Quan ly thu vien` thao tac truc tiep len file system, chua qua API trung gian.

### 17.4 Service khoi tao trong UI co the nang
Lan dau mo page `Xu ly anh` hoac `Xu ly video`, viec load model co the mat thoi gian.

## 18. Troubleshooting

### UI mo len nhung bao loi khoi tao service
Kiem tra:
- `models/yolov8_person_detection.pt`
- `models/best_model_state_dict.pth`
- moi truong Python da cai du thu vien

### Khong tai duoc gallery
Kiem tra:
- co file `known_gallery/gallery_embeddings.npz`
- file khong bi hong

### Khong detect duoc nguoi trong anh
Kiem tra:
- anh co nguoi ro rang hay khong
- anh co bi mo, qua nho, qua toi hay khong

### Video xu ly xong nhung khong co ket qua
Kiem tra:
- video co chua nguoi hay khong
- track co du chat luong de tao best frame hay khong
- gallery da duoc build hay chua

## 19. Suggested Maintenance Notes

Neu mo rong Streamlit UI, nen giu nguyen cac nguyen tac sau:
- tiep tuc dung `session_state` de tranh reload model khong can thiet
- neu them setting that su, can dong bo settings vao service instance
- neu them export ket qua, uu tien dung `st.download_button`
- neu tach file UI thanh nhieu module, nen tach theo page

## 20. Related Files

- `streamlit_app.py`
- `src/pipeline/service.py`
- `source_documentation.md`
- `docs/COMPLETE_DOCUMENTATION.md`
