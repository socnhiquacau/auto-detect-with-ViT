"""
Giao diện Streamlit cho Hệ thống Phát hiện và Nhận dạng Người
"""
import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
import shutil
import hashlib
from PIL import Image, ImageOps
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import cv2
import numpy as np

# Page config
st.set_page_config(
    page_title="Phát hiện & Nhận dạng Người",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        border-radius: 5px;
        padding: 0.5rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    /* Result cards styling */
    .stContainer {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
    }
    /* Thumbnail hover effect */
    img {
        transition: transform 0.2s;
        cursor: pointer;
    }
    img:hover {
        transform: scale(1.05);
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 5px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'service' not in st.session_state:
    st.session_state.service = None
if 'gallery_loaded' not in st.session_state:
    st.session_state.gallery_loaded = False
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'last_video_upload_signature' not in st.session_state:
    st.session_state.last_video_upload_signature = None


def build_fixed_preview(image_source, size=(220, 220), background=(245, 247, 250)):
    if isinstance(image_source, Image.Image):
        image = image_source.convert("RGB")
    else:
        image = Image.open(image_source).convert("RGB")

    fitted = ImageOps.contain(image, size)
    canvas = Image.new("RGB", size, background)
    offset = ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def clear_result_frame_dir(result_dir="result_frame") -> int:
    """
    Remove all previous track images/folders before processing a new video.
    Returns number of removed entries.
    """
    target = Path(result_dir)
    if not target.exists() or not target.is_dir():
        return 0

    removed = 0
    for entry in target.iterdir():
        try:
            if entry.is_file() or entry.is_symlink():
                entry.unlink()
                removed += 1
            elif entry.is_dir():
                shutil.rmtree(entry)
                removed += 1
        except Exception:
            # Ignore cleanup failure for one item to avoid blocking UI.
            continue
    return removed

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Điều hướng")
    page = st.radio(
        "Chọn trang",
        ["🏠 Trang chủ", "📸 Xử lý ảnh", "🎬 Xử lý video",
            "👥 Quản lý thư viện", "⚙️ Cài đặt"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📊 Trạng thái hệ thống")

    # Check models
    yolo_exists = os.path.exists("models/yolov8_person_detection.pt")
    reid_exists = os.path.exists("models/best_model_state_dict.pth")

    st.markdown(f"**Mô hình YOLO:** {'✅' if yolo_exists else '❌'}")
    st.markdown(f"**Mô hình ReID:** {'✅' if reid_exists else '❌'}")
 

# =====================================
# HOME PAGE
# =====================================
if page == "🏠 Trang chủ":
    st.markdown('<h1 class="main-header">🎯 Phát hiện & Nhận dạng Người</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    ### Chào mừng đến với Hệ thống Phát hiện & Nhận dạng Người!

    Ứng dụng này sử dụng các mô hình AI tiên tiến để phát hiện và nhận dạng người trong ảnh và video.
    """)

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Người đã biết", len([d for d in os.listdir("known_gallery")
                  if os.path.isdir(f"known_gallery/{d}") and not d.startswith('.')]) if os.path.exists("known_gallery") else 0)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_images = sum([len([f for f in os.listdir(f"known_gallery/{d}")
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                           for d in os.listdir("known_gallery")
                           if os.path.isdir(f"known_gallery/{d}") and not d.startswith('.')]) if os.path.exists("known_gallery") else 0
        st.metric("Ảnh thư viện", total_images)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Mô hình", "2/2" if yolo_exists and reid_exists else "⚠️")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Trạng thái", "✅ Sẵn sàng" if yolo_exists and reid_exists else "⚠️ Cần cài đặt")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Features
    st.markdown("### 🚀 Tính năng")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 📸 Xử lý Ảnh
        - Tải lên một ảnh
        - Phát hiện tất cả người
        - Khớp với thư viện đã biết
        - Xem điểm tin cậy
        """)

    with col2:
        st.markdown("""
        #### 🎬 Xử lý Video
        - Xử lý video đầy đủ
        - Theo dõi người qua các khung hình
        - Tạo báo cáo
        - Xuất kết quả

        #### ⚙️ Tính năng Nâng cao
        - Điều chỉnh ngưỡng
        - Chỉ số chất lượng
        - Xử lý hàng loạt
        - Nhiều mô hình
        """)

    st.markdown("---")

    # Getting started
    st.markdown("### 🎓 Hướng dẫn Bắt đầu")

    with st.expander("1️⃣ Xử lý Ánh"):
        st.markdown("""
        1. Đi đến trang **📸 Xử lý Ảnh**
        2. Tải lên một ảnh
        3. Xem người được phát hiện và kết quả khớp
        4. Tải xuống kết quả
        """)

    with st.expander("3️⃣ Xử lý Video"):
        st.markdown("""
        1. Đi đến trang **🎬 Xử lý Video**
        2. Tải lên một file video
        3. Chờ xử lý (có thể mất vài phút)
        4. Xem kết quả theo dõi và thống kê
        """)


# =====================================
# PROCESS IMAGE PAGE
# =====================================
elif page == "📸 Xử lý ảnh":
    st.markdown('<h1 class="main-header">📸 Xử lý Ảnh</h1>',
                unsafe_allow_html=True)

    # Initialize service
    if st.session_state.service is None:
        with st.spinner("Đang khởi tạo dịch vụ..."):
            try:
                from src.pipeline.service import ReIDPipelineService
                st.session_state.service = ReIDPipelineService(
                    yolo_model_path="models/yolov8_person_detection.pt",
                    reid_weights_path="models/best_model_state_dict.pth",
                    known_threshold=0.8
                )
                st.markdown(
                    '<div class="success-box">✅ Khởi tạo dịch vụ thành công!</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(
                    f'<div class="error-box">❌ Lỗi khởi tạo dịch vụ: {e}</div>', unsafe_allow_html=True)
                st.stop()

    # Load gallery if not loaded
    gallery_path = "known_gallery/gallery_embeddings.npz"
    if not st.session_state.gallery_loaded and os.path.exists(gallery_path):
        with st.spinner("Đang tải thư viện..."):
            try:
                st.session_state.service.load_known_gallery(gallery_path)
                st.session_state.gallery_loaded = True
                st.success("✅ Tải thư viện thành công!")
            except Exception as e:
                st.warning(f"⚠️ Không thể tải thư viện: {e}")
    elif not os.path.exists(gallery_path):
        st.warning(
            '⚠️ Không tìm thấy thư viện. Vui lòng xây dựng thư viện trước ở trang "👥 Quản lý thư viện".')
        st.info(
            "💡 Hướng dẫn: Vào trang 👥 Quản lý thư viện → Thêm ảnh người → Nhấn 🔨 Xây dựng Thư viện")

    # Upload image
    uploaded_file = st.file_uploader(
        "Tải lên ảnh", type=['jpg', 'jpeg', 'png'])

    col1, col2 = st.columns([1, 2])

    with col1:
        topk = st.slider("Số kết quả khớp", min_value=1, max_value=10, value=5)
        process_btn = st.button("🔍 Phát hiện & Nhận dạng", type="primary")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        if process_btn:
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            with st.spinner("Đang xử lý ảnh..."):
                try:
                    result = st.session_state.service.process_query_image(
                        tmp_path, topk=topk)

                    # Layout: Left column for uploaded image, Right column for results
                    left_col, right_col = st.columns([1, 1.5])

                    with left_col:
                        st.markdown("### 📷 Ảnh đã tải lên")
                        st.image(image, use_container_width=True)

                    with right_col:
                        st.markdown("### 🎯 Kết quả nhận dạng")

                        if result and 'topk' in result:
                            matches = result['topk']

                            if matches and len(matches) > 0:
                                # Display results
                                st.markdown(
                                    f"**Tìm thấy {len(matches)} kết quả khớp:**")

                                for i, match in enumerate(matches, 1):
                                    # Result card with thumbnail
                                    with st.container():
                                        st.markdown(f"""
                                        <div style="background-color: #f8f9fa; padding: 1rem;
                                                    border-radius: 10px; border-left: 4px solid #667eea;
                                                    margin-bottom: 1rem;">
                                        """, unsafe_allow_html=True)

                                        result_cols = st.columns(
                                            [0.4, 1.6, 0.8])

                                        with result_cols[0]:
                                            st.markdown(f"**#{i}**")

                                        with result_cols[1]:
                                            st.markdown(
                                                f"**Người:** `{match['person_id']}`")
                                            similarity = match['similarity']
                                            st.markdown(
                                                f"**Độ tương đồng:** {similarity:.2%}")
                                            st.progress(similarity)

                                        with result_cols[2]:
                                            img_path = match.get(
                                                'image_path') or match.get('img_path', '')
                                            if img_path and os.path.exists(img_path):
                                                ref_img = Image.open(img_path)
                                                # Display small thumbnail
                                                st.image(
                                                    ref_img, width=120, caption="Tham chiếu")

                                                # Expandable full-size image
                                                with st.expander("🔍 Xem ảnh lớn"):
                                                    st.image(ref_img, use_container_width=True,
                                                             caption=f"Ảnh tham chiếu - {match['person_id']}")
                                            else:
                                                st.info("Không có ảnh")

                                        st.markdown(
                                            "</div>", unsafe_allow_html=True)

                                # Chart below (full width)
                                st.markdown("### 📊 Biểu đồ Độ tương đồng")
                                df = pd.DataFrame(matches)
                                fig = px.bar(
                                    df,
                                    x='person_id',
                                    y='similarity',
                                    title='Kết quả Khớp Tốt Nhất Theo Độ Tương Đồng',
                                    labels={'similarity': 'Điểm Tương Đồng',
                                            'person_id': 'ID Người'},
                                    color='similarity',
                                    color_continuous_scale='RdYlGn'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.markdown(
                                    '<div class="warning-box">⚠️ Không tìm thấy kết quả khớp nào trên ngưỡng.</div>',
                                    unsafe_allow_html=True)
                        elif result and 'error' in result:
                            st.markdown(
                                f'<div class="error-box">❌ Lỗi: {result["error"]}</div>',
                                unsafe_allow_html=True)
                        elif result is None or not result:
                            st.markdown(
                                '<div class="warning-box">⚠️ Không detect được người trong ảnh. Hãy thử ảnh khác có người rõ ràng hơn.</div>',
                                unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f'<div class="warning-box">⚠️ Không có kết quả trả về. Kiểm tra: {result}</div>',
                                unsafe_allow_html=True)

                except Exception as e:
                    st.markdown(
                        f'<div class="error-box">❌ Lỗi xử lý ảnh: {e}</div>', unsafe_allow_html=True)
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    # Cleanup
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        else:
            # Show uploaded image preview before processing
            st.markdown("### 📷 Ảnh đã tải lên")
            st.image(image, caption="Preview - Nhấn '🔍 Phát hiện & Nhận dạng' để xử lý",
                     use_container_width=True)


# =====================================
# PROCESS VIDEO PAGE
# =====================================
elif page == "🎬 Xử lý video":
    st.markdown('<h1 class="main-header">🎬 Xử lý Video</h1>',
                unsafe_allow_html=True)

    # Initialize service
    if st.session_state.service is None:
        with st.spinner("Đang khởi tạo dịch vụ..."):
            try:
                from src.pipeline.service import ReIDPipelineService
                st.session_state.service = ReIDPipelineService()
                st.markdown(
                    '<div class="success-box">✅ Khởi tạo dịch vụ thành công!</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(
                    f'<div class="error-box">❌ Lỗi: {e}</div>', unsafe_allow_html=True)
                st.stop()

    # Load gallery
    gallery_path = "known_gallery/gallery_embeddings.npz"
    if not st.session_state.gallery_loaded and os.path.exists(gallery_path):
        with st.spinner("Đang tải thư viện..."):
            try:
                st.session_state.service.load_known_gallery(gallery_path)
                st.session_state.gallery_loaded = True
            except Exception as e:
                st.warning(f"Không thể tải thư viện: {e}")

    # Upload video
    uploaded_video = st.file_uploader(
        "Tải lên video", type=['mp4', 'avi', 'mov'])

    col1, col2 = st.columns([1, 2])

    with col1:
        use_classification_model = st.checkbox(
            "Dùng model phân loại",
            value=False,
            key="video_use_classification_model",
            help="Bật để phân loại Quen/Lạ. Khi bật sẽ ẩn so sánh giống ai và độ tương đồng.",
        )
        st.caption("YOLO config co dinh: confidence=0.85, IoU=0.70, min_box_area_ratio=0.0200")
        if use_classification_model:
            topk = 1
            st.caption("Chế độ phân loại: top-k và độ tương đồng được tắt.")
        else:
            topk = st.slider("Số kết quả khớp", min_value=1,
                             max_value=10, value=5, key="video_topk")
        process_btn = st.button("🎬 Xử lý Video", type="primary")

    if uploaded_video is not None:
        video_bytes = uploaded_video.getvalue()
        current_video_signature = (
            f"{uploaded_video.name}:{len(video_bytes)}:"
            f"{hashlib.md5(video_bytes).hexdigest()}"
        )

        if st.session_state.last_video_upload_signature != current_video_signature:
            removed_count = clear_result_frame_dir("result_frame")
            st.session_state.last_video_upload_signature = current_video_signature
            if removed_count > 0:
                st.caption(f"🧹 Đã dọn {removed_count} ảnh/nhóm track cũ trong thư mục `result_frame`.")

        st.video(uploaded_video)

        if process_btn:
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_bytes)
                tmp_path = tmp_file.name

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text(
                    "Đang xử lý video... Có thể mất vài phút ⏳")

                raw_result = st.session_state.service.process_video(
                    tmp_path,
                    topk=topk,
                    use_classification_model=use_classification_model,
                )
                formatted = st.session_state.service.format_video_result(
                    raw_result)

                progress_bar.progress(100)
                status_text.text("✅ Xử lý hoàn tất!")

                st.markdown("---")
                st.markdown("### 🎯 Kết quả")

                # Summary stats
                col1, col2, col3 = st.columns(3)

                num_tracks = formatted.get('num_tracks', 0)
                results = formatted.get('results', [])
                known_tracks = len(
                    [r for r in results if r.get('prediction', {}).get('label') == 'known'])
                unknown_tracks = max(num_tracks - known_tracks, 0)

                with col1:
                    st.metric("Tổng số Track", num_tracks)
                with col2:
                    st.metric("Người đã biết", known_tracks)
                with col3:
                    st.metric("Chưa biết", unknown_tracks)

                # Results table
                if results:
                    is_classification_mode = formatted.get('mode', 'similarity') == 'classification'
                    detail_rows = []
                    for item in results:
                        prediction = item.get('prediction', {})
                        label = prediction.get('label', 'unknown')

                        if is_classification_mode:
                            classification_confidence = float(
                                prediction.get('classification_confidence', prediction.get('best_similarity', 0.0))
                            )
                            final_name = 'Nguoi quen' if label == 'known' else 'Nguoi la'
                            detail_rows.append(
                                {
                                    "track_id": item.get('track_id', -1),
                                    "frame_index": item.get('frame_index', -1),
                                    "ket_luan": final_name,
                                    "nhan": "Quen" if label == 'known' else "La",
                                    "do_tin_cay_phan_loai": classification_confidence,
                                }
                            )
                        else:
                            person_id = prediction.get('person_id') or 'unknown'
                            final_name = person_id if label == 'known' else 'Nguoi la'
                            detail_rows.append(
                                {
                                    "track_id": item.get('track_id', -1),
                                    "frame_index": item.get('frame_index', -1),
                                    "ket_luan": final_name,
                                    "nhan": "Da biet" if label == 'known' else "Nguoi la",
                                    "do_tuong_dong_cao_nhat": prediction.get('best_similarity', 0.0),
                                    "nguong": prediction.get('threshold', formatted.get('threshold', 0.0)),
                                }
                            )

                    df_results = pd.DataFrame(detail_rows)

                    st.markdown("### 📋 Chi tiết Phát hiện")
                    st.dataframe(df_results, use_container_width=True)

                    if is_classification_mode:
                        st.markdown("### Tong quan do tin cay phan loai")
                        fig_summary = px.bar(
                            df_results,
                            x='track_id',
                            y='do_tin_cay_phan_loai',
                            color='nhan',
                            text='ket_luan',
                            title='Do tin cay phan loai theo tung track',
                            labels={
                                'track_id': 'Track ID',
                                'do_tin_cay_phan_loai': 'Do tin cay',
                                'nhan': 'Phan loai',
                            },
                            color_discrete_map={'Quen': '#2E8B57', 'La': '#DC3545'},
                        )
                    else:
                        st.markdown("### Tổng quan độ tương đồng")
                        fig_summary = px.bar(
                            df_results,
                            x='track_id',
                            y='do_tuong_dong_cao_nhat',
                            color='nhan',
                            text='ket_luan',
                            title='Độ tương đồng cao nhất theo từng track',
                            labels={
                                'track_id': 'Track ID',
                                'do_tuong_dong_cao_nhat': 'Độ tương đồng',
                                'nhan': 'Phân loại',
                            },
                            color_discrete_map={'Đã biết': '#2E8B57', 'Người lạ': '#DC3545'},
                        )

                    fig_summary.update_traces(textposition='outside')
                    fig_summary.update_yaxes(range=[0, 1])
                    chart_left, chart_center, chart_right = st.columns([1, 3, 1])
                    with chart_center:
                        st.plotly_chart(fig_summary, use_container_width=True)

                    person_counts = pd.Series(
                        [row['ket_luan'] for row in detail_rows]
                    ).value_counts()
                    fig_distribution = px.pie(
                        values=person_counts.values,
                        names=person_counts.index,
                        title='Phân bố kết luận nhận dạng'
                    )
                    pie_left, pie_center, pie_right = st.columns([1, 3, 1])
                    with pie_center:
                        st.plotly_chart(fig_distribution, use_container_width=True)

                    st.markdown("### Chi tiết từng track")
                    for index, item in enumerate(results, start=1):
                        prediction = item.get('prediction', {})
                        top_matches = prediction.get('topk', [])
                        label = prediction.get('label', 'unknown')
                        track_id = item.get('track_id', -1)
                        frame_index = item.get('frame_index', -1)

                        if is_classification_mode:
                            classification_confidence = float(
                                prediction.get('classification_confidence', prediction.get('best_similarity', 0.0))
                            )
                            final_name = 'Nguoi quen' if label == 'known' else 'Nguoi la'
                            expander_title = (
                                f"Track #{track_id} | Ket luan: {final_name} | "
                                f"Do tin cay: {classification_confidence:.2%}"
                            )
                        else:
                            person_id = prediction.get('person_id') or 'unknown'
                            best_similarity = float(prediction.get('best_similarity', 0.0))
                            threshold = float(prediction.get('threshold', formatted.get('threshold', 0.0)))
                            final_name = person_id if label == 'known' else 'Người lạ'
                            expander_title = (
                                f"Track #{track_id} | Kết luận: {final_name} | "
                                f"Độ tương đồng: {best_similarity:.2%}"
                            )

                        with st.expander(expander_title, expanded=index == 1):
                            left_col, right_col = st.columns([1, 1.5])

                            with left_col:
                                st.markdown("#### Ảnh cắt từ video")
                                track_image_path = item.get('track_url', '')
                                if track_image_path and os.path.exists(track_image_path):
                                    track_img = build_fixed_preview(track_image_path, size=(240, 240))
                                    st.image(
                                        track_img,
                                        caption=f"Track #{track_id} - Frame {frame_index}",
                                        width=240,
                                    )
                                else:
                                    st.info("Không có ảnh track để hiển thị")

                                st.markdown(f"**Frame:** `{frame_index}`")

                            with right_col:
                                box_class = "success-box" if label == "known" else "warning-box"

                                if is_classification_mode:
                                    classification_confidence = float(
                                        prediction.get('classification_confidence', prediction.get('best_similarity', 0.0))
                                    )
                                    conclusion_text = (
                                        "Ket luan: <b>Nguoi quen</b>" if label == "known"
                                        else "Ket luan: <b>Nguoi la</b>"
                                    )
                                    st.markdown(
                                        (
                                            f'<div class="{box_class}">{conclusion_text}<br>'
                                            f'Do tin cay phan loai: <b>{classification_confidence:.2%}</b></div>'
                                        ),
                                        unsafe_allow_html=True,
                                    )
                                    st.markdown("**Do tin cay phan loai:**")
                                    st.progress(min(max(classification_confidence, 0.0), 1.0))
                                else:
                                    best_similarity = float(prediction.get('best_similarity', 0.0))
                                    threshold = float(prediction.get('threshold', formatted.get('threshold', 0.0)))
                                    conclusion_text = (
                                        f"Kết luận: <b>{final_name}</b>"
                                        if label == "known"
                                        else "Kết luận: <b>Người lạ</b>"
                                    )
                                    st.markdown(
                                        (
                                            f'<div class="{box_class}">{conclusion_text}<br>'
                                            f'Độ tương đồng cao nhất: <b>{best_similarity:.2%}</b><br>'
                                            f'Ngưỡng nhận dạng: <b>{threshold:.2%}</b></div>'
                                        ),
                                        unsafe_allow_html=True,
                                    )
                                    st.markdown("**Độ tương đồng cao nhất:**")
                                    st.progress(min(max(best_similarity, 0.0), 1.0))

                                    if top_matches:
                                        match_cols = st.columns(min(len(top_matches), 3))
                                        for rank, match in enumerate(top_matches, start=1):
                                            similarity = float(match.get('similarity', 0.0))
                                            match_person = match.get('person_id', 'unknown')
                                            img_path = match.get('image_path') or match.get('img_path', '')
                                            with match_cols[(rank - 1) % len(match_cols)]:
                                                st.markdown(
                                                    """
                                                    <div style="background-color: #f8f9fa; padding: 0.75rem;
                                                                border-radius: 10px; border-left: 4px solid #667eea;
                                                                margin-bottom: 1rem;">
                                                    """,
                                                    unsafe_allow_html=True,
                                                )
                                                st.markdown(f"**#{rank} - {match_person}**")
                                                if img_path and os.path.exists(img_path):
                                                    ref_img = build_fixed_preview(img_path, size=(180, 180))
                                                    st.image(
                                                        ref_img,
                                                        width=180,
                                                        caption="Tham chiếu",
                                                    )
                                                else:
                                                    st.info("Không có ảnh")
                                                st.markdown(f"**Độ tương đồng:** {similarity:.2%}")
                                                st.progress(min(max(similarity, 0.0), 1.0))
                                                st.markdown("</div>", unsafe_allow_html=True)

                                        chart_df = pd.DataFrame(
                                            [
                                                {
                                                    "person_id": match.get('person_id', 'unknown'),
                                                    "similarity": float(match.get('similarity', 0.0)),
                                                }
                                                for match in top_matches
                                            ]
                                        )
                                        fig_track = px.bar(
                                            chart_df,
                                            x='person_id',
                                            y='similarity',
                                            title=f'Biểu đồ độ tương đồng - Track #{track_id}',
                                            labels={
                                                'similarity': 'Độ tương đồng',
                                                'person_id': 'ID người',
                                            },
                                            color='similarity',
                                            color_continuous_scale='RdYlGn'
                                        )
                                        fig_track.update_yaxes(range=[0, 1])
                                        track_chart_left, track_chart_center, track_chart_right = st.columns([1, 3, 1])
                                        with track_chart_center:
                                            st.plotly_chart(fig_track, use_container_width=True)
                                    else:
                                        st.markdown(
                                            '<div class="warning-box">Không có ảnh tham chiếu phù hợp cho track này.</div>',
                                            unsafe_allow_html=True,
                                        )


            except Exception as e:
                st.markdown(
                    f'<div class="error-box">❌ Lỗi: {e}</div>', unsafe_allow_html=True)
                import traceback
                st.code(traceback.format_exc())
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    else:
        st.session_state.last_video_upload_signature = None


# =====================================
# MANAGE GALLERY PAGE
# =====================================
elif page == "👥 Quản lý thư viện":
    st.markdown('<h1 class="main-header">👥 Quản lý Thư viện</h1>',
                unsafe_allow_html=True)

    gallery_path = "known_gallery/gallery_embeddings.npz"

    # Gallery stats
    if os.path.exists("known_gallery"):
        persons = [d for d in os.listdir("known_gallery")
                   if os.path.isdir(f"known_gallery/{d}") and not d.startswith('.')]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("👥 Người", len(persons))

        total_images = 0
        for person in persons:
            person_dir = f"known_gallery/{person}"
            images = [f for f in os.listdir(person_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_images += len(images)

        with col2:
            st.metric("📸 Tổng số Ảnh", total_images)

        with col3:
            gallery_exists = os.path.exists(gallery_path)
            st.metric("💾 Trạng thái Thư viện",
                      "✅ Đã xây dựng" if gallery_exists else "⏳ Chưa xây dựng")

        st.markdown("---")

        # Add new person / Add images to existing person
        st.markdown("### ➕ Thêm Ảnh Nhân Vật")

        tab1, tab2 = st.tabs(
            ["🆕 Tạo nhân vật mới", "📸 Thêm ảnh vào nhân vật có sẵn"])

        with tab1:
            st.markdown("**Tạo thư mục nhân vật mới và upload ảnh**")

            col1, col2 = st.columns([2, 1])

            with col1:
                new_person_name = st.text_input(
                    "Tên nhân vật (VD: nguyen_van_a, cristiano_ronaldo)",
                    key="new_person_name",
                    placeholder="Nhập tên không dấu, dùng _ thay khoảng trắng"
                )

            with col2:
                st.markdown("**Tự động:**")
                st.info("📁 Tạo folder\n📸 Lưu ảnh\n✅ Hoàn tất")

            uploaded_new_images = st.file_uploader(
                "Upload ảnh cho nhân vật mới (có thể chọn nhiều ảnh)",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key="new_person_uploader"
            )

            if st.button("🚀 Tạo nhân vật & Lưu ảnh", type="primary", key="create_person"):
                if not new_person_name or new_person_name.strip() == "":
                    st.error("❌ Vui lòng nhập tên nhân vật!")
                elif not uploaded_new_images:
                    st.error("❌ Vui lòng upload ít nhất 1 ảnh!")
                else:
                    # Clean name
                    clean_name = new_person_name.strip().replace(" ", "_")
                    person_dir = f"known_gallery/{clean_name}"

                    if os.path.exists(person_dir):
                        st.error(
                            f"❌ Nhân vật '{clean_name}' đã tồn tại! Vui lòng dùng tab 'Thêm ảnh vào nhân vật có sẵn'")
                    else:
                        try:
                            # Create directory
                            os.makedirs(person_dir, exist_ok=True)

                            # Save images
                            saved_count = 0
                            for i, uploaded_file in enumerate(uploaded_new_images, 1):
                                # Generate filename
                                ext = uploaded_file.name.split('.')[-1]
                                filename = f"{clean_name}_{i:02d}.{ext}"
                                filepath = f"{person_dir}/{filename}"

                                # Save file
                                with open(filepath, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                saved_count += 1

                            st.markdown(
                                f'<div class="success-box">✅ Đã tạo nhân vật <b>{clean_name}</b> với <b>{saved_count}</b> ảnh!<br>📁 Thư mục: {person_dir}</div>',
                                unsafe_allow_html=True
                            )
                            st.info(
                                "💡 Nhớ nhấn nút '🔨 Xây dựng Thư viện' bên dưới để cập nhật!")

                            # Force refresh
                            import time
                            time.sleep(1)
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Lỗi khi tạo nhân vật: {e}")

        with tab2:
            st.markdown("**Thêm ảnh vào nhân vật đã có**")

            if persons:
                selected_person = st.selectbox(
                    "Chọn nhân vật",
                    persons,
                    key="select_person_for_add"
                )

                # Show current images count
                person_dir = f"known_gallery/{selected_person}"
                current_images = [f for f in os.listdir(person_dir)
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                st.info(f"📊 Hiện tại có {len(current_images)} ảnh")

                uploaded_add_images = st.file_uploader(
                    f"Upload thêm ảnh cho {selected_person}",
                    type=['jpg', 'jpeg', 'png'],
                    accept_multiple_files=True,
                    key="add_images_uploader"
                )

                if st.button("💾 Lưu ảnh vào thư mục", type="primary", key="add_images"):
                    if not uploaded_add_images:
                        st.error("❌ Vui lòng chọn ảnh để upload!")
                    else:
                        try:
                            saved_count = 0
                            start_index = len(current_images) + 1

                            for i, uploaded_file in enumerate(uploaded_add_images, start_index):
                                ext = uploaded_file.name.split('.')[-1]
                                filename = f"{selected_person}_{i:02d}.{ext}"
                                filepath = f"{person_dir}/{filename}"

                                with open(filepath, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                saved_count += 1

                            st.markdown(
                                f'<div class="success-box">✅ Đã thêm <b>{saved_count}</b> ảnh vào <b>{selected_person}</b>!<br>📁 Tổng cộng: {len(current_images) + saved_count} ảnh</div>',
                                unsafe_allow_html=True
                            )
                            st.info(
                                "💡 Nhớ nhấn nút '🔨 Xây dựng Thư viện' để cập nhật embeddings!")

                            import time
                            time.sleep(1)
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Lỗi khi thêm ảnh: {e}")
            else:
                st.warning(
                    "⚠️ Chưa có nhân vật nào. Vui lòng tạo nhân vật mới ở tab bên trái!")

        st.markdown("---")

        # Show persons
        if persons:
            st.markdown("### 👤 Người Hiện tại")

            for person in persons:
                with st.expander(f"📁 {person}"):
                    person_dir = f"known_gallery/{person}"
                    images = [f for f in os.listdir(person_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                    st.markdown(f"**Số ảnh:** {len(images)}")

                    # Display images
                    if images:
                        cols = st.columns(5)
                        # Show max 10
                        for i, img_name in enumerate(images[:10]):
                            with cols[i % 5]:
                                img_path = f"{person_dir}/{img_name}"
                                img = Image.open(img_path)
                                st.image(img, caption=img_name,
                                         use_container_width=True)
        else:
            st.markdown(
                '<div class="warning-box">⚠️ Không tìm thấy người nào trong thư mục thư viện.</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Build gallery
        st.markdown("### 🔧 Thao tác Thư viện")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔨 Xây dựng Thư viện", type="primary", use_container_width=True):
                if persons and total_images > 0:
                    with st.spinner("Đang xây dựng embeddings thư viện..."):
                        try:
                            from src.pipeline.service import ReIDPipelineService
                            service = ReIDPipelineService()
                            service.build_known_gallery(
                                "known_gallery", gallery_path)
                            st.markdown(
                                '<div class="success-box">✅ Xây dựng thư viện thành công!</div>', unsafe_allow_html=True)
                            st.session_state.gallery_loaded = False  # Force reload
                            st.rerun()
                        except Exception as e:
                            st.markdown(
                                f'<div class="error-box">❌ Lỗi: {e}</div>', unsafe_allow_html=True)
                else:
                    st.warning("Vui lòng thêm thư mục người với ảnh trước!")

        with col2:
            if st.button("🔄 Tải lại Thư viện", use_container_width=True):
                st.session_state.gallery_loaded = False
                st.session_state.service = None
                st.markdown(
                    '<div class="success-box">✅ Thư viện sẽ được tải lại khi sử dụng lần tiếp.</div>', unsafe_allow_html=True)

    else:
        st.markdown(
            '<div class="warning-box">⚠️ Không tìm thấy thư mục thư viện. Đang tạo...</div>', unsafe_allow_html=True)
        os.makedirs("known_gallery", exist_ok=True)
        st.rerun()

    st.markdown("---")

    # Instructions
    st.markdown("### 📖 Hướng dẫn Sử dụng")
    st.markdown("""
    **Cách 1: Upload ảnh trực tiếp (Khuyến nghị)**
    1. Vào tab **🆕 Tạo nhân vật mới**
    2. Nhập tên nhân vật (VD: `nguyen_van_a`, `messi`, `cristiano_ronaldo`)
    3. Upload nhiều ảnh của người đó (`.jpg`, `.png`)
    4. Nhấn **🚀 Tạo nhân vật & Lưu ảnh**
    5. Nhấn **🔨 Xây dựng Thư viện** để hoàn tất
    
    **Cách 2: Thêm ảnh thủ công**
    1. Tạo thư mục trong `known_gallery/` với tên người
    2. Copy ảnh vào thư mục đó
    3. Nhấn **🔨 Xây dựng Thư viện**
    
    **Ví dụ cấu trúc:**
    ```
    known_gallery/
    ├── nguyen_van_a/
    │   ├── photo1.jpg
    │   ├── photo2.jpg
    │   └── photo3.jpg
    └── tran_thi_b/
        ├── img1.jpg
        └── img2.jpg
    ```
    
    **💡 Mẹo:**
    - Upload 5-10 ảnh/người để độ chính xác tốt hơn
    - Dùng ảnh nhiều góc độ, điều kiện ánh sáng khác nhau
    - Tên nhân vật nên không dấu, dùng _ thay khoảng trắng
    """)


# =====================================
# SETTINGS PAGE
# =====================================
elif page == "⚙️ Cài đặt":
    st.markdown('<h1 class="main-header">⚙️ Cài đặt</h1>',
                unsafe_allow_html=True)

    st.markdown("### 🎛️ Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**YOLO Model**")
        yolo_path = st.text_input(
            "YOLO Model Path", "models/yolov8_person_detection.pt")
        yolo_exists = os.path.exists(yolo_path)
        st.markdown(f"Status: {'✅ Found' if yolo_exists else '❌ Not found'}")

    with col2:
        st.markdown("**ReID Model**")
        reid_path = st.text_input(
            "ReID Model Path", "models/best_model_state_dict.pth")
        reid_exists = os.path.exists(reid_path)
        st.markdown(f"Status: {'✅ Found' if reid_exists else '❌ Not found'}")

    st.markdown("---")

    st.markdown("### 🏚️ Tham số Phát hiện")

    col1, col2 = st.columns(2)

    with col1:
        threshold = st.slider("Ngưỡng Nhận dạng", 0.0, 1.0, 0.8, 0.05)
        st.info(f"Hiện tại: {threshold}")
        st.markdown("Cao hơn = khớp chặt chẽ hơn")

    with col2:
        device = st.selectbox("Thiết bị", ["auto", "cpu", "cuda"])
        st.info(f"Sẽ dùng: {device}")

    st.markdown("---")

    st.markdown("### 📊 Thông tin Hệ thống")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Thư mục:**")
        for folder in ["models", "known_gallery", "uploads", "output", "temp"]:
            exists = os.path.exists(folder)
            st.markdown(f"- `{folder}`: {'✅' if exists else '❌'}")

    with col2:
        st.markdown("**Thư viện:**")
        try:
            import torch
            st.markdown(f"- PyTorch: ✅ {torch.__version__}")
        except:
            st.markdown("- PyTorch: ❌")

        try:
            import cv2
            st.markdown(f"- OpenCV: ✅ {cv2.__version__}")
        except:
            st.markdown("- OpenCV: ❌")

        try:
            from ultralytics import YOLO
            st.markdown("- Ultralytics: ✅")
        except:
            st.markdown("- Ultralytics: ❌")

    st.markdown("---")

    if st.button("🔄 Đặt lại Tất cả", type="secondary"):
        st.session_state.service = None
        st.session_state.gallery_loaded = False
        st.session_state.last_result = None
        st.markdown('<div class="success-box">✅ Đặt lại hoàn tất!</div>',
                    unsafe_allow_html=True)
        st.rerun()


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    🎯 Hệ thống Phát hiện & Nhận dạng Người | 2026
</div>
""", unsafe_allow_html=True)
