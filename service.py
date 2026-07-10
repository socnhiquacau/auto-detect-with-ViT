import os
import shutil
import random
from pathlib import Path

def split_dataset_clean(src_dir, out_dir, train=0.70, val=0.15, test=0.15, seed=42):
    """
    Chia data KHÔNG trùng lặp giữa các split.
    Mỗi ảnh chỉ xuất hiện đúng 1 lần trong toàn bộ dataset.
    """
    random.seed(seed)

    # Xóa thư mục cũ nếu có (tránh lẫn data cũ)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        print(f'🗑️  Đã xóa thư mục cũ: {out_dir}')

    for class_name in os.listdir(src_dir):
        class_path = Path(src_dir) / class_name
        if not class_path.is_dir():
            continue

        # Lấy toàn bộ ảnh
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        imgs = sorted([f for f in class_path.iterdir()
                       if f.suffix.lower() in exts])

        # Shuffle ngẫu nhiên
        random.shuffle(imgs)

        # Tính điểm cắt
        n = len(imgs)
        n_train = int(n * train)
        n_val   = int(n * val)

        # Chia KHÔNG trùng
        splits = {
            'train': imgs[:n_train],
            'val'  : imgs[n_train : n_train + n_val],
            'test' : imgs[n_train + n_val:]           # phần còn lại
        }

        # Copy vào thư mục tương ứng
        for split_name, files in splits.items():
            dst = Path(out_dir) / split_name / class_name
            dst.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dst / f.name)

        print(f'✅ {class_name:10s}: '
              f'train={len(splits["train"]):4d} | '
              f'val={len(splits["val"]):4d} | '
              f'test={len(splits["test"]):4d} | '
              f'tổng={n:4d}')


# ============================================================
# 🚀 CHẠY LẠI
# ============================================================
split_dataset_clean(
    src_dir = r'V:\Doc\Thesis\classification_vit',          # thư mục gốc known/ unknown/
    out_dir = r'V:\Doc\Thesis\classification_vit/dataset_split_v2',   # thư mục mới
    train=0.70, val=0.15, test=0.15,
    seed=42
)