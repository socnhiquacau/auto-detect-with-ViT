# 🔄 Code Comparison - Before & After

## 1️⃣ Imports & Config

### ❌ BEFORE
```python
import os
import sys
from pathlib import Path

# Import DataLoader for unified model loading
from common import DataLoader, YOLO_MODEL_PATH, VIT_MODEL_PATH

# Hardcoded defaults with long URLs
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:admin123@localhost:27017/video_detection_db?authSource=admin")
DATABASE_NAME = os.getenv("DATABASE_NAME", "video_detection_db")
```

### ✅ AFTER
```python
import re
from pathlib import Path
from common import DataLoader, MONGODB_URL, DATABASE_NAME

# Centralized config from common.py
MONGODB_URL = os.getenv("MONGODB_URL", MONGODB_URL)
DATABASE_NAME = os.getenv("DATABASE_NAME", DATABASE_NAME)

# Hardcoded folder path (not dynamic from args)
KNOWN_PERSONS_DIR = "known_persons"
```

**Thay Đổi:**
- ❌ Xóa `import sys` (không dùng sys.argv)
- ✅ Thêm `import re` (sanitize ID)
- ✅ Import `MONGODB_URL, DATABASE_NAME` từ `common.py`
- ✅ Thêm `KNOWN_PERSONS_DIR = "known_persons"` (cứng hóa)

---

## 2️⃣ Main Entry Point

### ❌ BEFORE
```python
async def main():
    """
    Main entry point for adding known persons to database.

    Supports operations:
    - add <folder>: Add persons from folder (auto-detect structure)
    - list: List all known persons
    - delete <person_id>: Delete a person by ID
    """
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python add_known_persons.py add <folder_path>       - Add from folder")
        print("  python add_known_persons.py list                    - List all persons")
        print("  python add_known_persons.py delete <person_id>      - Delete person")
        return

    operation = sys.argv[1].lower()

    try:
        if operation == "add" and len(sys.argv) >= 3:
            folder_path = sys.argv[2]
            await add_persons_from_folder(folder_path)

        elif operation == "list":
            await list_known_persons()

        elif operation == "delete" and len(sys.argv) >= 3:
            person_id = sys.argv[2]
            await delete_person(person_id)

        else:
            print(f"❌ Unknown operation: {operation}")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
```

### ✅ AFTER
```python
async def main():
    """
    Điểm khởi động chính - Gọi các hàm trực tiếp, không cần arg[].
    
    Main entry point - Call functions directly without CLI arguments.
    
    Các hàm có sẵn:
    - add_known_persons() - Thêm người từ KNOWN_PERSONS_DIR
    - list_known_persons() - Liệt kê tất cả
    - delete_person_by_id(person_id) - Xóa theo ID
    - delete_all_persons() - Xóa tất cả
    - update_person_by_id(person_id, new_name) - Cập nhật tên
    """
    # === CHỈNH SỬA ĐÂY - EDIT HERE ===
    
    # Thêm người từ known_persons / Add from known_persons
    await add_known_persons()
    
    # Liệt kê tất cả / List all
    await list_known_persons()
    
    # Xóa người theo ID / Delete by ID
    # await delete_person_by_id("person_name")
    
    # Xóa tất cả / Delete all
    # await delete_all_persons()
    
    # Cập nhật tên / Update name
    # await update_person_by_id("person_id", "new_name")
    
    # === HẾT CHỈNH SỬA ===


if __name__ == "__main__":
    """
    Điểm vào của script / Script entry point.
    
    Chạy: python add_known_persons.py
    Run: python add_known_persons.py
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n⚠️  Bị gián đoạn bởi người dùng / Interrupted by user")
    except Exception as e:
        print(f"\n❌ Lỗi / Error: {e}")
        import traceback
        traceback.print_exc()
```

**Thay Đổi:**
- ✅ Không dùng `sys.argv` check
- ✅ Gọi hàm trực tiếp (uncomment để chạy)
- ✅ Comment 2 ngôn ngữ
- ✅ Clear instructions ("EDIT HERE")

---

## 3️⃣ Manager Class - Folder Method

### ❌ BEFORE
```python
async def add_from_folder(self, folder_path: str, structure: str = "auto"):
    """
    Add known persons from folder.

    Args:
        folder_path: Path to folder containing person images
        structure: "flat", "nested", or "auto" (auto-detect)
    """
    folder = Path(folder_path)

    if not folder.exists():
        print(f"❌ Folder not found: {folder_path}")
        return

    # Auto-detect folder structure if not specified
    if structure == "auto":
        structure = self._detect_structure(folder)
        print(f"📁 Detected structure: {structure}")

    if structure == "flat":
        await self._process_flat_structure(folder)
    elif structure == "nested":
        await self._process_nested_structure(folder)
    else:
        print(f"❌ Unknown structure: {structure}")
```

### ✅ AFTER
```python
async def add_from_known_persons_folder(self):
    """
    Thêm người nổi tiếng từ thư mục KNOWN_PERSONS_DIR.
    
    Add known persons from KNOWN_PERSONS_DIR (hardcoded path).
    
    Quy trình:
    1. Kiểm tra thư mục có tồn tại không
    2. Tự động phát hiện cấu trúc (flat hoặc nested)
    3. Xử lý cấu trúc phù hợp
    4. In kết quả
    """
    folder = Path(KNOWN_PERSONS_DIR)

    if not folder.exists():
        print(f"❌ Thư mục không tồn tại: {KNOWN_PERSONS_DIR}")
        print(f"❌ Folder not found: {KNOWN_PERSONS_DIR}")
        return False

    print(f"\n🚀 Bắt đầu xử lý thư mục: {KNOWN_PERSONS_DIR}")
    print(f"🚀 Starting processing folder: {KNOWN_PERSONS_DIR}")

    # Tự động phát hiện cấu trúc thư mục
    # Auto-detect folder structure
    structure = self._detect_structure(folder)
    print(f"📁 Phát hiện cấu trúc: {structure} / Detected structure: {structure}")

    # Xử lý cấu trúc phù hợp
    # Process detected structure
    if structure == "flat":
        return await self._process_flat_structure(folder)
    elif structure == "nested":
        return await self._process_nested_structure(folder)
    else:
        print(f"❌ Cấu trúc không hợp lệ / Invalid structure: {structure}")
        return False
```

**Thay Đổi:**
- ✅ Tên hàm: `add_from_folder()` → `add_from_known_persons_folder()`
- ✅ Xóa tham số `folder_path` (dùng cứng hóa `KNOWN_PERSONS_DIR`)
- ✅ Xóa tham số `structure` (luôn auto-detect)
- ✅ Thêm return type: `-> bool`
- ✅ Comment 2 ngôn ngữ
- ✅ Chi tiết steps

---

## 4️⃣ Helper Functions - Add Function

### ❌ BEFORE
```python
async def add_persons_from_folder(folder_path: str):
    """
    Add known persons from folder to database.

    Steps:
    1. Connect to MongoDB
    2. Initialize FeatureExtractor using DataLoader
    3. Load images from folder
    4. Extract features
    5. Save to database

    Args:
        folder_path: Path to folder containing person images
    """
    print(f"\n🚀 Adding persons from: {folder_path}")

    # Connect to MongoDB
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]

    # Initialize DataLoader for unified model loading
    print("📦 Initializing models...")
    data_loader = DataLoader()
    feature_extractor = data_loader.get_feature_extractor()

    # Initialize manager
    manager = KnownPersonsManager(db, feature_extractor)

    # Process folder
    await manager.add_from_folder(folder_path, structure="auto")

    client.close()

    print("\n✅ Done!")
```

### ✅ AFTER
```python
async def add_known_persons():
    """
    Thêm người nổi tiếng từ thư mục KNOWN_PERSONS_DIR vào database.
    
    Add known persons from KNOWN_PERSONS_DIR to database.
    
    Quy trình:
    1. Kết nối MongoDB / Connect to MongoDB
    2. Khởi tạo FeatureExtractor từ DataLoader / Initialize FeatureExtractor
    3. Tạo KnownPersonsManager / Create manager
    4. Xử lý thư mục người / Process known persons folder
    5. Đóng kết nối / Close connection
    """
    print(f"\n{'='*60}")
    print(f"🚀 THÊM NGƯỜI NỖI TIẾNG")
    print(f"🚀 ADD KNOWN PERSONS")
    print(f"{'='*60}")

    try:
        # Kết nối MongoDB / Connect to MongoDB
        print(f"\n📡 Kết nối database / Connecting to database...")
        client = AsyncIOMotorClient(MONGODB_URL)
        db = client[DATABASE_NAME]
        print(f"✅ Kết nối thành công / Connected")

        # Khởi tạo DataLoader & FeatureExtractor / Initialize models
        print(f"\n📦 Khởi tạo mô hình / Initializing models...")
        data_loader = DataLoader()
        feature_extractor = data_loader.get_feature_extractor()
        print(f"✅ Mô hình sẵn sàng / Models ready")

        # Tạo manager / Create manager
        manager = KnownPersonsManager(db, feature_extractor)

        # Xử lý thư mục / Process folder
        success = await manager.add_from_known_persons_folder()

        if success:
            print(f"\n✅ HOÀN TẤT THÀNH CÔNG / COMPLETED SUCCESSFULLY")
        else:
            print(f"\n⚠️  KHÔNG CÓ DỮ LIỆU / NO DATA PROCESSED")

        return success

    except Exception as e:
        print(f"\n❌ LỖI / ERROR: {e}")
        return False

    finally:
        # Đóng kết nối / Close connection
        if 'client' in locals():
            client.close()
            print(f"\n📡 Đóng kết nối / Connection closed")
```

**Thay Đổi:**
- ✅ Tên hàm: `add_persons_from_folder()` → `add_known_persons()`
- ✅ Xóa tham số `folder_path` (xài `KNOWN_PERSONS_DIR`)
- ✅ Thêm try-except-finally (proper error handling)
- ✅ Thêm return type: `-> bool`
- ✅ Better status output (headers, steps)
- ✅ Comment 2 ngôn ngữ

---

## 5️⃣ Delete Function

### ❌ BEFORE
```python
async def delete_person(person_id: str):
    """Delete a person from database"""
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]
    collection = db.known_persons

    result = await collection.delete_one({"person_id": person_id})

    if result.deleted_count > 0:
        print(f"✅ Deleted person: {person_id}")
    else:
        print(f"❌ Person not found: {person_id}")

    client.close()
```

### ✅ AFTER
```python
async def delete_person_by_id(person_id: str):
    """
    Xóa một người khỏi database theo ID.
    
    Delete a person from database by ID.

    Args:
        person_id: ID duy nhất người / Unique person ID
    """
    print(f"\n{'='*60}")
    print(f"🗑️  XÓA NGƯỜI NỖI TIẾNG")
    print(f"🗑️  DELETE KNOWN PERSON")
    print(f"{'='*60}")

    try:
        # Kết nối database / Connect to database
        client = AsyncIOMotorClient(MONGODB_URL)
        db = client[DATABASE_NAME]
        collection = db.known_persons

        # Xóa người / Delete person
        result = await collection.delete_one({"person_id": person_id})

        if result.deleted_count > 0:
            print(f"\n✅ Xóa thành công / Deleted: {person_id}")
        else:
            print(f"\n❌ Không tìm thấy / Not found: {person_id}")

    except Exception as e:
        print(f"\n❌ LỖI / ERROR: {e}")

    finally:
        if 'client' in locals():
            client.close()
```

**Thay Đổi:**
- ✅ Tên hàm: `delete_person()` → `delete_person_by_id()`
- ✅ Thêm try-except-finally
- ✅ Better docstring
- ✅ Comment 2 ngôn ngữ
- ✅ Better formatting (headers)

---

## 6️⃣ Thêm Hàm Mới

### ✅ AFTER ONLY
```python
# Hàm này không có trong version cũ
async def delete_all_persons():
    """
    Xóa tất cả người khỏi database.
    
    Delete all persons from database.
    
    ⚠️  CẢNH BÁO: Thao tác này không thể hoàn tác!
    ⚠️  WARNING: This action cannot be undone!
    """
    # ... implementation ...

async def update_person_by_id(person_id: str, person_name: str = None):
    """
    Cập nhật thông tin người (chỉ tên, không update vector).
    
    Update person info (only name, not feature vector).
    """
    # ... implementation ...
```

**Bổ Sung:**
- ✅ `delete_all_persons()` - Xóa tất cả với cảnh báo
- ✅ `update_person_by_id()` - Cập nhật tên người

---

## 📊 Tóm Tắt Thay Đổi

| Aspect | Trước | Sau | Lợi Ích |
|--------|-------|-----|---------|
| **Import** | `sys` (CLI args) | `re` (sanitize) | ✅ Không cần CLI |
| **Config** | Hardcoded URLs | From common.py | ✅ Centralized |
| **Folder** | Dynamic (arg) | Fixed (cứng hóa) | ✅ Rõ ràng |
| **Main()** | CLI arg parsing | Direct function calls | ✅ Đơn giản |
| **Error Handling** | Minimal | try-except-finally | ✅ Robust |
| **Comments** | English only | Việt + Anh | ✅ Rõ ràng |
| **Functions** | 5 | 6 (+ 2 new) | ✅ More options |
| **Documentation** | None | Extensive | ✅ Clear |

---

## 🎯 Kết Quả

**Trước:**
- ❌ Phức tạp: phải nhớ CLI syntax
- ❌ Khó bảo trì: config rải rác
- ❌ Ít documentation

**Sau:**
- ✅ Đơn giản: gọi hàm trực tiếp
- ✅ Dễ bảo trì: config tập trung
- ✅ Chi tiết: comment 2 ngôn ngữ

---

**✨ Refactoring thành công!**


