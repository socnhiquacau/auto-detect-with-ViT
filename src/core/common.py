import os
import ssl
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from dotenv import load_dotenv

# ⚠️  SSL bypass for development only (allows torch.hub downloads)
# Remove this in production!
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# =========================
# CONFIG
# =========================
TARGET_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Environment Configuration
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
KNOWN_PERSON_THRESHOLD = float(os.getenv("KNOWN_PERSON_THRESHOLD", "0.95"))
FRAME_EXTRACTION_FPS = int(os.getenv("FRAME_EXTRACTION_FPS", "2"))
YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH", "models/yolov8_person_detection.pt")
VIT_MODEL_PATH = os.getenv("VIT_MODEL_PATH", "models/vit_final_model.pth")
CLASSIFIER_MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH", "best_vit.pth")

# =========================
# RESIZE + PAD
# =========================


def resize_with_padding(
    img: Image.Image,
    target_size: int = TARGET_SIZE,
    fill_color=(0, 0, 0)
) -> Image.Image:
    """
    Resize image to (target_size x target_size) using letterbox:
    - Keep aspect ratio
    - No distortion

    Strategy: Calculate scaling factor based on the largest dimension, resize image,
    then center it on a blank canvas to maintain aspect ratio.
    """
    w, h = img.size
    # Calculate scale factor based on longest dimension to fit within target_size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image with bilinear interpolation for better quality
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Create blank canvas and paste resized image in center
    new_img = Image.new("RGB", (target_size, target_size), fill_color)
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    new_img.paste(img, (left, top))
    return new_img


# =========================
# TRANSFORM (AFTER RESIZE)
# =========================
# ImageNet normalization: these are standard mean/std values used to normalize
# images for pre-trained Vision Transformer models. Convert pixel values (0-1)
# to standardized distribution expected by model training
to_tensor = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to [0,1] float tensor
    # Normalize using ImageNet statistics
    transforms.Normalize(mean=MEAN, std=STD)
])


# =========================
# PREPROCESS HELPERS
# =========================

def preprocess_pil(
    img: Image.Image,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Preprocess a PIL image: resize+pad -> normalize -> tensor [1,3,224,224]
    """
    img = img.convert("RGB")
    img = resize_with_padding(img)
    tensor = to_tensor(img).unsqueeze(0).to(device)
    return tensor


def preprocess_image(
    img_path: str,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Read image from path -> preprocess -> tensor
    """
    img = Image.open(img_path)
    return preprocess_pil(img, device=device)


def preprocess_cv2(
    img_bgr: np.ndarray,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Preprocess an OpenCV BGR image (numpy array)
    """
    # BGR -> RGB
    import cv2
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    return preprocess_pil(img, device=device)


# =========================
# MODEL LOADER (LOCAL ONLY)
# =========================

def _models_dir() -> str:
    # Use models/ folder in repository root.
    return str(PROJECT_ROOT / "models")


def find_model_file(preferred_name: str = None) -> str:
    """
    Find a model file inside the local models/ directory.

    Behavior improvements vs. previous implementation:
    - If `preferred_name` starts with "models/" or "models\\" the prefix will be stripped.
    - If `preferred_name` is an absolute path it will be accepted only if it resides inside the models/ folder.
    - If `preferred_name` is a relative path, it will be resolved relative to the repository root and checked.
    - If no exact match is found, the function will do a recursive search under models/ for any .pt/.pth file that
      contains `preferred_name` as a substring in the filename or path.
    - If preferred_name is None, returns the first top-level .pt/.pth file under models/ (unchanged behavior).
    """
    models_dir = _models_dir()
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"models/ directory not found at {models_dir}")

    # helper: gather all model files (recursive)
    all_model_files = []
    for root, _, files in os.walk(models_dir):
        for f in files:
            if f.lower().endswith((".pt", ".pth")):
                all_model_files.append(os.path.join(root, f))

    # If user didn't request a specific name, return the first top-level candidate
    if not preferred_name:
        # prefer files directly under models/ (non-recursive) if any
        top_level = [f for f in os.listdir(
            models_dir) if f.lower().endswith((".pt", ".pth"))]
        if top_level:
            return os.path.join(models_dir, top_level[0])
        # otherwise fallback to recursive picks
        if all_model_files:
            return all_model_files[0]
        raise FileNotFoundError(
            f"No .pt/.pth model files found in {models_dir}")

    # Normalize preferred_name: strip models/ prefix if present
    norm_name = preferred_name.replace("\\", "/")
    if norm_name.startswith("models/"):
        norm_name = norm_name[len("models/"):]

    # If an absolute path was given, accept it only if it exists and is inside models_dir
    if os.path.isabs(preferred_name):
        if os.path.exists(preferred_name):
            # ensure inside models dir
            if os.path.commonpath([os.path.abspath(preferred_name), os.path.abspath(models_dir)]) == os.path.abspath(models_dir):
                return os.path.abspath(preferred_name)
            else:
                raise ValueError(
                    f"Provided absolute model path is outside models/: {preferred_name}")
        else:
            raise FileNotFoundError(
                f"Model file not found at absolute path: {preferred_name}")

    # First check direct file under models_dir using normalized name
    direct_path = os.path.join(models_dir, norm_name)
    if os.path.exists(direct_path):
        return direct_path

    # Next, check top-level simple filename match (e.g., user passed 'dino_vits16.pth')
    candidates = [os.path.join(models_dir, f) for f in os.listdir(
        models_dir) if f.lower().endswith((".pt", ".pth"))]
    for c in candidates:
        if os.path.basename(c).lower() == norm_name.lower():
            return c

    # Finally, fallback to recursive substring match in full recursive listing
    for p in all_model_files:
        if norm_name.lower() in os.path.basename(p).lower() or norm_name.lower() in p.lower():
            return p

    # Not found: prepare helpful diagnostic message
    available = "\n".join([os.path.relpath(p, models_dir)
                          for p in all_model_files]) if all_model_files else "(none)"
    raise FileNotFoundError(
        f"Model {preferred_name} not found inside {models_dir}.\n"
        f"Searched direct path: {direct_path}\n"
        f"Available model files under models/:\n{available}\n"
        "Tip: pass just the filename (e.g. 'dino_vits16_epoch100.pth') or a substring that appears in the filename."
    )


def _build_model_from_state_dict(state_dict: dict, device: torch.device = None):
    """
    Xây dựng và load VisionTransformer từ state_dict.

    Cách hoạt động:
    - Lấy model architecture từ torch.hub (DINO ViT-S/16)
    - Load state_dict (weights) vào model
    - Xử lý prefix "module." nếu train bằng DataParallel
    - Xử lý prefix "vit." nếu state_dict được lưu từ wrapper module

    Args:
        state_dict: Dictionary chứa weights model
        device: torch device (default: cpu)

    Returns:
        VisionTransformer model đã load weights
    """
    device = device or torch.device("cpu")

    # 1. Xây dựng architecture từ torch.hub (DINO ViT-S/16)
    # Đây là cách standard để lấy model architecture chuẩn từ Facebook Research
    try:
        print("   🔄 Building ViT-S/16 architecture from torch.hub...")
        vit = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vits16",
            pretrained=False  # Không load pretrained weights, chỉ architecture
        )
    except Exception as e:
        print(f"   ⚠️  Warning: Could not load from torch.hub: {e}")
        print("   Trying offline fallback...")
        raise RuntimeError(
            f"Cannot build model architecture. Please ensure torch.hub can access DINO model.\n"
            f"Error: {e}"
        )

    # 2. Xử lý prefix trong state_dict
    # - Loại bỏ prefix "module." nếu training dùng DataParallel (single GPU -> multi-GPU)
    # - Loại bỏ prefix "vit." nếu state_dict được lưu từ wrapper module
    processed_state_dict = {}
    for k, v in state_dict.items():
        # Xử lý DataParallel prefix
        name = k[7:] if k.startswith('module.') else k
        # Xử lý wrapper module prefix (vit.)
        name = name[4:] if name.startswith('vit.') else name
        processed_state_dict[name] = v

    # 3. Load weights (strict=False để an toàn với layer dư thừa)
    try:
        missing, unexpected = vit.load_state_dict(
            processed_state_dict, strict=False)

        if missing:
            print(f"   [Warning] Missing keys: {missing[:3]}...")
        if unexpected:
            print(f"   [Warning] Unexpected keys: {unexpected[:3]}...")
    except RuntimeError as e:
        print(f"   ⚠️  Could not load all weights: {e}")
        print("   Continuing with partial weights...")

    vit.eval()
    vit.to(device)
    return vit


def load_model_from_models(
        filename: str = None,
        model_arch: torch.nn.Module = None,
        device: torch.device = None
) -> torch.nn.Module:
    """
    Load model từ thư mục models/ - tất cả models PHẢI nằm trong folder này.

    Hỗ trợ:
    - Full models (nn.Module object)
    - TorchScript models
    - State dict (weights) - tự động xây dựng architecture
    - Checkpoint dict (chứa state_dict + metadata)

    Args:
        filename: Tên model hoặc pattern (e.g., 'vit_final_model.pth')
        model_arch: Kiến trúc model để load weights (nếu file là state_dict)
        device: torch device (default: cuda if available, else cpu)

    Returns:
        Loaded model instance (nn.Module)
    """
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    path = find_model_file(filename)
    print(f"Loading model from: {path}")

    # 1) Try TorchScript (.jit)
    try:
        model = torch.jit.load(path, map_location=device)
        model.eval()
        model.to(device)
        print(f"✅ Loaded TorchScript model")
        return model
    except Exception:
        pass

    # 2) Try torch.load
    obj = torch.load(path, map_location=device)

    # Case A: File chứa full model (nn.Module)
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        obj.to(device)
        print(f"✅ Loaded full model (nn.Module)")
        return obj

    # Case B: File chứa Dictionary (state_dict hoặc checkpoint)
    if isinstance(obj, dict):
        # Tìm state_dict thực sự từ checkpoint (thường là 'model', 'state_dict', etc.)
        state_dict = None
        for key in ("model", "net", "module", "state_dict"):
            if key in obj:
                candidate = obj[key]
                # Nếu tìm được model object thì trả về ngay
                if isinstance(candidate, torch.nn.Module):
                    candidate.eval()
                    candidate.to(device)
                    print(f"✅ Loaded model from checkpoint key '{key}'")
                    return candidate
                # Nếu là dict, đó là state_dict
                elif isinstance(candidate, dict):
                    state_dict = candidate
                    break

        # Nếu chưa tìm được, giả sử toàn bộ obj là state_dict
        if state_dict is None:
            state_dict = obj

        # Nếu người dùng truyền vào kiến trúc model, load weights vào
        if model_arch is not None:
            # Xử lý prefix "module."
            processed_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                processed_state_dict[name] = v

            missing, unexpected = model_arch.load_state_dict(
                processed_state_dict, strict=False)
            if missing:
                print(f"[Warning] Missing keys: {missing[:3]}...")

            model_arch.eval()
            model_arch.to(device)
            print(f"✅ Loaded state_dict into provided architecture")
            return model_arch

        # Nếu không có kiến trúc, tự động xây dựng model từ state_dict
        # (dành cho ViT models)
        model = _build_model_from_state_dict(state_dict, device=device)
        print(f"✅ Built model from state_dict and loaded weights")
        return model

    raise RuntimeError(f"Unrecognized model format in {path}")


# =========================
# DATA LOADER CLASS (UNIFIED MODEL LOADING)
# =========================

class DataLoader:
    """
    Centralized data and model loading class for the entire project.

    This class encapsulates:
    - YOLO person detection model loading
    - ViT feature extractor initialization
    - Image preprocessing pipelines

    Usage:
        loader = DataLoader(device="cuda")
        yolo_model = loader.get_yolo_model()
        feature_extractor = loader.get_feature_extractor()
    """

    def __init__(self,
                 yolo_path: str = None,
                 vit_path: str = None,
                 classifier_path: str = None,
                 device: torch.device = None):
        """
        Initialize DataLoader with model paths from environment or parameters.

        Args:
            yolo_path: Path to YOLO model (default from YOLO_MODEL_PATH env var)
            vit_path: Path to ViT model (default from VIT_MODEL_PATH env var)
            device: torch device (default: cuda if available, else cpu)
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_path = yolo_path or YOLO_MODEL_PATH
        self.vit_path = vit_path or VIT_MODEL_PATH
        self.classifier_path = classifier_path or CLASSIFIER_MODEL_PATH

        # Initialize model cache
        self._yolo_model = None
        self._feature_extractor = None
        self._known_unknown_classifier = None

    def get_yolo_model(self):
        """
        Get YOLO model for person detection. Lazy loads on first access.

        Returns:
            YOLO: Loaded YOLO model instance
        """
        if self._yolo_model is None:
            from ultralytics import YOLO
            print(f"🎯 Loading YOLO model from: {self.yolo_path}")
            self._yolo_model = YOLO(self.yolo_path)
            print("✅ YOLO model loaded successfully")
        return self._yolo_model

    def get_feature_extractor(self, model_name: str = None):
        """
        Get FeatureExtractor instance for extracting feature vectors.
        Lazy loads on first access.

        Args:
            model_name: Optional specific model name (overrides vit_path)

        Returns:
            FeatureExtractor: Instance for feature extraction
        """
        if self._feature_extractor is None:
            # Import here to avoid circular dependency
            from src.pipeline.feature_extractor import FeatureExtractor

            model_to_load = model_name or self.vit_path
            print(f"📦 Loading FeatureExtractor with model: {model_to_load}")
            self._feature_extractor = FeatureExtractor(
                model_name=model_to_load,

                device=self.device
            )
            print("✅ FeatureExtractor loaded successfully")
        return self._feature_extractor

    def get_known_unknown_classifier(self, model_name: str = None):
        """
        Get classifier for known/unknown prediction.
        Lazy loads on first access.

        Args:
            model_name: Optional specific model filename inside models/

        Returns:
            KnownUnknownClassifier: Classifier instance
        """
        if self._known_unknown_classifier is None:
            from known_unknown_classifier import KnownUnknownClassifier

            model_to_load = model_name or self.classifier_path
            print(f"📦 Loading KnownUnknownClassifier with model: {model_to_load}")
            self._known_unknown_classifier = KnownUnknownClassifier(
                model_name=model_to_load,
                device=self.device,
            )
            print("✅ KnownUnknownClassifier loaded successfully")
        return self._known_unknown_classifier

    @staticmethod
    def preprocess_image(img_path: str, device: torch.device = None) -> torch.Tensor:
        """
        Preprocess image from file path.

        Args:
            img_path: Path to image file
            device: torch device (default: cpu)

        Returns:
            torch.Tensor: Preprocessed image tensor [1, 3, 224, 224]
        """
        return preprocess_image(img_path, device or torch.device("cpu"))

    @staticmethod
    def preprocess_pil(img: Image.Image, device: torch.device = None) -> torch.Tensor:
        """
        Preprocess PIL Image.

        Args:
            img: PIL Image instance
            device: torch device (default: cpu)

        Returns:
            torch.Tensor: Preprocessed image tensor [1, 3, 224, 224]
        """
        return preprocess_pil(img, device or torch.device("cpu"))

    @staticmethod
    def preprocess_cv2(img_bgr: np.ndarray, device: torch.device = None) -> torch.Tensor:
        """
        Preprocess OpenCV BGR image (numpy array).

        Args:
            img_bgr: OpenCV BGR image (H, W, 3)
            device: torch device (default: cpu)

        Returns:
            torch.Tensor: Preprocessed image tensor [1, 3, 224, 224]
        """
        return preprocess_cv2(img_bgr, device or torch.device("cpu"))
