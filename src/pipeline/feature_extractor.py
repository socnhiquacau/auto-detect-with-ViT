import torch
import numpy as np
import cv2
from PIL import Image
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from src.core import common
from src.core.common import load_model_from_models


class FeatureExtractor:
    """
    Extract feature vectors using a ViT-style model loaded from local models/ directory.

    Supports two model types:
    1. DINO-style models: have get_intermediate_layers() method
       - Extracts CLS token from transformer layer as feature
    2. Standard models: use model.forward() output
       - Flattens output to 1D feature vector

    Features:
    - Always loads models from models/ directory (no external downloads)
    - Normalizes feature vectors to unit length (L2-norm = 1)
    - Supports batch and single image feature extraction
    """

    def __init__(self, model_name: str = None, device: torch.device = None):
        """
        Initialize FeatureExtractor with model from models/ directory.

        Args:
            model_name: Optional model filename or pattern (e.g., 'vit_final_model.pth')
                       If None, loads first available .pt/.pth file from models/
            device: torch device (default: cuda if available, else cpu)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name is None:
            print("📦 Loading default model from models/ (first .pt/.pth file)")
        else:
            print(f"📦 Loading FeatureExtractor model: {model_name}")

        # Load model strictly from models/ directory using common.load_model_from_models()
        # This function automatically handles:
        # - Finding the model file in models/
        # - Loading full models or state_dicts
        # - Building architecture from state_dict if needed
        # - Removing DataParallel "module." prefix if present
        self.model = load_model_from_models(model_name, device=self.device)

        # Ensure model is in eval mode and on correct device
        self.model.eval()
        self.model.to(self.device)


    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from a single OpenCV BGR image.

        Args:
            image: OpenCV BGR image (H, W, 3) as numpy array

        Returns:
            Feature vector as 1D numpy array (384-dim for ViT)
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray (OpenCV BGR image)")

        # 1. Preprocess image: Convert BGR -> RGB, resize+pad, normalize
        img = Image.fromarray(image[:, :, ::-1])  # BGR -> RGB
        img = common.to_tensor(img).unsqueeze(0).to(self.device)


        # 2. Forward pass through model (inference mode - no grad)
        with torch.no_grad():
            feat = self.model(img)

            if isinstance(feat, (list, tuple)):
                feat = feat[0]

            feat = feat.squeeze(0)
            feat = torch.nn.functional.normalize(feat, p=2, dim=0)

        return feat.detach().cpu().numpy()


    def extract_batch(self, images: list) -> np.ndarray:
        """
        Extract features from a list of OpenCV BGR images.

        Args:
            images: List of OpenCV BGR images (each is H x W x 3 numpy array)

        Returns:
            Feature matrix as numpy array of shape (N, D) where:
            - N = number of images
            - D = feature dimension (384 for DINO, depends on model for others)
        """
        if not isinstance(images, (list, tuple)):
            raise TypeError("images must be a list or tuple of numpy arrays")
        features = [self.extract(img) for img in images]
        return np.stack(features, axis=0)

    def get_feature_dimension(self) -> int:
        """
        Get the feature vector dimension of the model.

        This is useful for:
        - Pre-allocating arrays for feature storage
        - Database schema definition
        - Feature matching algorithms that need dimension info

        Method:
        1. Create dummy input tensor
        2. Pass through model
        3. Return size of output feature dimension

        Returns:
            Feature dimension (e.g., 384 for ViT-B models), or 384 as fallback
        """
        try:
            # Create dummy input tensor with expected shape [1, 3, 224, 224]
            # This allows the model to run without real image data
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            with torch.no_grad():
                if hasattr(self.model, "get_intermediate_layers"):
                    feat = self.model.get_intermediate_layers(dummy, n=1)[0][:, 0]
                else:
                    out = self.model(dummy)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    if out.dim() > 2:
                        out = out.view(out.size(0), -1)
                    feat = out
            return int(feat.shape[1])
        except Exception:
            # Fallback: many ViT models have 384-dim features
            return 384
