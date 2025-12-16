import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

class FeatureExtractor:
    """Extract feature vectors using Vision Transformer with DINO"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load your trained ViT-DINO model
        if model_path and os.path.exists(model_path):
            print(f"📦 Loading ViT model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            print("✅ Checkpoint loaded successfully")
            
            # Create base model
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            
            # Load checkpoint into model
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
                print("✅ Loaded state_dict from 'model' key")
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
                print("✅ Loaded state_dict from 'state_dict' key")
            else:
                # Assume checkpoint is state_dict directly
                self.model.load_state_dict(checkpoint)
                print("✅ Loaded state_dict directly")
        else:
            # Fallback: Use pretrained DINO model from torch hub
            print("⚠️  No custom model found, using pretrained DINO model")
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        
        self.model.eval()
        self.model.to(self.device)
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from person image
        
        Args:
            image: numpy array (BGR format from OpenCV)
        
        Returns:
            feature_vector: numpy array of features
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply transformations
        image_tensor = self.transform(pil_image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(image_tensor)
        
        # Convert to numpy and normalize to unit vector
        feature_vector = features.cpu().numpy().flatten()
        feature_vector = self._normalize_vector(feature_vector)
        
        return feature_vector
    
    def extract_batch(self, images: list) -> np.ndarray:
        """
        Extract feature vectors from multiple images
        
        Args:
            images: list of numpy arrays
        
        Returns:
            feature_vectors: numpy array of shape (n_images, feature_dim)
        """
        feature_vectors = []
        
        for image in images:
            feature_vector = self.extract(image)
            feature_vectors.append(feature_vector)
        
        return np.array(feature_vectors)
    
    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        """Normalize feature vector to unit length"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def get_feature_dimension(self) -> int:
        """Return the dimension of feature vectors"""
        # For ViT-S/16, typical dimension is 384
        # Adjust based on your model
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            features = self.model(dummy_input)
        return features.shape[1]