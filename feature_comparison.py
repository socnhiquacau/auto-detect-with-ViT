import cv2
import numpy as np
from common import DataLoader


class FeatureComparison:
    """
    Class để so sánh feature vectors của 2 ảnh - TEST ONLY

    Uses centralized DataLoader from common.py for:
    - Unified model loading
    - Consistent image preprocessing
    - Normalized feature extraction
    """

    def __init__(self):
        print(f"🎯 Loading ViT model...")
        # Use DataLoader for unified model loading from common.py
        self.data_loader = DataLoader()
        self.feature_extractor = self.data_loader.get_feature_extractor("vit_final_model.pth")
        print("✅ ViT model loaded\n")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file using OpenCV.

        Args:
            image_path: Path to image file

        Returns:
            OpenCV BGR image as numpy array
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"❌ Could not read image: {image_path}")
        print(f"✅ Loaded: {image_path}")
        return image
    
    def extract_feature(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from image.

        Uses FeatureExtractor from DataLoader which handles:
        - Image preprocessing (resize+pad, normalization)
        - Model forward pass
        - Feature normalization (L2-norm = 1)

        Args:
            image: OpenCV BGR image (H, W, 3)

        Returns:
            Normalized feature vector (384-dim for DINO)
        """
        feature = self.feature_extractor.extract(image)
        print(f"   Feature shape: {feature.shape}")
        return feature
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Formula: similarity = (vec1 · vec2) / (||vec1|| × ||vec2||)

        Properties:
        - Range: [-1, 1] where 1 = identical vectors
        - For normalized vectors (norm=1), equals dot product
        - Robust to vector magnitude differences

        Args:
            vec1: First feature vector
            vec2: Second feature vector

        Returns:
            Cosine similarity score
        """
        # Calculate norms before normalization for debugging
        norm1_before = np.linalg.norm(vec1)
        norm2_before = np.linalg.norm(vec2)
        print(f"   Norms BEFORE normalize - vec1: {norm1_before:.6f}, vec2: {norm2_before:.6f}")
        
        # Normalize vectors to unit length
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        vec1_normalized = vec1 / norm1
        vec2_normalized = vec2 / norm2
        
        # Verify normalization
        print(f"   Norms AFTER normalize - vec1: {np.linalg.norm(vec1_normalized):.6f}, vec2: {np.linalg.norm(vec2_normalized):.6f}")
        
        # Calculate cosine similarity (dot product of normalized vectors)
        similarity = np.dot(vec1_normalized, vec2_normalized)
        print(f"   Raw dot product: {similarity:.6f}")
        return float(similarity)
    
    def compare_two_images(self, path1: str, path2: str) -> float:
        """
        Compare two images and return similarity score.

        Pipeline:
        1. Load images using cv2 (OpenCV)
        2. Extract features using FeatureExtractor (from DataLoader)
        3. Calculate cosine similarity
        4. Display results and interpretation

        Args:
            path1: Path to first image
            path2: Path to second image

        Returns:
            Similarity score (0-1)
        """
        # Load and process first image
        print(f"📷 Image 1: {path1}")
        img1 = self.load_image(path1)
        print("📊 Extracting features...")
        feature1 = self.extract_feature(img1)
        
        # Load and process second image
        print(f"\n📷 Image 2: {path2}")
        img2 = self.load_image(path2)
        print("📊 Extracting features...")
        feature2 = self.extract_feature(img2)
        
        # Print debug information
        print(f"\n🔍 DEBUG INFO:")
        print(f"   Feature1 - Min: {np.min(feature1):.6f}, Max: {np.max(feature1):.6f}, Mean: {np.mean(feature1):.6f}")
        print(f"   Feature2 - Min: {np.min(feature2):.6f}, Max: {np.max(feature2):.6f}, Mean: {np.mean(feature2):.6f}")
        print(f"   Distance (L2): {np.linalg.norm(feature1 - feature2):.6f}")
        
        # Calculate similarity
        print("\n⚖️  Calculating similarity...")
        similarity = self.cosine_similarity(feature1, feature2)
        
        # Display results
        print(f"\n{'='*50}")
        print(f"🎯 Similarity Score: {similarity:.6f}")
        if similarity >= 0.8:
            print("✅ MATCH - Same person!")
        elif similarity >= 0.6:
            print("⚠️  Possible match")
        else:
            print("❌ NO MATCH - Different persons")
        print(f"{'='*50}\n")
        
        return similarity


if __name__ == "__main__":
    # Test configuration (hardcoded paths for testing)
    image1 = "known_persons/test1.png"
    image2 = "known_persons/test3.png"

    # Create comparator using centralized DataLoader from common.py
    comparator = FeatureComparison()

    # Compare two images
    comparator.compare_two_images(image1, image2)


