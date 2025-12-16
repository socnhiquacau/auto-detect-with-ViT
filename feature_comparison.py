import cv2
import numpy as np
from feature_extractor import FeatureExtractor


class FeatureComparison:
    """Class để so sánh 2 ảnh - TEST ONLY"""
    
    def __init__(self):
        print(f"🎯 Loading ViT model...")
        self.feature_extractor = FeatureExtractor("models/dino_vits16_epoch100.pth")
        print("✅ ViT model loaded\n")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Đọc ảnh"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"❌ Could not read image: {image_path}")
        print(f"✅ Loaded: {image_path}")
        return image
    
    def extract_feature(self, image: np.ndarray) -> np.ndarray:
        """Trích xuất feature"""
        feature = self.feature_extractor.extract(image)
        print(f"   Feature shape: {feature.shape}")
        return feature
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Tính cosine similarity"""
        # Tính norm trước normalize
        norm1_before = np.linalg.norm(vec1)
        norm2_before = np.linalg.norm(vec2)
        print(f"   Norms BEFORE normalize - vec1: {norm1_before:.6f}, vec2: {norm2_before:.6f}")
        
        # Chuẩn hóa vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        vec1_normalized = vec1 / norm1
        vec2_normalized = vec2 / norm2
        
        # Kiểm tra norms sau normalize
        print(f"   Norms AFTER normalize - vec1: {np.linalg.norm(vec1_normalized):.6f}, vec2: {np.linalg.norm(vec2_normalized):.6f}")
        
        # Tính cosine similarity
        similarity = np.dot(vec1_normalized, vec2_normalized)
        print(f"   Raw dot product: {similarity:.6f}")
        return float(similarity)
    
    def compare_two_images(self, path1: str, path2: str):
        """So sánh 2 ảnh"""
        print(f"📷 Image 1: {path1}")
        img1 = self.load_image(path1)
        print("📊 Extracting features...")
        feature1 = self.extract_feature(img1)
        
        print(f"\n📷 Image 2: {path2}")
        img2 = self.load_image(path2)
        print("📊 Extracting features...")
        feature2 = self.extract_feature(img2)
        
        # Debug info
        print(f"\n🔍 DEBUG INFO:")
        print(f"   Feature1 - Min: {np.min(feature1):.6f}, Max: {np.max(feature1):.6f}, Mean: {np.mean(feature1):.6f}")
        print(f"   Feature2 - Min: {np.min(feature2):.6f}, Max: {np.max(feature2):.6f}, Mean: {np.mean(feature2):.6f}")
        print(f"   Distance (L2): {np.linalg.norm(feature1 - feature2):.6f}")
        
        print("\n⚖️  Calculating similarity...")
        similarity = self.cosine_similarity(feature1, feature2)
        
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
    # Hardcode đường dẫn ảnh test
    image1 = "known_persons/cr7.png"
    image2 = "known_persons/messi.png"
    
    comparator = FeatureComparison()
    comparator.compare_two_images(image1, image2)
