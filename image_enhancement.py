# image_enhancement.py
import cv2
import numpy as np

class ImageEnhancer:
    """Image enhancement for better detection quality"""
    
    def __init__(self, gamma: float = 1.2):
        if gamma <= 0:
            raise ValueError("gamma must be greater than zero")

        self.gamma = gamma
        # CLAHE parameters
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @staticmethod
    def _gamma_correct(image: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction to an image."""
        inverse_gamma = 1.0 / gamma
        lookup_table = np.array(
            [(value / 255.0) ** inverse_gamma * 255 for value in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(image, lookup_table)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Apply all enhancement techniques to the image"""

        if image is None or image.size == 0:
            raise ValueError("image must be a non-empty numpy array")

        gamma_corrected = self._gamma_correct(image, gamma=self.gamma)

        # # 1. Denoise - Bilateral filter (preserves edges)
        # denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        #
        # # 2. Balance brightness
        # balanced = self._balance_brightness(denoised)
        #
        # # 3. Enhance contrast with CLAHE
        # enhanced = self._apply_clahe(balanced)
        #
        # # 4. Sharpen
        # sharpened = self._sharpen(enhanced)
        
        return gamma_corrected
    
    def _balance_brightness(self, image: np.ndarray) -> np.ndarray:
        """Balance brightness for over/underexposed images"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate mean brightness
        mean_l = np.mean(l)
        
        # Adjust if too dark or too bright
        if mean_l < 80:  # Too dark
            l = cv2.add(l, int(80 - mean_l))
        elif mean_l > 180:  # Too bright
            l = cv2.subtract(l, int(mean_l - 180))
        
        # Merge channels and convert back
        balanced_lab = cv2.merge([l, a, b])
        balanced = cv2.cvtColor(balanced_lab, cv2.COLOR_LAB2BGR)
        
        return balanced
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_clahe = self.clahe.apply(l)
        
        # Merge and convert back
        enhanced_lab = cv2.merge([l_clahe, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """Sharpen blurry images"""
        # Create sharpening kernel
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        
        # Apply kernel
        sharpened = cv2.filter2D(image, -1, kernel)
        
        return sharpened
    
    def denoise_gaussian(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply Gaussian blur for denoising (alternative method)"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for varying lighting conditions"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)


# -------------------------
# Convenience API
# -------------------------
# Provide a simple module-level function `enhance(img)` so older code that calls
# `image_enhancement.enhance(img)` continues to work. The implementation uses a
# module-level singleton `ImageEnhancer` to avoid re-instantiation overhead.
#
# Bilingual comment:
# EN: Call `image_enhancement.enhance(img)` for a quick enhancement pipeline.
# VI: Gọi `image_enhancement.enhance(img)` để xử lý ảnh một cách đơn giản.

_default_enhancer = ImageEnhancer()

def enhance(image: np.ndarray) -> np.ndarray:
    """Convenience wrapper around ImageEnhancer.enhance

    Args:
        image: OpenCV BGR image (numpy.ndarray)
    Returns:
        Enhanced image (same type as input)
    """
    return _default_enhancer.enhance(image)
