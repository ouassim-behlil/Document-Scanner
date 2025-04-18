import cv2
import numpy as np

class ImageQualityChecker:
    def __init__(self):
        pass

    def check_resolution(self, image_path, min_width=640, min_height=480):
        """
        Checks if the image resolution meets the minimum requirements.

        Args:
            image_path (str): Path to the input image file.
            min_width (int): Minimum acceptable width. Default is 640.
            min_height (int): Minimum acceptable height. Default is 480.

        Returns:
            bool: True if resolution is sufficient, False otherwise.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at path: {image_path}")
        
        height, width = image.shape[:2]
        return width >= min_width and height >= min_height

    def check_blur(self, image_path, threshold=100):
        """
        Checks if the image is blurry using the Laplacian variance.

        Args:
            image_path (str): Path to the input image file.
            threshold (float): Threshold for blur detection. Lower values indicate blurrier images.

        Returns:
            bool: True if the image is not blurry, False otherwise.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at path: {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var > threshold

    def check_contrast(self, image_path, threshold=30):
        """
        Checks if the image has sufficient contrast by measuring pixel intensity variation.

        Args:
            image_path (str): Path to the input image file.
            threshold (float): Threshold for contrast detection. Lower values indicate lower contrast.

        Returns:
            bool: True if the image has sufficient contrast, False otherwise.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at path: {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()  # Standard deviation of pixel intensities
        return contrast > threshold

    def check_noise(self, image_path, threshold=50):
        """
        Checks if the image has excessive noise using entropy.

        Args:
            image_path (str): Path to the input image file.
            threshold (float): Threshold for noise detection. Higher values indicate more noise.

        Returns:
            bool: True if the image has acceptable noise levels, False otherwise.
        """
        from scipy.stats import entropy
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at path: {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_normalized = hist / hist.sum()
        image_entropy = entropy(hist_normalized)
        return image_entropy < threshold


# Example Usage
if __name__ == "__main__":
    # Paths
    image_path = "data/images/12.jpg"

    try:
        # Initialize the ImageQualityChecker
        quality_checker = ImageQualityChecker()

        # Check resolution
        resolution_ok = quality_checker.check_resolution(image_path)
        print(f"Resolution OK: {resolution_ok}")

        # Check blur
        blur_ok = quality_checker.check_blur(image_path)
        print(f"Not Blurry: {blur_ok}")

        # Check contrast
        contrast_ok = quality_checker.check_contrast(image_path)
        print(f"Sufficient Contrast: {contrast_ok}")

        # Check noise
        noise_ok = quality_checker.check_noise(image_path)
        print(f"Acceptable Noise Levels: {noise_ok}")

    except Exception as e:
        print(f"Error during image quality verification: {e}")