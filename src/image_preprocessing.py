import cv2
import numpy as np

class ImagePreprocessor:
    """
    A class to preprocess images for OCR tasks.
    Includes methods for grayscale conversion, noise reduction, contrast enhancement, and resizing.
    """

    def __init__(self, scale_factor=1000):
        """
        Initializes the ImagePreprocessor with default settings.
        
        Args:
            scale_factor (int): Maximum dimension (width or height) for resizing. Default is 1000 pixels.
        """
        # Private variable: Scale factor for resizing
        self.__scale_factor = scale_factor

    def preprocess(self, image_path):
        """
        Preprocesses the input image to enhance OCR accuracy.
        
        Steps:
        1. Convert to grayscale.
        2. Apply Gaussian blur to reduce noise.
        3. Enhance contrast using adaptive thresholding.
        4. Resize the image to ensure consistent dimensions.
        
        Args:
            image_path (str): Path to the input image file.
        
        Returns:
            np.ndarray: Preprocessed image as a NumPy array.
        """
        # Step 1: Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at path: {image_path}")
        
        # Step 2: Convert to grayscale
        gray = self._convert_to_grayscale(image)
        
        # Step 3: Apply Gaussian blur to reduce noise
        blurred = self._apply_gaussian_blur(gray)
        
        # Step 4: Enhance contrast using adaptive thresholding
        binary = self._enhance_contrast(blurred)
        
        # Step 5: Resize the image to ensure consistent dimensions
        resized = self._resize_image(binary)
        
        return resized

    @staticmethod
    def save_image(image, output_path):
        """
        Saves the preprocessed image to the specified output path.
        
        Args:
            image (np.ndarray): Preprocessed image as a NumPy array.
            output_path (str): Path to save the preprocessed image.
        """
        cv2.imwrite(output_path, image)

    # Private method: Convert to grayscale
    def _convert_to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Private method: Apply Gaussian blur
    def _apply_gaussian_blur(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    # Private method: Enhance contrast using adaptive thresholding
    def _enhance_contrast(self, image):
        return cv2.adaptiveThreshold(
            image,
            255,  # Maximum pixel value
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,  # Size of the neighborhood area
            C=2  # Constant subtracted from the mean
        )

    # Private method: Resize the image
    def _resize_image(self, image):
        height, width = image.shape[:2]
        scale_factor = self.__scale_factor / max(height, width)  # Scale to a maximum dimension
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


# # Example Usage
# if __name__ == "__main__":
#     # Paths
#     input_image_path = "data/images/3.jpg"
#     output_image_path = "outputs/preprocessed_receipt.jpg"

#     try:
#         # Initialize the ImagePreprocessor
#         preprocessor = ImagePreprocessor(scale_factor=1000)

#         # Preprocess the image
#         preprocessed_image = preprocessor.preprocess(input_image_path)

#         # Save the preprocessed image
#         ImagePreprocessor.save_image(preprocessed_image, output_image_path)
#         print(f"Preprocessed image saved to {output_image_path}")

#     except Exception as e:
#         print(f"Error during preprocessing: {e}")