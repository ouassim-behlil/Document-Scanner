import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Preprocesses the input image to enhance OCR accuracy.
    
    Steps:
    1. Convert to grayscale.
    2. Apply Gaussian blur to reduce noise.
    3. Enhance contrast using histogram equalization or adaptive thresholding.
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 4: Enhance contrast using adaptive thresholding
    # Adaptive thresholding works better than global thresholding for receipts
    binary = cv2.adaptiveThreshold(
        blurred, 
        255,                     # Maximum pixel value
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        blockSize=11,            # Size of the neighborhood area
        C=2                      # Constant subtracted from the mean
    )
    
    # Step 5: Resize the image to ensure consistent dimensions
    # Resizing helps with scaling issues in OCR
    height, width = binary.shape[:2]
    scale_factor = 1000 / max(height, width)  # Scale to a maximum dimension of 1000px
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    resized = cv2.resize(binary, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized


def save_preprocessed_image(image, output_path):
    """
    Saves the preprocessed image to the specified output path.
    
    Args:
        image (np.ndarray): Preprocessed image as a NumPy array.
        output_path (str): Path to save the preprocessed image.
    """
    cv2.imwrite(output_path, image)


# Example Usage
if __name__ == "__main__":
    # Paths
    input_image_path = "data/images/3.jpg"
    output_image_path = "outputs/preprocessed_receipt.jpg"
    
    # Preprocess the image
    try:
        preprocessed_image = preprocess_image(input_image_path)
        
        # Save the preprocessed image
        save_preprocessed_image(preprocessed_image, output_image_path)
        print(f"Preprocessed image saved to {output_image_path}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")