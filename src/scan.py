"""
DocScanner: An advanced document scanner that automatically detects, crops, and enhances documents in images.

This module provides functionality to:
1. Detect document boundaries using corner detection and contour analysis
2. Transform the perspective to obtain a top-down view
3. Apply image processing to enhance readability
4. Save the processed document

Author: Improved by Claude, based on original code
"""

import os
import cv2
import numpy as np
import math
import itertools
import argparse
from scipy.spatial import distance as dist
from typing import List, Tuple, Optional, Union, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocScanner:
    """
    A document scanner that detects, crops, and enhances documents in images.
    
    Attributes:
        min_quad_area_ratio: Minimum area ratio for a valid quadrilateral
        max_quad_angle_range: Maximum angle range for a valid quadrilateral
        canny_threshold: Threshold for Canny edge detection
        morph_size: Size of morphological kernel
        gaussian_blur_size: Size of Gaussian blur kernel
        corner_min_dist: Minimum distance between filtered corners
        rescaled_height: Height to which images are rescaled for processing
    """
    
    def __init__(
        self, 
        min_quad_area_ratio: float = 0.25, 
        max_quad_angle_range: float = 40,
        canny_threshold: int = 84,
        morph_size: int = 9,
        gaussian_blur_size: int = 7,
        corner_min_dist: int = 20,
        rescaled_height: float = 500.0
    ):
        """
        Initialize the document scanner with customizable parameters.
        
        Args:
            min_quad_area_ratio: Minimum ratio of quadrilateral area to image area
            max_quad_angle_range: Maximum range of angles in the quadrilateral
            canny_threshold: Upper threshold for Canny edge detection
            morph_size: Size of the kernel for morphological operations
            gaussian_blur_size: Size of the kernel for Gaussian blur
            corner_min_dist: Minimum distance between corners
            rescaled_height: Height to which images are rescaled for processing
        """
        self.min_quad_area_ratio = min_quad_area_ratio
        self.max_quad_angle_range = max_quad_angle_range
        self.canny_threshold = canny_threshold
        self.morph_size = morph_size
        self.gaussian_blur_size = gaussian_blur_size
        self.corner_min_dist = corner_min_dist
        self.rescaled_height = rescaled_height

    def filter_corners(self, corners: List[Tuple[int, int]], min_dist: int = None) -> List[Tuple[int, int]]:
        """
        Filter corners to remove duplicates within a minimum distance.
        
        Args:
            corners: List of corner points as (x, y) tuples
            min_dist: Minimum Euclidean distance between corners
            
        Returns:
            List of filtered corner points
        """
        if min_dist is None:
            min_dist = self.corner_min_dist
            
        filtered = []
        for c in corners:
            if all(dist.euclidean(c, existing) >= min_dist for existing in filtered):
                filtered.append(c)
        return filtered

    def angle_between_vectors_degrees(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Calculate the angle between two vectors in degrees.
        
        Args:
            u: First vector
            v: Second vector
            
        Returns:
            Angle in degrees
        """
        # Handle edge case when vectors are too small (would cause division by zero)
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
        
        if u_norm < 1e-6 or v_norm < 1e-6:
            return 0.0
            
        # Calculate dot product and divide by product of magnitudes
        cos_angle = np.clip(np.dot(u, v) / (u_norm * v_norm), -1.0, 1.0)
        return np.degrees(math.acos(cos_angle))

    def get_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate the angle between three points.
        
        Args:
            p1: First point
            p2: Second point (vertex)
            p3: Third point
            
        Returns:
            Angle in degrees
        """
        # Convert to radians for consistent calculation
        a = p1 if len(np.shape(p1)) == 1 else p1[0]
        b = p2 if len(np.shape(p2)) == 1 else p2[0]
        c = p3 if len(np.shape(p3)) == 1 else p3[0]
        
        return self.angle_between_vectors_degrees(a - b, c - b)

    def angle_range(self, quad: np.ndarray) -> float:
        """
        Calculate the range of angles in a quadrilateral.
        
        Args:
            quad: Quadrilateral points in order [top-left, top-right, bottom-right, bottom-left]
            
        Returns:
            Range of angles (max angle - min angle)
        """
        # Ensure each point is a flat (x, y) shape
        tl, tr, br, bl = [p if len(np.shape(p)) == 1 else p[0] for p in quad]
        
        angles = [
            self.get_angle(tl, tr, br),  # Top-right angle
            self.get_angle(tr, br, bl),  # Bottom-right angle
            self.get_angle(br, bl, tl),  # Bottom-left angle
            self.get_angle(bl, tl, tr)   # Top-left angle
        ]
        
        return np.ptp(angles)  # Peak-to-peak (max - min)

    def detect_lines_and_corners(self, edged: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect lines and find their intersections as potential corners.
        
        Args:
            edged: Edge-detected image
            
        Returns:
            List of corner points as (x, y) tuples
        """
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edged, 
            rho=1, 
            theta=np.pi/180, 
            threshold=100, 
            minLineLength=100, 
            maxLineGap=10
        )
        
        corners = []
        
        if lines is not None:
            # Find intersections of lines as potential corners
            for i, line1 in enumerate(lines):
                for line2 in lines[i+1:]:  # Optimization: avoid duplicate calculations
                    x1, y1, x2, y2 = line1[0]
                    x3, y3, x4, y4 = line2[0]
                    
                    # Calculate determinant to check if lines are parallel
                    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                    
                    if abs(denom) < 1e-6:  # Lines are nearly parallel
                        continue
                        
                    # Calculate intersection point
                    px = ((x1 * y2 - y1 * x2) * (x3 - x4) -
                          (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                    py = ((x1 * y2 - y1 * x2) * (y3 - y4) -
                          (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
                    
                    # Only add corners that are within the image bounds
                    if 0 <= px < edged.shape[1] and 0 <= py < edged.shape[0]:
                        corners.append((int(px), int(py)))
        
        return self.filter_corners(corners)

    def is_valid_contour(self, cnt: np.ndarray, width: int, height: int) -> bool:
        """
        Check if a contour is a valid document quadrilateral.
        
        Args:
            cnt: Contour points
            width: Image width
            height: Image height
            
        Returns:
            True if the contour is a valid document quadrilateral
        """
        # Check if contour has exactly 4 points
        if len(cnt) != 4:
            return False
            
        # Check if contour area is significant compared to image area
        if cv2.contourArea(cnt) <= width * height * self.min_quad_area_ratio:
            return False
            
        # Check if angles are reasonable (not too skewed)
        if self.angle_range(cnt) >= self.max_quad_angle_range:
            return False
            
        return True

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order quadrilateral points in top-left, top-right, bottom-right, bottom-left order.
        
        Args:
            pts: Array of quadrilateral points
            
        Returns:
            Ordered array of points
        """
        # Sort points by x-coordinate
        x_sorted = pts[np.argsort(pts[:, 0]), :]
        
        # Grab left-most and right-most points
        left_most = x_sorted[:2, :]
        right_most = x_sorted[2:, :]
        
        # Sort left-most by y-coordinate to get top-left and bottom-left
        left_most = left_most[np.argsort(left_most[:, 1]), :]
        (tl, bl) = left_most
        
        # Calculate distance from top-left to right-most points
        # to determine top-right and bottom-right
        distances = [dist.euclidean(tl, right_pt) for right_pt in right_most]
        right_most = right_most[np.argsort(distances)[::-1], :]
        (br, tr) = right_most
        
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def get_contour(self, image: np.ndarray) -> np.ndarray:
        """
        Find the document contour in the image.
        
        Args:
            image: Input image
            
        Returns:
            Array of contour points representing the document boundary
        """
        height, width = image.shape[:2]
        
        # Convert to grayscale and apply blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.gaussian_blur_size, self.gaussian_blur_size), 0)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_size, self.morph_size))
        dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Detect edges
        edged = cv2.Canny(dilated, 0, self.canny_threshold)
        
        # Try corner detection approach first
        corners = self.detect_lines_and_corners(edged)
        candidates = []
        
        if len(corners) >= 4:
            # Generate all possible quadrilaterals from corners
            for quad in itertools.combinations(corners, 4):
                quad_array = np.array(quad, dtype=np.float32)
                ordered_quad = self.order_points(quad_array)
                candidates.append(ordered_quad)
            
            # Sort candidates by area (descending) and then by angle range
            if candidates:
                candidates = sorted(candidates, key=lambda x: cv2.contourArea(x), reverse=True)[:5]
                candidates = sorted(candidates, key=self.angle_range)
                
                # Return first valid candidate
                if candidates and self.is_valid_contour(candidates[0], width, height):
                    return candidates[0]
        
        # Fallback to contour detection if corner approach fails
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if cnts:
            # Sort contours by area (descending)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
            
            for c in cnts:
                # Approximate contour to a polygon
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                
                # Check if approximation is a valid quadrilateral
                if len(approx) == 4 and self.is_valid_contour(approx, width, height):
                    return approx.reshape(4, 2)
        
        # Fallback to full image if no valid contour is found
        logger.warning("No valid document contour found. Using full image.")
        return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    def four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Apply perspective transform to obtain a top-down view of the document.
        
        Args:
            image: Input image
            pts: Four corner points of the document
            
        Returns:
            Transformed (warped) image
        """
        # Order points in top-left, top-right, bottom-right, bottom-left order
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        
        # Compute width of new image
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        # Compute height of new image
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        # Set destination points for perspective transform
        dst = np.array([
            [0, 0],                      # Top-left
            [max_width - 1, 0],          # Top-right
            [max_width - 1, max_height - 1], # Bottom-right
            [0, max_height - 1]          # Bottom-left
        ], dtype=np.float32)
        
        # Compute perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        
        return warped

    def enhance_document(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance the document image for better readability.
        
        Args:
            image: Input document image
            
        Returns:
            Enhanced document image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive sharpening
        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            sharpened, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 
            21, 
            15
        )
        
        return thresh

    def scan(self, image_path: str, output_dir: str = "output", show_steps: bool = False) -> str:
        """
        Scan a document from an image file.
        
        Args:
            image_path: Path to the input image file
            output_dir: Directory to save the output image
            show_steps: Whether to show intermediate processing steps
            
        Returns:
            Path to the output image file
            
        Raises:
            FileNotFoundError: If the input image file doesn't exist
            ValueError: If the image can't be loaded
        """
        # Check if input file exists
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Get image dimensions and calculate resize ratio
        orig_height, orig_width = image.shape[:2]
        ratio = orig_height / self.rescaled_height
        
        # Keep original for final transform
        orig = image.copy()
        
        # Resize image for faster processing
        rescaled = cv2.resize(image, (int(orig_width / ratio), int(self.rescaled_height)))
        
        # Get document contour
        logger.info(f"Processing image: {os.path.basename(image_path)}")
        try:
            screenCnt = self.get_contour(rescaled)
            
            # Scale back contour points to original image size
            screenCnt = screenCnt * ratio
            
            # Apply perspective transform
            warped = self.four_point_transform(orig, screenCnt)
            
            # Enhance document
            enhanced = self.enhance_document(warped)
            
            # Save output
            os.makedirs(output_dir, exist_ok=True)
            basename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, basename)
            cv2.imwrite(output_path, enhanced)
            
            logger.info(f"✅ Successfully processed: {basename}")
            
            # Optional visualization of intermediate steps
            if show_steps:
                # Draw contour on original image
                contour_image = orig.copy()
                cv2.drawContours(contour_image, [np.int0(screenCnt)], -1, (0, 255, 0), 2)
                
                # Create debug output directory
                debug_dir = os.path.join(output_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                
                # Save intermediate images
                cv2.imwrite(os.path.join(debug_dir, f"contour_{basename}"), contour_image)
                cv2.imwrite(os.path.join(debug_dir, f"warped_{basename}"), warped)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error processing {os.path.basename(image_path)}: {str(e)}")
            raise

    def scan_batch(self, input_dir: str, output_dir: str = "output", show_steps: bool = False) -> List[str]:
        """
        Scan multiple documents from a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output images
            show_steps: Whether to show intermediate processing steps
            
        Returns:
            List of paths to output image files
        """
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
        valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
        output_paths = []
        
        # Get all image files in directory
        image_files = [
            f for f in os.listdir(input_dir) 
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        for image_file in image_files:
            try:
                input_path = os.path.join(input_dir, image_file)
                output_path = self.scan(input_path, output_dir, show_steps)
                output_paths.append(output_path)
            except Exception as e:
                logger.error(f"Failed to process {image_file}: {str(e)}")
                
        return output_paths


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="DocScanner: Automatically detect, crop, and enhance documents in images"
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", 
        help="Path to a single image to scan"
    )
    input_group.add_argument(
        "--images", 
        help="Directory of images to scan"
    )
    
    # Output options
    parser.add_argument(
        "--output", 
        default="output",
        help="Directory to save processed images (default: output)"
    )
    
    # Processing options
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Save intermediate processing steps"
    )
    parser.add_argument(
        "--min-area-ratio", 
        type=float, 
        default=0.25,
        help="Minimum area ratio for valid document contour (default: 0.25)"
    )
    parser.add_argument(
        "--max-angle-range", 
        type=float, 
        default=40,
        help="Maximum angle range for valid document contour (default: 40)"
    )
    
    args = parser.parse_args()
    
    # Initialize scanner with parameters
    scanner = DocScanner(
        min_quad_area_ratio=args.min_area_ratio,
        max_quad_angle_range=args.max_angle_range
    )
    
    try:
        if args.image:
            scanner.scan(args.image, args.output, args.debug)
        else:
            scanner.scan_batch(args.images, args.output, args.debug)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())