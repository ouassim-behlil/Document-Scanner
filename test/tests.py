import unittest
import os
import json
import shutil
import tempfile
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from scan import DocScanner
from extract_text import TextExtractor


class TestDocScanner(unittest.TestCase):
    """Tests for the DocScanner class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.scanner = DocScanner()
        # Create a temporary directory for test outputs
        self.test_output_dir = tempfile.mkdtemp()
        
        # Create a simple test image with a quadrilateral shape (representing a document)
        self.test_image_path = os.path.join(self.test_output_dir, "test_document.jpg")
        # Create a 600x800 white image
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        # Draw a black quadrilateral representing a document (slightly rotated rectangle)
        points = np.array([[100, 100], [600, 150], [550, 450], [150, 400]], dtype=np.int32)
        cv2.fillPoly(img, [points], (0, 0, 0))
        cv2.imwrite(self.test_image_path, img)

    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary directory
        shutil.rmtree(self.test_output_dir)

    def test_filter_corners(self):
        """Test corner filtering functionality."""
        corners = [(10, 10), (15, 15), (100, 100), (105, 105)]
        filtered = self.scanner.filter_corners(corners, min_dist=10)
        self.assertEqual(len(filtered), 2)
        self.assertIn((10, 10), filtered)  # First point should be kept
        self.assertIn((100, 100), filtered)  # Third point should be kept

    def test_angle_between_vectors_degrees(self):
        """Test angle calculation between vectors."""
        u = np.array([1, 0])  # Horizontal vector
        v = np.array([0, 1])  # Vertical vector
        angle = self.scanner.angle_between_vectors_degrees(u, v)
        self.assertAlmostEqual(angle, 90.0)  # Should be 90 degrees

    def test_order_points(self):
        """Test ordering of quadrilateral points."""
        # Define points in random order
        pts = np.array([
            [100, 200],  # Bottom-left
            [300, 100],  # Top-right
            [100, 100],  # Top-left
            [300, 200],  # Bottom-right
        ], dtype=np.float32)
        
        ordered = self.scanner.order_points(pts)
        
        # Check if points are properly ordered (tl, tr, br, bl)
        self.assertTrue(np.array_equal(ordered[0], [100, 100]))  # Top-left
        self.assertTrue(np.array_equal(ordered[1], [300, 100]))  # Top-right
        self.assertTrue(np.array_equal(ordered[2], [300, 200]))  # Bottom-right
        self.assertTrue(np.array_equal(ordered[3], [100, 200]))  # Bottom-left

    def test_get_contour(self):
        """Test document contour detection."""
        # Load test image
        image = cv2.imread(self.test_image_path)
        
        # Get contour
        contour = self.scanner.get_contour(image)
        
        # Contour should have 4 points for a quadrilateral
        self.assertEqual(len(contour), 4)
        
        # Check if contour has reasonable area
        area = cv2.contourArea(contour)
        img_area = image.shape[0] * image.shape[1]
        self.assertGreater(area, img_area * self.scanner.min_quad_area_ratio)

    def test_scan(self):
        """Test full document scanning process."""
        output_path = self.scanner.scan(
            self.test_image_path, 
            self.test_output_dir
        )
        
        # Check if output file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Check if output is a valid image
        output_img = cv2.imread(output_path)
        self.assertIsNotNone(output_img)
        self.assertTrue(output_img.shape[0] > 0 and output_img.shape[1] > 0)

    @patch('cv2.findContours')
    def test_scan_no_contour_fallback(self, mock_find_contours):
        """Test scanner fallback when no valid contour is found."""
        # Mock findContours to return no contours
        mock_find_contours.return_value = ([], None)
        
        output_path = self.scanner.scan(
            self.test_image_path, 
            self.test_output_dir
        )
        
        # Check if output file exists despite no contour
        self.assertTrue(os.path.exists(output_path))

    def test_scan_batch(self):
        """Test batch document scanning."""
        # Create a batch directory with test images
        batch_dir = os.path.join(self.test_output_dir, "batch_input")
        os.makedirs(batch_dir)
        
        # Copy test image multiple times with different names
        for i in range(3):
            shutil.copy(
                self.test_image_path, 
                os.path.join(batch_dir, f"test_doc_{i}.jpg")
            )
        
        # Process batch
        output_paths = self.scanner.scan_batch(
            batch_dir, 
            os.path.join(self.test_output_dir, "batch_output")
        )
        
        # Check if all outputs exist
        self.assertEqual(len(output_paths), 3)
        for path in output_paths:
            self.assertTrue(os.path.exists(path))


class TestTextExtractor(unittest.TestCase):
    """Tests for the TextExtractor class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Use a low-overhead mock since we don't want to initialize PaddleOCR in tests
        self.extractor = TextExtractor()
        self.extractor._initialized = True
        self.extractor.ocr = MagicMock()
        
        # Create temporary directories
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir)
        
        # Create a simple test image
        self.test_image_path = os.path.join(self.test_dir, "test_text.jpg")
        img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Test Text", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.imwrite(self.test_image_path, img)
        
        # Sample OCR results that would be returned by PaddleOCR
        self.sample_ocr_result = [[
            [
                [[50, 100], [300, 100], [300, 200], [50, 200]],  # Box coordinates
                ["Test Text", 0.95]  # Text and confidence
            ]
        ]]

    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.test_dir)

    def test_extract_text(self):
        """Test text extraction from a single image."""
        # Mock OCR results
        self.extractor.ocr.ocr.return_value = self.sample_ocr_result
        
        # Extract text
        result = self.extractor.extract_text(self.test_image_path)
        
        # Check results
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Test Text")
        self.assertAlmostEqual(result[0]["confidence"], 0.95)
        self.assertEqual(len(result[0]["box"]), 4)  # Should have 4 corner points

    def test_extract_text_no_results(self):
        """Test behavior when no text is found."""
        # Mock OCR with no results
        self.extractor.ocr.ocr.return_value = [[]]
        
        result = self.extractor.extract_text(self.test_image_path)
        
        # Should return empty list
        self.assertEqual(result, [])

    def test_extract_text_nonexistent_file(self):
        """Test handling of nonexistent files."""
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_text("nonexistent_file.jpg")

    def test_process_batch(self):
        """Test batch processing of images."""
        # Create a batch directory with test images
        batch_dir = os.path.join(self.test_dir, "batch_input")
        os.makedirs(batch_dir)
        
        # Create multiple test images
        for i in range(3):
            img_path = os.path.join(batch_dir, f"test_img_{i}.jpg")
            shutil.copy(self.test_image_path, img_path)
        
        # Mock OCR results
        self.extractor.ocr.ocr.return_value = self.sample_ocr_result
        
        # Process batch
        results = self.extractor.process_batch(
            batch_dir, 
            self.output_dir, 
            output_format="json"
        )
        
        # Check results
        self.assertEqual(len(results), 3)
        
        # Check output files
        json_files = [f for f in os.listdir(self.output_dir) if f.endswith('.json')]
        self.assertEqual(len(json_files), 3)
        
        # Check content of first JSON file
        with open(os.path.join(self.output_dir, json_files[0]), 'r') as f:
            data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["text"], "Test Text")

    def test_process_batch_with_txt_output(self):
        """Test batch processing with text output format."""
        # Create a batch directory with a test image
        batch_dir = os.path.join(self.test_dir, "txt_batch")
        os.makedirs(batch_dir)
        shutil.copy(self.test_image_path, os.path.join(batch_dir, "test.jpg"))
        
        # Mock OCR results
        self.extractor.ocr.ocr.return_value = self.sample_ocr_result
        
        # Process batch with txt output
        self.extractor.process_batch(
            batch_dir, 
            self.output_dir, 
            output_format="txt"
        )
        
        # Check output files
        txt_files = [f for f in os.listdir(self.output_dir) if f.endswith('.txt')]
        self.assertEqual(len(txt_files), 1)
        
        # Check content of text file
        with open(os.path.join(self.output_dir, txt_files[0]), 'r') as f:
            content = f.read().strip()
            self.assertEqual(content, "Test Text")

    def test_extract_structured_text(self):
        """Test structured text extraction."""
        # Mock OCR with multiple text blocks
        multi_text_result = [[
            [[[50, 50], [300, 50], [300, 100], [50, 100]], ["Document Title", 0.98]],
            [[[50, 150], [400, 150], [400, 200], [50, 200]], ["First paragraph text.", 0.95]],
            [[[50, 220], [400, 220], [400, 270], [50, 270]], ["Second paragraph text.", 0.94]]
        ]]
        self.extractor.ocr.ocr.return_value = multi_text_result
        
        # Extract structured text
        result = self.extractor.extract_structured_text(self.test_image_path)
        
        # Check structure
        self.assertIn("title", result)
        self.assertIn("paragraphs", result)
        self.assertEqual(result["title"], "Document Title")
        self.assertEqual(len(result["paragraphs"]), 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full document processing pipeline."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create temporary test directories
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.test_dir, "input")
        self.processed_dir = os.path.join(self.test_dir, "processed")
        self.extracted_text_dir = os.path.join(self.test_dir, "extracted_text")
        
        os.makedirs(self.input_dir)
        os.makedirs(self.processed_dir)
        os.makedirs(self.extracted_text_dir)
        
        # Create a test document image
        self.test_image_path = os.path.join(self.input_dir, "test_document.jpg")
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        # Add some text to the image
        cv2.putText(img, "Test Document", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.putText(img, "Sample text for OCR", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        cv2.imwrite(self.test_image_path, img)
        
        # Initialize components with mocks
        self.scanner = DocScanner()
        self.extractor = TextExtractor()
        self.extractor._initialized = True
        self.extractor.ocr = MagicMock()
        
        # Sample OCR results
        self.sample_ocr_result = [[
            [[[50, 50], [300, 50], [300, 100], [50, 100]], ["Test Document", 0.97]],
            [[[50, 250], [400, 250], [400, 350], [50, 350]], ["Sample text for OCR", 0.95]]
        ]]
        self.extractor.ocr.ocr.return_value = self.sample_ocr_result

    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.test_dir)

    @patch.object(DocScanner, 'get_contour')
    def test_full_pipeline(self, mock_get_contour):
        """Test the complete document processing pipeline."""
        # Mock the contour detection to return a simple quadrilateral
        mock_contour = np.array([[100, 100], [700, 100], [700, 500], [100, 500]], dtype=np.float32)
        mock_get_contour.return_value = mock_contour
        
        # Step 1: Scan document
        processed_path = self.scanner.scan(
            self.test_image_path, 
            self.processed_dir
        )
        
        # Check if processed document exists
        self.assertTrue(os.path.exists(processed_path))
        
        # Step 2: Extract text
        text_data = self.extractor.extract_text(processed_path)
        
        # Check text extraction results
        self.assertEqual(len(text_data), 2)
        self.assertEqual(text_data[0]["text"], "Test Document")
        self.assertEqual(text_data[1]["text"], "Sample text for OCR")
        
        # Step 3: Save extracted text
        self.extractor.process_batch(
            self.processed_dir,
            self.extracted_text_dir,
            output_format="both"
        )
        
        # Check output files
        json_files = [f for f in os.listdir(self.extracted_text_dir) if f.endswith('.json')]
        txt_files = [f for f in os.listdir(self.extracted_text_dir) if f.endswith('.txt')]
        
        self.assertEqual(len(json_files), 1)
        self.assertEqual(len(txt_files), 1)


if __name__ == '__main__':
    unittest.main()
