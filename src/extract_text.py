"""
Text Extractor: A module to extract text from processed document images using PaddleOCR.

This module provides functionality to:
1. Extract text from single or multiple images
2. Process the extracted text with custom filters
3. Export the results to various formats (text, JSON)
4. Visualize the detected text regions

Requirements:
- paddlepaddle
- paddleocr
- opencv-python
- numpy

Install dependencies:
pip install paddlepaddle paddleocr opencv-python numpy

Author: Claude
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Union, Optional, Any
from paddleocr import PaddleOCR, draw_ocr

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextExtractor:
    """
    Extract text from document images using PaddleOCR.
    
    Attributes:
        ocr: PaddleOCR instance for text recognition
        lang: Language for OCR detection
        use_gpu: Whether to use GPU for inference
        _initialized: Whether the OCR engine has been initialized
    """
    
    def __init__(
        self, 
        lang: str = 'en',
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        enable_mkldnn: bool = False,
        rec_batch_num: int = 6,
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.5
    ):
        """
        Initialize the text extractor with customizable parameters.
        
        Args:
            lang: Language for OCR, e.g., 'en', 'ch', 'fr', etc.
            use_angle_cls: Whether to use text direction classification
            use_gpu: Whether to use GPU for inference
            enable_mkldnn: Whether to use MKLDNN for CPU acceleration
            rec_batch_num: Text recognition batch size
            det_db_thresh: Text detection threshold
            det_db_box_thresh: Text detection box threshold
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self._initialized = False
        self.ocr_params = {
            'use_angle_cls': use_angle_cls,
            'lang': lang,
            'use_gpu': use_gpu,
            'enable_mkldnn': enable_mkldnn,
            'rec_batch_num': rec_batch_num,
            'det_db_thresh': det_db_thresh,
            'det_db_box_thresh': det_db_box_thresh
        }
        # Lazy initialization of OCR to save memory if not used
        self.ocr = None
    
    def _initialize_ocr(self):
        """
        Lazy initialization of PaddleOCR.
        """
        if not self._initialized:
            logger.info(f"Initializing PaddleOCR with language: {self.lang}")
            try:
                self.ocr = PaddleOCR(**self.ocr_params)
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
                raise
    
    def extract_text(self, image_path: str, cls_thresh: float = 0.9) -> List[Dict[str, Any]]:
        """
        Extract text from a single image.
        
        Args:
            image_path: Path to the image file
            cls_thresh: Classification confidence threshold
            
        Returns:
            List of dictionaries containing extracted text information
        """
        # Initialize OCR if not already done
        self._initialize_ocr()
        
        # Check if image exists
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return []
        
        # Extract text
        logger.info(f"Extracting text from: {os.path.basename(image_path)}")
        try:
            result = self.ocr.ocr(image_path, cls=True)
            
            # Process results
            extracted_info = []
            if result is not None and len(result) > 0:
                for idx, line in enumerate(result[0]):
                    if line is not None and len(line) >= 2:
                        box = line[0]
                        text_info = line[1]
                        text = text_info[0]
                        confidence = text_info[1]
                        
                        # Only include results with confidence above threshold
                        if confidence >= cls_thresh:
                            extracted_info.append({
                                'id': idx,
                                'text': text,
                                'confidence': float(confidence),
                                'box': box,
                                'position': {
                                    'x': int(sum(point[0] for point in box) / 4),
                                    'y': int(sum(point[1] for point in box) / 4)
                                }
                            })
            
            return extracted_info
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {str(e)}")
            return []
    
    def process_batch(
        self, 
        image_dir: str, 
        output_dir: str = "extracted_text",
        output_format: str = "json",
        cls_thresh: float = 0.9,
        visualize: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process multiple images in a directory.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save output files
            output_format: Output format ('json', 'txt', or 'both')
            cls_thresh: Classification confidence threshold
            visualize: Whether to save visualizations of detected text
            
        Returns:
            Dictionary with image filenames as keys and extracted info as values
        """
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = [
            f for f in os.listdir(image_dir) 
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        results = {}
        for image_file in image_files:
            try:
                image_path = os.path.join(image_dir, image_file)
                extracted_info = self.extract_text(image_path, cls_thresh)
                
                results[image_file] = extracted_info
                
                # Save results
                base_name = os.path.splitext(image_file)[0]
                
                # Save as JSON
                if output_format in ['json', 'both']:
                    json_path = os.path.join(output_dir, f"{base_name}.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(extracted_info, f, ensure_ascii=False, indent=2)
                
                # Save as plain text
                if output_format in ['txt', 'both']:
                    txt_path = os.path.join(output_dir, f"{base_name}.txt")
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        for item in extracted_info:
                            f.write(f"{item['text']}\n")
                
                # Create visualization
                if visualize and extracted_info:
                    self._visualize_results(image_path, extracted_info, output_dir)
                
                logger.info(f"✅ Processed: {image_file}")
            except Exception as e:
                logger.error(f"Failed to process {image_file}: {str(e)}")
        
        return results
    
    def _visualize_results(
        self, 
        image_path: str, 
        results: List[Dict[str, Any]], 
        output_dir: str
    ):
        """
        Create visualization of OCR results.
        
        Args:
            image_path: Path to the original image
            results: Extracted text information
            output_dir: Directory to save visualization
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image for visualization: {image_path}")
                return
            
            # Prepare data for visualization
            boxes = [item['box'] for item in results]
            texts = [item['text'] for item in results]
            scores = [item['confidence'] for item in results]
            
            # Create visualization using PaddleOCR's draw_ocr
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts/simfang.ttf')
            if not os.path.exists(font_path):
                # Default fallback font path for Linux/Mac
                font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
                # For Windows
                if os.name == 'nt':
                    font_path = 'C:/Windows/Fonts/Arial.ttf'
            
            # Draw OCR results if we have a valid font
            if os.path.exists(font_path):
                result_img = draw_ocr(image, boxes, texts, scores, font_path=font_path)
                
                # Convert back to OpenCV format if necessary
                if isinstance(result_img, np.ndarray):
                    vis_img = result_img
                else:
                    vis_img = np.array(result_img)
                
                # Save visualization
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                vis_path = os.path.join(vis_dir, f"{base_name}_ocr.jpg")
                cv2.imwrite(vis_path, vis_img)
            else:
                logger.warning("Font file not found. Skipping visualization.")
        
        except Exception as e:
            logger.error(f"Error creating visualization for {image_path}: {str(e)}")
    
    def extract_structured_text(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text and attempt to identify document structure (for advanced use).
        This is a simplified version that organizes text by vertical position.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with structured text information
        """
        # Extract basic text
        extracted_info = self.extract_text(image_path)
        
        if not extracted_info:
            return {'title': '', 'paragraphs': []}
        
        # Sort text blocks by vertical position (y-coordinate)
        extracted_info.sort(key=lambda x: x['position']['y'])
        
        # Simple heuristic: first block is title, rest are paragraphs
        # This is a very basic approach - real-world documents need more sophisticated analysis
        title = extracted_info[0]['text'] if extracted_info else ''
        paragraphs = []
        current_paragraph = []
        
        # Group text blocks into paragraphs based on vertical distance
        for i in range(1, len(extracted_info)):
            prev_block = extracted_info[i-1]
            curr_block = extracted_info[i]
            
            # If vertical distance is small, add to current paragraph
            if curr_block['position']['y'] - prev_block['position']['y'] < 30:
                if i == 1:  # First non-title block
                    current_paragraph = [prev_block['text'], curr_block['text']]
                else:
                    current_paragraph.append(curr_block['text'])
            else:
                # New paragraph
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [curr_block['text']]
        
        # Add the last paragraph if there is one
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return {
            'title': title,
            'paragraphs': paragraphs
        }


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Extract text from document images using PaddleOCR"
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", 
        help="Path to a single image to extract text from"
    )
    input_group.add_argument(
        "--images", 
        help="Directory of images to extract text from"
    )
    
    # Output options
    parser.add_argument(
        "--output", 
        default="extracted_text",
        help="Directory to save extracted text (default: extracted_text)"
    )
    parser.add_argument(
        "--format", 
        choices=["json", "txt", "both"],
        default="json",
        help="Output format (default: json)"
    )
    
    # Visualization option
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Generate visualizations of detected text"
    )
    
    # OCR options
    parser.add_argument(
        "--lang", 
        default="en",
        help="Language for OCR (default: en)"
    )
    parser.add_argument(
        "--threshold", 
        type=float,
        default=0.9,
        help="Confidence threshold for text detection (default: 0.9)"
    )
    parser.add_argument(
        "--use-gpu", 
        action="store_true",
        help="Use GPU for inference"
    )
    parser.add_argument(
        "--structured", 
        action="store_true",
        help="Extract structured text (experimental)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize text extractor
        extractor = TextExtractor(
            lang=args.lang,
            use_gpu=args.use_gpu
        )
        
        # Process input
        if args.image:
            if args.structured:
                # Extract structured text
                result = extractor.extract_structured_text(args.image)
                
                # Create output directory
                os.makedirs(args.output, exist_ok=True)
                
                # Save result
                base_name = os.path.splitext(os.path.basename(args.image))[0]
                output_path = os.path.join(args.output, f"{base_name}_structured.json")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                # Print summary
                logger.info(f"Extracted structured text saved to: {output_path}")
                if result['title']:
                    logger.info(f"Title: {result['title']}")
                logger.info(f"Paragraphs: {len(result['paragraphs'])}")
            else:
                # Extract basic text
                result = extractor.extract_text(args.image, args.threshold)
                
                # Create output directory
                os.makedirs(args.output, exist_ok=True)
                
                # Save result
                base_name = os.path.splitext(os.path.basename(args.image))[0]
                
                # Save in requested format(s)
                if args.format in ['json', 'both']:
                    json_path = os.path.join(args.output, f"{base_name}.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                
                if args.format in ['txt', 'both']:
                    txt_path = os.path.join(args.output, f"{base_name}.txt")
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        for item in result:
                            f.write(f"{item['text']}\n")
                
                # Create visualization if requested
                if args.visualize and result:
                    extractor._visualize_results(args.image, result, args.output)
                
                # Print summary
                logger.info(f"Extracted {len(result)} text regions from {os.path.basename(args.image)}")
        else:
            # Process batch
            results = extractor.process_batch(
                args.images, 
                args.output, 
                args.format,
                args.threshold,
                args.visualize
            )
            
            # Print summary
            total_images = len(results)
            total_text_regions = sum(len(texts) for texts in results.values())
            logger.info(f"Processed {total_images} images, extracted {total_text_regions} text regions")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())