# Document Processing System

A comprehensive document processing system that combines automatic document scanning, perspective correction, and text extraction capabilities. This project provides a complete pipeline for transforming images of documents into machine-readable text.

## Features

- **Document Detection**: Automatically identify document boundaries in images
- **Perspective Correction**: Transform skewed documents to a proper top-down view
- **Image Enhancement**: Apply image processing techniques to improve readability
- **Text Extraction**: Extract text from processed documents using OCR
- **Batch Processing**: Process multiple documents at once
- **Structured Output**: Save extracted text in various formats (JSON, TXT)
- **Visualization**: Generate visualizations of detected text regions

## Components

The system consists of two main components:

1. **DocScanner**: Handles document detection, perspective correction, and image enhancement
2. **TextExtractor**: Extracts text from processed document images using OCR

## Installation

### Prerequisites

- Python 3.6+
- OpenCV
- NumPy
- SciPy
- PaddlePaddle
- PaddleOCR

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ouassim-behlil/Receipt-OCR.git
   cd Receipt-OCR
   ```

2. Install the required packages:
   ```bash
   pip install paddlepaddle paddleocr opencv-python numpy scipy
   ```

## Usage

### Document Scanning

The `DocScanner` class provides functionality to detect and process document images.

#### Command-line Usage:

```bash
# Process a single image
python scan.py --image path/to/image.jpg --output processed_docs

# Process multiple images in a directory
python scan.py --images path/to/image/directory --output processed_docs

# Show intermediate processing steps
python scan.py --image path/to/image.jpg --output processed_docs --debug

# Adjust document detection parameters
python scan.py --image path/to/image.jpg --min-area-ratio 0.3 --max-angle-range 35
```

#### From Python:

```python
from scan import DocScanner

# Initialize the scanner
scanner = DocScanner()

# Process a single image
output_path = scanner.scan("input.jpg", "output_dir")

# Process multiple images
output_paths = scanner.scan_batch("input_dir", "output_dir")
```

### Text Extraction

The `TextExtractor` class extracts text from processed document images.

#### Command-line Usage:

```bash
# Extract text from a single image
python extract_text.py --image path/to/processed/image.jpg --output extracted_text

# Extract text from multiple images
python extract_text.py --images path/to/processed/images --output extracted_text

# Save output in different formats
python extract_text.py --image path/to/image.jpg --format json
python extract_text.py --image path/to/image.jpg --format txt
python extract_text.py --image path/to/image.jpg --format both

# Visualize detected text regions
python extract_text.py --image path/to/image.jpg --visualize

# Extract structured text (experimental)
python extract_text.py --image path/to/image.jpg --structured
```

#### From Python:

```python
from extract_text import TextExtractor

# Initialize the extractor
extractor = TextExtractor(lang="en")

# Extract text from a single image
text_info = extractor.extract_text("processed_image.jpg")

# Process a batch of images
results = extractor.process_batch("processed_images_dir", "output_dir", output_format="json")

# Extract structured text (experimental)
structured_text = extractor.extract_structured_text("processed_image.jpg")
```

## Complete Processing Pipeline

For a complete document processing pipeline, you can combine both components:

```python
from scan import DocScanner
from extract_text import TextExtractor
import os

# Initialize components
scanner = DocScanner()
extractor = TextExtractor()

# Process image with document scanner
processed_image = scanner.scan("document.jpg", "processed")

# Extract text from processed image
text_data = extractor.extract_text(processed_image)

# Print extracted text
for item in text_data:
    print(f"{item['text']} (Confidence: {item['confidence']:.2f})")
```

## Command-line Pipeline

You can also create a pipeline using command-line tools:

```bash
# Step 1: Scan and process the document
python scan.py --image input.jpg --output processed

# Step 2: Extract text from the processed document
python extract_text.py --image processed/input.jpg --output extracted_text
```

## Parameter Tuning

### DocScanner Parameters

- `min_quad_area_ratio`: Minimum ratio of quadrilateral area to image area (default: 0.25)
- `max_quad_angle_range`: Maximum range of angles in the quadrilateral (default: 40)
- `canny_threshold`: Upper threshold for Canny edge detection (default: 84)
- `morph_size`: Size of the kernel for morphological operations (default: 9)
- `gaussian_blur_size`: Size of the kernel for Gaussian blur (default: 7)
- `corner_min_dist`: Minimum distance between corners (default: 20)
- `rescaled_height`: Height to which images are rescaled for processing (default: 500.0)

### TextExtractor Parameters

- `lang`: Language for OCR, e.g., 'en', 'ch', 'fr', etc. (default: 'en')
- `use_angle_cls`: Whether to use text direction classification (default: True)
- `use_gpu`: Whether to use GPU for inference (default: False)
- `enable_mkldnn`: Whether to use MKLDNN for CPU acceleration (default: False)
- `rec_batch_num`: Text recognition batch size (default: 6)
- `det_db_thresh`: Text detection threshold (default: 0.3)
- `det_db_box_thresh`: Text detection box threshold (default: 0.5)

## Output Formats

### Text Extraction JSON Output

The JSON output from text extraction contains detailed information about each text region:

```json
[
  {
    "id": 0,
    "text": "Sample text",
    "confidence": 0.9542,
    "box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "position": {"x": 100, "y": 200}
  },
  ...
]
```

### Structured Text Output (Experimental)

```json
{
  "title": "Document Title",
  "paragraphs": [
    "First paragraph content...",
    "Second paragraph content...",
    ...
  ]
}
```

## Limitations

- Document detection works best with high-contrast documents against a contrasting background
- Text extraction accuracy depends on image quality, contrast, and text clarity
- The structured text extraction feature is experimental and may not accurately detect document structure in all cases
- Currently supports common languages included in PaddleOCR; for specialized languages, additional language models may be needed



## License

This project is licensed under the MIT License - see the LICENSE file for details.

