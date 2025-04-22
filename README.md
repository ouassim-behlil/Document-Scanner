# Document Processing System 

[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.7.0-red.svg)](https://github.com/PaddlePaddle/PaddleOCR)

A powerful, end-to-end document processing system that automatically detects document boundaries, corrects perspective, enhances images, and extracts text using OCR. Perfect for digitizing physical documents, automating data entry, and building document management systems.

<!-- ![Document Processing Flow](https://via.placeholder.com/800x200.png?text=Document+Processing+Flow) -->

##  Features

- **Automatic Document Detection** - Intelligently identifies document boundaries in images
- **Perspective Correction** - Transforms skewed documents to a proper top-down view
- **Image Enhancement** - Applies adaptive processing to improve document quality
- **Text Extraction** - Extracts text with high accuracy using state-of-the-art OCR
- **Batch Processing** - Process multiple documents at once efficiently
- **Structured Output** - Save extracted text in various formats (JSON, TXT)
- **Visualization** - Generate informative visualizations of detected text regions
- **Docker Support** - Run the entire system in containers without complex setup

## 📋 Table of Contents

- [Installation](#-installation)
  - [Standard Installation](#standard-installation)
  - [Docker Installation](#docker-installation)
- [Usage](#-usage)
  - [Command Line Interface](#command-line-interface)
  - [Docker Usage](#docker-usage)
  - [Python API](#python-api)
- [Advanced Configuration](#-advanced-configuration)
- [Output Formats](#-output-formats)
- [Real-world Applications](#-real-world-applications)
- [Performance Tips](#-performance-tips)
- [Contributing](#-contributing)
- [License](#-license)

##  Installation

### Standard Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ouassim-behlil/Receipt-OCR.git
   cd Receipt-OCR
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Docker Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ouassim-behlil/Receipt-OCR.git
   cd Receipt-OCR
   ```

2. Build the Docker image:
   ```bash
   docker-compose build
   ```

3. Ensure input, output, and extracted_text directories exist:
   ```bash
   mkdir -p input output extracted_text
   ```

##  Usage

### Command Line Interface

#### Document Scanning

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

#### Text Extraction

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

# Use custom confidence threshold
python extract_text.py --image path/to/image.jpg --threshold 0.75

# Use GPU acceleration
python extract_text.py --image path/to/image.jpg --use-gpu

# Use different language
python extract_text.py --image path/to/image.jpg --lang fr
```

#### Full Pipeline

```bash
# Process document and extract text in one go
python scan.py --image input.jpg --output processed && \
python extract_text.py --image processed/input.jpg --output extracted_text
```

### Docker Usage

#### Using the Convenience Script

We provide a simple shell script to handle common operations:

```bash
# Make the script executable
chmod +x run.sh

# Process a single document
./run.sh document.jpg

# Process all documents in the input directory
./run.sh --batch
```

#### Using Docker Compose

```bash
# Scan a document
docker-compose run --rm document-processor python scan.py --image /app/input/example.jpg --output /app/processed

# Extract text
docker-compose run --rm document-processor python extract_text.py --image /app/processed/example.jpg --output /app/extracted_text

# Process multiple documents
docker-compose run --rm document-processor python scan.py --images /app/input --output /app/processed

# Full pipeline in one command
docker-compose run --rm document-processor bash -c "python scan.py --image /app/input/example.jpg --output /app/processed && python extract_text.py --image /app/processed/example.jpg --output /app/extracted_text"
```

#### Using Docker Directly

```bash
# Scan a document
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/processed document-processor python scan.py --image /app/input/example.jpg --output /app/processed

# Extract text
docker run --rm -v $(pwd)/output:/app/processed -v $(pwd)/extracted_text:/app/extracted_text document-processor python extract_text.py --image /app/processed/example.jpg --output /app/extracted_text
```

### Python API

#### Document Scanning

```python
from scan import DocScanner

# Initialize the scanner
scanner = DocScanner()

# Process a single image
output_path = scanner.scan("input.jpg", "output_dir")

# Process multiple images
output_paths = scanner.scan_batch("input_dir", "output_dir")
```

#### Text Extraction

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

#### Complete Pipeline

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

##  Advanced Configuration

### DocScanner Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `min_quad_area_ratio` | Minimum ratio of quadrilateral area to image area | 0.25 |
| `max_quad_angle_range` | Maximum range of angles in the quadrilateral | 40 |
| `canny_threshold` | Upper threshold for Canny edge detection | 84 |
| `morph_size` | Size of the kernel for morphological operations | 9 |
| `gaussian_blur_size` | Size of the kernel for Gaussian blur | 7 |
| `corner_min_dist` | Minimum distance between corners | 20 |
| `rescaled_height` | Height to which images are rescaled for processing | 500.0 |

### TextExtractor Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lang` | Language for OCR ('en', 'ch', 'fr', etc.) | 'en' |
| `use_angle_cls` | Whether to use text direction classification | True |
| `use_gpu` | Whether to use GPU for inference | False |
| `enable_mkldnn` | Whether to use MKLDNN for CPU acceleration | False |
| `rec_batch_num` | Text recognition batch size | 6 |
| `det_db_thresh` | Text detection threshold | 0.3 |
| `det_db_box_thresh` | Text detection box threshold | 0.5 |

##  Output Formats

### Text Extraction JSON Output

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

##  Real-world Applications

- **Document Digitization**: Convert physical documents to digital format
- **Form Processing**: Extract data from forms for automatic database entry
- **Receipt Scanning**: Process receipts for expense management
- **Business Card Scanner**: Extract contact information from business cards
- **Academic Research**: Digitize printed research papers and books
- **Legacy Document Conversion**: Convert old documents for archival purposes
- **Invoice Processing**: Automate accounts payable workflows

##  Performance Tips

- **GPU Acceleration**: Enable GPU support for significantly faster text extraction
  ```bash
  docker-compose run --gpus all --rm document-processor python extract_text.py --use-gpu ...
  ```

- **Batch Processing**: Process multiple documents at once for better throughput

- **Image Resolution**: For optimal results, use images with at least 300 DPI

- **Docker Resource Allocation**: For large documents, increase Docker memory allocation:
  ```bash
  docker run --memory=4g ...
  ```

- **Pre-Processing**: For challenging documents, adjust parameters:
  ```bash
  python scan.py --min-area-ratio 0.2 --max-angle-range 45 ...
  ```

## 👨 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/ouassim-behlil">Ouassim Behlil</a>
</p>