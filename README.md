# Document Processing System

<!-- Badges -->

[![Python](https://img.shields.io/badge/Python-3.6%2B-blue)](https://www.python.org/) [![Docker](https://img.shields.io/badge/Docker-%3E%3D20-blue)](https://www.docker.com/) [![Docker Compose](https://img.shields.io/badge/Docker%20Compose-latest-blue)](https://docs.docker.com/compose/) [![PaddleOCR](https://img.shields.io/badge/PaddleOCR-latest-orange)](https://github.com/PaddlePaddle/PaddleOCR) [![OpenCV](https://img.shields.io/badge/OpenCV-latest-brightgreen)](https://opencv.org/)

A lightweight tool to scan and extract text from document images using Python and Docker.

## Features

* Automatic document detection, perspective correction, and enhancement (`scan.py`)
* High‑accuracy text extraction with PaddleOCR (`extract_text.py`)
* Batch processing support
* Docker & Docker Compose setup for easy deployment

## Requirements

* Python 3.6+
* Docker and Docker Compose (optional)
* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

## Usage

### 1. Scan a document

```bash
python scan.py --image path/to/image.jpg --output processed/
```

### 2. Extract text

```bash
python extract_text.py --image processed/image.jpg --output extracted_text/ --format both
```

### 3. Full pipeline (single command)

```bash
./run-script.sh document.jpg
```

## Docker

```bash
docker-compose build
docker-compose run --rm document-processor python scan.py --image /app/input/doc.jpg --output /app/processed
# then text extraction...
```



## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
