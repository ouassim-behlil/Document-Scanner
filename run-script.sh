#!/bin/bash

# Script to run document processing pipeline in Docker

# Check if input file argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 input_file [--batch]"
    echo "Example: $0 document.jpg       # Process a single document"
    echo "         $0 --batch            # Process all images in input directory"
    exit 1
fi

# Ensure input and output directories exist
mkdir -p input output extracted_text

# Process based on arguments
if [ "$1" == "--batch" ]; then
    echo "Batch processing all images in input directory..."
    
    # Step 1: Run document scanner on all images
    docker-compose run --rm document-processor python scan.py --images /app/input --output /app/processed
    
    # Step 2: Extract text from all processed images
    docker-compose run --rm document-processor python extract_text.py --images /app/processed --output /app/extracted_text --format both
    
    echo "✅ Batch processing complete!"
    echo "- Processed images available in: ./output/"
    echo "- Extracted text available in: ./extracted_text/"
else
    # Single file processing
    INPUT_FILE=$1
    FILENAME=$(basename "$INPUT_FILE")
    
    # Check if file exists
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Error: Input file not found: $INPUT_FILE"
        exit 1
    fi
    
    # Copy input file to input directory
    echo "Copying input file to processing directory..."
    cp "$INPUT_FILE" "input/$FILENAME"
    
    echo "Processing document: $FILENAME"
    
    # Step 1: Run document scanner
    echo "Step 1/2: Scanning document..."
    docker-compose run --rm document-processor python scan.py --image "/app/input/$FILENAME" --output /app/processed
    
    # Step 2: Extract text
    echo "Step 2/2: Extracting text..."
    docker-compose run --rm document-processor python extract_text.py --image "/app/processed/$FILENAME" --output /app/extracted_text --format both
    
    echo "✅ Processing complete!"
    echo "- Processed image available at: ./output/$FILENAME"
    echo "- Extracted text available at: ./extracted_text/$(basename "$FILENAME" .jpg).json and ./extracted_text/$(basename "$FILENAME" .jpg).txt"
fi

echo "Done."
