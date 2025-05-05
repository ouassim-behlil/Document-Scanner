# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    unzip \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Add a font for the visualization feature
RUN mkdir -p /app/fonts && \
    wget -q https://github.com/googlefonts/roboto/releases/download/v2.138/roboto-unhinted.zip && \
    unzip -q roboto-unhinted.zip -d /tmp && \
    cp /tmp/RobotoTTF/Roboto-Regular.ttf /app/fonts/simfang.ttf && \
    rm -rf /tmp/RobotoTTF roboto-unhinted.zip

# Copy application code
COPY scan.py extract_text.py ./

# Create directories for inputs and outputs
RUN mkdir -p /app/input /app/output /app/processed /app/extracted_text

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - print help information
CMD ["python", "-c", "import os; os.system('echo \"Document Processing System\\n\\nAvailable commands:\\n  python scan.py --image /app/input/your_image.jpg --output /app/processed\\n  python extract_text.py --image /app/processed/your_image.jpg --output /app/extracted_text\\n\\nMount volumes to access inputs and outputs from host.\"')"]