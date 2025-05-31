# Use Python 3.10-slim base image
FROM python:3.10-slim

# Set environment variables to avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app
ENV HOME=/app

# Install system dependencies required for pdfplumber (poppler-utils) and others
RUN apt-get update && \
    apt-get install -y poppler-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create required directories with appropriate permissions
RUN mkdir -p /app/.EasyOCR && \
    mkdir -p /app/temp && \
    chmod -R 777 /app/.EasyOCR /app/temp

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# Expose port 7860
EXPOSE 7860

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
