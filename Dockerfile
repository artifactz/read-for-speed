# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy contents into the container
COPY . .

# Install required Python packages
RUN pip install --no-cache-dir gunicorn flask msgpack pdfplumber pypdf2 reportlab tqdm numpy pillow torch torchvision

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Start the Flask app using Gunicorn for better performance
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]
