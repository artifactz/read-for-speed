# Use an official Python runtime as a parent image
FROM python:3.13.3-slim

# Set the working directory
WORKDIR /app

# Install required Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy contents into the container
COPY . .

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Start the Flask app
CMD ["python", "main.py"]
