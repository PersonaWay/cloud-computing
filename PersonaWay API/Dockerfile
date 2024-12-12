# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and application
COPY requirements.txt .
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Menyusun variabel lingkungan untuk konfigurasi
ENV DB_HOST=34.101.104.28
ENV DB_USER=root
ENV DB_PASSWORD=12345
ENV DB_NAME=psikologi
ENV GCS_BUCKET_NAME=image-disc

# Expose the application port
EXPOSE 8080

# Start the Flask app
CMD ["python", "app.py"]