FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed (added OpenCV dependencies)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose the port (Spaces use 7860)
EXPOSE 7860

# Run your Flask app
CMD ["python", "app.py"]
