# SatQuery AI — Container Build for Remote Sensing Prototype
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

EXPOSE 5000

# Run with Gunicorn on production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "4", "--timeout", "120", "wsgi:app"]
