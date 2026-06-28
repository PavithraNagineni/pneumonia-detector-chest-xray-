# Use a lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker cache)
COPY requirements.txt .

# Install CPU version of PyTorch
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app.py .
COPY index.html .
COPY outputs ./outputs
COPY samples ./samples

# Copy any remaining project files
COPY . .

# Expose Railway port
EXPOSE 8000

# Start FastAPI
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
