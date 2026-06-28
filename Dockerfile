FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-prod.txt

COPY app.py index.html ./
COPY outputs/pneumonia/ ./outputs/pneumonia/

ENV PORT=8001
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT', '8001') + '/health')"

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
