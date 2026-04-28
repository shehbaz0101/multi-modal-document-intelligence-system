# DocIntel — production Dockerfile
# Single-stage CPU build. Runtime image ~1.8 GB.
#
# Build:  docker build -t docintel:latest .
# Run:    docker run -p 8000:8000 --env-file .env docintel:latest

FROM python:3.11-slim AS runtime

# System deps for pdfplumber (needs libpoppler) and pytesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU torch FIRST — pinned to CPU index so we get the small wheel
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install rest of dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so cold start is fast
# This adds ~25MB to the image but eliminates first-query latency
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY app/ ./app/
COPY config/ ./config/
COPY embedding/ ./embedding/
COPY evaluation/ ./evaluation/
COPY generation/ ./generation/
COPY indexing/ ./indexing/
COPY ingestion/ ./ingestion/
COPY parsing/ ./parsing/
COPY retrieval/ ./retrieval/
COPY scripts/ ./scripts/
COPY ui.html .

# Create data directory (mounted as a volume in production)
RUN mkdir -p /app/data

# Non-root user (security best practice)
RUN useradd -m -u 1000 docintel && chown -R docintel:docintel /app
USER docintel

EXPOSE 8000

# Healthcheck — orchestrators (k8s, ECS, Docker Swarm) use this
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]