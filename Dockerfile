FROM python:3.10-slim

WORKDIR /app
ENV HOME=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create models directory structure with proper permissions
RUN mkdir -p /app/models/transformers_cache && \
    mkdir -p /app/models/sentence_transformers && \
    chmod -R 777 /app/models

# Set environment variables for model caching to local directory
ENV TRANSFORMERS_CACHE=/app/models/transformers_cache
ENV HF_HOME=/app/models/transformers_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/models/sentence_transformers
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONWARNINGS=ignore

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure models directory has correct permissions after copy
RUN chmod -R 777 /app/models

# Optional: Pre-download models during build (comment out if causing issues)
RUN python download_models.py || echo "Model pre-download failed, will download at runtime"

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false"]
