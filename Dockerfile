# =============================================================================
# NGSE: Next-Gen Search Engine - Dockerfile
# Multi-stage build for production deployment
# =============================================================================

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"



RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt
# Stage 2: Runtime
FROM python:3.10-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash ngse
RUN chown -R ngse:ngse /app

# Copy application code
COPY --chown=ngse:ngse . .

# Create data directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/indices /app/logs && \
    chown -R ngse:ngse /app/data /app/logs

# Switch to non-root user
USER ngse

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    PORT=8000 \
    HOST=0.0.0.0 \
    PYTHONMALLOC=malloc

# Expose port
EXPOSE 8000

# Health check (Increased start_period to 300s for model loading)
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application (Increased timeout to 300s)
# Note: No --preload to keep peak memory lower during startup
CMD gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 300
