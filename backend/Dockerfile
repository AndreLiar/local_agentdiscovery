# Multi-stage Docker build for Local Discovery Agent

# Stage 1: Base Python image with dependencies
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r app && useradd -r -g app app

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Production image
FROM base as production

# Copy application code
COPY app/ ./app/
COPY main_new.py .

# Create necessary directories
RUN mkdir -p /app/logs && chown -R app:app /app

# Switch to app user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "main_new.py"]

# Stage 3: Development image with additional tools
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    httpx \
    black \
    isort \
    flake8

# Copy application code
COPY . .

# Switch to app user
USER app

# Expose port for development
EXPOSE 8000

# Run in development mode with auto-reload
CMD ["python", "-m", "uvicorn", "app.core.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--reload"]