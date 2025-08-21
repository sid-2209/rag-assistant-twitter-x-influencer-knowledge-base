# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENABLE_WEB_UI=true

WORKDIR /app

# System deps (optional: for prod TLS, locales, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY infra/requirements.txt /app/infra/requirements.txt
RUN pip install --no-cache-dir -r /app/infra/requirements.txt

# Copy source
COPY . /app

# Create models directory for persistence
RUN mkdir -p /app/models

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/healthz || exit 1

# Default command
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
