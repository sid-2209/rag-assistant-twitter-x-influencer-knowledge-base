# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional: for prod TLS, locales, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY infra/requirements.txt /app/infra/requirements.txt
RUN pip install --no-cache-dir -r /app/infra/requirements.txt

# Copy source
COPY . /app

EXPOSE 8000

# Default command
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
