FROM python:3.12-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

# Create models directory for checkpoint
RUN mkdir -p /app/models

# Cloud Run expects PORT environment variable
ENV PORT=8080

ENTRYPOINT uvicorn group_56.api:app --host 0.0.0.0 --port ${PORT:-8080}
