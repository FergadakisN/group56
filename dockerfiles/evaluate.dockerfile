# 1. Reuse the same base image
FROM python:3.10-slim AS base
WORKDIR /app

# 2. System dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# 3. Copy requirements (same as train)
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# 4. Install dependencies (using cache mount for speed)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# 5. Copy the source code
COPY src src/

# 6. Install the project
RUN pip install . --no-deps --no-cache-dir

# 7. Change the Entrypoint to the evaluation script
ENTRYPOINT ["python", "-u", "-m", "group_56.evaluate"]
