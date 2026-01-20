FROM python:3.10-slim AS base
WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

COPY src src/

# ... previous steps ...

# Line 15 & 16: Install the requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Line 18: Install your local project (Make sure this is a NEW line starting with RUN)
RUN pip install . --no-deps --no-cache-dir --verbose

# Line 20: The Entrypoint
ENTRYPOINT ["python", "-u", "-m", "group_56.train"]
