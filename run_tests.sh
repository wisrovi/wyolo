#!/bin/bash
# Runs pytest inside a Docker container
echo "Running tests in Docker container for wyolo..."
mkdir -p /tmp/pip-cache
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v /tmp/pip-cache:/root/.cache/pip -v $(pwd):/app -w /app python:3.10-slim bash -c "apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/* && pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision pytest pytest-cov && pip install -e . && pytest tests/"
