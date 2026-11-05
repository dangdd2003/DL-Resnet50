# syntax=docker/dockerfile:1.7
#
# Build the image:
#   docker build -t dlresnet .
#
# Train on a GPU, with datasets and outputs mounted from the host:
#   docker run --gpus all --rm \
#     -v "$PWD/data:/workspace/data" \
#     -v "$PWD/runs:/workspace/runs" \
#     dlresnet train --config configs/birds_scratch.yaml

ARG TF_VERSION=2.21.0
FROM tensorflow/tensorflow:${TF_VERSION}-gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    KERAS_BACKEND=tensorflow

WORKDIR /workspace

# Install the package first so that code-only changes reuse the dependency layer.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --upgrade pip && pip install .

COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Free uid 1000 if the base image already uses it (e.g. the `ubuntu` user on 24.04 images).
RUN if existing="$(getent passwd 1000 | cut -d: -f1)" && [ -n "$existing" ]; then \
        userdel --remove "$existing" 2>/dev/null || userdel "$existing"; \
    fi \
    && useradd --create-home --uid 1000 trainer \
    && mkdir -p /workspace/data /workspace/runs \
    && chown -R trainer:trainer /workspace

USER trainer

VOLUME ["/workspace/data", "/workspace/runs"]

ENTRYPOINT ["dlresnet"]
CMD ["--help"]
