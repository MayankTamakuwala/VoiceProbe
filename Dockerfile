FROM python:3.12-slim

# onnxruntime and ctranslate2 (faster-whisper) need openmp
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    MODELS_DIR=/app/models

# deps first so code changes don't redo the slow layer
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev
COPY scenarios ./scenarios

EXPOSE 8000 8001
CMD ["voiceprobe", "--help"]
