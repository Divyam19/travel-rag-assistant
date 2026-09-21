# One image serves both the API and the built React UI, so a deployment is a single service on a
# single origin (no CORS, no second proxy).

# ---- stage 1: build the frontend
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- stage 2: the API
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken
WORKDIR /app

# Editable install on purpose: config.ROOT is found relative to the package file, so a normal
# install into site-packages would point ROOT at the wrong place.
COPY pyproject.toml ./
COPY src ./src
RUN pip install -e .

# Download the tokenizer now. Otherwise the first request after every deploy fetches it from the
# internet and fails if that fetch does.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY db ./db
COPY --from=web /web/dist ./frontend/dist

# Recorded Tavily responses are written under data/fixtures at runtime, so it must be writable.
RUN useradd --create-home app \
    && mkdir -p data/fixtures/tavily \
    && chown -R app:app /app /opt/tiktoken
USER app

# Railway supplies $PORT. A single worker on purpose: the rate limiter keeps its counters in memory.
CMD ["sh", "-c", "exec uvicorn travelrag.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
