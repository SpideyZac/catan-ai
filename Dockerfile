# syntax=docker/dockerfile:1
# Production image: Rust engine wheel + FastAPI server + prebuilt web client.
#
#   docker build -t catan-ai .
#   docker build -t catan-ai --build-arg WITH_TORCH=1 .   # enable neural bots (CPU torch)
#   docker run -p 8000:8000 -v catan-data:/data -v $PWD/models:/models catan-ai

# ---------------------------------------------------------------- web client
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---------------------------------------------------------------- engine wheel
FROM python:3.12-slim AS wheel
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH
RUN pip install --no-cache-dir "maturin>=1.8,<2"
WORKDIR /src
COPY Cargo.toml Cargo.lock rustfmt.toml pyproject.toml README.md ./
COPY crates crates
COPY python python
RUN maturin build --release --out /dist

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim
ARG WITH_TORCH=0
WORKDIR /app
COPY --from=wheel /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl \
    && if [ "$WITH_TORCH" = "1" ]; then pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; fi \
    && useradd --create-home --uid 1000 catan && mkdir -p /data/rooms /models && chown -R catan /data /models
COPY --from=web /web/dist /app/static
ENV CATAN_STATIC_DIR=/app/static \
    CATAN_DATA_DIR=/data/rooms \
    CATAN_MODELS_DIR=/models \
    PYTHONUNBUFFERED=1
USER catan
VOLUME ["/data", "/models"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["catan-server", "--host", "0.0.0.0", "--port", "8000"]
