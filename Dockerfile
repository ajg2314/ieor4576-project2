FROM python:3.12-slim

WORKDIR /app

# System deps: matplotlib needs these for the Agg backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock* ./

# Install dependencies (no dev deps, no editable install for speed)
RUN uv sync --frozen --no-dev --no-editable

# Copy application code (respects .dockerignore)
COPY . .

# Ensure artifacts and data dirs exist; artifacts is writable at runtime
RUN mkdir -p artifacts data/sector_knowledge data/report_examples

# Use non-interactive Agg backend for matplotlib (no display needed)
ENV MPLBACKEND=Agg

# Disable LiteLLM telemetry
ENV LITELLM_TELEMETRY=False

# Structured logging
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "600"]
