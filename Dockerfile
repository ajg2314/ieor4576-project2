FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock* ./

# Install dependencies (no dev deps, frozen lockfile)
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Ensure artifacts directory exists at runtime
RUN mkdir -p artifacts

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
