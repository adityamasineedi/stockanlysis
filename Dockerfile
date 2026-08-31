FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libxml2 \
        libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY prompts ./prompts
COPY data/portfolio ./data/portfolio
COPY data/portfolio/sip_portfolios.json ./config/portfolio/sip_portfolios.json

RUN uv sync --frozen --no-dev --no-editable

CMD ["uv", "run", "stockbot-bot"]
