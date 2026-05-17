FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app" \
    PATH="/app/.venv/bin:${PATH}"

RUN pip install --no-cache-dir uv \
    && true

COPY pyproject.toml uv.lock /app/
RUN uv sync --no-dev

COPY params.yaml /app/params.yaml
COPY alembic.ini /app/alembic.ini
COPY src /app/src
COPY README.md /app/README.md
COPY src/deploy/dockerfiles/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
