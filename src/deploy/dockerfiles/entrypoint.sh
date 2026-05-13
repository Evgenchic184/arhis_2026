#!/bin/sh
set -eu

export PYTHONPATH=/app${PYTHONPATH:+:$PYTHONPATH}

echo "Waiting for database migrations..."

attempt=1
max_attempts=12
sleep_seconds=3

while [ "$attempt" -le "$max_attempts" ]; do
  if uv run alembic -c /app/alembic.ini upgrade head; then
    echo "Migrations applied successfully."
    break
  fi

  if [ "$attempt" -eq "$max_attempts" ]; then
    echo "Migration failed after $max_attempts attempts."
    exit 1
  fi

  echo "Migration attempt $attempt failed, retrying in ${sleep_seconds}s..."
  attempt=$((attempt + 1))
  sleep "$sleep_seconds"
done

exec uvicorn src.app.main:app --host 0.0.0.0 --port 8000
