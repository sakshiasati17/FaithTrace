#!/bin/sh
set -e

# Only the API service should run migrations (set RUN_MIGRATIONS=true in docker-compose api env).
# Workers skip this to avoid race conditions on startup.
if [ "${RUN_MIGRATIONS:-false}" = "true" ]; then
  echo "Running Alembic migrations..."
  alembic upgrade head
  echo "Migrations complete."
fi

echo "Starting application..."
exec "$@"
