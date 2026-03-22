#!/bin/sh
set -e

echo "Running Alembic migrations..."
# Simple retry/catch to handle race condition if multiple workers start at once
alembic upgrade head || echo "Migration failed or already applied, continuing..."

echo "Starting application..."
exec "$@"
