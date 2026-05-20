#!/bin/bash
set -euo pipefail

# Safer .env loading
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

echo "Starting database..."
docker compose up -d

echo "Waiting for Neo4j to be ready..."
MAX_WAIT=60   # seconds
ELAPSED=0

until docker exec uozp_neo4j cypher-shell -u "$DB_USER" -p "$DB_PASSWORD" "RETURN 1" > /dev/null 2>&1; do
  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    echo "ERROR: Neo4j did not become ready within ${MAX_WAIT}s."
    echo "--- Last cypher-shell output ---"
    docker exec uozp_neo4j cypher-shell -u "$DB_USER" -p "$DB_PASSWORD" "RETURN 1" || true
    echo "--- Neo4j container logs ---"
    docker logs --tail 30 uozp_neo4j
    exit 1
  fi
  echo "  Still waiting... (${ELAPSED}s elapsed)"
  sleep 5
  ELAPSED=$((ELAPSED + 5))
done

echo "Neo4j is ready."

HAS_DATA=$(docker exec uozp_neo4j cypher-shell -u "$DB_USER" -p "$DB_PASSWORD" \
  "MATCH (n) RETURN count(n) as count" | tail -n 1)

if [ "$HAS_DATA" -eq "0" ]; then
  echo "Importing data..."
  uv run python -m geonews.db.ingest
else
  echo "Database already contains data. Skipping import."
fi

echo "Database accessible at http://localhost:${NEO4J_HTTP_PORT}"