#!/bin/bash
# Re-point the dataset mounts without rebuilding the image. Use after editing
# CSV_SOURCES in .env, or when a mounted host dir was replaced rather than
# edited in place. Edits *inside* a mounted dir need nothing — bind mounts are live.
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] || { echo "no .env — copy .env.example to .env first"; exit 1; }
set -a; source .env; set +a
source ./volumes.sh

before="$(cat docker-compose.override.yml 2>/dev/null || true)"
generate_override
detect_stale

if [ "$before" != "$(cat docker-compose.override.yml)" ]; then
  echo "mount config changed — recreating" >&2
elif [ ${#stale[@]} -gt 0 ]; then
  echo "stale mount(s) — recreating to re-point: ${stale[*]}" >&2
else
  echo "mounts already up to date — nothing to do"
  exit 0
fi

# --no-build so a dirty working tree can't sneak a new image in here
docker compose up -d --no-build --force-recreate csv-analyst
