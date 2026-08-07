#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] || { echo "no .env — copy .env.example to .env first"; exit 1; }
set -a; source .env; set +a
source ./volumes.sh

generate_override
detect_stale

if [ ${#stale[@]} -gt 0 ]; then
  echo "stale mount(s) — recreating to re-point: ${stale[*]}" >&2
  docker compose up -d --build --force-recreate
else
  docker compose up -d --build
fi
