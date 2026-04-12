#!/usr/bin/env bash
# Spherize the Fetch URDF using foam.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

RESOURCES="$ROOT/fetch_planning/resources/robot/fetch"

python "$ROOT/third_party/foam/scripts/generate_sphere_urdf.py" \
    "$RESOURCES/fetch.urdf" \
    --output "$RESOURCES/fetch_spherized.urdf" \
    --database "$ROOT/third_party/foam/sphere_database.json" \
    "$@"