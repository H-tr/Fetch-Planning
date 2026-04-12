#!/usr/bin/env bash
# Build planning-ready robot description from Fetch URDF.
# The Fetch URDF and meshes are pre-packaged in fetch_planning/resources/.
# This script distributes them to cricket and vamp resource directories
# so the FK generation pipeline can find them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

RESOURCES="$ROOT/fetch_planning/resources/robot/fetch"
CRICKET="$ROOT/third_party/cricket/resources/fetch"
VAMP="$ROOT/third_party/vamp/resources/fetch"

echo "Distributing Fetch robot descriptions..."
for DEST in "$CRICKET" "$VAMP"; do
    mkdir -p "$DEST"
    cp "$RESOURCES/fetch_spherized.urdf" "$DEST/"
    cp "$RESOURCES/fetch.srdf" "$DEST/"
    echo "  copied to $DEST"
done

echo "Done."