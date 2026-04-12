#!/usr/bin/env bash
# Generate FK/collision-checking C++ code for vamp using cricket.
# Generates the 11-DOF FetchWholeBody model and installs it in
# ext/ompl_vamp/include/vamp/robots/ (NOT inside the vamp submodule).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

CRICKET="$ROOT/third_party/cricket"
RESOURCES="$ROOT/fetch_planning/resources/robot/fetch"
FK_DEST="$ROOT/ext/ompl_vamp/include/vamp/robots"

# 1. Distribute robot descriptions to cricket resources
echo "[1/3] Distributing robot descriptions..."
CRICKET_RES="$CRICKET/resources/fetch"
mkdir -p "$CRICKET_RES"
cp "$RESOURCES/fetch_spherized.urdf" "$CRICKET_RES/"
cp "$RESOURCES/fetch.srdf" "$CRICKET_RES/"
echo "  copied to $CRICKET_RES"

# 2. Run cricket FK code generation
echo "[2/3] Generating FK code..."
CONFIG="$CRICKET/resources/fetch_whole_body.json"
if [ -f "$CONFIG" ]; then
    echo "  generating FK for fetch_whole_body..."
    "$CRICKET/build/fkcc_gen" "$CONFIG"
else
    echo "  ERROR: $CONFIG not found. Run pixi run cricket-build first."
    exit 1
fi

# 3. Copy generated header to ext/ompl_vamp/include (NOT into vamp)
echo "[3/3] Installing FK header..."
HEADER="fetch_whole_body_fk.hh"
if [ -f "$HEADER" ]; then
    mkdir -p "$FK_DEST"
    cp "$HEADER" "$FK_DEST/fetch_whole_body.hh"
    echo "  installed $FK_DEST/fetch_whole_body.hh"
else
    echo "  ERROR: $HEADER not generated."
    exit 1
fi

echo "Done! Rebuild the package to use the new FK code."