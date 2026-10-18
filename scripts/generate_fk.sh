#!/usr/bin/env bash
# Generate FK/collision-checking C++ code for vamp using cricket.
# Generates two VAMP robot models and installs them in
# ext/ompl_vamp/include/vamp/robots/ (NOT inside the vamp submodule):
#
#   - FetchWholeBody  (11 DOF)  from fetch_spherized.urdf + fetch.srdf
#   - FetchBase       (3  DOF)  from fetch_base.urdf + fetch_base.srdf
#                               (trimmed URDF — base_link only, no arm)
#                               Used as a proper-relaxation collision
#                               model at the lower level of multilevel
#                               whole-body planning.
#
# WARNING: cricket's CppAD codegen is not bit-stable across runs — the
# emitted formulas are mathematically equivalent but use different
# intermediate variable orderings and slightly different floating-point
# rounding.  Regenerating fetch_whole_body.hh can shift sphere positions
# by ~1e-6 m, which is enough to flip borderline collision checks for
# HOME_JOINTS in tight scenes.  After regenerating, rerun the demos and
# expect to need to re-tune any tight collision waypoints.  Regenerate
# only when the URDF or SRDF actually changes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

CRICKET="$ROOT/third_party/cricket"
RESOURCES="$ROOT/fetch_planning/resources/robot/fetch"
CRICKET_CONFIGS="$SCRIPT_DIR/cricket_configs"
FK_DEST="$ROOT/ext/ompl_vamp/include/vamp/robots"

# 1. Distribute robot descriptions and cricket configs into the cricket
#    submodule (kept out of submodule git tracking on purpose — the
#    canonical copies live in this repo).
echo "[1/3] Distributing robot descriptions..."
CRICKET_RES="$CRICKET/resources/fetch"
mkdir -p "$CRICKET_RES"
cp "$RESOURCES/fetch_spherized.urdf" "$CRICKET_RES/"
cp "$RESOURCES/fetch.srdf"           "$CRICKET_RES/"
cp "$RESOURCES/fetch_base.urdf"      "$CRICKET_RES/"
cp "$RESOURCES/fetch_base.srdf"      "$CRICKET_RES/"
cp "$CRICKET_CONFIGS/"*.json         "$CRICKET/resources/"
echo "  copied to $CRICKET_RES and $CRICKET/resources/"

# 2. Run cricket FK code generation
echo "[2/3] Generating FK code..."
mkdir -p "$FK_DEST"

for MODEL in fetch_whole_body fetch_base; do
    CONFIG="$CRICKET/resources/${MODEL}.json"
    if [ ! -f "$CONFIG" ]; then
        echo "  ERROR: $CONFIG not found."
        exit 1
    fi
    echo "  generating FK for $MODEL..."
    (cd "$CRICKET" && "$CRICKET/build/fkcc_gen" "resources/${MODEL}.json")

    HEADER="$CRICKET/${MODEL}_fk.hh"
    if [ ! -f "$HEADER" ]; then
        echo "  ERROR: $HEADER not generated."
        exit 1
    fi
    cp "$HEADER" "$FK_DEST/${MODEL}.hh"
    echo "    installed $FK_DEST/${MODEL}.hh"
done

echo "[3/3] Done! Rebuild the package to use the new FK code."
