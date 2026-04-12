#!/usr/bin/env bash
# Download scene assets (RLS environment meshes and point clouds).
# The Fetch robot description is pre-packaged in fetch_planning/resources/
# and does not need to be downloaded separately.
set -euo pipefail

echo "Note: Fetch robot description is already included in the package."
echo "This script downloads optional scene assets for the rls_pick_place_demo."
echo ""
echo "If you have the assets.zip URL, run:"
echo "  wget -O assets.zip <URL>"
echo "  unzip assets.zip"
echo "  rm assets.zip"