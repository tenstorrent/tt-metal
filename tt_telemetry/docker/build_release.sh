#!/bin/bash

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Metal root is three levels up from tt_telemetry/docker/
METAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Using Metal root: $METAL_ROOT"

# Build release image (pulls from GitHub, builds inside container)
docker build --target release -t ghcr.io/btrzynadlowski-tt/tt-telemetry:latest -f "$METAL_ROOT/tt_telemetry/docker/Dockerfile" "$METAL_ROOT"
