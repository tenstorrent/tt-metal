#!/bin/bash

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Metal root is two levels up from tt_telemetry/docker/
METAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Using Metal root: $METAL_ROOT"

# Validate paths exist
DOCKERFILE="$METAL_ROOT/tt_telemetry/docker/Dockerfile"
if [ ! -f "$DOCKERFILE" ]; then
    echo "Error: Dockerfile not found at $DOCKERFILE"
    exit 1
fi

if [ ! -d "$METAL_ROOT" ]; then
    echo "Error: Metal root directory not found at $METAL_ROOT"
    exit 1
fi

# Build release image (pulls from GitHub, builds inside container)
if ! docker build --target release \
    -t ghcr.io/btrzynadlowski-tt/tt-telemetry:latest \
    -f "$DOCKERFILE" "$METAL_ROOT"; then
    echo "Error: docker build failed for Dockerfile $DOCKERFILE (exit code: $?)"
    exit 1
fi

echo "Docker image built successfully"
