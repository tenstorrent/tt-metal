#!/bin/bash

#
# Builds a development build (target: dev-telemetry) by first building the telemetry server locally,
# then creating a minimal Docker image that contains only the binary.
#

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Metal root is three levels up from tt_telemetry/docker/
METAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Using Metal root: $METAL_ROOT"

# Build telemetry server locally first
echo "Building telemetry server locally..."
cd "$METAL_ROOT"
./build_metal.sh --build-telemetry --build-static-libs

# Check if the binary was built successfully
if [ ! -f "$METAL_ROOT/build/tt_telemetry/tt_telemetry_server" ]; then
    echo "Error: tt_telemetry_server binary not found after build"
    exit 1
fi

echo "Local build successful. Creating minimal Docker image..."

# Build minimal Docker image with just the binary
docker build --target dev \
    -t ghcr.io/btrzynadlowski-tt/tt-telemetry-dev:latest \
    -f "$METAL_ROOT/tt_telemetry/docker/Dockerfile" "$METAL_ROOT"
