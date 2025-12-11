#!/bin/bash

#
# Builds a development build (target: dev-telemetry) by first building the telemetry server locally,
# then creating a minimal Docker image that contains only the binary.
#

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

# Build telemetry server locally first
echo "Building telemetry server locally..."
cd "$METAL_ROOT"
./build_metal.sh --build-telemetry --build-static-libs

# Check if the binary was built successfully and is executable
BINARY="$METAL_ROOT/build/tt_telemetry/tt_telemetry_server"
if [ ! -x "$BINARY" ]; then
    echo "Error: tt_telemetry_server binary not found or not executable at $BINARY"
    exit 1
fi

echo "Local build successful. Creating minimal Docker image..."

# Build minimal Docker image with just the binary
if ! docker build --target dev \
    -t ghcr.io/btrzynadlowski-tt/tt-telemetry-dev:latest \
    -f "$DOCKERFILE" "$METAL_ROOT"; then
    echo "Error: docker build failed for Dockerfile $DOCKERFILE (exit code: $?)"
    exit 1
fi

echo "Docker image built successfully"
