#!/bin/bash

#
# Builds a development build (target: dev-telemetry) by first building the telemetry server locally,
# then creating a minimal Docker image that contains only the binary.
#

set -e

# Check if TT_METAL_HOME is set
if [ -z "$TT_METAL_HOME" ]; then
    echo "Error: TT_METAL_HOME environment variable is not set"
    exit 1
fi

# Check if TT_METAL_HOME directory exists
if [ ! -d "$TT_METAL_HOME" ]; then
    echo "Error: TT_METAL_HOME directory does not exist: $TT_METAL_HOME"
    exit 1
fi

# Build telemetry server locally first
echo "Building telemetry server locally..."
cd $TT_METAL_HOME
./build_metal.sh --build-telemetry --build-static-libs

# Check if the binary was built successfully
if [ ! -f "$TT_METAL_HOME/build/tt_telemetry/tt_telemetry_server" ]; then
    echo "Error: tt_telemetry_server binary not found after build"
    exit 1
fi

echo "Local build successful. Creating minimal Docker image..."

# Build minimal Docker image with just the binary
docker build --target dev-telemetry \
    -t ghcr.io/btrzynadlowski-tt/tt-telemetry-dev:latest \
    -f $TT_METAL_HOME/tt_telemetry/docker/Dockerfile $TT_METAL_HOME
