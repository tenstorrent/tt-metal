#!/bin/bash

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

# Build dev-telemetry image, which requires that the tt-metal repo ($TT_METAL_HOME) be the context directory
docker build --target dev-telemetry -t ghcr.io/btrzynadlowski-tt/tt-telemetry-dev:latest -f $TT_METAL_HOME/tt_telemetry/docker/Dockerfile $TT_METAL_HOME
