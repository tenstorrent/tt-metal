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


# Build release image (pulls from GitHub, uses TT_METAL_HOME only to locate Dockerfile)
docker build --target release-t ghcr.io/btrzynadlowski-tt/tt-telemetry:latest -f $TT_METAL_HOME/tt_telemetry/docker/Dockerfile $TT_METAL_HOME
