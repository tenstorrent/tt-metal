#!/bin/bash

set -e

# Build release-telemetry image (pulls from GitHub, doesn't need TT_METAL_HOME)
docker build --target release-telemetry -t ghcr.io/btrzynadlowski-tt/tt-telemetry-release:latest -f $TT_METAL_HOME/tt_telemetry/docker/Dockerfile .
