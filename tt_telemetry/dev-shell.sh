#!/bin/bash

# dev-shell.sh
# Convenience script to launch the tt_telemetry development container
#
# Usage:
#   ./tt_telemetry/dev-shell.sh [options]
#
# Options:
#   --build         Build the Docker image first
#   --ports         Forward ports 8080 and 8081 to host
#   --cache         Use persistent build cache volume
#   --help          Show this help message

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="tt-telemetry-dev"

# Parse arguments
BUILD_IMAGE=0
FORWARD_PORTS=0
USE_CACHE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD_IMAGE=1
            shift
            ;;
        --ports)
            FORWARD_PORTS=1
            shift
            ;;
        --cache)
            USE_CACHE=1
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Launch an interactive development shell for tt_telemetry"
            echo ""
            echo "Options:"
            echo "  --build    Build the Docker image first"
            echo "  --ports    Forward ports 8080 and 8081 to host"
            echo "  --cache    Use persistent build cache volume"
            echo "  --help     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Launch interactive shell"
            echo "  $0 --build            # Build image then launch shell"
            echo "  $0 --ports            # Launch with port forwarding"
            echo "  $0 --build --ports    # Build and launch with ports"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build image if requested
if [ $BUILD_IMAGE -eq 1 ]; then
    echo "Building Docker image: ${IMAGE_NAME}..."
    cd "${TT_METAL_ROOT}"
    docker build -t "${IMAGE_NAME}" -f tt_telemetry/Dockerfile .
    echo "Build complete!"
    echo ""
fi

# Check if image exists
if ! docker image inspect "${IMAGE_NAME}" &> /dev/null; then
    echo "Error: Docker image '${IMAGE_NAME}' not found"
    echo "Build it first with: $0 --build"
    exit 1
fi

# Build docker run command
DOCKER_RUN_CMD="docker run -it --rm"

# Set hostname to match host machine (required for distributed telemetry)
DOCKER_RUN_CMD="${DOCKER_RUN_CMD} -h $(hostname)"

# Mount tt-metal source directory
DOCKER_RUN_CMD="${DOCKER_RUN_CMD} -v ${TT_METAL_ROOT}:/tt-metal"

# Mount host /tmp to share UNIX sockets between containers
DOCKER_RUN_CMD="${DOCKER_RUN_CMD} -v /tmp:/tmp"

# Pass through Tenstorrent device (required for UMD)
if [ -e /dev/tenstorrent ]; then
    DOCKER_RUN_CMD="${DOCKER_RUN_CMD} --device /dev/tenstorrent"
else
    echo "Warning: /dev/tenstorrent not found - hardware access will not be available"
fi

# Pass through hugepages (required for DMA on non-IOMMU systems)
if [ -e /dev/hugepages-1G ]; then
    DOCKER_RUN_CMD="${DOCKER_RUN_CMD} -v /dev/hugepages-1G:/dev/hugepages-1G"
fi

# Add port forwarding if requested
if [ $FORWARD_PORTS -eq 1 ]; then
    DOCKER_RUN_CMD="${DOCKER_RUN_CMD} -p 8080:8080 -p 8081:8081"
    echo "Port forwarding enabled:"
    echo "  - Web UI: http://localhost:8080"
    echo "  - Collection endpoint: ws://localhost:8081"
    echo ""
fi

# Add persistent cache if requested
if [ $USE_CACHE -eq 1 ]; then
    DOCKER_RUN_CMD="${DOCKER_RUN_CMD} -v tt-metal-build-cache:/tt-metal/build"
    echo "Using persistent build cache volume"
    echo ""
fi

# Add image name
DOCKER_RUN_CMD="${DOCKER_RUN_CMD} ${IMAGE_NAME}"

echo "Launching interactive development shell..."
echo "Hostname: $(hostname)"
echo "Working directory: /tt-metal (mounted from ${TT_METAL_ROOT})"
echo "Shared /tmp: /tmp (mounted from host)"
echo "  - UNIX sockets in /tmp are shared across all containers"
echo "  - gRPC socket: /tmp/tt_telemetry.sock"
echo ""
echo "Hardware access:"
if [ -e /dev/tenstorrent ]; then
    echo "  ✓ /dev/tenstorrent (Tenstorrent device)"
else
    echo "  ✗ /dev/tenstorrent not found (hardware telemetry unavailable)"
fi
if [ -e /dev/hugepages-1G ]; then
    echo "  ✓ /dev/hugepages-1G (hugepages for DMA)"
fi
echo ""
echo "First time setup (ensure git tags are available):"
echo "  git fetch --tags 2>/dev/null || git tag -a v0.1-alpha0 -m 'Dev' 2>/dev/null || true"
echo ""
echo "To build tt_telemetry inside the container:"
echo "  ./build_metal.sh --build-telemetry"
echo ""
echo "To run the telemetry server:"
echo "  ./build/tt_telemetry/tt_telemetry_server --mock-telemetry"
echo ""
echo "To test gRPC from another container (in parallel):"
echo "  # Terminal 2: Launch another dev container"
echo "  ./tt_telemetry/dev-shell.sh"
echo "  # Inside: Test the gRPC connection"
echo "  cd tt_telemetry/scripts"
echo "  python3 telemetry_client.py"
echo ""

# Run the container
exec $DOCKER_RUN_CMD
