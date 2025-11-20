# TT-Telemetry Interactive Development Dockerfile
#
# This Dockerfile creates an interactive development environment for tt_telemetry.
# It does not build anything - instead, it expects the tt-metal directory to be
# mounted at runtime.
#
# Usage:
#   docker build -t tt-telemetry-dev -f tt_telemetry/Dockerfile .
#   docker run -it --rm -v $(pwd):/tt-metal tt-telemetry-dev

# Use the published tt-metal base image
# This includes basic build tools (cmake, ninja, g++, mpi-ulfm)
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest

# Install gRPC and Protobuf dependencies for tt_telemetry
RUN apt-get update && apt-get install -y \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python gRPC tools for client scripts
RUN pip3 install grpcio grpcio-tools protobuf

# Set up TT_METAL_HOME environment
ENV TT_METAL_HOME=/tt-metal

# Set TT_METAL_RUNTIME_ROOT, required by Metal run-time options
ENV TT_METAL_RUNTIME_ROOT=/tt-metal

# Set working directory to TT_METAL_HOME
# This directory will be mounted at runtime
WORKDIR /tt-metal

# Configure git to trust the mounted directory
# (Newer git versions block operations on directories with different ownership)
RUN git config --global --add safe.directory /tt-metal

# Expose ports for telemetry server
# Port 8080: Web server (Prometheus metrics, UI)
# Port 8081: Collection endpoint (WebSocket for aggregators)
EXPOSE 8080 8081

# Launch bash for interactive development
CMD ["/bin/bash"]
