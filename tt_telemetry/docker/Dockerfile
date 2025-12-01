# TT-Telemetry Dockerfile

#############################################################

# Use the published tt-metal base image instead of rebuilding
# This includes basic build tools (cmake, ninja, g++, mpi-ulfm)
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest AS base

#############################################################

FROM base AS release

# Set working directory
WORKDIR /tt-metal

# Clone tt-metal
RUN /bin/bash -c "git clone --filter=blob:none --recurse-submodules --tags \
    https://github.com/tenstorrent/tt-metal.git /tt-metal"

# Set TT_METAL_RUNTIME_ROOT, required by Metal run-time options to be set to tt_metal/ base
ENV TT_METAL_RUNTIME_ROOT=/tt-metal

# Build tt_telemetry then clear cache
RUN /bin/bash -c "./build_metal.sh --build-telemetry \
    && rm -rf /root/.cache /root/.cargo /tt-metal/.cpmcache"

# Expose ports for telemetry server
# Port 8080: Web server
# Port 8081: Intra-process websocket connections
EXPOSE 8080 8081

# Set the telemetry server as the entrypoint
# This allows passing command line arguments directly to docker run
ENTRYPOINT ["./build/tt_telemetry/tt_telemetry_server"]

#############################################################

FROM base AS dev

# Set working directory
WORKDIR /tt-metal

# Set TT_METAL_RUNTIME_ROOT, required by Metal run-time options to be set to tt_metal/ base
ENV TT_METAL_RUNTIME_ROOT=/tt-metal

# Copy the pre-built telemetry server binary and frontend files
COPY build/tt_telemetry/ ./build/tt_telemetry/

# Copy libs
COPY build_Release/lib ./build_Release/lib

# Copy the entire tt_telemetry directory (provides fallback path for frontend files and source tree access)
COPY tt_telemetry/ ./tt_telemetry/

# Set library search path to find the copied libraries
ENV LD_LIBRARY_PATH=/tt-metal/build_Release/lib

# Make sure the binary is executable
RUN chmod +x ./build/tt_telemetry/tt_telemetry_server

# Expose ports for telemetry server
# Port 8080: Web server
# Port 8081: Intra-process websocket connections
EXPOSE 8080 8081

# Set the telemetry server as the entrypoint
# This allows passing command line arguments directly to docker run
ENTRYPOINT ["./build/tt_telemetry/tt_telemetry_server"]
