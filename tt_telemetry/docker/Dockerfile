# TT-Telemetry Dockerfile

#############################################################

# Use the published tt-metal base image instead of rebuilding
# This includes basic build tools (cmake, ninja, g++, mpi-ulfm)
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest AS base

#############################################################

FROM base AS release

# Set up TT_METAL_HOME
# (PYTHON_ENV_DIR is already set in the base image)
ENV TT_METAL_HOME=/tt-metal

# Set TT_METAL_RUNTIME_ROOT
ENV TT_METAL_RUNTIME_ROOT=/tt-metal

# Clone tt-metal
RUN /bin/bash -c "git clone --filter=blob:none --recurse-submodules --tags \
    https://github.com/tenstorrent/tt-metal.git ${TT_METAL_HOME} \
    && cd ${TT_METAL_HOME}"

WORKDIR /tt-metal

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

# Set up TT_METAL_HOME environment
ENV TT_METAL_HOME=/tt-metal

# TT_METAL_RUNTIME_ROOT
ENV TT_METAL_RUNTIME_ROOT=/tt-metal

# Set working directory to TT_METAL_HOME
WORKDIR ${TT_METAL_HOME}

# Copy the pre-built telemetry server binary
COPY build/tt_telemetry/tt_telemetry_server ./build/tt_telemetry/tt_telemetry_server

# Copy libs
COPY build_Release/lib ./build_Release/lib

# Copy the entire tt_telemetry directory (contains frontend files, etc.)
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
