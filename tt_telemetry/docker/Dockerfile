# TT-Telemetry Dockerfile

#############################################################

# Use the published tt-metal base image instead of rebuilding
# This includes basic build tools (cmake, ninja, g++, mpi-ulfm)
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest AS base

#############################################################

FROM base AS release-telemetry

# Set up TT_METAL_HOME
# (PYTHON_ENV_DIR is already set in the base image)
ENV TT_METAL_HOME=/tt-metal
ENV PYTHONPATH=/tt-metal

# Clone tt-metal
RUN /bin/bash -c "git clone --filter=blob:none --recurse-submodules --tags \
    https://github.com/tenstorrent/tt-metal.git ${TT_METAL_HOME} \
    && cd ${TT_METAL_HOME}"

WORKDIR /tt-metal

# Build tt_telemetry then clear cache
RUN /bin/bash -c "./build_metal.sh --build-telemetry \
    && rm -rf /root/.cache /root/.cargo /tt-metal/.cpmcache"

# Extra system deps
# RUN apt-get update && apt-get install -y \
#     # required
#     gosu \
#     libgl1 \
#     libsndfile1 \
#     && rm -rf /var/lib/apt/lists/*

# Expose ports for telemetry server
# Port 8080: Web server
# Port 8081: Intra-process websocket connections
EXPOSE 8080 8081

# Set the telemetry server as the default command
# This allows passing command line arguments when running the container
CMD ["./build/tt_telemetry/tt_telemetry_server"]

#############################################################

FROM base AS dev-telemetry

# Set up TT_METAL_HOME for local development
# (PYTHON_ENV_DIR is already set in the base image)
ENV TT_METAL_HOME=/tt-metal
ENV PYTHONPATH=/tt-metal

# Set working directory to TT_METAL_HOME
WORKDIR ${TT_METAL_HOME}

# Copy the entire local tt-metal repository
# This copies from your build context (local $TT_METAL_HOME) to /tt-metal/ in container
COPY . .

# Build tt_telemetry from local source, then clear cache. We need to clean first because we
# will have copied over some stale build files from the context directory.
RUN /bin/bash -c "./build_metal.sh --clean \
    && ./build_metal.sh --build-telemetry \
    && rm -rf /root/.cache /root/.cargo /tt-metal/.cpmcache"

# Expose ports for telemetry server
# Port 8080: Web server
# Port 8081: Intra-process websocket connections
EXPOSE 8080 8081

# Set the telemetry server as the default command
# This allows passing command line arguments when running the container
CMD ["./build/tt_telemetry/tt_telemetry_server"]
