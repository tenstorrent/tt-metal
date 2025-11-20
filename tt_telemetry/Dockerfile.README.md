# Interactive Development Dockerfile

This Dockerfile provides an interactive development environment for working on `tt_telemetry` with all necessary dependencies pre-installed.

## Features

- Based on `tt-metalium-ubuntu-22.04-release` base image
- Includes gRPC and Protobuf build dependencies
- Includes Python gRPC tools for client scripts
- Does not clone or build tt-metal (uses bind mount instead)
- Launches bash for interactive development

## Building the Image

From the **tt-metal root directory**, run:

```bash
docker build -t tt-telemetry-dev -f tt_telemetry/Dockerfile .
```

## Running the Container

### Interactive Shell

Mount your local tt-metal directory and launch an interactive shell:

```bash
docker run -it --rm \
  -h $(hostname) \
  -v $(pwd):/tt-metal \
  -v /tmp:/tmp \
  --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  tt-telemetry-dev
```

This will:
- Set container hostname to match host machine (`-h $(hostname)`)
- Mount your current directory (tt-metal) to `/tt-metal` in the container
- Mount host `/tmp` to share UNIX sockets between containers
- Pass through `/dev/tenstorrent` for hardware access (required for UMD)
- Mount `/dev/hugepages-1G` for DMA on non-IOMMU systems
- Drop you into a bash shell
- Remove the container when you exit (`--rm`)

**Note**: The `-v /tmp:/tmp` mount allows multiple containers to access the same UNIX socket at `/tmp/tt_telemetry.sock`.

**Convenience Script**: Use `./tt_telemetry/dev-shell.sh` which handles all these options automatically and includes checks for available devices.

### Building tt_telemetry

Once inside the container:

```bash
# Verify git is working (should show current tag)
git describe --abbrev=10 --first-parent

# Build tt-metal with telemetry
./build_metal.sh --build-telemetry
```

**Troubleshooting VERSION Error**

If you get a CMake error like `No VERSION specified for WRITE_BASIC_CONFIG_VERSION_FILE()`, the git version parsing failed.

**Quick Fix**:
```bash
# Ensure you have proper git tags
git fetch --tags  # If you have a remote

# Or create a dummy tag if needed
git tag v0.1-alpha0 2>/dev/null || true

# Verify it works
git describe --abbrev=10 --first-parent
# Should output something like: v0.1-alpha0 or v0.65.0-dev20251120-21-g4678325952

# Now build
./build_metal.sh --build-telemetry
```

**Common Causes**:
- Repository is a shallow clone (missing tags)
- .git directory not properly mounted
- No tags in the repository
- Git security settings blocking operations on mounted directories

**Alternative**: If git issues persist, you can configure git to trust the mounted directory:
```bash
git config --global --add safe.directory /tt-metal
```

### Running the Telemetry Server

```bash
# Run with mock telemetry
./build/tt_telemetry/tt_telemetry_server --mock-telemetry

# Run with real hardware (requires FSD file)
./build/tt_telemetry/tt_telemetry_server --fsd /path/to/fsd.textproto
```

### Testing the gRPC Client

```bash
# Generate Python gRPC code
cd tt_telemetry/scripts
python3 -m grpc_tools.protoc -I../include/server --python_out=. --grpc_python_out=. ../include/server/telemetry_service.proto

# Run the client (in another terminal or after starting the server in background)
python3 telemetry_client.py
```

## Port Forwarding

To access the web UI from your host machine, use the convenience script with `--ports`:

```bash
./tt_telemetry/dev-shell.sh --ports
```

Or manually:
```bash
docker run -it --rm \
  -h $(hostname) \
  -v $(pwd):/tt-metal \
  -v /tmp:/tmp \
  --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -p 8080:8080 \
  -p 8081:8081 \
  tt-telemetry-dev
```

Then access:
- Web UI: http://localhost:8080
- Prometheus metrics: http://localhost:8080/metrics
- gRPC: unix:///tmp/tt_telemetry.sock (shared via `/tmp` mount)

**Note**: Since `/tmp` is mounted from the host, the gRPC socket at `/tmp/tt_telemetry.sock` is accessible from:
- Other containers with `-v /tmp:/tmp` mount
- Host machine directly (if gRPC client supports it)

## Persistent Build Cache

To persist CMake build cache between container runs:

```bash
docker run -it --rm \
  -v $(pwd):/tt-metal \
  -v tt-metal-build-cache:/tt-metal/build \
  tt-telemetry-dev
```

## Development Workflow

1. **Edit code** on your host machine using your favorite editor/IDE
2. **Build and test** inside the container
3. **Changes persist** because the directory is bind-mounted

### Single Container Workflow

```bash
# Terminal 1: Start container (use convenience script)
./tt_telemetry/dev-shell.sh --ports

# Or manually:
# docker run -it --rm \
#   -h $(hostname) \
#   -v $(pwd):/tt-metal \
#   -v /tmp:/tmp \
#   --device /dev/tenstorrent \
#   -v /dev/hugepages-1G:/dev/hugepages-1G \
#   -p 8080:8080 -p 8081:8081 \
#   tt-telemetry-dev

# Inside container: Build and run
./build_metal.sh --build-telemetry
./build/tt_telemetry/tt_telemetry_server --mock-telemetry -p 8080

# Terminal 2: Edit code on host
vim tt_telemetry/server/grpc_telemetry_server.cpp

# Terminal 1: Rebuild (inside container)
cmake --build build --target tt_telemetry_server
# Restart server...
```

### Multi-Container Workflow (Server + Client)

The `/tmp` mount allows multiple containers to share UNIX sockets:

```bash
# Terminal 1: Start server container
./tt_telemetry/dev-shell.sh
# Inside: Build and run server
./build_metal.sh --build-telemetry
./build/tt_telemetry/tt_telemetry_server --mock-telemetry

# Terminal 2: Start client container (while server is running)
./tt_telemetry/dev-shell.sh
# Inside: Test gRPC connection
cd tt_telemetry/scripts
python3 -m grpc_tools.protoc -I../include/server --python_out=. --grpc_python_out=. ../include/server/telemetry_service.proto
python3 telemetry_client.py
# Should successfully ping the server in the other container!
```

This is useful for:
- Running the server in one container, clients in others
- Testing distributed telemetry scenarios
- Debugging client-server interactions

## Differences from docker/Dockerfile

The `docker/Dockerfile` is designed for **production deployment**:
- Clones tt-metal from GitHub
- Builds everything from scratch
- Creates minimal runtime images
- Has separate `dev` and `release` targets

This `Dockerfile` is designed for **interactive development**:
- Uses bind mount for tt-metal (no cloning)
- Doesn't build anything
- Includes all development tools
- Single simple target
- Launches bash for interactive work

## Hardware Requirements

### Required
- **`/dev/tenstorrent`**: UMD device access (required for hardware telemetry)
  - Automatically passed through by `dev-shell.sh` if available
  - Warning shown if not present

### Optional
- **`/dev/hugepages-1G`**: Hugepages for DMA on non-IOMMU systems
  - Automatically mounted by `dev-shell.sh` if available
  - Not required for all systems

### Hostname
The container hostname should match the host machine for distributed telemetry aggregation to work correctly. The `dev-shell.sh` script automatically sets this with `-h $(hostname)`.

## Installed Packages

The container includes:
- CMake, Ninja, GCC (from base image)
- gRPC C++ development libraries
- Protocol Buffers compiler
- gRPC Protocol Buffers plugin
- Python 3 with pip
- Python gRPC tools (`grpcio`, `grpcio-tools`)
- Git (for version parsing and development)
