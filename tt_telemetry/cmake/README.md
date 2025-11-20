# CMake Modules

This directory contains custom CMake modules for finding and configuring dependencies.

## Modules

- **FindSystemgRPC.cmake** - Finds gRPC and Protobuf packages (modern and legacy systems)
- **BuildTelemetryProto.cmake** - Builds the telemetry gRPC proto library

## FindSystemgRPC.cmake

Finds system-installed gRPC and Protobuf packages. This module supports both modern and older Linux distributions:

### Modern Systems (Ubuntu 24.04+)
- Uses CMake config files (`gRPCConfig.cmake`, `ProtobufConfig.cmake`)
- Found via `find_package(gRPC CONFIG)`

### Older Systems (Ubuntu 22.04, etc.)
- Falls back to `pkg-config` when CMake config files are not available
- Requires packages: `libgrpc++-dev`, `libprotobuf-dev`, `protobuf-compiler`, `protobuf-compiler-grpc`

### Exported Variables

The module sets the following variables for use in the main CMakeLists.txt:

- `_PROTOBUF_LIBPROTOBUF` - Protobuf library target
- `_PROTOBUF_PROTOC` - Path to protoc executable
- `_GRPC_GRPCPP` - gRPC C++ library target
- `_GRPC_CPP_PLUGIN_EXECUTABLE` - Path to grpc_cpp_plugin executable
- `_REFLECTION` - gRPC reflection library (if available)
- `GRPC_FOUND_VIA_CMAKE` - Boolean indicating which method was used

### Installing gRPC on Ubuntu 22.04

```bash
sudo apt-get update
sudo apt-get install -y \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc
```

### Installing gRPC on Ubuntu 24.04+

```bash
sudo apt-get update
sudo apt-get install -y \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler-grpc
```

The newer versions come with proper CMake config files.

## BuildTelemetryProto.cmake

Builds the `telemetry_grpc_proto` library from the proto file at `include/server/telemetry_service.proto`.

### Requirements

This module must be included **after** `FindSystemgRPC.cmake` as it depends on the variables it exports.

### What It Does

1. Locates the proto file at `include/server/telemetry_service.proto`
2. Creates a library target `telemetry_grpc_proto`
3. Generates C++ code from the proto file (handles both modern and legacy systems)
4. Links against gRPC and Protobuf libraries
5. Exports generated headers via `${CMAKE_CURRENT_BINARY_DIR}`

### Usage

```cmake
# In CMakeLists.txt
include(FindSystemgRPC)      # Must come first
include(BuildTelemetryProto) # Creates telemetry_grpc_proto target
```

Then link your executable:

```cmake
target_link_libraries(your_target PRIVATE telemetry_grpc_proto)
```
