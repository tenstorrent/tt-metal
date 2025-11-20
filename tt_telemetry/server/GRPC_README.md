# gRPC Telemetry Service

This document explains the gRPC telemetry service implementation using UNIX domain sockets.

## Overview

The gRPC telemetry service provides a high-performance IPC mechanism for other processes on the same machine to interact with the telemetry server. It runs on a UNIX domain socket at `/tmp/tt_telemetry.sock`.

## Architecture

### Components

1. **Protocol Definition** (`include/server/telemetry_service.proto`)
   - Defines the gRPC service interface using Protocol Buffers
   - Currently implements a simple Ping/Pong service for RTT measurement
   - Located in `include/` as it's an interface definition

2. **Server Implementation** (`server/grpc_telemetry_server.cpp/hpp`)
   - Implements `GrpcTelemetryServer` class that derives from `TelemetrySubscriber`
   - Automatically receives telemetry updates through the subscriber interface
   - Runs as part of the main telemetry server process

3. **CMake Build Logic** (`cmake/BuildTelemetryProto.cmake`)
   - Handles proto file compilation for both modern and legacy systems
   - Generates C++ code from the proto file
   - Creates the `telemetry_grpc_proto` library

4. **Python Client** (`scripts/telemetry_client.py`)
   - Example client demonstrating how to connect and use the gRPC service
   - Measures RTT through multiple Ping/Pong round trips

### UNIX Domain Socket

**Socket Path**: `/tmp/tt_telemetry.sock` (defined as `GRPC_TELEMETRY_SOCKET_PATH` in the header)

**Key Points**:
- The socket is created when the server starts
- Permissions are set to `0666` (read/write for all users)
- The socket file is cleaned up on server shutdown
- Stale socket files are removed at startup

**Connection Format**: `unix:///tmp/tt_telemetry.sock` (note the triple slash for absolute path)

## When the Server Runs

The gRPC server is instantiated **only in collector mode** (not aggregator mode):

```cpp
// From main.cpp
if (!aggregator_mode) {
    grpc_server = std::make_shared<GrpcTelemetryServer>();
    grpc_server->start();
    subscribers.push_back(grpc_server);
}
```

It starts right before the telemetry collector thread begins, ensuring it's ready to receive telemetry updates.

## Current Implementation: Ping/Pong Service

### Service Definition

```protobuf
service TelemetryService {
  rpc Ping(PingRequest) returns (PingResponse) {}
}

message PingRequest {
  int64 timestamp = 1;  // Client timestamp (milliseconds since epoch)
}

message PingResponse {
  int64 timestamp = 1;  // Echo of the client's timestamp
}
```

### Purpose

The Ping/Pong service allows clients to:
- Verify connectivity to the telemetry server
- Measure round-trip time (RTT) for gRPC calls over UNIX sockets
- Test that the service is responsive

### How It Works

1. Client sends a `PingRequest` with current timestamp
2. Server immediately echoes the timestamp back in `PingResponse`
3. Client calculates RTT = (current_time - sent_timestamp)

## Building

### Dependencies

The build system requires system-installed gRPC and Protobuf packages.

**Ubuntu 22.04 / Debian-based systems:**
```bash
sudo apt-get update
sudo apt-get install -y \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc
```

**Ubuntu 24.04+ (with CMake config):**
```bash
sudo apt-get update
sudo apt-get install -y \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler-grpc
```

**Other requirements:**
- CMake 3.22 or later

### CMake Configuration

The build system uses two custom CMake modules:

1. **`cmake/FindSystemgRPC.cmake`**
   - Finds gRPC and Protobuf packages
   - First tries modern CMake config files (Ubuntu 24.04+)
   - Falls back to pkg-config for older systems (Ubuntu 22.04)

2. **`cmake/BuildTelemetryProto.cmake`**
   - Builds the proto library from `include/server/telemetry_service.proto`
   - Generates C++ code from the `.proto` file using protoc
   - Creates a `telemetry_grpc_proto` library with the generated code
   - Links the main server against gRPC libraries

See `cmake/README.md` for details on these modules.

## Using the Service

### From C++

```cpp
#include <grpcpp/grpcpp.h>
#include "telemetry_service.grpc.pb.h"

// Connect to server
auto channel = grpc::CreateChannel(
    "unix:///tmp/tt_telemetry.sock",
    grpc::InsecureChannelCredentials()
);
auto stub = tt::telemetry::TelemetryService::NewStub(channel);

// Make a Ping call
tt::telemetry::PingRequest request;
request.set_timestamp(get_current_timestamp_ms());

tt::telemetry::PingResponse response;
grpc::ClientContext context;

grpc::Status status = stub->Ping(&context, request, &response);
if (status.ok()) {
    std::cout << "Pong received! Timestamp: " << response.timestamp() << std::endl;
}
```

### From Python

A Python client is provided in `tt_telemetry/scripts/telemetry_client.py`. See `scripts/README.md` for setup instructions.

Quick start:
```bash
cd tt_telemetry/scripts
pip install -r requirements.txt
python3 -m grpc_tools.protoc -I../server --python_out=. --grpc_python_out=. ../server/telemetry_service.proto
python3 telemetry_client.py
```

Example usage in code:
```python
from telemetry_client import TelemetryClient

client = TelemetryClient("/tmp/tt_telemetry.sock")
success, timestamp, rtt_ms = client.ping()
if success:
    print(f"RTT: {rtt_ms} ms")
client.close()
```

## Why UNIX Domain Sockets?

Compared to TCP sockets:
- **Faster**: No network stack overhead
- **Simpler**: No port conflicts or firewall issues
- **Secure**: Controlled by filesystem permissions
- **Local only**: Perfect for IPC on the same machine

## Future Enhancements

The current Ping/Pong service is a foundation. Future additions could include:

1. **GetMetric RPC**: Request current value of a specific metric
2. **SubscribeToMetric RPC**: Stream updates for a metric (server streaming)
3. **GetAllMetrics RPC**: Bulk retrieval of telemetry data
4. **QueryMetrics RPC**: Query with filters/time ranges

All of these can leverage the existing `TelemetrySubscriber` interface to access the accumulated telemetry state.

## Implementation Details

### Thread Safety

- The `GrpcTelemetryServer` derives from `TelemetrySubscriber`
- It receives telemetry updates through `on_telemetry_updated()`
- Access to `telemetry_state_` is protected by `state_mutex_`
- gRPC handles incoming requests on its own thread pool

### Lifecycle

1. **Construction**: Server object created, subscriber thread started
2. **Start**: gRPC server starts listening on UNIX socket
3. **Operation**: Processes RPCs and telemetry updates concurrently
4. **Shutdown**: Server stops gracefully, socket file removed

### Error Handling

- Socket file conflicts are resolved by `unlink()` at startup
- RPC timeouts can be set by clients
- Server logs errors through the TT logging system
- Graceful shutdown with 5-second deadline for pending RPCs

## Debugging

**Check if socket exists**:
```bash
ls -l /tmp/tt_telemetry.sock
# Should show: srwxrw-rw- ... (s = socket)
```

**Check server logs**:
```bash
# Server startup should log:
# "gRPC telemetry server listening on UNIX socket: /tmp/tt_telemetry.sock"
```

**Test connectivity**:
```bash
# Use the example client or grpcurl:
grpcurl -plaintext -unix /tmp/tt_telemetry.sock tt.telemetry.TelemetryService/Ping
```
