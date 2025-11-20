# TT Telemetry Python Client

Python client for interacting with the TT Telemetry gRPC service over UNIX domain sockets.

## Setup

### 1. Install Dependencies

```bash
cd tt_telemetry/scripts
pip install -r requirements.txt
```

### 2. Generate Python gRPC Code

The Python client requires generated code from the `.proto` file. Run this command from the `scripts` directory:

```bash
python3 -m grpc_tools.protoc -I../include/server --python_out=. --grpc_python_out=. ../include/server/telemetry_service.proto
```

This will generate two files:
- `telemetry_service_pb2.py` - Protocol Buffer message definitions
- `telemetry_service_pb2_grpc.py` - gRPC service stub

### 3. Ensure Server is Running

The telemetry server must be running in **collector mode** for the gRPC service to be available:

```bash
# From tt_telemetry directory
./run_telemetry_server.sh --fsd <path-to-fsd-file>
```

The server will create a UNIX socket at `/tmp/tt_telemetry.sock`.

## Usage

### Basic Ping Test

```bash
python3 telemetry_client.py
```

This will:
- Connect to `/tmp/tt_telemetry.sock`
- Send 10 ping requests
- Measure and report RTT statistics

### Custom Socket Path

```bash
python3 telemetry_client.py /path/to/custom.sock
```

### Custom Number of Pings

```bash
python3 telemetry_client.py -n 100
```

### Custom Delay Between Pings

```bash
python3 telemetry_client.py -d 0.5  # 500ms delay
```

### Full Example

```bash
python3 telemetry_client.py /tmp/tt_telemetry.sock -n 20 -d 0.05
```

## Example Output

```
Connecting to telemetry server at: /tmp/tt_telemetry.sock

Sending 10 pings...
  Ping 1: 0 ms
  Ping 2: 0 ms
  Ping 3: 1 ms
  Ping 4: 0 ms
  Ping 5: 0 ms
  Ping 6: 1 ms
  Ping 7: 0 ms
  Ping 8: 0 ms
  Ping 9: 1 ms
  Ping 10: 0 ms

Results:
  Successful: 10/10
  Average RTT: 0.30 ms
  Min RTT: 0 ms
  Max RTT: 1 ms
  Std Dev: 0.46 ms
```

## Using the Client in Your Code

```python
from telemetry_client import TelemetryClient

# Connect to the server
client = TelemetryClient("/tmp/tt_telemetry.sock")

# Send a ping
success, timestamp, rtt_ms = client.ping()
if success:
    print(f"Ping successful! RTT: {rtt_ms} ms")

# Close the connection
client.close()
```

## Troubleshooting

### ImportError: No module named 'telemetry_service_pb2'

**Problem**: Generated Python code is missing.

**Solution**: Run the protoc command to generate the Python gRPC code (see step 2 above).

### grpc.RpcError: failed to connect to all addresses

**Problem**: Cannot connect to the UNIX socket.

**Possible causes**:
1. Server is not running - start the telemetry server in collector mode
2. Socket file doesn't exist - check if `/tmp/tt_telemetry.sock` exists
3. Permission denied - check socket file permissions with `ls -l /tmp/tt_telemetry.sock`

### Socket file exists but connection fails

**Problem**: Stale socket file from crashed server.

**Solution**:
```bash
rm /tmp/tt_telemetry.sock
# Restart the telemetry server
```

## Notes

- UNIX sockets only work on the same machine (no network access)
- The socket path must be absolute
- In Python, use `unix:/path` (not `unix:///path` like in C++)
- RTT measurements include Python overhead, so expect slightly higher values than C++
