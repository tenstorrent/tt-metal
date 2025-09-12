#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
WebSocket Test Client for TT Telemetry Server

This script tests the WebSocket functionality of the tt_telemetry_server.
It connects to the WebSocket server, sends test messages, and displays
received telemetry data.

Usage:
    python3 websocket_test_client.py [--host HOST] [--port PORT] [--verbose]

Requirements:
    pip install websockets asyncio
"""

import asyncio
import json
import sys
import argparse
import time
from datetime import datetime
from typing import Optional

try:
    import websockets
except ImportError:
    print("Error: websockets library not found.")
    print("Install with: pip install websockets")
    sys.exit(1)


class TelemetryWebSocketClient:
    def __init__(self, host: str = "localhost", port: int = 8081, verbose: bool = False, path: str = ""):
        self.host = host
        self.port = port
        self.path = path
        self.verbose = verbose
        self.uri = f"ws://{host}:{port}{path}"
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.running = False

        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.hello_received = False
        self.telemetry_count = 0
        self.last_telemetry_time = None

    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if level == "ERROR":
            print(f"[{timestamp}] ❌ {message}")
        elif level == "SUCCESS":
            print(f"[{timestamp}] ✅ {message}")
        elif level == "RECEIVED":
            print(f"[{timestamp}] ⬇️  {message}")
        elif level == "SENT":
            print(f"[{timestamp}] ⬆️  {message}")
        elif self.verbose or level == "INFO":
            print(f"[{timestamp}] ℹ️  {message}")

    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            self.log(f"Connecting to {self.uri}...")
            # Simple connection without extra parameters for maximum compatibility
            self.websocket = await websockets.connect(self.uri)
            self.running = True
            self.log("Connected successfully!", "SUCCESS")
            return True
        except websockets.exceptions.InvalidURI as e:
            self.log(f"Invalid URI: {e}", "ERROR")
            return False
        except websockets.exceptions.ConnectionClosed as e:
            self.log(f"Connection closed during handshake: {e}", "ERROR")
            return False
        except OSError as e:
            self.log(f"Network error: {e}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Failed to connect: {e}", "ERROR")
            return False

    async def disconnect(self):
        """Disconnect from the WebSocket server"""
        if self.websocket:
            self.running = False
            await self.websocket.close()
            self.log("Disconnected")

    async def send_message(self, message: str):
        """Send a message to the server"""
        if not self.websocket:
            self.log("Not connected to server", "ERROR")
            return False

        try:
            await self.websocket.send(message)
            self.messages_sent += 1
            self.log(f"Sent: {message}", "SENT")
            return True
        except Exception as e:
            self.log(f"Failed to send message: {e}", "ERROR")
            return False

    async def receive_messages(self):
        """Listen for messages from the server"""
        if not self.websocket:
            return

        try:
            async for message in self.websocket:
                self.messages_received += 1

                # Handle different types of messages
                if message == "hello":
                    self.hello_received = True
                    self.log("Received hello message from server", "SUCCESS")
                else:
                    # Try to parse as JSON (telemetry data)
                    try:
                        data = json.loads(message)
                        self.telemetry_count += 1
                        self.last_telemetry_time = datetime.now()
                        self.log(f"Received telemetry data ({len(data)} fields)", "RECEIVED")

                        if self.verbose:
                            # Pretty print telemetry data
                            print("📊 Telemetry Data:")
                            for key, value in data.items():
                                if isinstance(value, list) and len(value) > 5:
                                    print(f"  {key}: [{len(value)} items]")
                                else:
                                    print(f"  {key}: {value}")
                            print()
                    except json.JSONDecodeError:
                        # Regular text message (echo)
                        self.log(f"Received: {message}", "RECEIVED")

        except websockets.exceptions.ConnectionClosed:
            self.log("Connection closed by server")
            self.running = False
        except Exception as e:
            self.log(f"Error receiving messages: {e}", "ERROR")
            self.running = False

    async def send_test_messages(self):
        """Send periodic test messages"""
        test_messages = ["ping", "test message 1", "hello from client", "status check", "test message 2"]

        for i, message in enumerate(test_messages):
            if not self.running:
                break

            await asyncio.sleep(2)  # Wait 2 seconds between messages
            await self.send_message(message)

    def print_statistics(self):
        """Print connection statistics"""
        print("\n" + "=" * 50)
        print("📈 CONNECTION STATISTICS")
        print("=" * 50)
        print(f"Messages sent:     {self.messages_sent}")
        print(f"Messages received: {self.messages_received}")
        print(f"Hello received:    {'✅ Yes' if self.hello_received else '❌ No'}")
        print(f"Telemetry count:   {self.telemetry_count}")
        if self.last_telemetry_time:
            print(f"Last telemetry:    {self.last_telemetry_time.strftime('%H:%M:%S')}")
        else:
            print(f"Last telemetry:    Never")
        print("=" * 50)

    async def run_test(self, duration: int = 30):
        """Run the complete WebSocket test"""
        print("🚀 TT Telemetry WebSocket Test Client")
        print(f"Target: {self.uri}")
        print(f"Test duration: {duration} seconds")
        print("-" * 50)

        # Connect to server
        if not await self.connect():
            return False

        try:
            # Start listening for messages
            receive_task = asyncio.create_task(self.receive_messages())

            # Wait a moment for hello message
            await asyncio.sleep(1)

            # Send test messages
            test_task = asyncio.create_task(self.send_test_messages())

            # Wait for specified duration or until connection closes
            start_time = time.time()
            while self.running and (time.time() - start_time) < duration:
                await asyncio.sleep(1)

            # Cancel tasks
            receive_task.cancel()
            test_task.cancel()

            try:
                await receive_task
            except asyncio.CancelledError:
                pass

            try:
                await test_task
            except asyncio.CancelledError:
                pass

        finally:
            await self.disconnect()
            self.print_statistics()

        return True


async def test_multiple_endpoints(host: str, port: int, verbose: bool = False):
    """Test multiple possible WebSocket endpoints"""
    endpoints_to_test = [
        "",  # ws://localhost:8081
        "/",  # ws://localhost:8081/
        "/ws",  # ws://localhost:8081/ws
        "/websocket",  # ws://localhost:8081/websocket
        "/api/ws",  # ws://localhost:8081/api/ws
        "/telemetry",  # ws://localhost:8081/telemetry
    ]

    print("🔍 Testing multiple WebSocket endpoints...")
    print("-" * 50)

    for path in endpoints_to_test:
        uri = f"ws://{host}:{port}{path}"
        print(f"Testing: {uri}")

        try:
            # Quick connection test (remove timeout for compatibility)
            websocket = await asyncio.wait_for(websockets.connect(uri), timeout=3)
            print(f"✅ SUCCESS: {uri} - Connection established!")
            await websocket.close()
            return path  # Return the working path
        except asyncio.TimeoutError:
            print(f"❌ FAILED: {uri} - Connection timeout")
        except Exception as e:
            print(f"❌ FAILED: {uri} - {type(e).__name__}: {e}")

    print("\n❌ No working endpoints found")
    return None


async def check_server_reachability(host: str, port: int):
    """Check if the server port is reachable at all"""
    import socket
    import aiohttp

    print(f"🔌 Checking if port {port} is reachable on {host}...")

    try:
        # Try to connect with raw socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            print(f"✅ Port {port} is open and accepting connections")

            # Try to make an HTTP request to test if uWebSockets is responding
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://{host}:{port}/test", timeout=3) as response:
                        text = await response.text()
                        print(f"✅ HTTP test endpoint works: {text}")
                        return True
            except ImportError:
                print("ℹ️  aiohttp not available, skipping HTTP test")
                return True
            except Exception as e:
                print(f"⚠️  Port is open but HTTP test failed: {e}")
                print("   This might indicate a WebSocket-only server or configuration issue")
                return True
        else:
            print(f"❌ Port {port} is not accepting connections (error code: {result})")
            return False
    except Exception as e:
        print(f"❌ Cannot reach {host}:{port} - {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="WebSocket test client for TT Telemetry")
    parser.add_argument("--host", default="localhost", help="WebSocket server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8081, help="WebSocket server port (default: 8081)")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds (default: 30)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test-endpoints", action="store_true", help="Test multiple endpoints to find working one")
    parser.add_argument("--path", default="", help="WebSocket path (default: empty)")

    args = parser.parse_args()

    # First, check if the server port is reachable
    if not await check_server_reachability(args.host, args.port):
        print("\n💡 Troubleshooting:")
        print("1. Make sure the telemetry server is running:")
        print("   ./tt_telemetry_server --enable-websocket --ws-port 8081 --mock-telemetry")
        print("2. Check if the server shows 'WebSocket server listening on port 8081'")
        print("3. Verify no firewall is blocking the port")
        return 1

    # If requested, test multiple endpoints
    working_path = args.path
    if args.test_endpoints:
        working_path = await test_multiple_endpoints(args.host, args.port, args.verbose)
        if working_path is None:
            print("\n💡 The port is reachable but no WebSocket endpoints work.")
            print("This suggests the WebSocket server is not properly configured.")
            return 1
        else:
            print(f"\n✅ Found working endpoint: ws://{args.host}:{args.port}{working_path}")

    client = TelemetryWebSocketClient(args.host, args.port, args.verbose, working_path)

    try:
        success = await client.run_test(args.duration)

        # Provide feedback on test results
        print("\n🎯 TEST RESULTS:")
        if not success:
            print("❌ Failed to connect to WebSocket server")
            print("\n💡 Troubleshooting:")
            print("1. Make sure the telemetry server is running:")
            print("   ./tt_telemetry_server --enable-websocket --ws-port 8081 --mock-telemetry")
            print("2. Check if the port is correct (default: 8081)")
            print("3. Verify the server shows 'WebSocket server listening on port 8081'")
            return 1

        elif not client.hello_received:
            print("⚠️  Connected but didn't receive hello message")
            print("   This might indicate a server-side issue")
            return 1

        elif client.telemetry_count == 0:
            print("⚠️  Connected and received hello, but no telemetry data")
            print("   Make sure to run server with --mock-telemetry flag")
            return 1

        else:
            print("✅ All tests passed!")
            print(f"   - Connected successfully")
            print(f"   - Received hello message")
            print(f"   - Received {client.telemetry_count} telemetry updates")
            print(f"   - Bidirectional communication working")
            return 0

    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
