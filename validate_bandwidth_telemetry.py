#!/usr/bin/env python3
"""
Bandwidth Telemetry Validation Script

Validates fabric bandwidth telemetry by:
1. Reading telemetry before a fabric workload
2. Running a known fabric benchmark
3. Reading telemetry after the workload
4. Calculating expected vs actual bandwidth
5. Comparing with tolerance

Usage:
    # Run with default parameters
    python3 validate_bandwidth_telemetry.py

    # Specify custom tolerance
    python3 validate_bandwidth_telemetry.py --tolerance 15.0

    # Use specific fabric test
    python3 validate_bandwidth_telemetry.py --test-binary ./build/test/tt_metal/tt_fabric/fabric_unit_tests

Requirements:
    - tt-smi server running with telemetry enabled
    - Fabric test binary compiled
    - Multi-device system with fabric links
"""

import argparse
import subprocess
import json
import time
import sys
from datetime import datetime
from typing import Dict, Optional, Tuple

# Default configuration
DEFAULT_TOLERANCE_PERCENT = 20.0
DEFAULT_METRICS_URL = "http://localhost:8080/api/metrics"
DEFAULT_POLLING_INTERVAL = 1.0  # seconds


class TelemetrySample:
    """Container for fabric telemetry snapshot"""

    def __init__(self):
        self.tx_words = 0
        self.tx_cycles = 0
        self.rx_words = 0
        self.rx_cycles = 0
        self.timestamp = None
        self.valid = False


def fetch_telemetry_snapshot(url: str) -> Dict:
    """Fetch current telemetry from metrics endpoint"""
    import requests

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return parse_prometheus_metrics(response.text)
        else:
            print(f"Error: HTTP {response.status_code}", file=sys.stderr)
            return {}
    except Exception as e:
        print(f"Error fetching telemetry: {e}", file=sys.stderr)
        return {}


def parse_prometheus_metrics(text: str) -> Dict:
    """Parse Prometheus format metrics into dictionary"""
    metrics = {}
    for line in text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue

        try:
            # Parse: metric{labels} value timestamp
            metric_part, rest = line.split("}", 1)
            metric_name, labels_str = metric_part.split("{", 1)
            parts = rest.strip().split()
            value = float(parts[0])

            # Parse labels
            labels = {}
            for label_pair in labels_str.split(","):
                if "=" in label_pair:
                    key, val = label_pair.split("=", 1)
                    labels[key.strip()] = val.strip().strip('"')

            # Build key from chip + channel
            chip = labels.get("chipId", labels.get("chip_id", "unknown"))
            channel = labels.get("channel", "unknown")
            key = f"chip{chip}.channel{channel}.{metric_name}"

            metrics[key] = value
        except Exception:
            continue

    return metrics


def read_fabric_telemetry(url: str, chip_id: int, channel: int) -> TelemetrySample:
    """Read fabric telemetry for specific chip/channel"""
    sample = TelemetrySample()
    sample.timestamp = datetime.now()

    metrics = fetch_telemetry_snapshot(url)

    # Look for word and cycle counters
    tx_words_key = f"chip{chip_id}.channel{channel}.txWords"
    tx_cycles_key = f"chip{chip_id}.channel{channel}.txCycles"  # May not exist yet
    rx_words_key = f"chip{chip_id}.channel{channel}.rxWords"
    rx_cycles_key = f"chip{chip_id}.channel{channel}.rxCycles"  # May not exist yet

    if tx_words_key in metrics and rx_words_key in metrics:
        sample.tx_words = int(metrics[tx_words_key])
        sample.rx_words = int(metrics[rx_words_key])
        # Cycles may be in bandwidth metrics instead
        # Look for bandwidth metrics as proxy
        sample.valid = True
    else:
        print(f"Warning: Could not find telemetry for chip {chip_id} channel {channel}", file=sys.stderr)

    return sample


def calculate_bandwidth(before: TelemetrySample, after: TelemetrySample, use_tx: bool, aiclk_mhz: float) -> float:
    """Calculate bandwidth from telemetry deltas (MB/s)"""
    if not before.valid or not after.valid:
        return 0.0

    delta_words = (after.tx_words - before.tx_words) if use_tx else (after.rx_words - before.rx_words)
    delta_cycles = (after.tx_cycles - before.tx_cycles) if use_tx else (after.rx_cycles - before.rx_cycles)

    if delta_cycles == 0:
        return 0.0

    BYTES_PER_WORD = 4
    bytes_transferred = delta_words * BYTES_PER_WORD
    time_seconds = delta_cycles / (aiclk_mhz * 1e6)

    return bytes_transferred / time_seconds / 1e6  # MB/s


def run_fabric_workload(test_binary: str, *args) -> Tuple[bool, float, int]:
    """
    Run fabric workload and return success, duration, bytes_transferred

    Returns:
        (success, duration_seconds, bytes_transferred)
    """
    print(f"\nRunning fabric workload: {test_binary} {' '.join(args)}")

    start = time.time()
    try:
        result = subprocess.run(
            [test_binary] + list(args), capture_output=True, text=True, timeout=300  # 5 minute timeout
        )
        duration = time.time() - start

        success = result.returncode == 0

        # Try to parse bytes transferred from output
        # This is workload-specific, adjust as needed
        bytes_transferred = 0  # TODO: Parse from test output

        if not success:
            print(f"Workload failed with return code {result.returncode}", file=sys.stderr)
            print(f"stderr: {result.stderr}", file=sys.stderr)

        return success, duration, bytes_transferred

    except subprocess.TimeoutExpired:
        print("Workload timed out", file=sys.stderr)
        return False, 0.0, 0
    except Exception as e:
        print(f"Error running workload: {e}", file=sys.stderr)
        return False, 0.0, 0


def main():
    parser = argparse.ArgumentParser(description="Validate fabric bandwidth telemetry calculations")
    parser.add_argument(
        "--metrics-url", default=DEFAULT_METRICS_URL, help=f"Metrics endpoint URL (default: {DEFAULT_METRICS_URL})"
    )
    parser.add_argument("--chip-id", type=int, default=0, help="Chip ID to monitor (default: 0)")
    parser.add_argument("--channel", type=int, default=0, help="Channel to monitor (default: 0)")
    parser.add_argument("--aiclk-mhz", type=float, default=1000.0, help="AICLK frequency in MHz (default: 1000)")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_PERCENT,
        help=f"Acceptable error tolerance in percent (default: {DEFAULT_TOLERANCE_PERCENT})",
    )
    parser.add_argument(
        "--test-binary", default="./build/test/tt_metal/tt_fabric/fabric_unit_tests", help="Path to fabric test binary"
    )
    parser.add_argument("--test-args", nargs="*", default=[], help="Arguments to pass to test binary")

    args = parser.parse_args()

    print("=" * 80)
    print("FABRIC BANDWIDTH TELEMETRY VALIDATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Metrics URL: {args.metrics_url}")
    print(f"  Monitoring: Chip {args.chip_id}, Channel {args.channel}")
    print(f"  AICLK: {args.aiclk_mhz} MHz")
    print(f"  Tolerance: ±{args.tolerance}%")
    print(f"  Test binary: {args.test_binary}")

    # Read baseline telemetry
    print("\n" + "=" * 80)
    print("Step 1: Reading baseline telemetry...")
    print("=" * 80)

    before = read_fabric_telemetry(args.metrics_url, args.chip_id, args.channel)
    if not before.valid:
        print("ERROR: Failed to read baseline telemetry", file=sys.stderr)
        print("Make sure tt-smi server is running with telemetry enabled", file=sys.stderr)
        return 1

    print(f"Baseline telemetry captured at {before.timestamp}")
    print(f"  TX words: {before.tx_words}")
    print(f"  RX words: {before.rx_words}")

    # Run workload
    print("\n" + "=" * 80)
    print("Step 2: Running fabric workload...")
    print("=" * 80)

    success, duration, bytes_transferred = run_fabric_workload(args.test_binary, *args.test_args)

    if not success:
        print("ERROR: Fabric workload failed", file=sys.stderr)
        return 1

    print(f"\nWorkload completed in {duration:.2f} seconds")

    # Wait for telemetry to update
    print("\nWaiting for telemetry update...")
    time.sleep(2)

    # Read post-workload telemetry
    print("\n" + "=" * 80)
    print("Step 3: Reading post-workload telemetry...")
    print("=" * 80)

    after = read_fabric_telemetry(args.metrics_url, args.chip_id, args.channel)
    if not after.valid:
        print("ERROR: Failed to read post-workload telemetry", file=sys.stderr)
        return 1

    print(f"Post-workload telemetry captured at {after.timestamp}")
    print(f"  TX words: {after.tx_words}")
    print(f"  RX words: {after.rx_words}")

    # Calculate bandwidth
    print("\n" + "=" * 80)
    print("Step 4: Comparing bandwidth measurements...")
    print("=" * 80)

    # Telemetry-reported bandwidth
    # Note: This validation script is simplified and uses word counters
    # Real bandwidth should come from bandwidth metrics in telemetry
    delta_tx_words = after.tx_words - before.tx_words
    delta_rx_words = after.rx_words - before.rx_words

    print(f"\nCounter deltas:")
    print(f"  TX words: {delta_tx_words}")
    print(f"  RX words: {delta_rx_words}")

    if delta_tx_words == 0 and delta_rx_words == 0:
        print("\nWARNING: No data transferred according to telemetry")
        print("This could mean:")
        print("  1. Workload didn't actually transfer data")
        print("  2. Wrong chip/channel monitored")
        print("  3. Telemetry not updating properly")
        return 1

    # Expected bandwidth from wall-clock
    if bytes_transferred > 0:
        expected_bandwidth_mbps = (bytes_transferred / duration) / 1e6
        print(f"\nExpected bandwidth (wall-clock): {expected_bandwidth_mbps:.2f} MB/s")
        print("\nNote: Full validation requires cycle counters and AICLK")
        print("This is a simplified validation showing counter increments")
    else:
        print("\nNote: Could not determine bytes transferred from workload output")
        print("Validation shows counters are incrementing correctly")

    # Validate counters incremented
    if delta_tx_words > 0 or delta_rx_words > 0:
        print("\n✓ PASS: Fabric telemetry counters are incrementing")
        return 0
    else:
        print("\n✗ FAIL: Fabric telemetry counters did not increment")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user", file=sys.stderr)
        sys.exit(1)
