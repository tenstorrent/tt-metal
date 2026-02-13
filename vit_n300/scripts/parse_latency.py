#!/usr/bin/env python3
"""
Parse watcher log ring buffer data to extract per-core max semaphore wait latencies.

The instrumented matmul kernels push latency data to the ring buffer with format:
    0xSSNNNNNN  where SS = site ID, NNNNNN = max cycles (24-bit)

Site IDs:
    0x10: BRISC - in1 mcast semaphore wait (waiting for in1 tiles from mcast sender)
    0x20: BRISC - bias mcast semaphore wait (waiting for bias tiles from mcast sender)
    0x30: BRISC - output CB wait, OUT_SHARDED final (waiting for compute to finish all tiles)
    0x40: BRISC - output CB wait, non-sharded per-subblock
    0x50: Compute - cb_wait_front(in0) (waiting for in0 tiles from dataflow)
    0x60: Compute - cb_wait_front(in1) (waiting for in1 tiles from dataflow)
    0x70: Compute - cb_wait_front(partials) in FUSE_BIAS path
    0x80: Compute - cb_reserve_back(out) in FUSE_BIAS path (back-pressure from writer)
    0x90: NCRISC - sender waiting for all receivers ready
    0xA0: NCRISC - receiver waiting for mcast data

Wormhole runs at 1GHz, so 1 cycle = 1ns.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

SITE_NAMES = {
    0x10: "BRISC: in1_mcast_sem_wait",
    0x20: "BRISC: bias_mcast_sem_wait",
    0x30: "BRISC: out_sharded_wait",
    0x40: "BRISC: out_cb_wait",
    0x50: "Compute: in0_cb_wait",
    0x60: "Compute: in1_cb_wait",
    0x70: "Compute: partials_cb_wait (bias)",
    0x80: "Compute: out_reserve (bias)",
    0x90: "NCRISC: sender_sem_wait",
    0xA0: "NCRISC: receiver_mcast_wait",
}


def parse_watcher_log(log_path: str):
    """Parse watcher log and extract ring buffer latency data per core.

    Reads ALL watcher dumps (not just the last) and takes the MAXIMUM
    latency seen for each core/site across all dumps. This is critical
    because the ring buffer wraps every ~3 kernel invocations, but the
    watcher polls ~100 times during a test run.
    """
    text = Path(log_path).read_text()

    core_pattern = re.compile(
        r"Device\s+(\d+)\s+\w+\s+core\(x=\s*(\d+),y=\s*(\d+)\)\s+virtual\(x=\s*(\d+),y=\s*(\d+)\)"
    )
    ring_buf_pattern = re.compile(r"debug_ring_buffer=\s*\n\s*\[(.*?)\]", re.DOTALL)

    # Accumulate MAX across all dumps for each (device, core_x, core_y)
    # Key: (device, core_x, core_y, virt_x, virt_y) -> {site_id: max_cycles}
    global_max = {}
    num_dumps = 0

    # Find all core headers
    core_matches = list(core_pattern.finditer(text))

    for i, core_match in enumerate(core_matches):
        device = int(core_match.group(1))
        core_x = int(core_match.group(2))
        core_y = int(core_match.group(3))
        virt_x = int(core_match.group(4))
        virt_y = int(core_match.group(5))

        # Search for ring buffer in the text between this core and the next
        start = core_match.end()
        end = core_matches[i + 1].start() if i + 1 < len(core_matches) else len(text)
        block = text[start:end]

        ring_match = ring_buf_pattern.search(block)
        if not ring_match:
            continue

        # Parse hex values
        raw = ring_match.group(1)
        hex_values = re.findall(r"0x([0-9a-fA-F]+)", raw)

        key = (device, core_x, core_y, virt_x, virt_y)
        if key not in global_max:
            global_max[key] = {}

        for hex_str in hex_values:
            val = int(hex_str, 16)
            site_id = (val >> 24) & 0xFF
            cycles = val & 0x00FFFFFF

            if site_id in SITE_NAMES:
                # Take MAX across all dumps for this core/site
                if site_id not in global_max[key] or cycles > global_max[key][site_id]:
                    global_max[key][site_id] = cycles

    # Count unique dumps
    dump_pattern = re.compile(r"Dump #(\d+)")
    num_dumps = len(dump_pattern.findall(text))

    # Convert to result list
    results = []
    for (device, core_x, core_y, virt_x, virt_y), latencies in global_max.items():
        if latencies:
            results.append((device, core_x, core_y, virt_x, virt_y, latencies))

    print(f"Watcher dumps found: {num_dumps}")
    print(f"(Global max taken across all dumps for each core/site)")

    return results


def analyze_and_print(results):
    """Analyze and print latency data."""
    if not results:
        print("No latency data found in watcher log.")
        return

    print(f"\n{'=' * 90}")
    print(f"SEMAPHORE WAIT LATENCY ANALYSIS")
    print(f"{'=' * 90}")
    print(f"Total cores with data: {len(results)}")
    print(f"Clock: 1 GHz (1 cycle = 1 ns)")
    print()

    # Aggregate by site
    site_data = defaultdict(list)  # site_id -> [(device, core_x, core_y, virt_x, virt_y, cycles), ...]

    for device, core_x, core_y, virt_x, virt_y, latencies in results:
        for site_id, cycles in latencies.items():
            site_data[site_id].append((device, core_x, core_y, virt_x, virt_y, cycles))

    # Print per-site summary
    for site_id in sorted(site_data.keys()):
        entries = site_data[site_id]
        site_name = SITE_NAMES.get(site_id, f"Unknown 0x{site_id:02X}")
        cycles_list = [e[5] for e in entries]

        if not cycles_list or max(cycles_list) == 0:
            continue

        avg = sum(cycles_list) / len(cycles_list)
        max_val = max(cycles_list)
        min_val = min(cycles_list)
        nonzero = [c for c in cycles_list if c > 0]

        print(f"--- Site 0x{site_id:02X}: {site_name} ---")
        print(f"  Cores reporting: {len(entries)}, non-zero: {len(nonzero)}")
        print(f"  Min: {min_val:>10,} cycles ({min_val/1000:.1f} us)")
        print(f"  Avg: {avg:>10,.0f} cycles ({avg/1000:.1f} us)")
        print(f"  Max: {max_val:>10,} cycles ({max_val/1000:.1f} us)")
        print(f"  Spread (max/avg): {max_val/avg:.1f}x" if avg > 0 else "")

        # Top 5 slowest cores
        entries_sorted = sorted(entries, key=lambda e: e[5], reverse=True)
        print(f"  Top 5 slowest cores:")
        for dev, cx, cy, vx, vy, cyc in entries_sorted[:5]:
            print(f"    dev={dev} core=({cx},{cy}) virtual=({vx},{vy}): {cyc:>10,} cycles ({cyc/1000:.1f} us)")
        print()

    # Print full heatmap for the highest-variance site
    if site_data:
        # Find site with highest max/avg ratio (most variance = most interesting)
        best_site = None
        best_ratio = 0
        for site_id, entries in site_data.items():
            cycles_list = [e[5] for e in entries]
            if max(cycles_list) == 0:
                continue
            avg = sum(cycles_list) / len(cycles_list)
            if avg > 0:
                ratio = max(cycles_list) / avg
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_site = site_id

        if best_site is not None:
            site_name = SITE_NAMES.get(best_site, f"Unknown 0x{best_site:02X}")
            print(f"{'=' * 90}")
            print(f"HIGHEST VARIANCE SITE: 0x{best_site:02X} ({site_name})")
            print(f"Max/Avg ratio: {best_ratio:.1f}x")
            print(f"{'=' * 90}")

            for dev_id in sorted(set(e[0] for e in site_data[best_site])):
                dev_entries = [(cx, cy, vx, vy, cyc) for d, cx, cy, vx, vy, cyc in site_data[best_site] if d == dev_id]
                if not dev_entries:
                    continue

                max_cx = max(e[0] for e in dev_entries)
                max_cy = max(e[1] for e in dev_entries)

                print(f"\nDevice {dev_id} - Heatmap (core coords, values in us):")
                # Build grid
                grid = {}
                for cx, cy, vx, vy, cyc in dev_entries:
                    grid[(cx, cy)] = cyc

                # Print header
                header = "     " + "".join(f"  y={y:<5d}" for y in range(max_cy + 1))
                print(header)
                for x in range(max_cx + 1):
                    row = f"x={x:<2d} "
                    for y in range(max_cy + 1):
                        cyc = grid.get((x, y))
                        if cyc is not None:
                            us = cyc / 1000
                            if us >= 100:
                                row += f" {us:>6.0f} "
                            elif us >= 1:
                                row += f" {us:>6.1f} "
                            else:
                                row += f" {us:>6.2f} "
                        else:
                            row += "    --  "
                    print(row)
            print()


def main():
    if len(sys.argv) < 2:
        log_path = "generated/watcher/watcher.log"
    else:
        log_path = sys.argv[1]

    print(f"Parsing: {log_path}")
    results = parse_watcher_log(log_path)
    analyze_and_print(results)


if __name__ == "__main__":
    main()
