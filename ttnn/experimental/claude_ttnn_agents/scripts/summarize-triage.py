#!/usr/bin/env python3
"""Summarize tt-triage output into a compact, agent-friendly format.

Parses the Rich table output from tt-triage and produces a structured summary
that groups cores by callstack pattern. This is critical for debugging hangs
where different cores may be stuck at different points (e.g., multicast sender
vs receiver patterns).

Usage: python3 summarize-triage.py /tmp/dev-test-triage.log
"""

import re
import sys
import os
from collections import defaultdict

# Dispatch infrastructure kernels — not user kernels, listed separately
DISPATCH_KERNELS = {"cq_prefetch", "cq_dispatch", "cq_dispatch_subordinate"}

# Firmware source files — frames in these are startup/idle, not user code
FIRMWARE_SOURCES = {"brisck.cc", "ncrisck.cc", "ncrisc.cc", "trisc.cc"}


def parse_sections(content):
    """Split triage output into named sections."""
    sections = {}
    current_section = None
    current_lines = []
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.endswith(".py:") and " " not in stripped:
            if current_section:
                sections[current_section] = "\n".join(current_lines)
            current_section = stripped.rstrip(":").removesuffix(".py")
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)
    if current_section:
        sections[current_section] = "\n".join(current_lines)
    return sections


def parse_checks(sections):
    """Extract pass/fail status from check sections."""
    checks = {}
    for name in [
        "check_noc_locations",
        "dump_watcher_ringbuffer",
        "check_cb_inactive",
        "check_noc_status",
        "check_core_magic",
        "check_binary_integrity",
    ]:
        if name in sections:
            text = sections[name].strip().lower()
            short = name.replace("check_", "").replace("dump_", "")
            checks[short] = "PASS" if "pass" in text else "FAIL"
    return checks


def parse_op_info(section_text):
    """Extract operation name, shape, dtype from dump_running_operations."""
    if not section_text:
        return None

    info = {}
    for line in section_text.split("\n"):
        if "│" not in line:
            continue
        fields = [f.strip() for f in line.split("│")]
        fields = [f for f in fields if f]
        if not fields:
            continue

        if fields[0].isdigit():
            info["op_id"] = fields[0]
            info["op_name"] = fields[1]
            if len(fields) >= 8:
                try:
                    info["core_count"] = int(fields[7])
                except (ValueError, IndexError):
                    pass
            continue

        param = fields[0] if len(fields) >= 1 else ""
        if "logical_shape:" in param:
            info["shape"] = param.split(":", 1)[1].strip()
        elif "dtype:" in param:
            info["dtype"] = param.split(":", 1)[1].strip()
        elif "memory_layout:" in param:
            info["memory_layout"] = param.split(":", 1)[1].strip()
        elif "buffer_type:" in param:
            info["buffer_type"] = param.split(":", 1)[1].strip()

    return info if info else None


def normalize_frame(frame_text):
    """Normalize a callstack frame for grouping.

    Handles two formats:
      #0 0x00008374 in cb_wait_front () at ./path/file.h 479:29
      #1 kernel_main () at ./path/file.cpp 41:22
    """
    # Format 1: "#N 0xADDR in func_name () at ./path line:col"
    m = re.search(r"in (\S+)\s*\(\)\s*at\s+\./(.+?)\s+(\d+):\d+", frame_text)
    if m:
        return m.group(1) + "()", f"{m.group(2)}:{m.group(3)}"

    # Format 2: "#N func_name () at ./path line:col" (no address, no "in")
    m = re.search(r"#\d+\s+(\S+)\s*\(\)\s*at\s+\./(.+?)\s+(\d+):\d+", frame_text)
    if m:
        return m.group(1) + "()", f"{m.group(2)}:{m.group(3)}"

    # Format 3: "#N 0xADDR in func_name () at ./path line:col" — try looser match
    m = re.search(r"(\w+)\s*\(\)\s*at\s+\./(.+?)\s+(\d+):\d+", frame_text)
    if m:
        return m.group(1) + "()", f"{m.group(2)}:{m.group(3)}"

    return "???", "???"


def is_firmware_entry_frame(func, loc):
    """Check if a frame is a generic firmware entry point (noise in callstacks).

    Only filters _start() and main() in firmware files. Keeps meaningful firmware
    functions like wait_for_brisc_notification() since those indicate where the
    RISC-V is stuck in the startup sequence.
    """
    basename = os.path.basename(loc.rsplit(":", 1)[0]) if ":" in loc else ""
    return basename in FIRMWARE_SOURCES and func in ("_start()", "main()")


def is_dispatch_kernel(kernel_name):
    """Check if a kernel is dispatch infrastructure."""
    return any(dk in kernel_name for dk in DISPATCH_KERNELS)


# Internal implementation functions that are poll-loop details.
# When these appear as frame #0, they're just the inner loop of the API call
# at frame #1. We strip them so cores caught at different points in the same
# poll loop group together.
_POLL_INTERNALS = {"reg_read()", "reg_write()", "invalidate_l1_cache()"}


def _normalize_for_grouping(frames):
    """Normalize callstack frames for grouping.

    Two normalizations:
    1. Strip internal poll-loop frames (reg_read inside cb_wait_front).
    2. For the topmost frame, use function name only (ignore exact line within
       the function). Different cores may be caught at different lines within
       the same spinning function — that's sampling noise, not a real difference.
       Deeper frames keep their full location since different call sites ARE
       meaningfully different.
    """
    result = list(frames)
    # Strip poll internals from top
    if len(result) >= 2 and result[0][0] in _POLL_INTERNALS:
        result = result[1:]
    # Normalize top frame: function name only (strip source line)
    if result:
        func, loc = result[0]
        # Keep just the function name for grouping; strip the specific line
        result[0] = (func, "")
    return result


def parse_callstacks(section_text):
    """Parse dump_callstacks table into per-core entries, then group by pattern."""
    if not section_text:
        return {}

    cores = []
    current_core = None

    for line in section_text.split("\n"):
        if "│" not in line:
            continue
        raw = [f.strip() for f in line.split("│")]
        if len(raw) < 12:
            continue

        loc = raw[2]
        riscv = raw[3]
        kernel_name = raw[5]
        waypoint = raw[9]
        callstack = raw[11] if len(raw) > 11 else ""

        if loc:  # New core entry
            current_core = {
                "loc": loc,
                "riscv": riscv,
                "kernel_name": kernel_name if kernel_name != "N/A" else "N/A",
                "waypoint": waypoint,
                "frames": [],
            }
            cores.append(current_core)
            if callstack and callstack.startswith("#"):
                current_core["frames"].append(normalize_frame(callstack))
        elif current_core and callstack and callstack.startswith("#"):
            current_core["frames"].append(normalize_frame(callstack))

    # Group by pattern
    # For N/A kernels (firmware idle), ignore exact callstack — they're all idle
    # groups maps grouping_key -> {"cores": [...], "display_frames": [...]}
    groups = defaultdict(lambda: {"cores": [], "display_frames": None})
    for core in cores:
        if core["kernel_name"] == "N/A":
            key = (core["riscv"], "N/A", "W", ())
        else:
            # Filter out generic firmware entry points (_start, main)
            meaningful_frames = [(f, l) for f, l in core["frames"] if not is_firmware_entry_frame(f, l)]
            # Normalized key for grouping (merges poll-loop sampling differences)
            grouping_frames = _normalize_for_grouping(meaningful_frames)
            key = (
                core["riscv"],
                core["kernel_name"],
                core["waypoint"],
                tuple(grouping_frames),
            )
            # Keep first core's actual frames for display
            if groups[key]["display_frames"] is None:
                groups[key]["display_frames"] = meaningful_frames

        coord_match = re.search(r"\((\d+,\d+)\)", core["loc"])
        coord = coord_match.group(1) if coord_match else core["loc"]
        groups[key]["cores"].append(coord)

    return groups


def format_summary(checks, op_info, callstack_groups, triage_path):
    """Format everything into a compact summary."""
    lines = []
    lines.append("=== TRIAGE SUMMARY ===")

    # Operation info
    if op_info:
        name = op_info.get("op_name", "???")
        op_id = op_info.get("op_id", "?")
        core_count = op_info.get("core_count", "?")
        lines.append(f"Operation: {name} (id={op_id}, {core_count} cores)")
        shape = op_info.get("shape", "")
        dtype = op_info.get("dtype", "")
        mem = op_info.get("memory_layout", "")
        buf = op_info.get("buffer_type", "")
        if shape or dtype:
            lines.append(f"  shape={shape} dtype={dtype} {mem}/{buf}")
    lines.append("")

    # Checks — only show if any failed, otherwise one-line summary
    if checks:
        failed = {k: v for k, v in checks.items() if v == "FAIL"}
        if failed:
            lines.append(f"Checks FAILED: {', '.join(failed.keys())}")
            passed = [k for k, v in checks.items() if v == "PASS"]
            if passed:
                lines.append(f"Checks passed: {', '.join(passed)}")
        else:
            lines.append(f"Checks: all passed ({', '.join(checks.keys())})")
        lines.append("")

    if not callstack_groups:
        lines.append("No callstack data found.")
        lines.append(f"Full triage: {triage_path}")
        lines.append("=== END TRIAGE SUMMARY ===")
        return "\n".join(lines)

    # Separate user kernels, dispatch, and firmware idle
    user_groups = {}
    dispatch_groups = {}
    idle_groups = {}
    for key, group_data in callstack_groups.items():
        riscv, kernel_name, waypoint, _ = key
        if kernel_name == "N/A":
            idle_groups[key] = group_data
        elif is_dispatch_kernel(kernel_name):
            dispatch_groups[key] = group_data
        else:
            user_groups[key] = group_data

    # Print user kernel patterns (the important ones)
    if user_groups:
        lines.append("Hung kernel callstacks:")
        sorted_groups = sorted(user_groups.items(), key=lambda x: -len(x[1]["cores"]))
        pattern_num = 0
        for (riscv, kernel_name, waypoint, _), group_data in sorted_groups:
            pattern_num += 1
            core_list = group_data["cores"]
            display_frames = group_data["display_frames"] or []
            count = len(core_list)
            cores_str = _format_core_list(core_list)

            lines.append(f"  [{pattern_num}] {riscv} ({count} cores) — {kernel_name} — waypoint: {waypoint}")
            if display_frames:
                for func, loc in display_frames:
                    lines.append(f"    {func:40s} {loc}")
            else:
                lines.append("    (no user-code frames captured)")
            lines.append(f"    Cores: {cores_str}")
            lines.append("")

    # Firmware idle — single collapsed line
    if idle_groups:
        total_idle = sum(len(v["cores"]) for v in idle_groups.values())
        idle_types = sorted(set(k[0] for k in idle_groups.keys()))
        lines.append(f"Firmware idle: {total_idle} entries ({', '.join(idle_types)}) — no kernel assigned")
        lines.append("")

    # Dispatch — collapsed unless interesting
    if dispatch_groups:
        dispatch_names = sorted(set(k[1] for k in dispatch_groups.keys()))
        lines.append(f"Dispatch cores: {', '.join(dispatch_names)} (infrastructure, usually not the cause)")
        lines.append("")

    lines.append(f"Full triage: {triage_path}")
    lines.append("=== END TRIAGE SUMMARY ===")
    return "\n".join(lines)


def _format_core_list(core_list):
    """Format a list of core coordinates compactly."""
    if len(core_list) <= 6:
        return " ".join(f"({c})" for c in core_list)
    shown = core_list[:3] + ["..."] + core_list[-1:]
    return " ".join(f"({c})" if c != "..." else c for c in shown)


def main():
    if len(sys.argv) < 2:
        print("Usage: summarize-triage.py <triage_log_path>", file=sys.stderr)
        sys.exit(1)

    triage_path = sys.argv[1]
    if not os.path.exists(triage_path):
        print(f"Triage log not found: {triage_path}", file=sys.stderr)
        sys.exit(1)

    with open(triage_path) as f:
        content = f.read()

    sections = parse_sections(content)
    checks = parse_checks(sections)
    op_info = parse_op_info(sections.get("dump_running_operations", ""))
    callstack_groups = parse_callstacks(sections.get("dump_callstacks", ""))

    summary = format_summary(checks, op_info, callstack_groups, triage_path)
    print(summary)


if __name__ == "__main__":
    main()
