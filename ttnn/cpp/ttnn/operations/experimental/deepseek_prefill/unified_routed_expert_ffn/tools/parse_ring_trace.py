#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Decode the grouped unified_routed_expert_ffn ring-buffer traces from a tt-metal watcher log.

The grouped reader/writer kernels push one word per protocol step with WATCHER_RING_BUFFER_PUSH
(no-ops without the watcher). Run a hanging case with the light watcher
(TT_METAL_WATCHER=2 TT_METAL_WATCHER_DISABLE_SANITIZE_NOC=1 TT_METAL_WATCHER_DISABLE_ASSERT=1), then:

    python parse_ring_trace.py generated/watcher/watcher.log ["x,y;x,y;y=N"]

Entries print newest first; val = semaphore value observed before the wait, seq = the counter waited on.
"""
import re, sys

TAGS = {
    1: "GU_TOP",
    2: "in0_ready.wait",
    3: "in0_valid.wait",
    4: "in1_ready.wait_min",
    5: "peer_valid.wait_min",
    6: "up_done.wait_min",
    7: "DN_TOP",
    8: "act_ready.wait",
    9: "act_valid.wait",
    0xA: "down_done.wait_min",
    0xB: "GU_PUSHED",
    0xC: "DN_PUSHED",
    0xD: "W.up_go.wait_min",
    0xE: "W.down_go.wait_min",
    0xF: "W.drain",
}


def dec(v):
    tag = v >> 28
    val = v & 0x0FFFFFFF
    name = TAGS.get(tag, f"tag{tag}")
    if tag in (1, 7, 0xB, 0xC, 0xF):
        return f"{name}(item={val>>16},chunk={(val>>8)&0xFF},kb={val&0xFF})"
    if tag in (4, 5, 6, 0xA, 0xD, 0xE):
        return f"{name}(val={val&0xFFF},seq={val>>12})"
    return f"{name}(val={val})"


want = set(sys.argv[2].split(";")) if len(sys.argv) > 2 else set()  # "x,y;x,y" and/or "y=N"
core = None
wp = ""
buf = []


def flush():
    if core and buf and (not want or core in want or ("y=" + core.split(",")[1]) in want):
        print(f"core ({core}) {wp}")
        for thr, hv in buf:
            print(f"   {thr:7s} {dec(int(hv,16))}")


for line in open(sys.argv[1]):
    m = re.match(r"Device 0 worker core\(x=\s*(\d+),y=\s*(\d+)\)\S*\s*(.*?)\s+rmsg", line)
    if m:
        flush()
        core = f"{m.group(1)},{m.group(2)}"
        wp = m.group(3)
        buf = []
        continue
    if "[BRISC]" in line or "[NCRISC]" in line or "[TRISC" in line:
        buf += re.findall(r"\[(\w+)\](0x[0-9a-fA-F]+)", line)
flush()
