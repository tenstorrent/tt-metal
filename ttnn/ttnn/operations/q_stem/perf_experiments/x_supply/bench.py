# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""q_stem x-supply bench: correctness (consumer checksum vs host) + DEVICE KERNEL DURATION per variant.

    XS_VARIANTS=baseline,slotflags XS_SPIN_NS=0 XS_ROWS=20 XS_QT=48 \\
        scripts/run_safe_pytest.sh --no-precompile --import-mode=prepend \\
        ttnn/ttnn/operations/q_stem/perf_experiments/x_supply/bench.py -s

XS_FULL=1 checksums every word of every round (slow; correctness only, not for timing).
Results are appended to results.jsonl next to this file.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import json
import socket
import sys
import time
from pathlib import Path

import pytest
import ttnn

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("q_stem_x_supply_bench", Path(__file__).with_name("x_supply.py"))
_xs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _xs
_spec.loader.exec_module(_xs)
VARIANTS, RECT, x_supply, _rect_cores = _xs.VARIANTS, _xs.RECT, _xs.x_supply, _xs._rect_cores

DUR = "DEVICE KERNEL DURATION [ns]"
ROWS = int(os.environ.get("XS_ROWS", "20"))
QT = int(os.environ.get("XS_QT", "48"))
SPIN_NS = int(os.environ.get("XS_SPIN_NS", "0"))
FULL = os.environ.get("XS_FULL", "0") == "1"
DEPTH = int(os.environ.get("XS_DEPTH", "3"))
TICKS_PER_NS = float(os.environ.get("XS_TICKS_PER_NS", "1.35"))  # wall clock = AICLK on Blackhole
NAMES = [n for n in os.environ.get("XS_VARIANTS", ",".join(VARIANTS)).split(",") if n]
OUT = Path(__file__).resolve().parent / "results.jsonl"


def _read_ns(device):
    ttnn.ReadDeviceProfiler(device)
    data = ttnn.get_latest_programs_perf_data() or {}
    vals = []
    for programs in data.values():
        for p in programs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(DUR)
            if e is not None:
                vals.append(float(e.duration))
    return vals[-1] if vals else None


def _inputs(device):
    import torch  # lazy: no global torch import under ttnn/

    t = torch.arange(ROWS).view(ROWS, 1)
    k = torch.arange(QT).view(1, QT)
    v = ((t * QT + k) * 7 % 251 + 1).float()  # one exact bf16 integer per tile
    xt = v.repeat_interleave(32, 0).repeat_interleave(32, 1)
    x = ttnn.from_torch(
        xt.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )  # fmt: skip
    bits = v.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF
    word = (bits | (bits << 16)).sum().item()
    expect = (word * (512 if FULL else 4)) & 0xFFFFFFFF
    return x, expect


@pytest.mark.parametrize("name", NAMES)
def test_x_supply(device, name):
    import torch  # lazy: no global torch import under ttnn/

    x, expect = _inputs(device)
    n = len(_rect_cores(RECT))
    out = ttnn.from_torch(
        torch.zeros(n, 16, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )  # fmt: skip
    _read_ns(device)  # drain
    x_supply(x, out, variant=name, depth=DEPTH, spin=int(SPIN_NS * TICKS_PER_NS), full=FULL)
    ttnn.synchronize_device(device)
    ns = _read_ns(device)
    got = ttnn.to_torch(out).to(torch.int64)[:, 0] & 0xFFFFFFFF
    bad = int((got != expect).sum())
    rec = dict(
        variant=name, rows=ROWS, qt=QT, spin_ns=SPIN_NS, depth=DEPTH, full=FULL, ns=ns,
        per_round_ns=ns / ROWS if ns else None, bad_cores=bad, host=socket.gethostname(), t=time.time(),
    )  # fmt: skip
    print("XS_RESULT", json.dumps(rec))
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if "noread" in name:
        return  # diagnostic: data is not read, the checksum is meaningless
    assert bad == 0, f"{name}: {bad}/{n} receivers checksum mismatch (expect {expect:#x}, got {got[:4].tolist()})"
