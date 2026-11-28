# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""A/B/stock device-time bench of the tensix_untilize_bf16 candidate on production cells.

Run (from the slot root, TT_METAL_HOME / PYTHONPATH exported to it):
    G6TU_CASES=c2036 G6TU_VARIANTS="op,cand,stock" scripts/run_safe_pytest.sh --profile --no-py-zones \
        ttnn/ttnn/operations/g6_to_layout/perf_experiments/tensix_untilize_bf16/test_g6tu_bench.py --import-mode=prepend
    python3 ttnn/ttnn/operations/g6_to_layout/perf_experiments/tensix_untilize_bf16/parse_bench.py <csv>

Variants (comma list, run in the given order, each wrapped in signposts "<case>|<variant>|start/end"):
    op       the op as it is today (R1 staged_reorder for these cells)
    stock    ttnn.to_layout(x, ROW_MAJOR_LAYOUT)
    cand[:k=v...]  the candidate; keys: c max_block_cols, b blocks_per_core, ra read_ahead,
             wb window_blocks, rd row_depth, f cb_format (uint16|bf16), p preserve_zero_flag
Every variant's first output is checked bit-exact against the input (bf16 int16 views; the candidate
with f=bf16 is reported, not asserted). G6TU_LAUNCHES launches per variant (default 4; the parser drops
the first).
"""

import os
import sys
from pathlib import Path

import pytest

# Test-only dependency; the ttnn package forbids a module-level torch import.
torch = pytest.importorskip("torch")
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
import g6tu_program as g6tu  # noqa: E402

from ttnn.operations.g6_to_layout import g6_to_layout  # noqa: E402

try:
    from tracy import signpost
except Exception:  # noqa: BLE001

    def signpost(*_args, **_kwargs):
        pass


CASES = {
    "c2036": ((1, 1024, 2560), "DRAM"),
    "c2025": ((1, 32, 512), "DRAM"),
    "c2006": ((1, 64, 2560), "DRAM"),
    "c273": ((1, 64, 128, 32), "DRAM"),
    "c295": ((1, 8192, 128), "DRAM"),
    "c304": ((1, 1024, 2048, 128), "DRAM"),
    "c2038": ((1, 32, 2560), "L1"),
    "c297": ((1, 128, 256, 32), "DRAM"),
    "c293": ((1, 1, 131072), "DRAM"),
    "c294": ((1, 131072), "DRAM"),
    "c2026": ((8, 1, 2560), "DRAM"),
    "c1993": ((16, 1, 2560), "DRAM"),
    "c269": ((1, 512), "DRAM"),
    "c288": ((1, 256), "DRAM"),
    "c290": ((1, 196), "DRAM"),
    "c309": ((1, 185), "DRAM"),
    "c311": ((1, 207), "DRAM"),
}

_KEYS = {
    "c": "max_block_cols",
    "b": "blocks_per_core",
    "ra": "read_ahead",
    "wb": "window_blocks",
    "rd": "row_depth",
    "f": "cb_format",
    "p": "preserve_zero_flag",
    "lr": "live_reads",
    "s": "short_max_block_cols",
}


def _knobs(spec):
    kwargs = {}
    for item in spec.split(":")[1:]:
        key, value = item.split("=")
        kwargs[_KEYS[key]] = value if key == "f" else int(value)
    return g6tu.Knobs(**kwargs)


def _read_profiler(device):
    try:
        ttnn.ReadDeviceProfiler(device)
    except Exception:  # noqa: BLE001
        pass


def _bits_equal(out, x, shape):
    got = ttnn.to_torch(out).reshape(shape)
    return int((got.view(torch.int16) != x.view(torch.int16)).sum())


@pytest.mark.parametrize("case", os.environ.get("G6TU_CASES", "c2036").split(","))
def test_bench(device, case):
    shape, buf = CASES[case]
    mem = ttnn.L1_MEMORY_CONFIG if buf == "L1" else ttnn.DRAM_MEMORY_CONFIG
    launches = int(os.environ.get("G6TU_LAUNCHES", "4"))
    variants = os.environ.get("G6TU_VARIANTS", "op,cand,stock").split(",")
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem, preserve_nan_values=True
    )
    for variant in variants:
        tag = f"{case}|{variant}"
        out0 = None
        plan = None
        if variant.startswith("cand"):
            knobs = _knobs(variant)
            plan = g6tu.plan_for(tt_x, mem, knobs)
            # compile + first dispatch outside the window, so the window holds timed launches only
        _read_profiler(device)
        signpost(f"{tag}|start")
        for i in range(launches):
            if variant == "op":
                out = g6_to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
            elif variant == "stock":
                out = ttnn.to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
            else:
                out = g6tu.run(tt_x, mem, knobs, plan=plan)
            ttnn.synchronize_device(device)
            if out0 is None:
                out0 = out
            else:
                ttnn.deallocate(out)
        signpost(f"{tag}|end")
        _read_profiler(device)
        bad = _bits_equal(out0, x, shape)
        ttnn.deallocate(out0)
        extra = f" {g6tu.describe(plan)}" if plan is not None else ""
        print(f"\nG6TU_BENCH {tag} shape={shape} buf={buf} launches={launches} mismatches={bad}{extra}")
        if variant != "stock" and not (variant.startswith("cand") and "f=bf16" in variant):
            assert bad == 0, f"{tag}: {bad} mismatching datums"
