# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end A/B of the op with its kernel dir swapped for e2e_kernels/<variant> (make_e2e_kernels.py).

Each case is the golden suite's own cell of a production INPUTS entry (c<id>), run through the
suite's runner, so every variant is checked by the suite's own gate. Launch 1 is the checked one;
launches 2..N are timed in-process (DEVICE KERNEL DURATION, median).

    export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
    E2E_CASES=c2025,c2006 E2E_VARIANTS=base,state_bulk scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/g6_to_layout/perf_experiments/row_noc_issue/test_e2e.py --import-mode=prepend -s

E2E_VARIANTS: names of e2e_kernels/ dirs, or "stock" (ttnn.to_layout, same inputs and keywords).
"""

import importlib
import os
import random
import statistics
import sys
import zlib
from pathlib import Path

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import pytest

# Test-only dependency; the ttnn package forbids a module-level torch import.
torch = pytest.importorskip("torch")
import ttnn

HERE = Path(__file__).resolve().parent
OP = "g6_to_layout"
SPEC = importlib.import_module(f"eval.golden_tests.{OP}.feature_spec")
HELPERS = importlib.import_module(f"eval.golden_tests.{OP}.helpers")
HARNESS = importlib.import_module("eval.golden_harness")
OP_PACKAGE = importlib.import_module(f"ttnn.operations.{OP}")
op_mod = sys.modules[f"ttnn.operations.{OP}.{OP}"]
import ttnn.operations.g6_to_layout.g6_to_layout_program_descriptor as pd  # noqa: E402

_KEY = "DEVICE KERNEL DURATION [ns]"
LAUNCHES = int(os.environ.get("E2E_LAUNCHES", "6"))
CASES = [c.strip() for c in os.environ.get("E2E_CASES", "c2025").split(",") if c.strip()]
VARIANTS = [v.strip() for v in os.environ.get("E2E_VARIANTS", "base,state_bulk").split(",") if v.strip()]


def _durations(device):
    ttnn.ReadDeviceProfiler(device)
    vals = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_KEY)
            if entry is not None:
                vals.append(float(entry.duration))
    return vals


def _case(case_id):
    production = {f"c{entry[0]['case']}": entry for entry in SPEC.INPUTS}
    (row,) = HARNESS.parametrize_cases(
        SPEC.TARGET,
        [production[case_id]],
        OP_PACKAGE.INPUT_TAGGERS,
        OP_PACKAGE.SUPPORTED,
        OP_PACKAGE.EXCLUSIONS,
        getattr(SPEC, "INVALID", []),
        shape_id=lambda _inputs: "bench",
    )
    inputs, axes = row.values
    return inputs, axes


def _tensors(value):
    if isinstance(value, ttnn.Tensor):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in _tensors(item)]
    return []


@pytest.mark.parametrize("case_id", CASES)
def test_e2e(case_id, device, monkeypatch):
    inputs, axes = _case(case_id)
    real_op = HELPERS.g6_to_layout
    results = {}
    for variant in VARIANTS:
        seed = zlib.crc32(case_id.encode()) & 0x7FFFFFFF
        torch.manual_seed(seed)
        random.seed(seed)
        if variant == "stock":
            op = lambda t, layout, **kw: ttnn.to_layout(t, layout, **kw)  # noqa: E731
        else:
            kdir = HERE / "e2e_kernels" / variant
            assert kdir.is_dir(), f"missing {kdir}: run make_e2e_kernels.py"
            monkeypatch.setattr(pd, "KERNEL_DIR", kdir)
            op = real_op
        op_mod._PLAN_CACHE.clear()
        op_mod._PROGRAM_CACHE.clear()
        timings = []

        def repeated(*args, **kwargs):
            given = _tensors(args) + _tensors(kwargs)
            if variant != "stock" and not timings:
                plan = op_mod.select_plan(*args, **kwargs)
                keys = ("regime", "num_cores", "block_rows_tiles", "block_cols_tiles")
                print(f"\nE2E_PLAN case={case_id} " + " ".join(f"{k}={plan.get(k)}" for k in keys), flush=True)
            kept = op(*args, **kwargs)
            ttnn.synchronize_device(device)
            _durations(device)
            for _ in range(LAUNCHES - 1):
                r = op(*args, **kwargs)
                ttnn.synchronize_device(device)
                for t in _tensors(r):
                    if not any(t is g for g in given + _tensors(kept)):
                        ttnn.deallocate(t)
            timings.extend(_durations(device))
            return kept

        monkeypatch.setattr(HELPERS, "g6_to_layout", repeated)
        HELPERS.run_g6_to_layout(inputs, device=device, extras=None, **axes)
        monkeypatch.setattr(HELPERS, "g6_to_layout", real_op)
        med = statistics.median(timings) if timings else float("nan")
        results[variant] = med
        print(
            f"\nE2E case={case_id} variant={variant} ns={med:.0f} n={len(timings)} "
            f"all={[round(v) for v in timings]} CHECK_OK",
            flush=True,
        )
    op_mod._PLAN_CACHE.clear()
    op_mod._PROGRAM_CACHE.clear()
    base = results.get(VARIANTS[0])
    print(
        "E2E_SUMMARY case="
        + case_id
        + " "
        + " ".join(f"{v}={results[v]:.0f}({base / results[v]:.3f}x)" for v in results),
        flush=True,
    )
