"""Decompose where the isclose Makora speedup actually comes from.

Measurements per shape:
  A. ttnn.add(a, b)  — single binary primitive (1 program). Baseline cost of
     "one bf16 DRAM->compute->DRAM cycle on this shape on this hardware."
  B. ttnn.isclose(a, b) — current composite (11 programs).
  C. Hand-rolled Python composite that mirrors the C++ composite step-for-step
     (also 11 programs). Verifies B's cost is purely program count, not anything
     internal to the C++ composite implementation.
  D. Makora's fused kernel (1 program).

Reported ratios:
  composite/single       — effective number of "add-equivalent" programs.
                           Should be ~11 if dispatch + DRAM dominate; less if
                           per-op compute is non-trivial vs. bandwidth.
  composite/manual       — sanity check; should be ~1.0.
  composite/makora       — total Makora win (matches earlier verify_makora.py).
  makora/single          — Makora's cost relative to the cheapest possible
                           single-primitive cost. ~1.0 = "as fast as one add."

Run:
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \\
  TT_METAL_PROFILER_CPP_POST_PROCESS=1 python verify_isclose_decomposition.py
"""

from __future__ import annotations

import importlib.util
import os
import re
import statistics
import sys
from pathlib import Path

import torch
import ttnn

DEVICE_KERNEL_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
SHAPES = [(32, 32), (4, 384), (4, 384, 4096), (4, 1, 384, 4096)]
ITERS = 10
WARMUP = 2

REQUIRED_ENV_VARS = (
    "TT_METAL_DEVICE_PROFILER",
    "TT_METAL_PROFILER_MID_RUN_DUMP",
    "TT_METAL_PROFILER_CPP_POST_PROCESS",
)

MAKORA_PATH = Path("/localdev/dnijemcevic/kernels/Tenstorrent/fusion_store/binary/isclose/isclose.py")


def _patch_includes(src: str) -> str:
    src = src.replace('#include "compute_kernel_api/', '#include "api/compute/')
    src = src.replace('#include "compute_kernel_api.h"', '#include "api/compute/compute_kernel_api.h"')
    src = src.replace('#include "dataflow_api.h"', '#include "api/dataflow/dataflow_api.h"')
    src = src.replace("where_fp32_tile(", "where_tile<DataFormat::Float32>(")
    src = re.sub(r"\bwhere_tile\(", "where_tile<DataFormat::Float16_b>(", src)
    src = re.sub(r"namespace\s+NAMESPACE\s*\{\s*\n", "\n", src)
    src = re.sub(r"\}\s*//\s*namespace\s+NAMESPACE.*\n", "\n", src)
    src = src.replace("void MAIN {", "void kernel_main() {")
    return src


def _load_makora_isclose():
    spec = importlib.util.spec_from_file_location("makora_isclose", MAKORA_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in dir(mod):
        if attr.endswith("_kernel_src") or attr.endswith("_src"):
            val = getattr(mod, attr)
            if isinstance(val, str) and "#include" in val:
                setattr(mod, attr, _patch_includes(val))
    return mod


def _measure_kernel_ns(device) -> int:
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    perf = ttnn.get_latest_programs_perf_data()
    chip_id = next(iter(perf))
    total = 0
    for prog in perf[chip_id]:
        result = prog.program_analyses_results.get(DEVICE_KERNEL_DURATION_KEY)
        total += int(result.duration)
    return total


def _run_and_median(call, device, iters=ITERS, warmup=WARMUP) -> int:
    for _ in range(warmup):
        out = call()
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)
    ttnn.ReadDeviceProfiler(device)
    _ = ttnn.get_latest_programs_perf_data()

    durs = []
    for _ in range(iters):
        out = call()
        durs.append(_measure_kernel_ns(device))
        ttnn.deallocate(out)
    return int(statistics.median(durs))


def _manual_isclose(a, b, rtol=1e-5, atol=1e-8):
    """Hand-rolled exact mirror of the C++ composite at binary_composite_op.cpp:49."""
    mask_a = ttnn.isnan(a)
    val_a = ttnn.where(mask_a, 1.0, a)
    mask_b = ttnn.isnan(b)
    val_b = ttnn.where(mask_b, 0.0, b)
    abs_diff = ttnn.abs(ttnn.subtract(val_a, val_b))
    abs_b = ttnn.abs(val_b)
    mul_b = ttnn.multiply(abs_b, rtol)
    rhs = ttnn.add(mul_b, atol)
    cmp_le = ttnn.le(abs_diff, rhs)
    return ttnn.where(cmp_le, 1.0, 0.0)


def main():
    missing = [v for v in REQUIRED_ENV_VARS if os.environ.get(v) != "1"]
    if missing:
        sys.stderr.write("ERROR: missing env vars: " + ", ".join(missing) + "\n")
        sys.exit(2)

    makora_mod = _load_makora_isclose()
    device = ttnn.open_device(device_id=0)
    try:
        warm = ttnn.ones((1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _ = ttnn.add(warm, warm)
        _ = _measure_kernel_ns(device)
        ttnn.deallocate(warm)

        print(
            f"{'shape':<20} {'add':>10} {'isclose':>10} {'manual':>10} {'makora':>10}  "
            f"{'cmp/add':>8} {'cmp/man':>8} {'cmp/mak':>8} {'mak/add':>8}"
        )

        for shape in SHAPES:
            torch.manual_seed(0)
            ta = torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)
            tb = torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)
            a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

            t_add = _run_and_median(lambda: ttnn.add(a, b), device)
            t_iscls = _run_and_median(lambda: ttnn.isclose(a, b), device)
            t_manual = _run_and_median(lambda: _manual_isclose(a, b), device)
            t_makora = _run_and_median(lambda: makora_mod.host(a, b), device)

            ttnn.deallocate(a)
            ttnn.deallocate(b)

            print(
                f"{str(shape):<20} {t_add:>7d} ns {t_iscls:>7d} ns {t_manual:>7d} ns {t_makora:>7d} ns  "
                f"{t_iscls/t_add:>7.2f}x {t_iscls/t_manual:>7.2f}x {t_iscls/t_makora:>7.2f}x {t_makora/t_add:>7.2f}x"
            )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
