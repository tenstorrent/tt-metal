"""Quantify how much of multigammaln's Makora speedup is from TTNN technical debt.

Three measurements per shape:
  A. ttnn.lgamma(x) — single SFPU primitive (one program).
  B. ttnn.multigammaln(x) — current composite (~150 programs through old _lgamma).
  C. "Uplifted" multigammaln built out of 4 SFPU lgamma calls + adds (the rewrite
     proposed for the `TODO: Remove this once the multigammaln is uplifted` comment).

Reports kernel-duration medians and ratios B/A, B/C, C/A.

Run with the same env vars as verify_makora.py:
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \\
  TT_METAL_PROFILER_CPP_POST_PROCESS=1 python verify_lgamma_uplift.py
"""

from __future__ import annotations

import math
import os
import statistics
import sys

import torch
import ttnn

DEVICE_KERNEL_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
SHAPES = [(32, 32), (32, 128), (5, 2240, 32)]
ITERS = 10
WARMUP = 2

REQUIRED_ENV_VARS = (
    "TT_METAL_DEVICE_PROFILER",
    "TT_METAL_PROFILER_MID_RUN_DUMP",
    "TT_METAL_PROFILER_CPP_POST_PROCESS",
)


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


def _uplifted_multigammaln(x):
    # The proposed "uplifted" multigammaln, built on the SFPU lgamma primitive.
    # Mirrors PyTorch identity for p=4: sum_{j=0..3} lgamma(x - j/2) + 3*log(pi).
    r = ttnn.lgamma(x)
    r = ttnn.add(r, ttnn.lgamma(ttnn.subtract(x, 0.5)))
    r = ttnn.add(r, ttnn.lgamma(ttnn.subtract(x, 1.0)))
    r = ttnn.add(r, ttnn.lgamma(ttnn.subtract(x, 1.5)))
    r = ttnn.add(r, math.log(math.pi) * 3.0)  # constant
    return r


def main():
    missing = [v for v in REQUIRED_ENV_VARS if os.environ.get(v) != "1"]
    if missing:
        sys.stderr.write("ERROR: missing env vars: " + ", ".join(missing) + "\n")
        sys.exit(2)

    device = ttnn.open_device(device_id=0)
    try:
        # Sanity warm.
        warm = ttnn.ones((1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _ = ttnn.add(warm, warm)
        _ = _measure_kernel_ns(device)
        ttnn.deallocate(warm)

        print(
            f"{'shape':<22} {'lgamma(prim)':>14} {'multigammaln':>14} {'uplifted':>10} "
            f"{'gap(B/A)':>10} {'old/uplifted':>13} {'uplifted/lgamma':>17}"
        )
        for shape in SHAPES:
            torch.manual_seed(0)
            t = torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)
            x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

            t_lgamma = _run_and_median(lambda: ttnn.lgamma(x), device)
            t_multigam = _run_and_median(lambda: ttnn.multigammaln(x), device)
            t_uplifted = _run_and_median(lambda: _uplifted_multigammaln(x), device)

            ttnn.deallocate(x)

            gap_BA = t_multigam / t_lgamma
            ratio_old_to_uplifted = t_multigam / t_uplifted
            ratio_uplifted_to_lgamma = t_uplifted / t_lgamma
            print(
                f"{str(shape):<22} {t_lgamma:>11d} ns {t_multigam:>11d} ns {t_uplifted:>7d} ns "
                f"{gap_BA:>9.1f}x {ratio_old_to_uplifted:>12.1f}x {ratio_uplifted_to_lgamma:>16.1f}x"
            )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
