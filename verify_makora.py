"""Harness: benchmark Makora's fused matmul+silu kernel against ttnn.matmul.

Quick start — one command, produces the full Makora-vs-TTNN comparison table
(6 shapes, per-shape ns + speedup + PCC + max_abs_diff + math util, GMEAN row):

  scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/matmul_silu/test_makora_bench.py::test_matmul_silu_readme_shapes -s

That pytest shim sets the profiler env vars + TT_METAL_LOGGER_LEVEL=Warning
and drives verify_makora.main() with --readme-shapes. Use --run-all if you
want pytest not to stop on first failure.

Sample output (Wormhole 8x8 grid):
  matmul_silu  shape=(32, 32, 32)             path=fused  ttnn=    5 us  makora=     6 us  speedup=0.80x  ttnn_util= 0.64%  makora_util= 0.51%
  matmul_silu  shape=(128, 128, 128)          path=fused  ttnn=    6 us  makora=     9 us  speedup=0.65x  ttnn_util= 2.09%  makora_util= 1.36%
  matmul_silu  shape=(4, 1024, 1024, 1024)    path=kwarg  ttnn=  419 us  makora=  3292 us  speedup=0.13x  ttnn_util=15.66%  makora_util= 1.99%
  matmul_silu  shape=(4096, 1024, 1024)       path=fused  ttnn=  380 us  makora=  3321 us  speedup=0.11x  ttnn_util=17.25%  makora_util= 1.97%
  matmul_silu  shape=(2, 1, 2048, 2048, 2048) path=kwarg  ttnn=  913 us  makora= 12691 us  speedup=0.07x  ttnn_util=28.70%  makora_util= 2.07%
  matmul_silu  shape=(4096, 2048, 2048)       path=fused  ttnn= 1054 us  makora= 12838 us  speedup=0.08x  ttnn_util=24.86%  makora_util= 2.04%
  matmul_silu  GMEAN over 6 shapes:                       ttnn=  129 us  makora=   686 us  speedup=0.19x  ttnn_util=14.87%  makora_util= 1.66%

speedup column is ttnn_time / makora_time, so smaller = TTNN faster.
path column: 'fused' = single device program (in-kernel SiLU via
core_grid trigger), 'kwarg' = 2-program post-op chain (used for batched B,
which TTNN's fused factory refuses at matmul_program_config.cpp:454).

Workflow per shape:
  1) Build random bf16 inputs.
  2) Warmup both kernels (sync + drain profiler).
  3) Run N measured iterations; for each: sync + ReadDeviceProfiler +
     ttnn.get_latest_programs_perf_data(), sum DEVICE KERNEL DURATION [ns]
     across all programs in the latest read.
  4) Compute PCC + max_abs_diff vs Makora.
  5) Compute math utilization (SDPA convention: matmul FLOPs /
     (active_cores × duration_ns × 2048) × 100, HiFi2).

Required env vars (set by the pytest shim above; needed if running directly):
  TT_METAL_DEVICE_PROFILER=1
  TT_METAL_PROFILER_MID_RUN_DUMP=1
  TT_METAL_PROFILER_CPP_POST_PROCESS=1

Direct (no pytest shim, no device-reset-on-exit):
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \\
  TT_METAL_PROFILER_CPP_POST_PROCESS=1 \\
  python verify_makora.py --readme-shapes --iters 5
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import statistics
import sys
from pathlib import Path

import torch

import ttnn

# Makora kernel source root. Defaults to the in-repo copy that ships with this
# branch; override with MAKORA_ROOT env var to point at an external mirror
# (e.g. /localdev/dnijemcevic/kernels/Tenstorrent/fusion_store).
MAKORA_ROOT = Path(
    os.environ.get(
        "MAKORA_ROOT",
        Path(__file__).resolve().parent / "tests/ttnn/unit_tests/operations/matmul_silu/makora",
    )
)
DEVICE_KERNEL_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

REQUIRED_ENV_VARS = (
    "TT_METAL_DEVICE_PROFILER",
    "TT_METAL_PROFILER_MID_RUN_DUMP",
    "TT_METAL_PROFILER_CPP_POST_PROCESS",
)


# `(*batch_dims, M, K, N)` — last 3 entries are matmul dims.
# Reference shapes (square M=K=N) from
# /localdev/dnijemcevic/kernels/Tenstorrent/references/matmul_fused_activation.py
# plus their batch-fused-into-M "unfolded" equivalents (same FMA count, b is 2D
# so TTNN's in-kernel fused-activation path accepts them).
README_SHAPES = [
    (32, 32, 32),
    (128, 128, 128),
    (4, 1024, 1024, 1024),
    (4096, 1024, 1024),
    (2, 1, 2048, 2048, 2048),
    (4096, 2048, 2048),
]


def _patch_includes(src: str) -> str:
    """Rewrite legacy kernel include paths to the current tt-metal layout.

    Makora kernels were written against an older tt-metal where headers lived
    under `compute_kernel_api/` and `dataflow_api.h` was on the bare include
    path. Current main moved them under `api/compute/` and `api/dataflow/`.
    """
    src = src.replace('#include "compute_kernel_api/', '#include "api/compute/')
    src = src.replace('#include "compute_kernel_api.h"', '#include "api/compute/compute_kernel_api.h"')
    src = src.replace('#include "dataflow_api.h"', '#include "api/dataflow/dataflow_api.h"')
    # Compute-kernel entry: legacy `namespace NAMESPACE { void MAIN { ... } }`
    # is now a free `void kernel_main() { ... }`.
    src = re.sub(r"namespace\s+NAMESPACE\s*\{\s*\n", "\n", src)
    src = re.sub(r"\}\s*//\s*namespace\s+NAMESPACE.*\n", "\n", src)
    src = src.replace("void MAIN {", "void kernel_main() {")
    return src


def _load_makora_module(category: str, name: str):
    # Two on-disk layouts:
    #   - legacy: fusion_store/<category>/<name>/<name>.py
    #   - flat   (web-app-generated kernels in fusion_store/new/): <category>/<name>.py
    candidates = [
        MAKORA_ROOT / category / name / f"{name}.py",
        MAKORA_ROOT / category / f"{name}.py",
    ]
    path = next((c for c in candidates if c.exists()), None)
    if path is None:
        raise FileNotFoundError("Makora kernel not found at any of:\n  " + "\n  ".join(str(c) for c in candidates))
    spec = importlib.util.spec_from_file_location(f"makora_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "host"):
        raise AttributeError(f"{path} does not expose host()")
    # Rewrite include paths in any *_kernel_src module attribute.
    for attr in dir(mod):
        if attr.endswith("_kernel_src") or attr.endswith("_src"):
            val = getattr(mod, attr)
            if isinstance(val, str) and "#include" in val:
                setattr(mod, attr, _patch_includes(val))
    return mod


def _b_is_batched(b) -> bool:
    """b has effective batch dims (rank > 2 with any leading dim > 1)?"""
    shape = list(b.shape)
    return len(shape) > 2 and any(d > 1 for d in shape[:-2])


def _ttnn_matmul_silu(a, b):
    """Apples-to-apples vs Makora's fused matmul+silu.

    bf16 in/out, DRAM-interleaved output, HiFi2 + fp32_dest_acc_en.

    Routing:
      - If b is NOT batched: pass `core_grid=device.core_grid` to trigger
        the in-kernel fused path (single device program, SiLU applied on
        DST between matmul accumulate and pack via
        bmm_large_block_zm_fused_bias_activation.cpp with SFPU_ACTIVATION=1).
      - If b IS batched: TTNN's in-kernel fused activation refuses
        (matmul_program_config.cpp:454: !fused_activation when batched B);
        omit core_grid so matmul.cpp:294 falls through to the 2-program
        post-op path (matmul then unary_chain SiLU, DRAM round-trip
        between).
    """
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
    )
    kwargs = dict(
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        activation="silu",
        compute_kernel_config=cfg,
    )
    if not _b_is_batched(b):
        kwargs["core_grid"] = a.device().core_grid
    return ttnn.matmul(a, b, **kwargs)


def _path_label(shape) -> str:
    """Which TTNN dispatch path will be taken for this shape's b tensor?"""
    if len(shape) > 3 and any(d > 1 for d in shape[:-3]):
        return "kwarg"
    return "fused"


def _make_inputs(shape, device, seed: int = 0):
    """`shape` is `(*batch_dims, M, K, N)`. Returns ((a_dev, b_dev), (a_t, b_t))."""
    if len(shape) < 3:
        raise ValueError(f"matmul_silu shape must be (*batch, M, K, N), got {shape!r}")
    batch_dims = tuple(shape[:-3])
    M, K, N = shape[-3], shape[-2], shape[-1]
    for label, dim in (("M", M), ("K", K), ("N", N)):
        if dim % 32 != 0:
            raise ValueError(f"matmul_silu requires {label} % 32 == 0, got {label}={dim}")
    g = torch.Generator().manual_seed(seed)
    a_t = (-1.0 + 2.0 * torch.rand(*batch_dims, M, K, generator=g, dtype=torch.float32)).to(torch.bfloat16)
    b_t = (-1.0 + 2.0 * torch.rand(*batch_dims, K, N, generator=g, dtype=torch.float32)).to(torch.bfloat16)
    a_dev = ttnn.from_torch(a_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b_dev = ttnn.from_torch(b_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return (a_dev, b_dev), (a_t, b_t)


def _check_env() -> None:
    missing = [v for v in REQUIRED_ENV_VARS if os.environ.get(v) != "1"]
    if missing:
        sys.stderr.write(
            "ERROR: missing required env vars: " + ", ".join(missing) + "\n"
            "Re-run with all of TT_METAL_DEVICE_PROFILER, TT_METAL_PROFILER_MID_RUN_DUMP, "
            "and TT_METAL_PROFILER_CPP_POST_PROCESS set to 1.\n"
        )
        sys.exit(2)


def _measure_one_kernel_duration_ns(device) -> int | None:
    """Sync, flush profiler, and sum DEVICE KERNEL DURATION [ns] across all programs."""
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    perf = ttnn.get_latest_programs_perf_data()
    if not perf:
        return None
    chip_id = next(iter(perf))
    programs = perf[chip_id]
    if not programs:
        return None
    total_ns = 0
    for prog in programs:
        result = prog.program_analyses_results.get(DEVICE_KERNEL_DURATION_KEY)
        if result is None:
            return None
        total_ns += int(result.duration)
    return total_ns


def _run_and_measure(callable_no_args, device, iters: int, warmup: int) -> list[int]:
    for _ in range(warmup):
        out = callable_no_args()
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)
    # Drain so warmup data doesn't pollute the first measurement.
    ttnn.ReadDeviceProfiler(device)
    _ = ttnn.get_latest_programs_perf_data()

    durations = []
    for _ in range(iters):
        out = callable_no_args()
        d = _measure_one_kernel_duration_ns(device)
        ttnn.deallocate(out)
        if d is None:
            raise RuntimeError(
                "No profiler data returned. Are TT_METAL_PROFILER_MID_RUN_DUMP=1 and "
                "TT_METAL_PROFILER_CPP_POST_PROCESS=1 set, and was tt-metal built with profiling?"
            )
        durations.append(d)
    return durations


def _check_numerics(t_a: ttnn.Tensor, t_b: ttnn.Tensor) -> tuple[float, float]:
    a = ttnn.to_torch(t_a).to(torch.float32)
    b = ttnn.to_torch(t_b).to(torch.float32)
    if a.shape != b.shape:
        return (float("nan"), float("nan"))
    diff = (a - b).abs().max().item()
    am, bm = a - a.mean(), b - b.mean()
    denom = (am.norm() * bm.norm()).item()
    pcc = (am * bm).sum().item() / denom if denom > 0 else 1.0
    return (pcc, diff)


_WORMHOLE_CORES = 64  # full 8×8 compute grid
_FLOPS_PER_CYCLE_PER_CORE = 2048  # HiFi2: 4096 base / 2 (FMAs counted as 2 FLOPs)
_CLOCK_GHZ = 1.0  # Wormhole_b0


def _math_utilization_pct(shape, duration_ns: float) -> float:
    """SDPA-style math util: matmul FLOPs / (active_cores × cycles × FLOPs/cycle).

    Active cores capped at min(output_tiles, full grid) — once every core has
    at least one output tile, more output tiles just stack onto the same cores.
    """
    if duration_ns <= 0:
        return 0.0
    M, K, N = shape[-3], shape[-2], shape[-1]
    batch = 1
    for d in shape[:-3]:
        batch *= d
    mm_flops = 2 * batch * M * K * N
    output_tiles = batch * (M // 32) * (N // 32)
    active_cores = min(output_tiles, _WORMHOLE_CORES)
    cycles = duration_ns * _CLOCK_GHZ
    peak_flops = active_cores * cycles * _FLOPS_PER_CYCLE_PER_CORE
    return (mm_flops / peak_flops) * 100.0 if peak_flops > 0 else 0.0


def _run_one_shape(shape, iters: int, warmup: int, device) -> dict:
    makora = _load_makora_module("new", "matmul_silu")
    (a_dev, b_dev), _torch_inputs = _make_inputs(shape, device)

    def call_ttnn():
        return _ttnn_matmul_silu(a_dev, b_dev)

    def call_makora():
        return makora.host(a_dev, b_dev)

    # One-off correctness check (no profiler).
    out_t = call_ttnn()
    ttnn.synchronize_device(device)
    out_m = call_makora()
    ttnn.synchronize_device(device)
    pcc, max_abs_diff = _check_numerics(out_m, out_t)
    ttnn.deallocate(out_t)
    ttnn.deallocate(out_m)

    ttnn_durs = _run_and_measure(call_ttnn, device, iters, warmup)
    makora_durs = _run_and_measure(call_makora, device, iters, warmup)

    ttnn_med = statistics.median(ttnn_durs)
    makora_med = statistics.median(makora_durs)
    return {
        "shape": shape,
        "path": _path_label(shape),
        "ttnn_median_ns": ttnn_med,
        "makora_median_ns": makora_med,
        "speedup": ttnn_med / makora_med if makora_med else float("inf"),
        "pcc": pcc,
        "max_abs_diff": max_abs_diff,
        "ttnn_util_pct": _math_utilization_pct(shape, ttnn_med),
        "makora_util_pct": _math_utilization_pct(shape, makora_med),
    }


def _print_row(r):
    print(
        f"  matmul_silu  shape={str(r['shape']):<28} "
        f"path={r['path']:<5}  "
        f"ttnn={int(r['ttnn_median_ns']):>8d} ns  "
        f"makora={int(r['makora_median_ns']):>9d} ns  "
        f"speedup={r['speedup']:>5.2f}x  "
        f"pcc={r['pcc']:.4f}  max_abs_diff={r['max_abs_diff']:.2e}  "
        f"ttnn_util={r['ttnn_util_pct']:>5.2f}%  makora_util={r['makora_util_pct']:>5.2f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=None,
        help="Matmul shape (*batch, M, K, N), space-separated (e.g. --shape 32 1024 1024).",
    )
    parser.add_argument("--readme-shapes", action="store_true", help="Run the predefined README shape set.")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    if not args.shape and not args.readme_shapes:
        parser.error("either --shape or --readme-shapes is required.")

    _check_env()

    shapes = README_SHAPES if args.readme_shapes else [tuple(args.shape)]

    device = ttnn.open_device(device_id=0)
    try:
        # Sanity-check that profiler data is being produced.
        warm = ttnn.ones((1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _ = ttnn.add(warm, warm)
        if _measure_one_kernel_duration_ns(device) is None:
            sys.stderr.write("ERROR: profiler returned no data on a warmup add. Was tt-metal built with profiling?\n")
            sys.exit(2)
        ttnn.deallocate(warm)

        print(f"Op: matmul_silu  iters={args.iters}  warmup={args.warmup}")
        speedups, ttnn_meds, makora_meds, ttnn_utils, makora_utils = [], [], [], [], []
        for shape in shapes:
            try:
                r = _run_one_shape(shape, args.iters, args.warmup, device)
                _print_row(r)
                speedups.append(r["speedup"])
                ttnn_meds.append(r["ttnn_median_ns"])
                makora_meds.append(r["makora_median_ns"])
                ttnn_utils.append(r["ttnn_util_pct"])
                makora_utils.append(r["makora_util_pct"])
            except Exception as e:
                print(f"  matmul_silu   shape={str(shape):<28} ERROR: {e}")

        if len(speedups) >= 2:
            gmean_speedup = statistics.geometric_mean(speedups)
            gmean_ttnn = statistics.geometric_mean(ttnn_meds)
            gmean_makora = statistics.geometric_mean(makora_meds)
            # Arithmetic mean for util (geometric mean of percentages is misleading).
            mean_ttnn_util = statistics.mean(ttnn_utils)
            mean_makora_util = statistics.mean(makora_utils)
            print(
                f"  matmul_silu   GMEAN over {len(speedups)} shapes:"
                f"            ttnn={int(gmean_ttnn):>8d} ns  "
                f"makora={int(gmean_makora):>8d} ns  "
                f"speedup={gmean_speedup:>5.2f}x"
                f"                                       "
                f"ttnn_util={mean_ttnn_util:>5.2f}%  makora_util={mean_makora_util:>5.2f}%"
            )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
