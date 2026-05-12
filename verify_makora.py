"""Hacky harness: verify Makora kernel speedup claims against TTNN baselines.

Workflow (per op, per shape):
  1) Build inputs (random torch -> ttnn).
  2) Warmup the op.
  3) Run N measured iterations; for each, sync + ReadDeviceProfiler +
     ttnn.get_latest_programs_perf_data() and pull "DEVICE KERNEL DURATION [ns]".
  4) Repeat for both TTNN baseline and Makora kernel.
  5) Optionally check numerical agreement (PCC + max-abs-diff).
  6) Print median(makora) vs. median(ttnn) and the ratio.

Required env vars (the script aborts if missing):
  TT_METAL_DEVICE_PROFILER=1
  TT_METAL_PROFILER_MID_RUN_DUMP=1
  TT_METAL_PROFILER_CPP_POST_PROCESS=1

Usage:
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \\
  TT_METAL_PROFILER_CPP_POST_PROCESS=1 \\
  python verify_makora.py isclose --shape 4 384 4096 --iters 20

  # use the README default shapes for `op`:
  python verify_makora.py multigammaln --readme-shapes --iters 20

Run with --list to see supported ops.
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

MAKORA_ROOT = Path("/localdev/dnijemcevic/kernels/Tenstorrent/fusion_store")
DEVICE_KERNEL_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

REQUIRED_ENV_VARS = (
    "TT_METAL_DEVICE_PROFILER",
    "TT_METAL_PROFILER_MID_RUN_DUMP",
    "TT_METAL_PROFILER_CPP_POST_PROCESS",
)


def _patch_includes(src: str) -> str:
    """Rewrite legacy kernel include paths to the current tt-metal layout.

    Makora kernels were written against an earlier tt-metal where headers lived
    under `compute_kernel_api/` and `dataflow_api.h` was on the bare include
    path. Current main moved them under `api/compute/` and `api/dataflow/`.
    """
    src = src.replace('#include "compute_kernel_api/', '#include "api/compute/')
    src = src.replace('#include "compute_kernel_api.h"', '#include "api/compute/compute_kernel_api.h"')
    src = src.replace('#include "dataflow_api.h"', '#include "api/dataflow/dataflow_api.h"')
    # SFPU API: typed `where_fp32_tile(...)` was templated to `where_tile<DataFormat::Float32>(...)`.
    src = src.replace("where_fp32_tile(", "where_tile<DataFormat::Float32>(")
    # Bare `where_tile(...)` (only in bf16 kernels) -> `where_tile<DataFormat::Float16_b>(...)`.
    # Avoid touching `where_tile_init` and already-templated `where_tile<...>` calls.
    src = re.sub(r"\bwhere_tile\(", "where_tile<DataFormat::Float16_b>(", src)
    # Compute-kernel entry: legacy `namespace NAMESPACE { void MAIN { ... } }` is now
    # a free `void kernel_main() { ... }`.
    src = re.sub(r"namespace\s+NAMESPACE\s*\{\s*\n", "\n", src)
    src = re.sub(r"\}\s*//\s*namespace\s+NAMESPACE.*\n", "\n", src)
    src = src.replace("void MAIN {", "void kernel_main() {")
    return src


def _load_makora_module(category: str, name: str):
    path = MAKORA_ROOT / category / name / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Makora kernel not found: {path}")
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


# README-provided benchmark shapes (one entry per op).
README_SHAPES: dict[str, list[tuple]] = {
    "atan2": [(32, 32), (4, 384), (4, 384, 4096), (4, 1, 384, 4096)],
    "isclose": [(32, 32), (4, 384), (4, 384, 4096), (4, 1, 384, 4096)],
    "nextafter": [(32, 32), (4, 384), (4, 384, 4096), (4, 1, 384, 4096)],
    "remainder": [(32, 32), (4, 384), (4, 384, 4096), (4, 1, 384, 4096)],
    "digamma": [(32, 128), (5, 2240, 32), (3, 2, 32, 5600)],
    "lgamma": [(32, 32), (32, 128), (5, 2240, 32)],
    "multigammaln": [(32, 32), (32, 128), (5, 2240, 32)],
    "polygamma": [(32, 32), (32, 128), (5, 2240, 32), (3, 2, 32, 5600)],
    "triu": [(32, 32), (32, 64), (4, 384, 4096), (4, 1, 384, 4096)],
    "glu": [(32, 32, 32, 64), (3, 2, 32, 4096)],
    "reglu": [(1, 1, 32, 64), (1, 1, 128, 512), (3, 2, 1024, 4096)],
    "swiglu": [(1, 1, 32, 64), (1, 1, 128, 512), (1, 1, 1024, 4096)],
    "multigammaln_lanczos": [(1, 1, 32, 32), (1, 1, 32, 128), (1, 5, 2240, 32)],
}


# Each entry: category, ttnn_baseline(a, b?), makora_caller(host_mod, a, b?), needs_b
def _ttnn_atan2(a, b):
    return ttnn.atan2(a, b)


def _ttnn_isclose(a, b):
    return ttnn.isclose(a, b)


def _ttnn_nextafter(a, b):
    return ttnn.nextafter(a, b)


def _ttnn_remainder(a, b):
    return ttnn.remainder(a, b)


def _ttnn_digamma(a):
    return ttnn.digamma(a)


def _ttnn_lgamma(a):
    return ttnn.lgamma(a)


def _ttnn_multigammaln(a):
    return ttnn.multigammaln(a)


def _ttnn_polygamma(a):
    return ttnn.polygamma(a, 2)


def _ttnn_triu(a):
    return ttnn.triu(a, diagonal=0)


def _ttnn_glu(a):
    return ttnn.glu(a, -1)


def _ttnn_reglu(a):
    return ttnn.reglu(a, -1)


def _ttnn_swiglu(a):
    return ttnn.swiglu(a, -1)


def _ttnn_multigammaln_lanczos(a):
    from ttnn.operations.multigammaln_lanczos import multigammaln_lanczos

    return multigammaln_lanczos(a)


# Per-op extras: `makora_op` overrides the Makora kernel folder (default = key);
# `dtype` overrides ttnn.bfloat16 default; `safe_domain` shifts random inputs.
OP_REGISTRY: dict[str, dict] = {
    "atan2": {"category": "binary", "binary": True, "ttnn": _ttnn_atan2, "makora_kwargs": {}},
    "isclose": {"category": "binary", "binary": True, "ttnn": _ttnn_isclose, "makora_kwargs": {}},
    "nextafter": {"category": "binary", "binary": True, "ttnn": _ttnn_nextafter, "makora_kwargs": {}},
    "remainder": {"category": "binary", "binary": True, "ttnn": _ttnn_remainder, "makora_kwargs": {}},
    "digamma": {"category": "unary", "binary": False, "ttnn": _ttnn_digamma, "makora_kwargs": {}},
    "lgamma": {"category": "unary", "binary": False, "ttnn": _ttnn_lgamma, "makora_kwargs": {}},
    "multigammaln": {"category": "unary", "binary": False, "ttnn": _ttnn_multigammaln, "makora_kwargs": {}},
    "polygamma": {"category": "unary", "binary": False, "ttnn": _ttnn_polygamma, "makora_kwargs": {"n": 2}},
    "triu": {"category": "unary", "binary": False, "ttnn": _ttnn_triu, "makora_kwargs": {"diag": 0}},
    "glu": {"category": "unary", "binary": False, "ttnn": _ttnn_glu, "makora_kwargs": {}},
    "reglu": {"category": "unary", "binary": False, "ttnn": _ttnn_reglu, "makora_kwargs": {}},
    "swiglu": {"category": "unary", "binary": False, "ttnn": _ttnn_swiglu, "makora_kwargs": {}},
    "multigammaln_lanczos": {
        "category": "unary",
        "binary": False,
        "ttnn": _ttnn_multigammaln_lanczos,
        "makora_kwargs": {},
        "makora_op": "multigammaln",
        "dtype": "float32",
        "safe_domain": (2.0, 10.0),
    },
}


def _check_env() -> None:
    missing = [v for v in REQUIRED_ENV_VARS if os.environ.get(v) != "1"]
    if missing:
        sys.stderr.write(
            "ERROR: missing required env vars: " + ", ".join(missing) + "\n"
            "Re-run with all of TT_METAL_DEVICE_PROFILER, TT_METAL_PROFILER_MID_RUN_DUMP, "
            "and TT_METAL_PROFILER_CPP_POST_PROCESS set to 1.\n"
        )
        sys.exit(2)


def _to_device(t: torch.Tensor, device, ttnn_dtype=ttnn.bfloat16) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _resolve_dtype(spec: dict):
    name = spec.get("dtype", "bfloat16")
    if name == "float32":
        return ttnn.float32, torch.float32
    return ttnn.bfloat16, torch.bfloat16


def _gen_input(shape, generator, torch_dtype, safe_domain=None):
    if safe_domain is not None:
        lo, hi = safe_domain
        a_f32 = torch.rand(*shape, generator=generator, dtype=torch.float32) * (hi - lo) + lo
    else:
        a_f32 = torch.randn(*shape, generator=generator, dtype=torch.float32)
    return a_f32 if torch_dtype == torch.float32 else a_f32.to(torch_dtype)


def _make_inputs(shape: tuple, binary: bool, op_name: str, device, spec: dict, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ttnn_dtype, torch_dtype = _resolve_dtype(spec)
    safe_domain = spec.get("safe_domain")

    a = _gen_input(shape, g, torch_dtype, safe_domain)
    a_dev = _to_device(a, device, ttnn_dtype)
    if not binary:
        return (a_dev,), a
    if op_name in ("atan2", "isclose", "nextafter"):
        b = _gen_input(shape, g, torch_dtype, safe_domain)
    elif op_name == "remainder":
        b_f32 = torch.rand(*shape, generator=g, dtype=torch.float32) + 0.5
        b = b_f32 if torch_dtype == torch.float32 else b_f32.to(torch_dtype)
    else:
        b = _gen_input(shape, g, torch_dtype, safe_domain)
    return (a_dev, _to_device(b, device, ttnn_dtype)), (a, b)


def _measure_one_kernel_duration_ns(device) -> int | None:
    """Sync, flush profiler, and pull the kernel duration of the latest program."""
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    perf = ttnn.get_latest_programs_perf_data()
    if not perf:
        return None
    # Single-chip case; if multi-chip, you'd want to aggregate.
    chip_id = next(iter(perf))
    programs = perf[chip_id]
    if not programs:
        return None
    # Sum across all programs in the latest read so chained-op baselines
    # (e.g. atan + mean) report the total kernel duration, not just the last.
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
    # Drain profiler so warmup data does not pollute first measurement.
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


def _check_numerics(makora_t: ttnn.Tensor, ttnn_t: ttnn.Tensor) -> tuple[float, float]:
    a = ttnn.to_torch(makora_t).to(torch.float32)
    b = ttnn.to_torch(ttnn_t).to(torch.float32)
    if a.shape != b.shape:
        return (float("nan"), float("nan"))
    diff = (a - b).abs().max().item()
    am, bm = a - a.mean(), b - b.mean()
    denom = (am.norm() * bm.norm()).item()
    pcc = (am * bm).sum().item() / denom if denom > 0 else 1.0
    return (pcc, diff)


def _run_one_shape(op: str, shape: tuple, iters: int, warmup: int, device) -> dict:
    spec = OP_REGISTRY[op]
    makora_op_name = spec.get("makora_op", op)
    makora = _load_makora_module(spec["category"], makora_op_name)
    inputs, _torch_inputs = _make_inputs(shape, spec["binary"], op, device, spec)

    def call_ttnn():
        return spec["ttnn"](*inputs)

    def call_makora():
        return makora.host(*inputs, **spec["makora_kwargs"])

    # One-off correctness check (doesn't bother with profiler).
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
        "ttnn_median_ns": ttnn_med,
        "makora_median_ns": makora_med,
        "speedup": ttnn_med / makora_med if makora_med else float("inf"),
        "pcc": pcc,
        "max_abs_diff": max_abs_diff,
        "ttnn_durs": ttnn_durs,
        "makora_durs": makora_durs,
    }


def _print_row(op, r):
    print(
        f"  {op:<14} shape={str(r['shape']):<28} "
        f"ttnn={int(r['ttnn_median_ns']):>8d} ns  "
        f"makora={int(r['makora_median_ns']):>8d} ns  "
        f"speedup={r['speedup']:>5.2f}x  "
        f"pcc={r['pcc']:.4f}  max_abs_diff={r['max_abs_diff']:.2e}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("op", nargs="?", help="Op name (e.g. isclose, multigammaln). Use --list to enumerate.")
    parser.add_argument(
        "--shape", type=int, nargs="+", default=None, help="Tensor shape, space-separated (e.g. --shape 4 384 4096)."
    )
    parser.add_argument("--readme-shapes", action="store_true", help="Run all README-listed shapes for the op.")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--list", action="store_true", help="List supported ops and exit.")
    args = parser.parse_args()

    if args.list:
        for op, spec in OP_REGISTRY.items():
            shapes = README_SHAPES.get(op, [])
            print(f"  {op:<14} category={spec['category']:<7} readme_shapes={shapes}")
        return

    if not args.op:
        parser.error("op is required (or pass --list)")
    if args.op not in OP_REGISTRY:
        parser.error(f"unknown op {args.op!r}; --list to enumerate.")

    _check_env()

    if args.readme_shapes:
        shapes = README_SHAPES.get(args.op)
        if not shapes:
            parser.error(f"no README shapes recorded for {args.op}; pass --shape explicitly.")
    elif args.shape:
        shapes = [tuple(args.shape)]
    else:
        parser.error("either --shape or --readme-shapes is required.")

    device = ttnn.open_device(device_id=0)
    try:
        # Sanity-check that profiler data is actually being produced.
        warm = ttnn.ones((1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _ = ttnn.add(warm, warm)
        if _measure_one_kernel_duration_ns(device) is None:
            sys.stderr.write("ERROR: profiler returned no data on a warmup add. Was tt-metal built with profiling?\n")
            sys.exit(2)
        ttnn.deallocate(warm)

        print(f"Op: {args.op}  iters={args.iters}  warmup={args.warmup}")
        speedups = []
        ttnn_meds = []
        makora_meds = []
        for shape in shapes:
            try:
                r = _run_one_shape(args.op, shape, args.iters, args.warmup, device)
                _print_row(args.op, r)
                speedups.append(r["speedup"])
                ttnn_meds.append(r["ttnn_median_ns"])
                makora_meds.append(r["makora_median_ns"])
            except Exception as e:
                print(f"  {args.op:<14} shape={str(shape):<28} ERROR: {e}")

        if len(speedups) >= 2:
            gmean_speedup = statistics.geometric_mean(speedups)
            gmean_ttnn = statistics.geometric_mean(ttnn_meds)
            gmean_makora = statistics.geometric_mean(makora_meds)
            print(
                f"  {args.op:<14} GMEAN over {len(speedups)} shapes:           "
                f"ttnn={int(gmean_ttnn):>8d} ns  "
                f"makora={int(gmean_makora):>8d} ns  "
                f"speedup={gmean_speedup:>5.2f}x"
            )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
