"""Decompose where the glu/reglu/swiglu Makora speedups come from.

All three TTNN composites (unary_composite_op.cpp:338/351/378) share the same
shape: split_tensor_for_glu (2 slices) + activation + multiply = 4 programs.
Activation differs: glu→sigmoid(ACCURATE), reglu→relu, swiglu→swish.

Per shape we time:
  - slice             : one slice on the input (one half) — known cost
  - act               : the activation alone on a half-tensor
  - multiply          : on the half-tensor (baseline single-binary cost)
  - manual chain      : slice + slice + act + multiply (mirror of composite)
  - ttnn op           : the actual composite (ttnn.glu/reglu/swiglu)
  - Makora            : the fused kernel

Usage:
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \\
  TT_METAL_PROFILER_CPP_POST_PROCESS=1 \\
  python verify_glu_decomposition.py [glu|reglu|swiglu|all]
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
ITERS = 10
WARMUP = 2

REQUIRED_ENV_VARS = (
    "TT_METAL_DEVICE_PROFILER",
    "TT_METAL_PROFILER_MID_RUN_DUMP",
    "TT_METAL_PROFILER_CPP_POST_PROCESS",
)

# README shapes per op (matches verify_makora.py).
OP_CONFIGS = {
    "glu": {
        "shapes": [(32, 32, 32, 64), (3, 2, 32, 4096)],
        "ttnn": lambda x: ttnn.glu(x, -1),
        "act": lambda h: ttnn.sigmoid(h),
        "makora": Path("/localdev/dnijemcevic/kernels/Tenstorrent/fusion_store/unary/glu/glu.py"),
    },
    "reglu": {
        "shapes": [(1, 1, 32, 64), (1, 1, 128, 512), (3, 2, 1024, 4096)],
        "ttnn": lambda x: ttnn.reglu(x, -1),
        "act": lambda h: ttnn.relu(h),
        "makora": Path("/localdev/dnijemcevic/kernels/Tenstorrent/fusion_store/unary/reglu/reglu.py"),
    },
    "swiglu": {
        "shapes": [(1, 1, 32, 64), (1, 1, 128, 512), (1, 1, 1024, 4096)],
        "ttnn": lambda x: ttnn.swiglu(x, -1),
        "act": lambda h: ttnn.swish(h),
        "makora": Path("/localdev/dnijemcevic/kernels/Tenstorrent/fusion_store/unary/swiglu/swiglu.py"),
    },
}


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


def _load_makora(path: Path):
    spec = importlib.util.spec_from_file_location(f"makora_{path.stem}", path)
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


def _slice_a(x):
    # Mirrors split_tensor_for_glu first slice: [..., 0 : N/2]
    s = x.padded_shape
    half = s[-1] // 2
    return ttnn.slice(x, [0, 0, 0, 0], [s[0], s[1], s[2], half], [1, 1, 1, 1])


def _slice_b(x):
    s = x.padded_shape
    half = s[-1] // 2
    return ttnn.slice(x, [0, 0, 0, half], [s[0], s[1], s[2], s[-1]], [1, 1, 1, 1])


def _make_manual(act_fn):
    """Build a hand-rolled chain that mirrors the C++ composite for the given activation."""

    def _manual(x):
        a = _slice_a(x)
        b = _slice_b(x)
        sb = act_fn(b)
        out = ttnn.multiply(a, sb)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        ttnn.deallocate(sb)
        return out

    return _manual


def _run_op(op_name: str, device):
    cfg = OP_CONFIGS[op_name]
    makora_mod = _load_makora(cfg["makora"])
    manual = _make_manual(cfg["act"])

    print(f"\n--- {op_name} ---")
    print(
        f"{'shape':<22} {'slice':>9} {'act':>9} {'mul':>9} {'sum_4':>9} "
        f"{'manual':>9} {op_name:>9} {'makora':>9}  {'op/mak':>8} {'op/(act+mul)':>13}"
    )

    for shape in cfg["shapes"]:
        torch.manual_seed(0)
        t = torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        half = _slice_a(x)
        ttnn.synchronize_device(device)

        t_slice = _run_and_median(lambda: _slice_a(x), device)
        t_act = _run_and_median(lambda: cfg["act"](half), device)
        t_mul = _run_and_median(lambda: ttnn.multiply(half, half), device)
        t_sum4 = 2 * t_slice + t_act + t_mul
        t_manual = _run_and_median(lambda: manual(x), device)
        t_op = _run_and_median(lambda: cfg["ttnn"](x), device)
        t_makora = _run_and_median(lambda: makora_mod.host(x), device)

        ttnn.deallocate(half)
        ttnn.deallocate(x)

        print(
            f"{str(shape):<22} {t_slice:>6d} ns {t_act:>6d} ns {t_mul:>6d} ns {t_sum4:>6d} ns "
            f"{t_manual:>6d} ns {t_op:>6d} ns {t_makora:>6d} ns "
            f"{t_op/t_makora:>7.2f}x {t_op/(t_act+t_mul):>12.2f}x"
        )


def main():
    missing = [v for v in REQUIRED_ENV_VARS if os.environ.get(v) != "1"]
    if missing:
        sys.stderr.write("ERROR: missing env vars: " + ", ".join(missing) + "\n")
        sys.exit(2)

    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    ops = list(OP_CONFIGS) if arg == "all" else [arg]
    if not all(o in OP_CONFIGS for o in ops):
        sys.stderr.write(f"unknown op; choices: {list(OP_CONFIGS)} or 'all'\n")
        sys.exit(2)

    device = ttnn.open_device(device_id=0)
    try:
        warm = ttnn.ones((1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _ = ttnn.add(warm, warm)
        _ = _measure_kernel_ns(device)
        ttnn.deallocate(warm)

        for op_name in ops:
            _run_op(op_name, device)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
