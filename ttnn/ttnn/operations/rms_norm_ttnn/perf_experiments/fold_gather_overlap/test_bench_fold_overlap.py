# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated bake-off: `fold_gather_overlap` -- fold the partials as they land.

Runs the op out of THIS DIRECTORY's private clone (descriptor + kernels), so the
shipped op is never touched.  Variants are selected by setting `PD.FOLD_RUNS`
(and `PD.MCAST_NO_COPY`) before each build; the descriptor is rebuilt on every
call, so a knob change is a fresh program.

Select what to run with env vars:
    RMS_BENCH_VARIANTS  comma list of variant labels   (default: base,r2,r4,r7)
    RMS_BENCH_CASES     comma list of case names       (default: G1)
    RMS_BENCH_TRIALS    profiled runs per variant/case (default: 3, median)
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import statistics
import sys
from pathlib import Path

import ttnn

_REPO = Path(__file__).resolve().parents[6]
for _p in (str(_REPO / "tt_metal" / "third_party" / "tt_ops_code_gen"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.sharding import shard_config  # noqa: E402

from ttnn.operations.rms_norm_ttnn.perf_experiments.fold_gather_overlap.rms_norm_ttnn import (  # noqa: E402
    rms_norm_ttnn,
    torch_rms_norm_ttnn,
)
from ttnn.operations.rms_norm_ttnn.perf_experiments.fold_gather_overlap import (  # noqa: E402
    rms_norm_ttnn_program_descriptor as PD,
)

_ML = ttnn.TensorMemoryLayout
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
N_TRIALS = int(os.environ.get("RMS_BENCH_TRIALS", "3"))


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def _cfg(fp32_dest):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = fp32_dest
    c.math_approx_mode = False
    return c


# name -> shape, (shard, grid), memory_layout, mode, fp32_dest
CASES = {
    # ---- THE FOCUS SHAPE: G=28, one combine round, flat root, BLOCK_ROWS=1 ----
    "G1": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "gamma", False),
    # ---- THE DE-SKEWED CONTROL: same geometry, no per-core gamma DRAM read ----
    "G1n": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "no_gamma", False),
    # ---- domain sweep: other combine geometries ----
    "W1024": ((1, 1, 32, 1024), ([32, 128], (8, 1)), _ML.WIDTH_SHARDED, "gamma", False),
    "W2304": ((1, 1, 32, 2304), ([32, 256], (9, 1)), _ML.WIDTH_SHARDED, "gamma", False),
    "W5120": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "gamma", False),
    "B8192": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, "gamma", False),
    "B7168": ((1, 1, 7168, 1024), ([896, 128], (8, 8)), _ML.BLOCK_SHARDED, "gamma_bias_residual", False),
    # the op's own fp32_dest_acc_en guard cell (tree combine + every optional operand)
    "W5120g": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "gamma_bias_residual", True),
    # NO combine at all -- the whole option-(d) body is `if constexpr (CROSS_CORE)`-gated
    "INT7168": ((1, 1, 32, 7168), None, _ML.INTERLEAVED, "gamma", False),
    "INT8192": ((1, 1, 8192, 1024), None, _ML.INTERLEAVED, "gamma", False),
}

# label -> knob dict applied to the private descriptor module
VARIANTS = {
    "base": {"FOLD_RUNS": 0, "MCAST_NO_COPY": False},
    "r2": {"FOLD_RUNS": 2, "MCAST_NO_COPY": False},
    "r4": {"FOLD_RUNS": 4, "MCAST_NO_COPY": False},
    "r7": {"FOLD_RUNS": 7, "MCAST_NO_COPY": False},
    "r14": {"FOLD_RUNS": 14, "MCAST_NO_COPY": False},
    "r8": {"FOLD_RUNS": 8, "MCAST_NO_COPY": False},
    "r16": {"FOLD_RUNS": 16, "MCAST_NO_COPY": False},
    # CONTROLS that price the prefix fold's OVERHEAD with none of its overlap
    "r2nopipe": {"FOLD_RUNS": 2, "FOLD_NOPIPE": True},
    "r7nopipe": {"FOLD_RUNS": 7, "FOLD_NOPIPE": True},
    "r14nopipe": {"FOLD_RUNS": 14, "FOLD_NOPIPE": True},
    "nocopy": {"FOLD_RUNS": 0, "MCAST_NO_COPY": True},
    # ABLATIONS -- perf ceilings, NOT correct programs.  pcc/rel-RMS are reported and
    # are expected to be garbage; they exist to bound what the real idea could buy.
    "abl_fold": {"FOLD_RUNS": 0, "ABLATE_ROOT_SUM": True},
    "abl_copy": {"FOLD_RUNS": 0, "ABLATE_MCAST_COPY": True},
    "r4nocopy": {"FOLD_RUNS": 4, "MCAST_NO_COPY": True},
    "r7nocopy": {"FOLD_RUNS": 7, "MCAST_NO_COPY": True},
    "r2nocopy": {"FOLD_RUNS": 2, "MCAST_NO_COPY": True},
    "r14nocopy": {"FOLD_RUNS": 14, "MCAST_NO_COPY": True},
}


def build(device, name):
    import torch  # function-local: ttnn/ttnn may not import torch globally (pre-commit)

    shape, shard, ml, mode, fp32_dest = CASES[name]
    torch.manual_seed(0)
    W = shape[-1]
    tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    lay = ttnn.TILE_LAYOUT
    if shard is None:
        mc = ttnn.DRAM_MEMORY_CONFIG
    else:
        mc = shard_config(shard[0], shard[1], ml, layout=lay, dtype=ttnn.bfloat16, device=device)
    x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=lay, device=device, memory_config=mc)
    kwargs = {"epsilon": 1e-12, "compute_kernel_config": _cfg(fp32_dest), "memory_config": x.memory_config()}
    ref = {"input_tensor": tx.float()}

    def _vec(seed):
        torch.manual_seed(seed)
        t = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
        return t, ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=lay, device=device)

    if "gamma" in mode and mode != "no_gamma":
        t, v = _vec(1)
        kwargs["weight"] = v
        ref["weight"] = t.float()
    if "bias" in mode:
        t, v = _vec(2)
        kwargs["bias"] = v
        ref["bias"] = t.float()
    if "residual" in mode:
        torch.manual_seed(3)
        tr = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        kwargs["residual_input_tensor"] = ttnn.from_torch(
            tr, dtype=ttnn.bfloat16, layout=lay, device=device, memory_config=mc
        )
        ref["residual_input_tensor"] = tr.float()
    expected = torch_rms_norm_ttnn(
        ref["input_tensor"],
        epsilon=1e-12,
        weight=ref.get("weight"),
        bias=ref.get("bias"),
        residual_input_tensor=ref.get("residual_input_tensor"),
    )
    live = [x] + [v for v in kwargs.values() if isinstance(v, ttnn.Tensor)]
    return (lambda: rms_norm_ttnn(x, **kwargs)), expected, live


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def relrms(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return float(((a - b).pow(2).mean().sqrt()) / (b.pow(2).mean().sqrt() + 1e-30))


def measure(device, name):
    import torch  # function-local: ttnn/ttnn may not import torch globally (pre-commit)

    run, expected, live = build(device, name)
    out = run()
    got = ttnn.to_torch(out)
    p, r = pcc(got, expected), relrms(got, expected)
    del out, got
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    samples = []
    for _ in range(N_TRIALS):
        run()
        ttnn.synchronize_device(device)
        v = _read_kernel_ns(device)
        if v is not None:
            samples.append(v)
    for t in live:
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return (statistics.median(samples) if samples else float("nan")), min(samples, default=float("nan")), p, r


def test_bench():
    labels = os.environ.get("RMS_BENCH_VARIANTS", "base,r2,r4,r7").split(",")
    names = os.environ.get("RMS_BENCH_CASES", "G1").split(",")
    device = ttnn.open_device(device_id=0)
    RES = {}
    try:
        for label in labels:
            knobs = VARIANTS[label]
            saved = {k: getattr(PD, k) for k in knobs}
            for k, v in knobs.items():
                setattr(PD, k, v)
            try:
                for name in names:
                    med, lo, p, r = measure(device, name)
                    RES[(name, label)] = (med, lo, p, r)
                    print(
                        f"RESULT {name:6s} {label:10s} median={med:9.0f} min={lo:9.0f} "
                        f"pcc={p:.7f} relrms={r:.6f} fold_runs={PD.LAST_FOLD_RUNS}"
                    )
            finally:
                for k, v in saved.items():
                    setattr(PD, k, v)
    finally:
        ttnn.close_device(device)

    base = labels[0]
    print("RESULT === speedup vs " + base + " (>1 = faster; median ns) ===")
    for name in names:
        b = RES[(name, base)][0]
        row = f"{name:6s}"
        for label in labels:
            row += f" {label}={b / RES[(name, label)][0]:.3f}"
        print("RESULT " + row)
