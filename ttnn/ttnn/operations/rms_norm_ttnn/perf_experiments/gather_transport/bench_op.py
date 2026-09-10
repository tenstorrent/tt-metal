# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# WHOLE-OP A/B for the `gather_transport` ideas that need no layout change.
#
# Every variant runs the SAME program descriptor, the SAME compile-time args, the SAME
# precision contract (HiFi2 / fp32_dest as the case declares / math_approx False) and the
# SAME tensors.  The ONLY difference is which directory the three kernel sources come from
# -- `rms_norm_ttnn_program_descriptor.KERNEL_DIR` is a module-level constant, so flipping
# it swaps the writer for a copy of itself and changes nothing else.
#
#   k_base  : byte-identical copy of the shipped kernels (the honest baseline; its
#             agreement with the shipped KERNEL_DIR is itself a control).
#   k_flush : the member's `noc_async_write_barrier()` before the gather's arrival atomic
#             becomes `Noc::async_writes_flushed()`.  Two line-for-line edits in the writer.
#
# Precision knobs are NEVER touched by a variant.

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import statistics
from pathlib import Path

import ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD
from eval.sharding import shard_config

HERE = Path(__file__).resolve().parent
_ML = ttnn.TensorMemoryLayout
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

N_WARMUP = int(os.environ.get("RMS_WARMUP", "2"))
N_TRIALS = int(os.environ.get("RMS_TRIALS", "5"))
N_REPS = int(os.environ.get("RMS_REPS", "3"))

VARIANTS = {
    "shipped": None,
    "base": HERE / "k_base",
    "flush": HERE / "k_flush",
}

# name: (shape, shard|None, memory_layout, mode, fp32_dest, ceiling_ns)
CASES = {
    "F_w7168_28c": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "gamma", False, 5481),
    "G_w1024_8c": ((1, 1, 32, 1024), ([32, 128], (8, 1)), _ML.WIDTH_SHARDED, "gamma", False, 4110),
    "G_w2304_9c": ((1, 1, 32, 2304), ([32, 256], (9, 1)), _ML.WIDTH_SHARDED, "gamma", False, 4617),
    "G_w5120_32c": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "gamma", False, 5267),
    "G_w5120_32c_gbr": (
        (1, 1, 32, 5120),
        ([32, 160], (8, 4)),
        _ML.WIDTH_SHARDED,
        "gamma_bias_residual",
        True,
        6555,
    ),
    "G_blk8192_64c": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, "gamma", False, 28619),
    "G_blk7168_gbr": (
        (1, 1, 7168, 1024),
        ([896, 128], (8, 8)),
        _ML.BLOCK_SHARDED,
        "gamma_bias_residual",
        False,
        34569,
    ),
    # a NON-combine guard: the gather does not exist here, so any delta is noise
    "N_int1024_prefill": ((1, 1, 8192, 1024), None, _ML.INTERLEAVED, "gamma", False, 89992),
}


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


def build(device, name):
    import torch  # function-local: ttnn/ttnn may not import torch globally (pre-commit)

    shape, shard, ml, mode, fp32_dest, ceiling = CASES[name]
    torch.manual_seed(0)
    W = shape[-1]
    tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    lay = ttnn.TILE_LAYOUT
    if shard is not None:
        mc = shard_config(shard[0], shard[1], ml, layout=lay, dtype=ttnn.bfloat16, device=device)
    else:
        mc = ttnn.DRAM_MEMORY_CONFIG
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
    return (lambda: rms_norm_ttnn(x, **kwargs)), expected, ceiling, live


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

    run, expected, ceiling, live = build(device, name)
    out = run()
    got = ttnn.to_torch(out)
    p, r = pcc(got, expected), relrms(got, expected)
    del out, got
    for _ in range(N_WARMUP):
        run()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    samples = []
    for _ in range(N_TRIALS):
        run()
        ttnn.synchronize_device(device)
        v = _read_kernel_ns(device)
        if v is not None:
            samples.append(v)
    ns = statistics.median(samples) if samples else float("nan")
    for t in live:
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return ns, p, r


def sweep(labels, names=None, device=None):
    names = names or list(CASES)
    shipped = PD.KERNEL_DIR
    RES = {}
    own = device is None
    if own:
        device = ttnn.open_device(device_id=0)
    try:
        for rep in range(N_REPS):
            for label in labels:
                PD.KERNEL_DIR = shipped if VARIANTS[label] is None else VARIANTS[label]
                try:
                    for name in names:
                        ns, p, r = measure(device, name)
                        RES.setdefault((name, label), []).append((ns, p, r))
                finally:
                    PD.KERNEL_DIR = shipped
        base = labels[0]
        print("RESULT " + f"{'case':20s}" + "".join(f"{l:>12s}" for l in labels))
        for name in names:
            row = f"{name:20s}" + "".join(f"{min(x[0] for x in RES[(name, l)]):12.0f}" for l in labels)
            print("RESULT " + row)
        print("RESULT --- speedup vs " + base + " (>1 = faster) ---")
        for name in names:
            b = min(x[0] for x in RES[(name, base)])
            row = f"{name:20s}" + "".join(f"{b / min(x[0] for x in RES[(name, l)]):12.3f}" for l in labels)
            print("RESULT " + row)
        for name in names:
            for l in labels:
                pmin = min(x[1] for x in RES[(name, l)])
                rmax = max(x[2] for x in RES[(name, l)])
                print(f"RESULT ACC {name:20s} {l:10s} pcc {pmin:.7f}  relrms {rmax:.4e}")
    finally:
        if own:
            ttnn.close_device(device)
    return RES
