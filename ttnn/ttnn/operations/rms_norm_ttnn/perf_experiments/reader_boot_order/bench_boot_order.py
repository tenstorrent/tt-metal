# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Isolated A/B bake-off for the `reader_boot_order` idea.
#
# WHAT IS ISOLATED.  Every variant runs the SAME program descriptor, the SAME
# compile-time args, the SAME writer and compute kernels, the SAME precision
# contract (HiFi2 / fp32_dest as the case declares / math_approx False) and the
# SAME tensors.  The ONLY difference is which directory the three kernel sources
# come from -- `rms_norm_ttnn_program_descriptor.KERNEL_DIR` is a module-level
# constant, so flipping it swaps the reader for a boot-reordered copy of itself
# and changes nothing else.  A measured delta is therefore attributable to the
# reorder alone.
#
#   k_base : byte-identical copy of the shipped reader (the honest baseline; its
#            agreement with the shipped KERNEL_DIR is itself a control).
#   k_a    : publish_native_shard hoisted ABOVE reader_scaler_boot.
#   k_b    : scaler boot -> publish -> per-channel DRAM read.
#   k_c    : per-channel NoC ISSUE -> publish -> deferred barrier + push.
#   k_d    : publish -> scaler -> per-channel ISSUE -> deferred barrier + push.
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
from eval.sharding import shard_config, auto_shard_config

HERE = Path(__file__).resolve().parent
_ML = ttnn.TensorMemoryLayout
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# One fresh-cache run per variant is the metric (/perf-measure: device kernel time
# has no warm-up transient).  N_TRIALS/N_REPS exist only to take a median when a
# call sits on the noise band.
N_WARMUP = int(os.environ.get("RMS_WARMUP", "2"))
N_TRIALS = int(os.environ.get("RMS_TRIALS", "5"))
N_REPS = int(os.environ.get("RMS_REPS", "3"))

VARIANTS = {
    "shipped": None,  # PD.KERNEL_DIR untouched
    "base": HERE / "k_base",
    "a_pub_first": HERE / "k_a",
    "b_scaler_pub": HERE / "k_b",
    "c_split": HERE / "k_c",
    "d_pub_first_split": HERE / "k_d",
}

# name: (shape, shard|"auto"|None, memory_layout, mode, fp32_dest, layout_rm, ceiling_ns)
CASES = {
    # ---- THE FOCUS SHAPE -------------------------------------------------
    "F_w7168_28c": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "gamma", False, False, 5481),
    # structural floor: gamma read AND multiply removed
    "F_w7168_28c_nog": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "no_gamma", False, False, 0),
    # ---- the STREAM / interleaved regimes --------------------------------
    "I_int7168_1row": ((1, 1, 32, 7168), None, _ML.INTERLEAVED, "gamma", False, False, 14894),
    "I_int1024_prefill": ((1, 1, 8192, 1024), None, _ML.INTERLEAVED, "gamma", False, False, 89992),
    "I_stream_gbr": ((1, 1, 1024, 16384), None, _ML.INTERLEAVED, "gamma_bias_residual", False, False, 0),
    # ---- guard geometries -------------------------------------------------
    "G_w1024_8c": ((1, 1, 32, 1024), ([32, 128], (8, 1)), _ML.WIDTH_SHARDED, "gamma", False, False, 3504),
    "G_w5120_32c": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "gamma", False, False, 4601),
    "G_w5120_32c_gbr": (
        (1, 1, 32, 5120),
        ([32, 160], (8, 4)),
        _ML.WIDTH_SHARDED,
        "gamma_bias_residual",
        True,
        False,
        6555,
    ),
    "G_blk8192": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, "gamma", False, False, 28619),
    "G_blk7168_gbr": (
        (1, 1, 7168, 1024),
        ([896, 128], (8, 8)),
        _ML.BLOCK_SHARDED,
        "gamma_bias_residual",
        False,
        False,
        34569,
    ),
    "G_band512_rm": ((1, 1, 256, 512), "auto", _ML.WIDTH_SHARDED, "gamma", False, True, 20000),
    "G_int4064_ragged": ((1, 1, 32, 4064), None, _ML.INTERLEAVED, "gamma", False, False, 0),
    # a ROW_MAJOR per-channel operand on the focus geometry: the FLAT RM form,
    # which variant (c)/(d) cannot split (the helper owns its barrier) and which
    # therefore falls back to variant (b) ordering there.  "_grm" suffix.
    "F_w7168_28c_grm": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "gamma", False, False, 0),
    # per-channel operand DTYPE sweep on the focus geometry: "_gb8" = bfloat8_b
    # weight (TRIM == 1, the half-page read), "_gf32" = float32 weight.
    "F_w7168_28c_gb8": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "gamma", False, False, 0),
    "F_w7168_28c_gf32": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "gamma", False, False, 0),
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

    shape, shard, ml, mode, fp32_dest, rm, ceiling = CASES[name]
    torch.manual_seed(0)
    W = shape[-1]
    tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    lay = ttnn.ROW_MAJOR_LAYOUT if rm else ttnn.TILE_LAYOUT
    if shard == "auto":
        mc = auto_shard_config(list(shape), ml, layout=lay, dtype=ttnn.bfloat16, device=device)
    elif shard is not None:
        mc = shard_config(shard[0], shard[1], ml, layout=lay, dtype=ttnn.bfloat16, device=device)
    else:
        mc = ttnn.DRAM_MEMORY_CONFIG
    x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=lay, device=device, memory_config=mc)
    kwargs = {"epsilon": 1e-12, "compute_kernel_config": _cfg(fp32_dest), "memory_config": x.memory_config()}
    ref = {"input_tensor": tx.float()}

    # "_grm": force the per-channel operands ROW_MAJOR while x stays TILE.
    pc_lay = ttnn.ROW_MAJOR_LAYOUT if name.endswith("_grm") else lay
    pc_dt = {"_gb8": ttnn.bfloat8_b, "_gf32": ttnn.float32}.get(name[-5:], None)
    if pc_dt is None:
        pc_dt = {"_gb8": ttnn.bfloat8_b, "_gf32": ttnn.float32}.get(name[-4:], ttnn.bfloat16)

    def _vec(seed):
        torch.manual_seed(seed)
        t = torch.randn(1, 1, 1, W, dtype=torch.float32)
        tq = t.to(torch.bfloat16) if pc_dt != ttnn.float32 else t
        return tq.float(), ttnn.from_torch(tq, dtype=pc_dt, layout=pc_lay, device=device)

    if "gamma" in mode and mode != "no_gamma":
        t, v = _vec(1)
        kwargs["weight"] = v
        ref["weight"] = t
    if "bias" in mode:
        t, v = _vec(2)
        kwargs["bias"] = v
        ref["bias"] = t
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
    own_device = device is None
    if own_device:
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
        base_label = labels[0]
        print("RESULT " + f"{'case':20s}" + "".join(f"{l:>13s}" for l in labels))
        for name in names:
            row = f"{name:20s}"
            for label in labels:
                row += f"{min(x[0] for x in RES[(name, label)]):13.0f}"
            print("RESULT " + row)
        print("RESULT --- speedup vs " + base_label + " (>1 = faster) ---")
        for name in names:
            b = min(x[0] for x in RES[(name, base_label)])
            row = f"{name:20s}"
            for label in labels:
                row += f"{b / min(x[0] for x in RES[(name, label)]):13.3f}"
            print("RESULT " + row)
        print("RESULT --- accuracy (worst pcc / worst rel-RMS over reps) ---")
        for name in names:
            for label in labels:
                pmin = min(x[1] for x in RES[(name, label)])
                rmax = max(x[2] for x in RES[(name, label)])
                print(f"RESULT ACC {name:20s} {label:13s} pcc {pmin:.6f}  relrms {rmax:.3e}")
    finally:
        if own_device:
            ttnn.close_device(device)
    return RES
