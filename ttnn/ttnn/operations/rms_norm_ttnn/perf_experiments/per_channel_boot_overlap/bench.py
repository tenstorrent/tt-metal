# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Isolated A/B bake-off for `per_channel_boot_overlap` (PERF round 2).
#
# WHAT IS ISOLATED.  Every variant runs the SAME program descriptor, the SAME
# writer and compute kernels, the SAME tensors and -- critically -- the SAME
# PRECISION CONTRACT (math_fidelity=HiFi2, fp32_dest_acc_en as the case declares,
# math_approx_mode=False, dtypes fixed).  A variant may change only:
#   * which directory the reader source comes from (`PD.KERNEL_DIR`), and
#   * the per-channel READ GRANULARITY knob (`PD.PER_CHANNEL_TRIM_GAMMA/_BIAS`),
#     which feeds ONE reader compile-time arg and nothing else (verified: the CB
#     table does not read it).
# So a measured delta is attributable to the reader's per-channel staging alone.
#
#   shipped    PD.KERNEL_DIR untouched -- the op as it stands today (control).
#   base       k_base: shipped + a DEAD `TRIM == 3` branch. The honest baseline.
#   stag       per-core ROTATION of the per-channel tile loop (issue order only).
#   split      per-channel ISSUE at boot, BARRIER under the first x chunk.
#   trim{0,1,3} granularity menu on the baseline reader.
#   split_* / *_stag  the combinations.
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
_D = PD.TRIM_DERIVED

N_WARMUP = int(os.environ.get("RMS_WARMUP", "1"))
N_TRIALS = int(os.environ.get("RMS_TRIALS", "3"))
N_REPS = int(os.environ.get("RMS_REPS", "1"))

# label: (kernel dir | None, gamma trim override, bias trim override)
VARIANTS = {
    "shipped": (None, _D, _D),
    "base": (HERE / "k_base", _D, _D),
    "stag": (HERE / "k_stag", _D, _D),
    "split": (HERE / "k_split", _D, _D),
    "split_stag": (HERE / "k_split_stag", _D, _D),
    "trim0": (HERE / "k_base", 0, 0),
    "trim1": (HERE / "k_base", 1, 1),
    "trim3": (HERE / "k_base", 3, 3),
    "split_trim3": (HERE / "k_split", 3, 3),
    "split_stag_trim3": (HERE / "k_split_stag", 3, 3),
    "stag_trim3": (HERE / "k_stag", 3, 3),
    "split_trim1": (HERE / "k_split", 1, 1),
    "split_trim0": (HERE / "k_split", 0, 0),
    "split_late": (HERE / "k_split_late", _D, _D),
    "split_late_trim3": (HERE / "k_split_late", 3, 3),
    # ABLATION ONLY (numerically wrong on purpose): the per-channel NoC PAYLOAD is
    # stubbed, the loop / reserve / barrier / push kept.  It is the FLOOR that any
    # read-granularity change can reach, and it is exempt from the pcc gate.
    "ablate": (HERE / "k_ablate", _D, _D),
}

# name: (shape, shard|None, memory_layout, mode, fp32_dest, weight_row_major[, weight_dtype])
CASES = {
    # ---- THE FOCUS SHAPE (worst measured/achievable of the perf group) ------
    "FOCUS_2304": ((1, 1, 8192, 2304), None, _ML.INTERLEAVED, "gamma", False, False),
    # structural floor: the per-channel operand removed entirely.
    "FOCUS_2304_nog": ((1, 1, 8192, 2304), None, _ML.INTERLEAVED, "none", False, False),
    # ---- the domain sweep --------------------------------------------------
    "I_1024_prefill": ((1, 1, 8192, 1024), None, _ML.INTERLEAVED, "gamma", False, False),
    "FOCUS_2304_gb": ((1, 1, 8192, 2304), None, _ML.INTERLEAVED, "gamma_bias", False, False),
    "STREAM_7168_gbr": ((1, 1, 8192, 7168), None, _ML.INTERLEAVED, "gamma_bias_residual", True, False),
    "R1_w7168_28c": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "gamma", False, False),
    "RM_4096_f32": ((1, 1, 128, 4096), None, _ML.INTERLEAVED, "gamma", True, True),
    "SMALL_1024": ((1, 1, 32, 1024), None, _ML.INTERLEAVED, "gamma", False, False),
    # ---- the per-channel operand DTYPE, on the focus geometry ---------------
    # The granularity option's LEGALITY question: TRIM == 3 is a PREFIX of the
    # tile, so it is only equal to TRIM == 2's two face-rows where the face
    # offset is a multiple of 64 B (D23's `legal_2`).  fp32 passes; bfloat8_b's
    # 272 B face does not, and must fall back to the half page.
    "F2304_gf32": ((1, 1, 8192, 2304), None, _ML.INTERLEAVED, "gamma", False, False, ttnn.float32),
    "F2304_gb8": ((1, 1, 8192, 2304), None, _ML.INTERLEAVED, "gamma", False, False, ttnn.bfloat8_b),
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

    spec = CASES[name]
    shape, shard, ml, mode, fp32_dest, w_rm = spec[:6]
    w_dtype = spec[6] if len(spec) > 6 else ttnn.bfloat16
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
    pc_lay = ttnn.ROW_MAJOR_LAYOUT if w_rm else lay

    def _vec(seed):
        torch.manual_seed(seed)
        t = torch.randn(1, 1, 1, W, dtype=torch.float32)
        tq = t if w_dtype == ttnn.float32 else t.to(torch.bfloat16)
        v = ttnn.from_torch(tq, dtype=w_dtype, layout=pc_lay, device=device)
        # bfloat8_b quantizes: compare against what the DEVICE actually holds, so
        # a granularity change is the only thing the pcc gate can catch.
        ref_t = ttnn.to_torch(v).float().reshape(1, 1, 1, W) if w_dtype == ttnn.bfloat8_b else tq.float()
        return ref_t, v

    if "gamma" in mode:
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
    run, expected, live = build(device, name)
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
    ns = min(samples) if samples else float("nan")
    for t in live:
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return ns, p, r


def sweep(labels, names=None, device=None):
    names = names or list(CASES)
    keep = (PD.KERNEL_DIR, PD.PER_CHANNEL_TRIM_GAMMA, PD.PER_CHANNEL_TRIM_BIAS)
    RES = {}
    own_device = device is None
    if own_device:
        device = ttnn.open_device(device_id=0)
    try:
        for _rep in range(N_REPS):
            for label in labels:
                kd, tg, tb = VARIANTS[label]
                PD.KERNEL_DIR = keep[0] if kd is None else kd
                PD.PER_CHANNEL_TRIM_GAMMA = tg
                PD.PER_CHANNEL_TRIM_BIAS = tb
                try:
                    for name in names:
                        ns, p, r = measure(device, name)
                        RES.setdefault((name, label), []).append((ns, p, r))
                        print(f"RESULT RAW {name:18s} {label:18s} {ns:12.0f} ns  pcc {p:.6f}", flush=True)
                finally:
                    PD.KERNEL_DIR, PD.PER_CHANNEL_TRIM_GAMMA, PD.PER_CHANNEL_TRIM_BIAS = keep
        base_label = labels[0]
        print("RESULT " + f"{'case':20s}" + "".join(f"{l:>18s}" for l in labels))
        for name in names:
            row = f"{name:20s}"
            for label in labels:
                row += f"{min(x[0] for x in RES[(name, label)]):18.0f}"
            print("RESULT " + row)
        print("RESULT --- speedup vs " + base_label + " (>1 = faster) ---")
        for name in names:
            b = min(x[0] for x in RES[(name, base_label)])
            row = f"{name:20s}"
            for label in labels:
                row += f"{b / min(x[0] for x in RES[(name, label)]):18.3f}"
            print("RESULT " + row)
        print("RESULT --- accuracy (worst pcc / worst rel-RMS) ---")
        for name in names:
            for label in labels:
                pmin = min(x[1] for x in RES[(name, label)])
                rmax = max(x[2] for x in RES[(name, label)])
                print(f"RESULT ACC {name:20s} {label:18s} pcc {pmin:.6f}  relrms {rmax:.3e}")
    finally:
        if own_device:
            ttnn.close_device(device)
    return RES
