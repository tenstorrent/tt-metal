"""passb_fusion — isolated A/B of pass B's DEST-window fusion.

VARIANTS (all three kernels come from a private dir; only the COMPUTE kernel
differs, and only in pass B):
  base      the shipped op, byte-for-byte              (k_base)
  fuse      normalize x gamma [+ bias] in ONE DEST window, one pack (k_fuse)
  fuse_ng   normalize x gamma fused, bias its own pass (k_fuse_ng)

The precision contract is FROZEN and identical across variants: every case pins
math_fidelity=HiFi2, math_approx_mode=False and its own fp32_dest_acc_en, and no
dtype changes.

Run:
  RMS_NAMES=focus scripts/tt-probe.sh rms_norm_ttnn <<'PY'
  import sys; sys.path.insert(0, "<this dir>"); import bench; bench.main()
  PY
"""

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

N_WARMUP = 2
N_TRIALS = int(os.environ.get("RMS_TRIALS", "5"))
N_REPS = int(os.environ.get("RMS_REPS", "2"))

# name: shape, shard|None, memory_layout, mode, fp32_dest, recorded_baseline_ns
CASES = {
    # ---- THE FOCUS SHAPE -------------------------------------------------
    "focus": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "gamma", False, 5408),
    # ---- domain sweep ----------------------------------------------------
    "w1024": ((1, 1, 32, 1024), ([32, 128], (8, 1)), _ML.WIDTH_SHARDED, "gamma", False, 3504),
    "w5120": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "gamma", False, 4601),
    "w5120_gbr32": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "gamma_bias_residual", True, 6076),
    "blk8192": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, "gamma", False, 22961),
    "blk7168_gbr": ((1, 1, 7168, 1024), ([896, 128], (8, 8)), _ML.BLOCK_SHARDED, "gamma_bias_residual", False, 32346),
    "int8192x1024": ((1, 1, 8192, 1024), None, _ML.INTERLEAVED, "gamma", False, 86992),
    "int8192x7168": ((1, 1, 8192, 7168), None, _ML.INTERLEAVED, "gamma", False, 578937),
    # ---- the fusion has nothing to fuse ----------------------------------
    "focus_nogamma": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "no_gamma", False, 0),
    # ---- extra: bias present on the focus geometry (option (b) vs (a)) ---
    "focus_gb": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "gamma_bias", False, 0),
    # ---- extra: STREAM regime (pass B re-reads x) ------------------------
    "stream_gbr": ((1, 1, 1024, 16384), None, _ML.INTERLEAVED, "gamma_bias_residual", False, 0),
    # ---- extra: interleaved small-W (PASS_B_BLK clamps to 1) -------------
    "int8192x256": ((1, 1, 8192, 256), None, _ML.INTERLEAVED, "gamma", False, 0),
}

DEFAULT_NAMES = ["focus"]


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

    shape, shard, ml, mode, fp32_dest, base_ns = CASES[name]
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
    return (lambda: rms_norm_ttnn(x, **kwargs)), expected, base_ns, live


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

    run, expected, base_ns, live = build(device, name)
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


def main():
    names = os.environ.get("RMS_NAMES", ",".join(DEFAULT_NAMES)).split(",")
    names = [n.strip() for n in names if n.strip()]
    want = [v for v in os.environ.get("RMS_VARIANTS", "base,fuse").split(",") if v]
    # "" (empty) = the op's own pass_b_blk choice
    blks = os.environ.get("RMS_BLKS", "").split(",")
    combos = [(v, b) for b in blks for v in want]

    saved_dir = PD.KERNEL_DIR
    saved_pc = PD._PC_NONE

    RES = {}
    device = ttnn.open_device(device_id=0)
    try:
        for rep in range(N_REPS):
            for label, blk in combos:
                PD.KERNEL_DIR = HERE / f"k_{label}"
                PD._PC_NONE = saved_pc._replace(subblock_w=int(blk)) if blk else saved_pc
                for name in names:
                    ns, p, r = measure(device, name)
                    RES.setdefault((name, label, blk), []).append((ns, p, r))
                PD.KERNEL_DIR = saved_dir
                PD._PC_NONE = saved_pc
    finally:
        PD.KERNEL_DIR = saved_dir
        PD._PC_NONE = saved_pc
        ttnn.close_device(device)

    cols = [f"{v}@{b or 'auto'}" for v, b in combos]
    print("RESULT " + f"{'case':16s}" + "".join(f"{c:>14s}" for c in cols))
    for name in names:
        row = f"{name:16s}" + "".join(f"{min(x[0] for x in RES[(name, v, b)]):14.0f}" for v, b in combos)
        print("RESULT " + row)
    v0, b0 = combos[0]
    print(f"RESULT --- speedup vs {cols[0]} (>1 = faster) ---")
    for name in names:
        ref = min(x[0] for x in RES[(name, v0, b0)])
        row = f"{name:16s}" + "".join(f"{ref / min(x[0] for x in RES[(name, v, b)]):14.3f}" for v, b in combos)
        print("RESULT " + row)
    print("RESULT --- pcc / rel-RMS vs torch fp32 ---")
    for name in names:
        for (v, b), c in zip(combos, cols):
            pc = min(x[1] for x in RES[(name, v, b)])
            rr = max(x[2] for x in RES[(name, v, b)])
            print(f"RESULT {name:16s} {c:14s} pcc={pc:.7f} relrms={rr:.6f}")
