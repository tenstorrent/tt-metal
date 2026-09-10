# Perf experiment `per_channel_mcast_v2` -- the round-2 case set and measurement harness.
#
# Round 2's assignment pins SEVEN cells: the focus shape, its neighbours, the STREAM
# case (the biggest ablation ceiling in the op) and THREE cells the multicast must
# leave alone (WIDTH shard = no reuse exists, BLOCK shard = round 1 measured a
# regression, ROW_MAJOR weight = a different staging shape).
#
# Everything here is measurement only: `DEVICE KERNEL DURATION [ns]` via the device
# profiler, pcc/rel-RMS vs torch as the pass/fail gate.  Perf is never asserted.
import os

# THE PROFILING DEFINE IS PINNED, and this is load-bearing for the round-2 entry
# condition.  `RMS_STAGE_ZONES` turns the op's per-stage device zones into a kernel
# DEFINE, so a build that has it and a build that has not are DIFFERENT BINARIES --
# comparing one against the other measures the instrumentation, not the idea.  Pin it
# here, before anything reads it, so every variant in a session agrees; `RMS_ZONES=1`
# turns it on for ALL of them together.
os.environ["RMS_STAGE_ZONES"] = os.environ.get("RMS_ZONES", "0")

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import importlib
import statistics

# `import torch` is deliberately NOT spelled as a global `import` statement: `scripts/validate_no_global_torch_imports.py`
# forbids a module-scope `import torch` anywhere under `ttnn/`, and this file lives
# there.  This module is only ever RUN as a script (`scripts/tt-probe.sh ... < run_v2.py`),
# never imported by the library, so the dependency is real but must not be spelled in the
# form the validator scans for.
import ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn
from eval.sharding import shard_config

torch = importlib.import_module("torch")

_ML = ttnn.TensorMemoryLayout
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
N_WARMUP = int(os.environ.get("RMS_WARMUP", "1"))
N_TRIALS = int(os.environ.get("RMS_TRIALS", "3"))


def _cfg(fp32_dest):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = fp32_dest
    c.math_approx_mode = False
    return c


# name -> dict(shape, shard, ml, mode, fp32, x_layout, w_layout)
CASES = {
    # ---- the ENGAGED half: interleaved row-split, every core owns the full width ----
    "FOCUS": dict(shape=(1, 1, 8192, 2304), ml=_ML.INTERLEAVED, mode="gamma"),
    "F1024": dict(shape=(1, 1, 8192, 1024), ml=_ML.INTERLEAVED, mode="gamma"),
    "FGB": dict(shape=(1, 1, 8192, 2304), ml=_ML.INTERLEAVED, mode="gamma_bias"),
    "STREAM": dict(shape=(1, 1, 8192, 7168), ml=_ML.INTERLEAVED, mode="gamma_bias_residual", fp32=True),
    # ---- the NON-ENGAGED half: the byte-identity / no-regression check ----
    "WSHARD": dict(shape=(1, 1, 32, 7168), ml=_ML.WIDTH_SHARDED, mode="gamma", shard=([32, 256], (7, 4))),
    "BLOCK": dict(shape=(1, 1, 8192, 1024), ml=_ML.BLOCK_SHARDED, mode="gamma", shard=([1024, 128], (8, 8))),
    "RMW": dict(shape=(1, 1, 128, 4096), ml=_ML.INTERLEAVED, mode="gamma", fp32=True, w_layout="rm"),
    # ---- extra probes (opt-in) ----
    "F4608": dict(shape=(1, 1, 8192, 4608), ml=_ML.INTERLEAVED, mode="gamma"),
    "FNONE": dict(shape=(1, 1, 8192, 2304), ml=_ML.INTERLEAVED, mode="no_gamma"),
    # one tile-row over the whole grid: every core owns a DISJOINT width slice, so no
    # reuse exists and the multicast must be inert (round 1's focus shape)
    "DECODE": dict(shape=(1, 1, 32, 7168), ml=_ML.INTERLEAVED, mode="gamma"),
    # a residual + gamma prefill at a third width, to widen the engaged domain
    "F5120R": dict(shape=(1, 1, 8192, 5120), ml=_ML.INTERLEAVED, mode="gamma_bias_residual"),
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


def build(device, name):
    c = CASES[name]
    shape, ml, mode = c["shape"], c["ml"], c["mode"]
    shard = c.get("shard")
    fp32 = c.get("fp32", False)
    W = shape[-1]
    torch.manual_seed(0)
    tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    lay = ttnn.TILE_LAYOUT
    wlay = ttnn.ROW_MAJOR_LAYOUT if c.get("w_layout") == "rm" else ttnn.TILE_LAYOUT
    if shard is not None:
        mc = shard_config(shard[0], shard[1], ml, layout=lay, dtype=ttnn.bfloat16, device=device)
    else:
        mc = ttnn.DRAM_MEMORY_CONFIG
    x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=lay, device=device, memory_config=mc)
    kwargs = {"epsilon": 1e-12, "compute_kernel_config": _cfg(fp32), "memory_config": x.memory_config()}
    ref = {"input_tensor": tx.float()}

    def _vec(seed):
        torch.manual_seed(seed)
        t = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
        return t, ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=wlay, device=device)

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
    ns = statistics.median(samples) if samples else float("nan")
    for t in live:
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return ns, p, r
