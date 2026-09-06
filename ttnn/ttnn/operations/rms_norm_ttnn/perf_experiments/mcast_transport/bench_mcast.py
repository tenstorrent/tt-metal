# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""mcast_transport -- isolated bake-off for the stat multicast's TRANSPORT.

ONE stage, held in isolation: `writer_mcast_send` on the combine's root (and the
matching `writer_mcast_recv` on every member).  Everything else in the op is held
byte-identical -- the kernels here are a scratch COPY of the shipped ones whose
only difference is an `MCV` switch inside that one block -- so the measured delta
is attributable to the transport alone.

Variants (see kernels/rms_norm_ttnn_writer.cpp's MCV header):
  baseline        MCV=0   the shipped local-publish + acked-barrier + push, then
                          EXCLUDE mcast + flag + async_writes_flushed().
  caller_managed  MCV=2   the same, minus the trailing flush; the source-L1 guard
                          is paid at the end of the kernel instead (in slack).
  loopback        MCV=3   no local publish; ONE MCAST_INCL_SRC transaction.
  abl_no_local    MCV=4   TIMING ABLATION, numerically wrong by construction.
  faces4          MCV=0 + COMBINE_MCAST_FACES=0 (whole 4096 B tile, not 3072 B).

  one_packet_local MCV=6  the LOCAL publish as noc_async_write<NOC_MAX_BURST_SIZE>.
  raw_one_packet  MCV=7   hand-written one-packet mcast + flag (bypasses SenderPipe).
  abl_no_transport MCV=5  TIMING ABLATION: the broadcast and every receive() DELETED.
                          The CEILING of this whole idea.

Run:  MCT_DIR=<this dir> scripts/tt-probe.sh rms_norm_ttnn <<'PYEOF'
      import os; exec(open(os.environ["MCT_DIR"] + "/bench_mcast.py").read())
      PYEOF
Env:  MCT_CASES=focus,w1024_g8   MCT_VARIANTS=baseline,caller_managed
      MCT_ZONES=1  (adds the writer_mcast_send sub-zones; attribution only)

===========================================================================
MEASURED (blackhole p150b @ 1350 MHz, in-process profiler, DEVICE KERNEL
DURATION [ns], best of 3 reps of median-of-5, fresh device per run)
===========================================================================
FOCUS (1,1,32,7168) WIDTH [32,256] (7,4) G=28, HiFi2, fp32_dest=False:

  baseline           5324 ns   1.000x   pcc 0.9999848  relrms 6.276e-03
  one_packet_local   5326 ns   1.001x   pcc 0.9999848  (NULL)
  caller_managed     5338 ns   0.997x   pcc 0.9999848  (NULL)
  faces4 (4096 B)    5375 ns   0.990x   pcc 0.9999848
  raw_one_packet     5380 ns   0.991x   pcc 0.9999848
  cm_faces4          5406 ns   0.985x   pcc 0.9999848
  loopback           5437 ns   0.979x   pcc 0.9999848
  ---- ablations (numerically invalid by construction) ----
  abl_no_local       5259 ns   1.017x   -- delete the root's local publish
  abl_no_transport   5101 ns   1.048x   -- delete the broadcast ENTIRELY  <== CEILING

Baseline run-to-run spread on the focus shape across every session: 5324..5382
(1.1%).  Anything inside +-1% is NOISE.

CEILING (abl_no_transport vs baseline) across the domain:
  focus       5347 -> 5101   1.048x   (246 ns)
  w1024_g8    3429 -> 3181   1.078x   (248 ns)
  w2304_g9    4174 -> 3897   1.071x   (277 ns)
  w5120_g32   4519 -> 4564   0.990x   (no headroom)
  w5120_gbr   5948 -> 5953   0.999x   (no headroom)
  blk8192    22910 -> 21874  1.047x  (1036 ns, many rounds)
  blk7168_gbr 32220 -> 30514 1.056x  (1706 ns, many rounds)

ZONE ATTRIBUTION of `writer_mcast_send` (zoned build, root core 1,2, focus):
  writer_mcast_send  3684 -> 4901   1216 ns   of which
      cb_wait_front(cb_stat_handoff)  ~810 ns  <- the ROOT'S OWN FOLD, not transport
      mcs_local (publish + acked bar)  108 ns
      mcs_issue (mcast + flag + flush) 234 ns
  Corroborated by the unzoned ablations: abl_no_transport removes 246 ns
  (~= mcs_issue 234), abl_no_local removes 66..88 ns (~= mcs_local 108).
  So of the 582 ns the shipped stage table attributes to `writer_mcast_send`,
  ~330 ns is transport and ~250 ns is the root waiting for its own finalize.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

ZONES = os.environ.get("MCT_ZONES", "0") not in ("", "0")
if ZONES:
    os.environ["RMS_STAGE_ZONES"] = "1"

import statistics
from pathlib import Path

import ttnn

from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD
from eval.sharding import shard_config

_ML = ttnn.TensorMemoryLayout
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

HERE = Path(os.environ["MCT_DIR"]) if os.environ.get("MCT_DIR") else Path(__file__).resolve().parent
N_WARMUP = 2
N_TRIALS = int(os.environ.get("MCT_TRIALS", "5"))
N_REPS = int(os.environ.get("MCT_REPS", "3"))

# ---------------------------------------------------------------------------
# Point the shipped descriptor at THIS experiment's scratch kernels, and give it
# a per-variant `-D MCV=` define.  Nothing under the op's own kernels/ is touched.
# ---------------------------------------------------------------------------
PD.KERNEL_DIR = HERE / "kernels"
_MCV = 0
_MCZ = 1 if ZONES else 0


def _defines():
    d = [("MCV", str(_MCV)), ("MCZ", str(_MCZ))]
    if PD.STAGE_ZONES:
        d.append(("RMS_STAGE_ZONES", "1"))
    return d


PD._kernel_defines = _defines

# label -> (MCV, {PD attr: value})
VARIANTS = {
    "baseline": (0, {}),
    "base_zoned": (1, {}),
    "caller_managed": (2, {}),
    "loopback": (3, {}),
    "abl_no_local": (4, {}),
    "faces4": (0, {"COMBINE_MCAST_FACES": 0}),
    "cm_faces4": (2, {"COMBINE_MCAST_FACES": 0}),
    "abl_no_transport": (5, {}),
    "one_packet_local": (6, {}),
    "raw_one_packet": (7, {}),
}

# name -> (shape, shard, memory_layout, mode, fp32_dest)
CASES = {
    # THE FOCUS SHAPE -- WIDTH shard, G=28, ONE combine round, identity path.
    "focus": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "gamma", False),
    # other combine geometries (domain sweep)
    "w1024_g8": ((1, 1, 32, 1024), ([32, 128], (8, 1)), _ML.WIDTH_SHARDED, "gamma", False),
    "w2304_g9": ((1, 1, 32, 2304), ([32, 256], (9, 1)), _ML.WIDTH_SHARDED, "gamma", False),
    "w5120_g32": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "gamma", False),
    "w5120_gbr": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "gamma_bias_residual", True),
    # BLOCK shards: MANY combine rounds, COMPACT path (whole-tile multicast)
    "blk8192": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, "gamma", False),
    "blk7168_gbr": ((1, 1, 7168, 1024), ([896, 128], (8, 8)), _ML.BLOCK_SHARDED, "gamma_bias_residual", False),
    # a NON-combine guard: must be byte-identical under every variant
    "int7168": ((1, 1, 8192, 7168), None, _ML.INTERLEAVED, "gamma", False),
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
    import torch  # function-local: ttnn/ttnn may not import torch globally (pre-commit)

    shape, shard, ml, mode, fp32_dest = CASES[name]
    W = shape[-1]
    torch.manual_seed(0)
    tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    lay = ttnn.TILE_LAYOUT
    mc = (
        shard_config(shard[0], shard[1], ml, layout=lay, dtype=ttnn.bfloat16, device=device)
        if shard is not None
        else ttnn.DRAM_MEMORY_CONFIG
    )
    x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=lay, device=device, memory_config=mc)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = fp32_dest
    cfg.math_approx_mode = False
    kwargs = {"epsilon": 1e-12, "compute_kernel_config": cfg, "memory_config": x.memory_config()}
    ref = {"input_tensor": tx.float()}

    def _vec(seed):
        torch.manual_seed(seed)
        t = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
        return t, ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=lay, device=device)

    if "gamma" in mode:
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
    global _MCV
    names = [n.strip() for n in os.environ.get("MCT_CASES", "focus").split(",") if n.strip()]
    labels = [v.strip() for v in os.environ.get("MCT_VARIANTS", "baseline,caller_managed").split(",") if v.strip()]
    RES = {}
    device = ttnn.open_device(device_id=0)
    saved = {}
    try:
        for rep in range(N_REPS):
            for label in labels:
                mcv, knobs = VARIANTS[label]
                _MCV = mcv
                for k, v in knobs.items():
                    saved.setdefault(k, getattr(PD, k))
                    setattr(PD, k, v)
                for name in names:
                    ns, p, r = measure(device, name)
                    RES.setdefault((name, label), []).append((ns, p, r))
                    print(f"RESULT rep{rep} {name:14s} {label:15s} {ns:9.0f} ns  pcc {p:.6f}  relrms {r:.3e}")
                for k, v in saved.items():
                    setattr(PD, k, v)
        base = labels[0]
        print("RESULT " + "=" * 78)
        hdr = f"{'case':14s}" + "".join(f"{l:>16s}" for l in labels)
        print("RESULT " + hdr)
        for name in names:
            row = f"{name:14s}"
            for label in labels:
                row += f"{min(x[0] for x in RES[(name, label)]):16.0f}"
            print("RESULT " + row)
        print("RESULT --- speedup vs " + base + " (>1 = faster) ---")
        for name in names:
            b = min(x[0] for x in RES[(name, base)])
            row = f"{name:14s}"
            for label in labels:
                row += f"{b / min(x[0] for x in RES[(name, label)]):16.4f}"
            print("RESULT " + row)
        print("RESULT --- pcc / relrms (worst over reps) ---")
        for name in names:
            for label in labels:
                p = min(x[1] for x in RES[(name, label)])
                r = max(x[2] for x in RES[(name, label)])
                print(f"RESULT {name:14s} {label:15s} pcc {p:.7f}  relrms {r:.4e}")
    finally:
        ttnn.close_device(device)
    return RES


main()
