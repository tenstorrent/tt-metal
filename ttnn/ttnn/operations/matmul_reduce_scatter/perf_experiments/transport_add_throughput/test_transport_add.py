# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness + device-ns bench for the transport-core fp32-DEST add variants.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/matmul_reduce_scatter/perf_experiments/transport_add_throughput/test_transport_add.py

Env knobs: TAT_VARIANTS=baseline,cand1  TAT_CASES="7:40,7:160,4:70"  (seg_tiles:num_segs)  TAT_NIN=2,3
TAT_REPS=1 (profiled launches per config; median reported).
Results: one JSON line per config appended to results.jsonl next to this file.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import importlib.util
import json
import socket
import statistics
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # test-only dependency (no global torch import inside ttnn/)
import ttnn
from loguru import logger

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("tat_bench", HERE / "bench.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)
_vspec = HERE / "variants.py"
if _vspec.exists():
    _s = importlib.util.spec_from_file_location("tat_variants", _vspec)
    _m = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(_m)
    _m.register_all(bench)

VARIANTS = os.environ.get("TAT_VARIANTS", "baseline").split(",")
CASES = [tuple(int(v) for v in c.split(":")) for c in os.environ.get("TAT_CASES", "7:40").split(",")]
NINS = [int(v) for v in os.environ.get("TAT_NIN", "2,3").split(",")]
REPS = int(os.environ.get("TAT_REPS", "1"))
TAG = os.environ.get("TAT_TAG", "")
COPY_SEGS = int(os.environ.get("TAT_COPY", "0"))  # must be a multiple of cap/seg_tiles (=8) to keep pages aligned
_KEY = "DEVICE KERNEL DURATION [ns]"


def _read_ns(device):
    ttnn.ReadDeviceProfiler(device)
    data = ttnn.get_latest_programs_perf_data()
    vals = []
    for programs in (data or {}).values():
        for p in programs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(_KEY)
            if e is not None:
                vals.append(float(e.duration))
    return vals


def _to_dev(t, device, cap):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=bench.mem_config(cap)
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n_in", NINS)
@pytest.mark.parametrize("case", CASES, ids=[f"seg{s}x{n}" for s, n in CASES])
def test_add(device, variant, n_in, case):
    seg_tiles, num_segs = case
    cap = bench.capacity(seg_tiles, 4)
    torch.manual_seed(1234)
    host = [(torch.randn(cap * 32, 32) * (1.0 + 3 * i)).to(torch.bfloat16) for i in range(2 if n_in == 22 else n_in)]
    ins = [_to_dev(h, device, cap) for h in host]
    out = _to_dev(torch.zeros(cap * 32, 32, dtype=torch.bfloat16), device, cap)
    _read_ns(device)  # flush
    samples = []
    for _ in range(REPS):
        bench.run(ins, out, variant=variant, n_in=n_in, seg_tiles=seg_tiles, num_segs=num_segs, num_copy_segs=COPY_SEGS)
        ttnn.synchronize_device(device)
        v = _read_ns(device)
        if v:
            samples.append(v[-1])
    got = ttnn.to_torch(out).float()
    exp32 = sum(h.float() for h in host)
    exp = exp32.to(torch.bfloat16).float()
    written = min(cap, seg_tiles * num_segs) * 32
    g, e, e32 = got[:written], exp[:written], exp32[:written]
    mism = int((g != e).sum())
    max_err = float((g - e32).abs().max())
    mag = sum(h.float().abs() for h in host)[:written]
    rel = float(((g - e32).abs() / (mag + 1e-6)).max())
    pcc = float(torch.corrcoef(torch.stack([g.flatten(), e32.flatten()]))[0, 1])
    ns = statistics.median(samples) if samples else None
    tiles = seg_tiles * num_segs
    rec = dict(
        tag=TAG,
        variant=variant,
        n_in=n_in,
        seg_tiles=seg_tiles,
        num_segs=num_segs,
        copy_segs=COPY_SEGS,
        tiles=tiles,
        ns=ns,
        samples=samples,
        ns_per_tile=(ns / tiles if ns else None),
        mismatch_vs_bf16_rounded=mism,
        max_abs_err_vs_fp32=max_err,
        max_rel_err=rel,
        pcc=pcc,
        box=socket.gethostname(),
        arch=str(device.arch()),
    )
    with open(HERE / "results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    logger.info(f"TAT {json.dumps(rec)}")
    # correctness gate: never worse than half-ulp-ish of the bf16 rounding of the fp32 sum.
    # (the baseline 3-input path rounds its intermediate through srcA; record mismatches, gate on PCC + max err)
    if "skipcompute" in variant or "nopack" in variant:
        return
    assert pcc > 0.9999, f"pcc {pcc}"
    assert rel < 0.02, f"max rel err {rel}"
