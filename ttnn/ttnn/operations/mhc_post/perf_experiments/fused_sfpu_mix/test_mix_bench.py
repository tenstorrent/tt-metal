# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness (bit-exactness vs the baseline) + device zone timing of the mix_block bench.

    MIXB_VARIANTS=baseline,fused MIXB_PASSES=4 scripts/run_safe_pytest.sh \
        ttnn/ttnn/operations/mhc_post/perf_experiments/fused_sfpu_mix/test_mix_bench.py -s
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import sys
from pathlib import Path

import ttnn

sys.path.insert(0, str(Path(__file__).parent))
import bench  # noqa: E402

_CSV = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated/profiler/.logs/profile_log_device.csv")


def _rows():
    with open(_CSV) as f:
        lines = f.read().splitlines()
    freq = 1000.0
    for part in lines[0].split(","):
        if "CHIP_FREQ" in part:
            freq = float(part.split(":")[1])
    rows = [[x.strip() for x in ln.split(",")] for ln in lines[2:] if ln.strip()]
    return [r for r in rows if len(r) >= 12], 1000.0 / freq


def _run_ids():
    if not os.path.exists(_CSV):
        return set()
    return {r[7] for r in _rows()[0]}


def _zone(seen, name="MIX"):
    rows, nspc = _rows()
    st, en = {}, {}
    for r in rows:
        risc, cyc, rid, zone, typ = r[3], int(r[5]), r[7], r[10], r[11]
        if rid in seen or zone != name:
            continue
        (st if typ == "ZONE_START" else en).setdefault(risc, []).append(cyc)
    per = {k: (max(en[k]) - min(st[k])) * nspc for k in st if k in en}
    span = (max(max(v) for v in en.values()) - min(min(v) for v in st.values())) * nspc if st else None
    return per, span


def _ulp_f32(a, b):
    import torch

    ai = a.contiguous().view(torch.int32).long()
    bi = b.contiguous().view(torch.int32).long()
    ai = torch.where(ai < 0, -(ai & 0x7FFFFFFF), ai)
    bi = torch.where(bi < 0, -(bi & 0x7FFFFFFF), bi)
    return (ai - bi).abs().max().item()


def _sel(name, default):
    return [s for s in os.environ.get(name, default).split(",") if s]


def test_mix_bench(device):
    import torch

    passes = int(os.environ.get("MIXB_PASSES", "4"))
    B = int(os.environ.get("MIXB_B", "4"))
    has_w = os.environ.get("MIXB_PLAIN", "0") != "1"
    fp32_io = os.environ.get("MIXB_FP32", "0") == "1"
    units = passes * B
    variants = _sel("MIXB_VARIANTS", "baseline,fused")
    F, X, post, comb = bench.make_inputs(units, f_dtype=torch.float32 if fp32_io else torch.bfloat16)
    gA, gD = bench.golden(F, X, post, comb, units, B)
    res = {}
    for v in variants:
        seen = _run_ids()
        out, bf, dsq = bench.run(device, v, F, X, post, comb, units, B, has_w=has_w, fp32_io=fp32_io)
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
        per, span = _zone(seen)
        o = ttnn.to_torch(out).float()
        b = ttnn.to_torch(bf).float()
        d = ttnn.to_torch(dsq).float()
        res[v] = (o, b, d, per, span)
        gbf = gA if fp32_io else gA.to(torch.bfloat16).float()
        print(
            f"\n[{v}] B={B} has_w={has_w} fp32_io={fp32_io} passes={passes} zone span {span:.0f} ns = {span / passes:.0f} ns/pass = {span / units:.0f} ns/unit;"
            f" per-RISC { {k: round(x) for k, x in per.items()} }"
        )
        print(
            f"[{v}] vs fp64-fma golden: out bf16 mismatches {(o != gbf).sum().item()} (max |d| {(o - gbf).abs().max().item():.3g}),"
            f" bf==out {torch.equal(o, b)}, dsq max ulp {_ulp_f32(d, gD)}"
        )
    if "baseline" in res:
        o0, b0, d0 = res["baseline"][:3]
        for v in variants:
            if v == "baseline" or v.startswith("abl_"):
                continue
            o, b, d = res[v][:3]
            eq_o, eq_b = torch.equal(o, o0), torch.equal(b, b0)
            ulp = _ulp_f32(d, d0)
            rel = ((d - d0).abs() / d0.abs().clamp_min(1e-30)).max().item()
            print(
                f"[{v}] vs baseline: X' out bit-exact {eq_o}, X' bf16 bit-exact {eq_b}, dsq max ulp {ulp} (max rel {rel:.3g})"
            )
            assert eq_o, f"{v}: X' differs from baseline"
            if has_w:
                assert eq_b, f"{v}: X' bf16 differs from baseline"
                assert rel < 1e-5, f"{v}: dsq deviates from baseline beyond reordering"
