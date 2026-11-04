# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""pv_rescale_fusion: isolated bake-off of the P.V matmul + online-softmax correction of ring-joint SDPA.

  PVR_CASES=focus,focus_large,...  (default: focus,focus_large)   PVR_VARIANTS=base,fused,l1acc
  PVR_OUT=results.json
  scripts/run_safe_pytest.sh --run-all <this file> -s

Single core, one attention unit walking n key blocks (resident L1 K/V slots), the op's compute kernel copied
(kernels/compute.cpp). Correctness (the only pass/fail): attention output vs a float64 torch reference over the
device's own (quantized) Q/K/V: PCC >= 0.99, and every candidate's rel-RMS within 1.1x (+0.005) of the baseline's
on the same inputs (baseline: rel-RMS <= 0.15). Every variant's precision is recorded. Perf: DEVICE KERNEL DURATION [ns] (one fresh run per variant) and per-zone medians.
"""
import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import importlib
import json
from pathlib import Path

torch = importlib.import_module("torch")
import ttnn
from loguru import logger

from ttnn.operations.ring_joint_scaled_dot_product_attention.perf_experiments.pv_rescale_fusion import bench  # noqa
from ttnn.operations.ring_joint_scaled_dot_product_attention import (
    ring_joint_scaled_dot_product_attention_program_descriptor as pd,
)

_KEY = "DEVICE KERNEL DURATION [ns]"
VARIANTS = os.environ.get("PVR_VARIANTS", "base,fused,l1acc").split(",")
OUT_JSON = Path(__file__).parent / os.environ.get("PVR_OUT", "results.json")
LOG = Path("generated/profiler/.logs/profile_log_device.csv")
FREQ_GHZ = 1.35

BF16, BF8, FP32 = ttnn.bfloat16, ttnn.bfloat8_b, True


def _case(
    dh=128,
    dv=None,
    n_kv_t=256,
    n_q_t=256,
    fp32=False,
    kv=BF16,
    nkb=32,
    R=None,
    mask_cols=24,
    ragged=0,
    joint=None,
    logit=1.0,
    fid=ttnn.MathFidelity.HiFi2,
):
    dv = dv or dh
    acc = ttnn.float32 if fp32 else ttnn.bfloat16
    Sq, Sk = pd.choose_block_extents(
        tensor_q_tiles=n_q_t,
        joint_q_tiles=0,
        tensor_kv_tiles=n_kv_t,
        b=1,
        hq=40,
        C=110,
        DHt=dh // 32,
        DVt=dv // 32,
        q_dtype=BF16,
        kv_dtype=kv,
        acc_dtype=acc,
    )
    valid = nkb * Sk - ragged
    segs = [(valid, mask_cols != 0)]
    if joint is not None:
        segs = [(valid, False), (joint, mask_cols != 0)]
    if fp32 and fid.value < pd.FP32_DEST_MIN_FIDELITY.value:
        fid = pd.FP32_DEST_MIN_FIDELITY  # the op's own floor for fp32 DEST
    return dict(
        dh=dh, dv=dv, Sq=Sq, Sk=Sk, fp32=fp32, kv=kv, segs=segs, R=R or Sq, mask_cols=mask_cols, logit=logit, fid=fid
    )


CASES = {
    "focus": _case(nkb=128),
    "focus_large": _case(nkb=128, logit=36.0),
    "focus32": _case(nkb=32),
    "fp32": _case(fp32=True),
    "fp32_large": _case(fp32=True, logit=36.0),
    "bf8kv": _case(kv=BF8),
    "dh64": _case(dh=64),
    "dh96": _case(dh=96),
    "dh256": _case(dh=256),
    "mla": _case(dh=192, dv=128),
    "small": _case(dh=64, n_q_t=4, n_kv_t=4, nkb=4, mask_cols=0),
    "ragged": _case(ragged=5),
    "tail4": _case(R=4),
    "tail1": _case(R=1),
    "joint": _case(joint=11),
    "fp32_tail2": _case(fp32=True, R=2, ragged=3),
}


def _kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    tot = None
    for progs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for p in progs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(_KEY)
            if e is not None:
                tot = (tot or 0.0) + float(e.duration)
    return tot


def _zones():
    """Median ns per zone (TRISC_1) of the latest run in the device CSV, if present."""
    try:
        import pandas as pd_

        df = pd_.read_csv(LOG, skiprows=1, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        run_col = "run host ID" if "run host ID" in df.columns else "run ID"
        df = df[df[run_col] == df[run_col].max()]
        df = df[df["RISC processor type"] == "TRISC_1"]
        t = "time[cycles since reset]"
        out = {}
        for zn, g in df[df["type"].isin(["ZONE_START", "ZONE_END"])].groupby("zone name"):
            if "KERNEL" in str(zn) or "FW" in str(zn):
                continue
            g = g.sort_values(t)
            st = g[g["type"] == "ZONE_START"][t].values
            en = g[g["type"] == "ZONE_END"][t].values
            n = min(len(st), len(en))
            if n:
                d = sorted((en[:n] - st[:n]) / FREQ_GHZ)
                out[str(zn)] = {"n": n, "median_ns": round(float(d[n // 2]), 0)}
        return out
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _inputs(c, gen):
    Sq, Sk, dh, dv = c["Sq"], c["Sk"], c["dh"], c["dv"]
    Q = torch.randn(Sq * 32, dh, generator=gen) * c["logit"]
    Q[c["R"] * 32 :] = 0
    Ks = [torch.randn(Sk * 32, dh, generator=gen) for _ in range(2)]
    Vs = [torch.randn(Sk * 32, dv, generator=gen) for _ in range(2)]
    return Q, Ks, Vs


def _reference(c, Qd, Ks, Vs, scale):
    Sk, R = c["Sk"], c["R"]
    keys, vals, mask = [], [], []
    kb = 0
    for valid, masked in c["segs"]:
        for t0 in range(0, valid, Sk):
            skv = min(Sk, valid - t0)
            keys.append(Ks[kb % 2][: skv * 32])
            vals.append(Vs[kb % 2][: skv * 32])
            m = torch.ones(skv * 32, dtype=torch.bool)
            if masked and t0 + skv == valid:
                m[(skv - 1) * 32 + c["mask_cols"] :] = False
            mask.append(m)
            kb += 1
    K = torch.cat(keys).double()
    V = torch.cat(vals).double()
    M = torch.cat(mask)
    S = (Qd[: R * 32].double() @ K.T) * scale
    S[:, ~M] = float("-inf")
    return torch.softmax(S, dim=-1) @ V


def _metrics(y, ref):
    y, ref = y.double().flatten(), ref.flatten()
    pcc = torch.corrcoef(torch.stack([y, ref]))[0, 1].item()
    rel = ((y - ref).norm() / ref.norm()).item()
    return {"pcc": round(pcc, 6), "rel_rms": round(rel, 5), "max_abs": round((y - ref).abs().max().item(), 4)}


def test_pv_rescale_fusion(device):
    cases = os.environ.get("PVR_CASES", "focus,focus_large").split(",")
    results = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    failures = []
    seeds = [int(x) for x in os.environ.get("PVR_SEEDS", "4242").split(",")]
    for name, seed in [(n, sd) for n in cases for sd in seeds]:
        c = CASES[name]
        gen = torch.Generator().manual_seed(seed)
        Q, Ks, Vs = _inputs(c, gen)
        Sq, Sk, DHt, DVt = c["Sq"], c["Sk"], c["dh"] // 32, c["dv"] // 32
        scale = c["dh"] ** -0.5
        scale_bf16 = torch.tensor(scale).to(torch.bfloat16).double().item()
        q_dev = bench.to_dev(Q, device, BF16)
        k_dev = bench.to_dev(torch.cat([bench.k_slot_layout(torch, k, Sk, DHt) for k in Ks]), device, c["kv"])
        v_dev = bench.to_dev(torch.cat(Vs), device, c["kv"])
        # reference over the device-quantized operands
        Qd = ttnn.to_torch(q_dev).float()
        kd = ttnn.to_torch(k_dev).float()
        Kd = [
            kd[s * DHt * 32 : (s + 1) * DHt * 32]
            .reshape(DHt, 32, Sk, 32)
            .permute(2, 1, 0, 3)
            .reshape(Sk * 32, DHt * 32)
            for s in range(2)
        ]
        vd = ttnn.to_torch(v_dev).float()
        Vd = [vd[s * Sk * 32 : (s + 1) * Sk * 32] for s in range(2)]
        ref = _reference(c, Qd, Kd, Vd, scale_bf16)
        res = results.setdefault(name if seed == 4242 else f"{name}@{seed}", {})
        res["shape"] = {k: (str(v) if not isinstance(v, (int, float, list, bool)) else v) for k, v in c.items()}
        for v in VARIANTS:
            out_dev = bench.to_dev(torch.zeros(Sq * 32, c["dv"]), device, BF16)
            _kernel_ns(device)  # flush
            device.clear_program_cache()  # generic_op caches on compile-time state; runtime-arg COUNTS differ per case
            nb = bench.run(
                device,
                variant=v,
                q_dev=q_dev,
                k_dev=k_dev,
                v_dev=v_dev,
                out_dev=out_dev,
                Sq=Sq,
                Sk=Sk,
                DHt=DHt,
                DVt=DVt,
                segments=c["segs"],
                R=c["R"],
                mask_cols=c["mask_cols"],
                scale=scale,
                fp32=c["fp32"],
                fidelity=c["fid"],
                q_dtype=BF16,
                kv_dtype=c["kv"],
            )
            ttnn.synchronize_device(device)
            ns = _kernel_ns(device)
            y = ttnn.to_torch(out_dev).float()[: c["R"] * 32]
            m = _metrics(y, ref)
            m.update(ns=ns, ns_per_block=round(ns / nb, 0) if ns else None, n_blocks=nb, zones=_zones())
            res[v] = m
            ttnn.deallocate(out_dev)
            logger.info(
                f"{name:12s} {v:6s} ns={ns} per_block={m['ns_per_block']} pcc={m['pcc']} rel={m['rel_rms']}"
                f" max_abs={m['max_abs']} zones={m['zones']}"
            )
            base_rel = res.get("base", {}).get("rel_rms")
            rel_gate = 0.15 if base_rel is None else 1.1 * base_rel + 0.005  # candidate: no worse than base
            if not (m["pcc"] >= 0.99 and m["rel_rms"] <= rel_gate) or not torch.isfinite(y).all():
                failures.append(f"{name}/{v}: {m}")
        OUT_JSON.write_text(json.dumps(results, indent=1))
        for t in (q_dev, k_dev, v_dev):
            ttnn.deallocate(t)
    assert not failures, "\n".join(failures)
