# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Step 1 — the TRUE isolated payload of rms_norm's `cp_rms_chain`.

    scripts/run_safe_pytest.sh --profile \
        ttnn/ttnn/operations/rms_norm/perf_experiments/sfpu_col0_chain/test_micro_chain.py -s --run-all

Correctness (column 0 vs torch) is the only assertion; perf is measured.
"""

from __future__ import annotations

import os
import statistics

import pytest
import ttnn

from ttnn.operations.rms_norm.perf_experiments.sfpu_col0_chain import micro_descriptor as M

TILE = 32
INV_W = 1.0 / 7168.0
EPS = 1e-6
ZONE = "RMS_CHAIN"

_DEVICE_CSV = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated/profiler/.logs/profile_log_device.csv")
_RISC_LABEL = {"TRISC_0": "unpack", "TRISC_1": "math", "TRISC_2": "pack"}


def _loose_cfg():
    """The focus case's EXACT precision corner — frozen, identical for every arm."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def _default_cfg():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


CFGS = {"loose": _loose_cfg, "default": _default_cfg}


def _make_input(device):
    """A REDUCE_ROW-shaped tile: 32 per-row sums-of-squares in COLUMN 0, zeros elsewhere."""
    import torch

    torch.manual_seed(0)
    t = torch.zeros((TILE, TILE), dtype=torch.float32)
    t[:, 0] = torch.rand(TILE) * 6000.0 + 500.0  # plausible sum(x^2) over W=7168
    dev = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=M.create_sharded_memory_config()
    )
    return dev, t


# --- per-TRISC zone timing ----------------------------------------------------
def _read_rows(path):
    with open(path) as f:
        lines = f.read().splitlines()
    freq = 1000.0
    for part in lines[0].split(","):
        if "CHIP_FREQ" in part:
            freq = float(part.split(":")[1])
    rows = [[x.strip() for x in ln.split(",")] for ln in lines[2:] if ln.strip()]
    return [r for r in rows if len(r) >= 12], (1000.0 / freq), freq


def _all_run_ids(path):
    if not os.path.exists(path):
        return set()
    rows, _, _ = _read_rows(path)
    return {r[7] for r in rows}


def _zone_engines(path, seen):
    rows, ns_per_cycle, _ = _read_rows(path)
    starts, ends = {}, {}
    for r in rows:
        risc, cyc, run_id, zone, typ = r[3], r[5], r[7], r[10], r[11]
        if run_id in seen or zone != ZONE:
            continue
        (starts if typ == "ZONE_START" else ends).setdefault(risc, []).append(int(cyc))
    out = {}
    for risc, s in starts.items():
        s.sort()
        e = sorted(ends.get(risc, []))
        d = [(ee - ss) * ns_per_cycle for ss, ee in zip(s, e)]
        if d:
            out[_RISC_LABEL.get(risc, risc)] = statistics.median(d)
    return out


def _measure(device, run_fn, reps, trials):
    run_fn()
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    samples = {}
    for _ in range(trials):
        seen = _all_run_ids(_DEVICE_CSV)
        run_fn()
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
        for eng, ns in _zone_engines(_DEVICE_CSV, seen).items():
            samples.setdefault(eng, []).append(ns / reps)
    return samples


def _check(out, ref_in, variant, cfg_name):
    """Column 0 must equal rsqrt(x/W + eps).  Other columns are DON'T-CARE by design."""
    import torch

    got = torch.Tensor(ttnn.to_torch(out)).to(torch.float32).reshape(TILE, TILE)
    want = torch.rsqrt(ref_in[:, 0].to(torch.float32) * INV_W + EPS)
    if variant == "none":
        return 0.0  # identity copy: nothing computed, not a candidate
    rel = ((got[:, 0] - want).abs() / want.abs()).max().item()
    assert torch.isfinite(got[:, 0]).all(), f"{cfg_name}/{variant}: non-finite column 0"
    assert rel < 5e-2, f"{cfg_name}/{variant}: column-0 rel err {rel:.4g}"
    return rel


@pytest.mark.timeout(2400)
def test_micro_chain(device):
    reps = int(os.environ.get("SC_REPS", "2000"))
    trials = int(os.environ.get("SC_TRIALS", "3"))
    cfg_names = (os.environ.get("SC_CFGS") or "loose,default").split(",")
    variants = (os.environ.get("SC_VARIANTS") or ",".join(M.VARIANTS)).split(",")

    x, ref_in = _make_input(device)
    print()
    for cfg_name in cfg_names:
        cfg = CFGS[cfg_name]()
        # correctness gate at reps=1 for every measured cell
        rels = {}
        for v in variants:
            out = M.run_micro(x, variant=v, inv_w=INV_W, eps=EPS, reps=1, compute_config=cfg)
            rels[v] = _check(out, ref_in, v, cfg_name)

        perf = {}
        for v in variants:
            perf[v] = _measure(
                device,
                lambda vv=v: M.run_micro(x, variant=vv, inv_w=INV_W, eps=EPS, reps=reps, compute_config=cfg),
                reps,
                trials,
            )

        base = statistics.median(perf["chain_rc"]["math"]) if "chain_rc" in perf else float("nan")
        print(
            f"\n=== isolated rms-chain, MATH (TRISC_1) ns per chain call — cfg={cfg_name} "
            f"reps={reps} trials={trials} ==="
        )
        print(
            f"  {'variant':<13} {'vec':>4} {'ns/call':>9} {'vs base':>8} {'ns/vec':>7} "
            f"{'unpack':>7} {'pack':>6}  {'col0 relerr':>11}  desc"
        )
        for v in variants:
            s = perf[v]
            ns = statistics.median(s["math"])
            up = statistics.median(s.get("unpack", [0.0]))
            pk = statistics.median(s.get("pack", [0.0]))
            nv = M.VEC_OPS[v]
            print(
                f"  {v:<13} {nv:>4} {ns:>9.1f} {base / ns if ns else 0:>7.2f}x "
                f"{(ns / nv if nv else 0):>7.2f} {up:>7.2f} {pk:>6.2f}  {rels[v]:>11.3e}  {M.DESC[v]}"
            )
