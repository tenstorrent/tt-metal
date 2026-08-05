# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness + on-device measurement for the rms_norm FINALIZE bake-off.

Correctness is the ONLY pass/fail.  Perf is measured and logged, never asserted.

  test_finalize_correctness   structural mode, every (variant x handoff x rows) cell:
                              PCC + rel-RMS on the lanes the variant leaves valid,
                              plus BIT-EQUALITY of column 0 against the op's current
                              approach for every non-fused variant (the proof that
                              scoping does not change the values that ARE read).
  test_stale_lane_inventory   logs, per variant, exactly what each stale lane holds
                              (raw sum(x^2) / touched-but-not-rsqrt'd / finalized) and
                              asserts none of them is inf/NaN.
  test_consumer_domain        runs pass B's REAL consumer
                              (BinaryFpu<x, stat, Mul, BroadcastDim::Col>) on top of
                              each variant's stat tile, with every column but 0 seeded
                              WRONG.  Passing PCC == the consumer provably reads only
                              column 0 == the scope is safe in the op.
  test_perf_isolated          MATH-thread-only SFPU cost per finalize call.
  test_perf_structural        the op's stage sequence: per-stage zone ns per tile,
                              copy/pack and CB handshake INCLUDED, rows in {1,2,10,32}.
"""

# NOTE ON THE PROFILER ENV: this module is auto-executed by
# `ttnn/ttnn/operations/__init__.py` (pkgutil.walk_packages) on every `import ttnn`, so it
# must have NO import-time side effects -- setting TT_METAL_* here would silently change
# the profiling behaviour of every other test in the repo, including a sibling
# experiment's measurements.  Pass the profiler env on the command line instead:
#
#   TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
#   TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
#   scripts/run_safe_pytest.sh <this file> -k test_perf_...
#
# The two perf tests report "no zones captured" instead of silently printing zeros when
# the env is missing.
import os
import math
import socket
import statistics
import struct

import pytest
import ttnn
from loguru import logger

from ttnn.operations.rms_norm.perf_experiments.root_finalize_scope.finalize_bench import (
    BASELINE,
    BASELINE_HANDOFF,
    HANDOFFS,
    NEEDS_POW4_W,
    PASSES,
    VALID_COLS,
    VARIANTS,
    VEC_OPS,
    ZONE_FINALIZE,
    ZONE_HANDOFF,
    ZONE_SFPU,
    create_sharded_memory_config,
    run_op,
)

TILE = 32


def _sel(env, allowed):
    """RFS_VARIANTS / RFS_HANDOFFS / RFS_ROWS narrow the sweep while iterating."""
    v = os.environ.get(env)
    if not v:
        return tuple(allowed)
    want = set(v.split(","))
    bad = want - set(str(a) for a in allowed)
    if bad:
        raise ValueError(f"unknown {env}: {sorted(bad)}; valid: {allowed}")
    return tuple(a for a in allowed if str(a) in want)


# The focus case's configuration: W = 1024 (the (1,1,8192,1024) target), the op's eps.
W = 1024
EPS = 1e-12
ROWS_SWEEP = (1, 2, 10, 32)  # 10 == the focus shape's BLOCK_ROWS; 32 == a core's whole assignment

# The focus case's SOFT gates.
PCC_GATE = 0.9995
RELRMS_GATE = 0.04

# Non-fused variants apply the identical math to column 0, so column 0 must be
# BIT-IDENTICAL to the op's current approach.  The fused variants keep the two
# intermediates in an fp32 LREG instead of round-tripping through a 16-bit DEST word,
# so they differ (in the accurate direction) and are gated on PCC/rel-RMS instead.
_BIT_EXACT_VS_BASE = ("rc_all", "base", "scope_c", "cskip3")
_FUSED = ("cskip2", "fused_c", "cskip_fused")


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


INV_W_BITS = _bits(1.0 / W)
EPS_BITS = _bits(EPS)
EPS_W_BITS = _bits(EPS * W)
# log2(W)/2, i.e. the exponent add that multiplies by sqrt(W); -1 when W is not a power
# of four (the fully-fused variants then refuse to build -- an INEXPRESSIBLE, not a bug).
HALF_LOG2_W = None  # set below (needs _half_log2)


# =============================================================================
# Inputs / golden
# =============================================================================
def _stat_torch(rows, seed=7):
    """A REDUCE_ROW-shaped stat block, fp32, `rows` tiles tall.

    Column 0 carries a realistic sum(x^2) over W bf16-ish normals (~W).  EVERY OTHER
    column carries a deliberately WRONG magnitude, so that a consumer reading any lane
    a scope leaves stale produces a visibly wrong answer instead of a lucky pass.
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    n = rows * TILE
    stat = torch.empty(n, TILE, dtype=torch.float32)
    # std 2 so sum(x^2)/W ~ 4 and rsqrt(4) ~ 0.5: that makes the three possible lane
    # states (raw ~4096 / touched ~4 / finalized ~0.5) unambiguously distinguishable.
    stat[:, 0] = ((2.0 * torch.randn(n, W, generator=g, dtype=torch.float32)) ** 2).sum(-1)
    decoy = 1.0e4 * (1.0 + torch.arange(1, TILE, dtype=torch.float32))  # 2e4 .. 3.2e5
    stat[:, 1:] = decoy.unsqueeze(0).expand(n, TILE - 1)
    return stat


def _finalize_golden(stat, w=None):
    """finalize(v) = rsqrt(v * (1/W) + eps), elementwise, in fp64."""
    import torch

    v = stat.to(torch.float64)
    return torch.rsqrt(v * (1.0 / (w or W)) + EPS)


def _half_log2(w):
    """log2(w)/2 -- the exponent add that multiplies by sqrt(w).  -1 when w is not a power
    of FOUR, which is exactly when the fully-fused bodies are inexpressible."""
    l2 = int(math.log2(w))
    return l2 // 2 if (2**l2 == w and l2 % 2 == 0) else -1


HALF_LOG2_W = _half_log2(W)


def _upload(device, t, dtype):
    import torch

    return ttnn.from_torch(
        t if dtype == ttnn.float32 else t.to(torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config(tuple(t.shape)),
    )


def _alloc(device, shape, dtype):
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, create_sharded_memory_config(tuple(shape))
    )


def _pcc(a, b):
    import torch

    a = a.to(torch.float64).flatten()
    b = b.to(torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a @ b) / d)


def _rel_rms(got, exp):
    import torch

    got = got.to(torch.float64)
    exp = exp.to(torch.float64)
    return float(((got - exp) ** 2).mean().sqrt() / (exp**2).mean().sqrt())


# =============================================================================
# Device runners
# =============================================================================
def _run_structural(device, stat_t, rows, variant, handoff):
    """Fresh stat upload each launch (the in-place variants MUTATE it), returns the
    handoff CB's contents as a torch fp32 tensor."""
    stat = _upload(device, stat_t, ttnn.float32)
    out = _alloc(device, (rows * TILE, TILE), ttnn.float32)
    res = run_op(
        stat,
        out,
        mode="structural",
        variant=variant,
        handoff=handoff,
        inv_w_bits=INV_W_BITS,
        eps_bits=EPS_BITS,
        eps_w_bits=EPS_W_BITS,
        half_log2_w=HALF_LOG2_W,
        rows=rows,
    )
    return ttnn.to_torch(res)


def _run_consumer(device, stat_t, x_t, rows, variant, finalize_first=True):
    stat = _upload(device, stat_t, ttnn.float32)
    x = _upload(device, x_t, ttnn.bfloat16)
    out = _alloc(device, (rows * TILE, TILE), ttnn.bfloat16)
    res = run_op(
        stat,
        out,
        mode="consumer",
        variant=variant,
        handoff="inplace_copy" if finalize_first else "xfer_raw",
        inv_w_bits=INV_W_BITS,
        eps_bits=EPS_BITS,
        eps_w_bits=EPS_W_BITS,
        half_log2_w=HALF_LOG2_W,
        rows=rows,
        x_tensor=x,
    )
    return ttnn.to_torch(res)


def _run_isolated(device, stat_t, variant, reps):
    stat = _upload(device, stat_t, ttnn.float32)
    out = _alloc(device, (TILE, TILE), ttnn.float32)
    return run_op(
        stat,
        out,
        mode="isolated",
        variant=variant,
        inv_w_bits=INV_W_BITS,
        eps_bits=EPS_BITS,
        eps_w_bits=EPS_W_BITS,
        half_log2_w=HALF_LOG2_W,
        rows=1,
        reps=reps,
    )


# =============================================================================
# Device-zone reader (DeviceZoneScopedN -> profile_log_device.csv)
# =============================================================================
_DEVICE_CSV = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated/profiler/.logs/profile_log_device.csv")
_RISC = {"TRISC_0": "unpack", "TRISC_1": "math", "TRISC_2": "pack"}


def _csv_rows(path):
    with open(path) as f:
        lines = f.read().splitlines()
    freq = 1000.0
    for part in lines[0].split(","):
        if "CHIP_FREQ" in part:
            freq = float(part.split(":")[1])
    rows = [[x.strip() for x in ln.split(",")] for ln in lines[2:] if ln.strip()]
    return [r for r in rows if len(r) >= 12], 1000.0 / freq, freq


def _run_ids(path):
    if not os.path.exists(path):
        return set()
    rows, _, _ = _csv_rows(path)
    return {r[7] for r in rows}


def _zones_for_new_run(path, seen, zone_name):
    """{engine: summed ns} for `zone_name` in the launch(es) whose run-host-id is new."""
    rows, ns_per_cycle, _ = _csv_rows(path)
    starts, ends = {}, {}
    for r in rows:
        risc, cyc, run_id, zone, typ = r[3], r[5], r[7], r[10], r[11]
        if run_id in seen or zone != zone_name:
            continue
        (starts if typ == "ZONE_START" else ends).setdefault(risc, []).append(int(cyc))
    out = {}
    for risc, s in starts.items():
        s.sort()
        e = sorted(ends.get(risc, []))
        durs = [(ee - ss) * ns_per_cycle for ss, ee in zip(s, e)]
        if durs:
            out[_RISC[risc]] = sum(durs)
    return out


def _require_profiler():
    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        raise RuntimeError(
            "device profiling is off -- re-run with TT_METAL_DEVICE_PROFILER=1 "
            "TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 "
            "(this module must not set them itself: `import ttnn` executes it)"
        )


def _measure(device, run_fn, zone_names, trials):
    """{zone: {engine: [ns per launch]}}.  One launch per trial; the zone is summed over
    all its entries in that launch (so a `rows`-trip loop yields the stage total)."""
    run_fn()  # warm the kernel cache / JIT so the measured launches are steady-state
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    samples = {z: {} for z in zone_names}
    for _ in range(trials):
        seen = _run_ids(_DEVICE_CSV)
        run_fn()
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
        for z in zone_names:
            for engine, ns in _zones_for_new_run(_DEVICE_CSV, seen, z).items():
                samples[z].setdefault(engine, []).append(ns)
    return samples


def _med(samples, zone, engine):
    v = samples.get(zone, {}).get(engine, [])
    return statistics.median(v) if v else 0.0


def _stage_ns(samples, zone):
    """A stage's cost = the busiest TRISC in its zone (the critical path through it)."""
    per = samples.get(zone, {})
    return max([statistics.median(v) for v in per.values()] or [0.0])


def _arch(device):
    a = str(device.arch()).rsplit(".", 1)[-1]
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH"}.get(a, a)


def _clock():
    try:
        return round(_csv_rows(_DEVICE_CSV)[2])
    except Exception:
        return None


def _stamp(device):
    return f"box={socket.gethostname()}  arch={_arch(device)}  clock={_clock()}MHz  cores=1  sharded-L1"


# =============================================================================
# 1. Correctness — the only pass/fail
# =============================================================================
def test_finalize_correctness(device):
    import torch

    variants, handoffs, rows_sweep = (
        _sel("RFS_VARIANTS", VARIANTS),
        _sel("RFS_HANDOFFS", HANDOFFS),
        _sel("RFS_ROWS", ROWS_SWEEP),
    )
    logger.info(f"\n{_stamp(device)}\n  W={W} eps={EPS} fp32 stat CB / HiFi2 / fp32_dest_acc_en=False")
    header = f"{'rows':>5} {'handoff':>13} {'variant':>12} {'valid cols':>11} {'pcc':>10} {'rel-RMS':>9} {'col0 vs base':>13}"
    lines = [header]
    for rows in rows_sweep:
        stat_t = _stat_torch(rows)
        golden = _finalize_golden(stat_t)
        base_col0 = None
        for handoff in handoffs:
            for variant in variants:
                out = _run_structural(device, stat_t, rows, variant, handoff)
                cols = VALID_COLS[variant]
                got = out[:, cols]
                exp = golden[:, cols]
                pcc = _pcc(got, exp)
                rr = _rel_rms(got, exp)

                if handoff == BASELINE_HANDOFF and variant == BASELINE:
                    base_col0 = out[:, 0].clone()
                eq = "—"
                if base_col0 is not None:
                    same = torch.equal(out[:, 0], base_col0)
                    eq = "bit-equal" if same else "differs"
                    if variant in _BIT_EXACT_VS_BASE:
                        assert same, (
                            f"{variant}/{handoff}/rows={rows}: column 0 is NOT bit-identical to the op's "
                            f"current approach — a scope must never change a lane it runs"
                        )
                lines.append(f"{rows:5d} {handoff:>13} {variant:>12} {len(cols):11d} {pcc:10.6f} {rr:9.5f} {eq:>13}")
                assert pcc >= PCC_GATE, f"{variant}/{handoff}/rows={rows}: pcc {pcc:.6f} < {PCC_GATE}"
                assert rr <= RELRMS_GATE, f"{variant}/{handoff}/rows={rows}: rel-RMS {rr:.5f} > {RELRMS_GATE}"
                assert torch.isfinite(out).all(), f"{variant}/{handoff}/rows={rows}: non-finite lane in the output"
    logger.info("\n" + "\n".join(lines))


# =============================================================================
# 2. What exactly is in the lanes a scope leaves behind?
# =============================================================================
def test_stale_lane_inventory(device):
    import torch

    rows = 2
    stat_t = _stat_torch(rows)
    golden = _finalize_golden(stat_t)
    # what a lane would hold if only `*(1/W) + eps` ran on it (no rsqrt)
    touched = stat_t.to(torch.float64) * (1.0 / W) + EPS
    raw = stat_t.to(torch.float64)

    lines = [f"{'variant':>12} {'col':>4} {'state':>26} {'sample':>14}"]
    for variant in _sel("RFS_VARIANTS", VARIANTS):
        out = _run_structural(device, stat_t, rows, variant, BASELINE_HANDOFF).to(torch.float64)
        assert torch.isfinite(out).all(), f"{variant}: a stale lane is inf/NaN"
        for col in (0, 1, 15, 16, 31):

            def close(ref):
                # 3% relative: bf16 DEST rounding alone is ~0.4% per pass, and the three
                # candidate states differ by ORDERS of magnitude, so this cannot alias.
                return bool((out[:, col] - ref[:, col]).abs().max() <= 3e-2 * ref[:, col].abs().max().clamp_min(1e-30))

            if close(golden):
                state = "FINALIZED (correct)"
            elif close(touched):
                state = "stale: *(1/W)+eps, no rsqrt"
            elif close(raw):
                state = "stale: raw sum(x^2)"
            else:
                state = "stale: other"
            lines.append(f"{variant:>12} {col:4d} {state:>26} {out[0, col]:14.6g}")
    logger.info("\n" + "\n".join(lines))


# =============================================================================
# 3. THE DOMAIN QUESTION — does pass B's real consumer ever read a stale lane?
# =============================================================================
def test_consumer_domain(device):
    """Every column but 0 of the stat tile is seeded WRONG (2e4..3.2e5 vs ~4e3).  If
    BroadcastDim::Col read anything but column 0, PCC would crater.

    Two experiments:
      raw   -- NO finalize: multiply x by the RAW stat tile.  This is the PURE lane test
               (which lanes of a tile does the FPU's column broadcast actually read?),
               independent of any scope.  Golden = x * stat[:, 0].
      final -- the op's real sequence: finalize with each variant, then consume.
               Golden = x * rsqrt(stat[:, 0]/W + eps).
    """
    import torch

    rows = 2
    stat_t = _stat_torch(rows)
    g = torch.Generator().manual_seed(11)
    x_t = torch.randn(rows * TILE, TILE, generator=g, dtype=torch.float32)
    x_bf = x_t.to(torch.bfloat16).to(torch.float64)

    lines = [f"{'stage':>6} {'variant':>12} {'pcc':>10} {'rel-RMS':>9}   consumer reads"]
    checks = []

    # --- the pure lane test -------------------------------------------------------
    exp_raw = x_bf * stat_t[:, 0:1].to(torch.float64)
    out = _run_consumer(device, stat_t, x_t, rows, BASELINE, finalize_first=False)
    checks.append(("raw", BASELINE, out, exp_raw))

    # --- the op's real sequence, per variant --------------------------------------
    exp_fin = x_bf * _finalize_golden(stat_t)[:, 0:1]
    for variant in _sel("RFS_VARIANTS", VARIANTS):
        out = _run_consumer(device, stat_t, x_t, rows, variant, finalize_first=True)
        checks.append(("final", variant, out, exp_fin))

    for stage, variant, out, exp in checks:
        pcc = _pcc(out, exp)
        rr = _rel_rms(out, exp)
        lines.append(
            f"{stage:>6} {variant:>12} {pcc:10.6f} {rr:9.5f}   "
            f"{'COLUMN 0 ONLY' if (pcc >= PCC_GATE and rr <= RELRMS_GATE) else 'LEAKS'}"
        )
    logger.info("\n" + _stamp(device) + "\n" + "\n".join(lines))

    for stage, variant, out, exp in checks:
        assert torch.isfinite(out).all(), f"{stage}/{variant}: consumer produced a non-finite value"
        pcc, rr = _pcc(out, exp), _rel_rms(out, exp)
        assert pcc >= PCC_GATE, f"{stage}/{variant}: consumer pcc {pcc:.6f} < {PCC_GATE} - a stale lane IS read"
        assert rr <= RELRMS_GATE, f"{stage}/{variant}: consumer rel-RMS {rr:.5f} > {RELRMS_GATE}"


# =============================================================================
# 4. Perf — isolated MATH-thread SFPU cost per finalize call
# =============================================================================
def test_perf_isolated(device):
    _require_profiler()
    reps = int(os.environ.get("RFS_REPS", "2000"))
    trials = int(os.environ.get("RFS_TRIALS", "3"))
    stat_t = _stat_torch(1)

    variants = _sel("RFS_VARIANTS", VARIANTS)
    samples = {}
    for variant in variants:
        samples[variant] = _measure(
            device, lambda v=variant: _run_isolated(device, stat_t, v, reps), [ZONE_SFPU], trials
        )

    base = (_med(samples.get(BASELINE, {}), ZONE_SFPU, "math") / reps) or 1.0
    lines = [
        f"{_stamp(device)}  N={trials} (median)  reps={reps}",
        "ISOLATED: MATH-thread (TRISC_1) ns per finalize call. copy(seed)+pack and every init are OUTSIDE",
        "the timed zone, so this is pure SFPU cycles for *(1/W), +eps, rsqrt.",
        "",
        f"{'variant':>12} {'vec ops':>8} {'passes':>7} {'math ns/call':>13} {'vs base':>8} {'ns/vec':>7} {'unpack':>7} {'pack':>6}",
    ]
    for variant in variants:
        m = _med(samples[variant], ZONE_SFPU, "math") / reps  # zone total / reps = per call
        spd = f"{base / m:.2f}x" if m else "-"
        lines.append(
            f"{variant:>12} {VEC_OPS[variant]:8d} {PASSES[variant]:7d} {m:13.1f} {spd:>8} "
            f"{m / VEC_OPS[variant]:7.1f} {_med(samples[variant], ZONE_SFPU, 'unpack') / reps:7.3f} "
            f"{_med(samples[variant], ZONE_SFPU, 'pack') / reps:6.3f}"
        )
    logger.info("\n" + "\n".join(lines))


# =============================================================================
# 5. Perf — the op's stage sequence (copy/pack + CB handshake included)
# =============================================================================
def test_perf_structural(device):
    _require_profiler()
    trials = int(os.environ.get("RFS_TRIALS", "3"))
    zones = [ZONE_FINALIZE, ZONE_HANDOFF]

    # (handoff, variant) cells worth measuring: the full variant ladder on the op's
    # current structure, plus both A->B structures across the ladder.
    # The two levers are orthogonal, so the full cross product is not needed: sweep the
    # whole (a) ladder on the op's CURRENT structure, and the (b) structures only on the
    # baseline finalize (b alone) and on the best (a) (a+b together).
    variants, handoffs = _sel("RFS_VARIANTS", VARIANTS), _sel("RFS_HANDOFFS", HANDOFFS)
    ab_variants = tuple(v for v in variants if v in (BASELINE, "cskip3", "cskip_fused"))
    cells = [(h, v) for h in handoffs for v in (variants if h == BASELINE_HANDOFF else ab_variants)]
    rows_sweep = _sel("RFS_ROWS", ROWS_SWEEP)

    results = {}  # (rows, handoff, variant) -> (finalize_ns, handoff_ns)
    for rows in rows_sweep:
        stat_t = _stat_torch(rows)
        for handoff, variant in cells:
            s = _measure(
                device,
                lambda r=rows, h=handoff, v=variant, st=stat_t: _run_structural(device, st, r, v, h),
                zones,
                trials,
            )
            results[(rows, handoff, variant)] = (_stage_ns(s, ZONE_FINALIZE), _stage_ns(s, ZONE_HANDOFF))

    lines = [
        f"{_stamp(device)}  N={trials} (median)  W={W}",
        "STRUCTURAL: the op's stage sequence.  ns = busiest TRISC inside the stage's zone, summed over the",
        "stage's whole `rows` loop; /tile divides by rows.  Includes unpack, pack, CB handshake and inits.",
        "  finalize   == the op's compute_finalize / compute_root_finalize  (this IS the LOCAL, non-combine",
        "               path's whole cost -- it has no handoff)",
        "  handoff    == the op's compute_stat_handoff (0 for the A->B structures: there is no second pass)",
        "",
        f"{'rows':>5} {'handoff':>13} {'variant':>12} {'fin ns':>9} {'hoff ns':>8} {'total ns':>9} {'ns/tile':>8} {'vs base':>8}",
    ]
    for rows in rows_sweep:
        base_total = sum(results.get((rows, BASELINE_HANDOFF, BASELINE), (0.0, 1.0)))
        for handoff, variant in cells:
            f_ns, h_ns = results[(rows, handoff, variant)]
            tot = f_ns + h_ns
            spd = f"{base_total / tot:.2f}x" if tot else "—"
            lines.append(
                f"{rows:5d} {handoff:>13} {variant:>12} {f_ns:9.0f} {h_ns:8.0f} {tot:9.0f} {tot / rows:8.1f} {spd:>8}"
            )
        lines.append("")
    logger.info("\n" + "\n".join(lines))


# =============================================================================
# 6. Domain: is the lever W-general?  (the (a) ladder's only W dependence)
# =============================================================================
def test_w_generality(device):
    """The finalize's only W dependence is the SCALAR 1/W, so the c_skip ladder is
    W-general by construction.  This nails that down at a NON-power-of-two W (768 --
    also the "W is not a multiple of the tile" regime's logical width), and shows the
    one genuine inexpressible: the FULLY fused bodies apply sqrt(W) as an EXPONENT ADD,
    so they refuse to build unless W is a power of FOUR.
    """
    rows, w_alt = 2, 768
    assert _half_log2(w_alt) == -1, "pick a W that is not a power of four for this test"
    stat_t = _stat_torch(rows)
    golden = _finalize_golden(stat_t, w=w_alt)

    lines = [f"W={w_alt} (not a power of two)   {'variant':>12} {'pcc':>10} {'rel-RMS':>9}  status"]
    for variant in VARIANTS:
        kw = dict(
            mode="structural",
            variant=variant,
            handoff=BASELINE_HANDOFF,
            inv_w_bits=_bits(1.0 / w_alt),
            eps_bits=EPS_BITS,
            eps_w_bits=_bits(EPS * w_alt),
            half_log2_w=_half_log2(w_alt),
            rows=rows,
        )
        stat = _upload(device, stat_t, ttnn.float32)
        out_t = _alloc(device, (rows * TILE, TILE), ttnn.float32)
        if variant in NEEDS_POW4_W:
            with pytest.raises(ValueError, match="power of FOUR"):
                run_op(stat, out_t, **kw)
            lines.append(f"{'':29} {variant:>12} {'-':>10} {'-':>9}  INEXPRESSIBLE (refused)")
            continue
        out = ttnn.to_torch(run_op(stat, out_t, **kw))
        cols = VALID_COLS[variant]
        pcc, rr = _pcc(out[:, cols], golden[:, cols]), _rel_rms(out[:, cols], golden[:, cols])
        lines.append(f"{'':29} {variant:>12} {pcc:10.6f} {rr:9.5f}  ok")
        assert pcc >= PCC_GATE, f"{variant} @ W={w_alt}: pcc {pcc:.6f} < {PCC_GATE}"
        assert rr <= RELRMS_GATE, f"{variant} @ W={w_alt}: rel-RMS {rr:.5f} > {RELRMS_GATE}"
    logger.info("\n" + "\n".join(lines))
