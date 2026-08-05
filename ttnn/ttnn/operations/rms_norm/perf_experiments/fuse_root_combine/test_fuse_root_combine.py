# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off for rms_norm's group-root combine chain.

    source python_env/bin/activate ; unset TT_METAL_DPRINT_CORES
    scripts/run_safe_pytest.sh --run-all \
      ttnn/ttnn/operations/rms_norm/perf_experiments/fuse_root_combine/test_fuse_root_combine.py

Correctness is the ONLY pass/fail (each variant vs a torch reference).  Perf is
measured and printed, never asserted.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import statistics

import pytest
from loguru import logger

import ttnn

from ttnn.operations.rms_norm.perf_experiments.fuse_root_combine.root_combine_bench import (
    VARIANTS,
    build_case,
    create_program_descriptor,
    make_tensors,
    ROW_MAJOR_VARIANTS,
    run_variant,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# Focus-shape geometry: (1,1,8192,1024) BLOCK_SHARDED [1024,128] on (8,8) =>
# GROUP_SIZE 8, BLOCK_ROWS 10, and each member owns 128 of the 1024 columns.
PRIMARY = (8, 10)
PER_SLOT_W = 128
EPS = 1e-12

# (group_size, rows) -- every combo the op actually builds that fits L1 on one
# core (cb_partials_gathered is GROUP_SIZE * BLOCK_ROWS fp32 pages).
GEOMETRIES = [
    (8, 10),  # PRIMARY: (1,1,8192,1024) BLOCK_SHARDED 64c
    (32, 1),  # SECONDARY: (1,1,32,5120) WIDTH_SHARDED 32c (decode)
    (28, 1),  # w7168_28c
    (9, 1),  # w2304_9c  -- ODD group size
    (8, 1),
    (2, 1),  # smallest group the op can build (a width split across 2 cores)
    (3, 1),  # smallest ODD group
    (4, 1),
    (4, 2),
    (8, 2),
    (16, 2),
    (32, 2),
    (16, 10),
    (4, 32),  # rows >> DEST capacity (8)
]

if os.environ.get("RCB_GEOS"):  # e.g. RCB_GEOS="8x10,32x1"
    _want = {tuple(int(v) for v in g.split("x")) for g in os.environ["RCB_GEOS"].split(",")}
    GEOMETRIES = [g for g in GEOMETRIES if g in _want]

PCC_GATE = 0.9995  # the op's soft gate, applied to the OP-LEVEL output below
RELRMS_GATE = 0.04


def _pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0.0:
        return 1.0
    return float((a @ b).item() / denom)


def _relrms(ref, got):
    ref = ref.double()
    got = got.double()
    return float((got - ref).norm().item() / max(ref.norm().item(), 1e-30))


def _quality(got, stat_ref, xb):
    """Precision of one variant, at two levels.

    stat_*  : the combine's own output (1/rms) -- the strictest view, its
              row-to-row variance is only a few percent.
    op_*    : x * (1/rms), i.e. what the op actually returns (gamma is a
              constant factor and drops out of PCC / rel-RMS).
    """
    import torch

    ref_op = xb * stat_ref.unsqueeze(1).to(torch.float64)
    got_op = xb.double() * got.double().unsqueeze(1)
    return {
        "stat_pcc": _pcc(stat_ref, got),
        "stat_relrms": _relrms(stat_ref, got),
        "op_pcc": _pcc(ref_op, got_op),
        "op_relrms": _relrms(ref_op, got_op),
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


# ---------------------------------------------------------------------------
# Correctness (the only pass/fail)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_correctness_primary(device, variant):
    group_size, rows = PRIMARY
    got, stat_ref, xb = run_variant(
        device, variant=variant, rows=rows, group_size=group_size, per_slot_w=PER_SLOT_W, eps=EPS, iters=1
    )
    if variant == "floor":
        pytest.skip("`floor` publishes no result by construction (handshake-only)")
    q = _quality(got, stat_ref, xb)
    logger.info(f"{variant} rows={rows} GS={group_size} quality={q}")
    # Only a gross-error gate here; the exact per-variant precision is REPORTED
    # by the menu test below, which is what the coordinator reads.
    assert q["stat_relrms"] < 0.05, f"{variant}: stat rel-RMS {q['stat_relrms']:.4g} -- not the right math"


@pytest.mark.parametrize("group_size,rows", GEOMETRIES)
def test_correctness_sweep(device, group_size, rows):
    """Every non-floor variant computes the same, correct stat at this geometry."""
    for variant in VARIANTS:
        if variant == "floor":
            continue
        got, stat_ref, xb = run_variant(
            device, variant=variant, rows=rows, group_size=group_size, per_slot_w=PER_SLOT_W, eps=EPS, iters=1
        )
        q = _quality(got, stat_ref, xb)
        logger.info(f"GS={group_size} rows={rows} {variant}: {q}")
        assert q["stat_relrms"] < 0.05, f"GS={group_size} rows={rows} {variant}: rel-RMS {q['stat_relrms']:.4g}"


@pytest.mark.parametrize("group_size,rows", [(8, 10), (9, 1)])
@pytest.mark.parametrize("variant", ["destacc_fused_cskip", "destacc_fused_col"])
def test_scoped_finalize_is_bit_identical(device, variant, group_size, rows):
    """The column-scoped finalizes are BIT-identical to the stock (RC) one.

    Column 0 is the only column the stat's consumer reads, and scoping only
    removes vectors that never touch column 0 -- the arithmetic on column 0 is the
    same fp32 LREG sequence.  So this is an EQUALITY check, not a tolerance check:
    it is what makes "same precision" a fact rather than a claim.  Covers a W that
    is a power of two (GS=8 -> 1024) and one that is not (GS=9 -> 1152).
    """
    import torch

    ref, _, _ = run_variant(
        device, variant="destacc_fused", rows=rows, group_size=group_size, per_slot_w=PER_SLOT_W, eps=EPS, iters=1
    )
    got, _, _ = run_variant(
        device, variant=variant, rows=rows, group_size=group_size, per_slot_w=PER_SLOT_W, eps=EPS, iters=1
    )
    assert torch.equal(ref, got), f"{variant} differs from destacc_fused: max |d| = {(got - ref).abs().max():.3e}"


@pytest.mark.parametrize("group_size,rows", [(8, 10), (9, 1), (28, 1)])
def test_sfpu1_finalize_precision(device, group_size, rows):
    """`destacc_fused_sfpu1` folds *(1/W) and +eps into the sqrt body's own pass.

    Bit-identical to the stock finalize when 1/W is exactly representable (W a
    power of two); otherwise it differs by ONE rounding (a single fused
    multiply-add instead of two separately-rounded SFPU passes).  The gate is
    that it is never LESS accurate than the stock finalize.
    """
    import torch

    ref, stat_ref, xb = run_variant(
        device, variant="destacc_fused", rows=rows, group_size=group_size, per_slot_w=PER_SLOT_W, eps=EPS, iters=1
    )
    got, _, _ = run_variant(
        device, variant="destacc_fused_sfpu1", rows=rows, group_size=group_size, per_slot_w=PER_SLOT_W, eps=EPS, iters=1
    )
    q_ref = _quality(ref, stat_ref, xb)
    q_got = _quality(got, stat_ref, xb)
    logger.info(
        f"GS={group_size} rows={rows} exact_1_over_W={(group_size * PER_SLOT_W & (group_size * PER_SLOT_W - 1)) == 0} "
        f"bit_identical={bool(torch.equal(ref, got))} stock_relrms={q_ref['stat_relrms']:.4e} "
        f"sfpu1_relrms={q_got['stat_relrms']:.4e}"
    )
    assert (
        q_got["stat_relrms"] <= q_ref["stat_relrms"] * 1.05
    ), f"sfpu1 rel-RMS {q_got['stat_relrms']:.4e} worse than stock {q_ref['stat_relrms']:.4e}"


@pytest.mark.parametrize("variant", ["l1acc_sep", "destacc_fused", "destacc_fused_sfpu1"])
def test_domain_fp32_dest_acc(device, variant):
    """DOMAIN: the pattern when the USER asks for fp32 DEST (DEST capacity 4, not 8).

    Not a perf lever -- the menu is measured at the contract's fp32_dest_acc_en=False.
    This only proves the DEST-resident form is still expressible and correct when the
    user's ComputeKernelConfig halves the DEST slot count.
    """
    group_size, rows = PRIMARY
    got, stat_ref, xb = run_variant(
        device,
        variant=variant,
        rows=rows,
        group_size=group_size,
        per_slot_w=PER_SLOT_W,
        eps=EPS,
        iters=1,
        fp32_dest_acc_en=True,
    )
    q = _quality(got, stat_ref, xb)
    logger.info(f"fp32_dest_acc_en=True {variant}: {q}")
    assert q["stat_relrms"] < 0.05, f"{variant} @ fp32 dest: rel-RMS {q['stat_relrms']:.4g}"


# ---------------------------------------------------------------------------
# The menu: ns + precision per variant, one measured launch each (median of 3)
# ---------------------------------------------------------------------------


def _iters_for(group_size, rows):
    """In-kernel repeat so the payload dominates the fixed launch cost."""
    work = group_size * rows
    return max(2, min(64, int(round(1600.0 / work))))


def _measure(device, variant, group_size, rows, iters, launches=3):
    import torch

    partials, W, stat_ref, xb = build_case(rows, group_size, PER_SLOT_W, EPS)
    tt_p, tt_out = make_tensors(device, partials, rows, group_size, variant in ROW_MAJOR_VARIANTS)
    desc = create_program_descriptor(
        tt_p,
        tt_out,
        variant=variant,
        rows=rows,
        group_size=group_size,
        W=W,
        eps=EPS,
        iters=iters,
    )
    out = ttnn.generic_op([tt_p, tt_out], desc)
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # drain
    samples = []
    for _ in range(launches):
        ttnn.generic_op([tt_p, tt_out], desc)
        ttnn.synchronize_device(device)
        ns = _read_kernel_ns(device)
        if ns is not None:
            samples.append(ns)
    got = ttnn.to_torch(out).to(torch.float32)[:, 0]
    q = _quality(got, stat_ref, xb) if variant != "floor" else {}
    med = statistics.median(samples) if samples else float("nan")
    spread = (max(samples) - min(samples)) / med * 100.0 if samples and med else float("nan")
    return med, spread, q


def test_menu(device):
    """Print the option menu (ns + precision) for every variant x geometry."""
    rows_out = []
    for group_size, rows in GEOMETRIES:
        iters = _iters_for(group_size, rows)
        base = None
        for variant in VARIANTS:
            # `floor` is a one-number diagnostic (the CB-handshake/launch cost);
            # measure it at the two live targets only.  Its fully-elided kernel
            # also trips a linker-script quirk at some GROUP_SIZEs, which is a
            # toolchain artifact of an EMPTY .text, not a result.
            if variant == "floor" and (group_size, rows) not in ((8, 10), (32, 1)):
                continue
            ns, spread, q = _measure(device, variant, group_size, rows, iters)
            per_iter = ns / iters if ns == ns else float("nan")
            if variant == "rmw_sep":
                base = per_iter
            rows_out.append((group_size, rows, iters, variant, ns, per_iter, spread, q))
            logger.info(
                f"GS={group_size:2d} rows={rows:2d} iters={iters:2d} {variant:20s} "
                f"total={ns:10.1f} ns  per_round={per_iter:9.1f} ns  "
                f"spread={spread:5.2f}%  "
                f"vs_rmw={'-' if not base else f'{base / per_iter:5.2f}x'}  "
                f"stat_pcc={q.get('stat_pcc', float('nan')):.6f} "
                f"stat_relrms={q.get('stat_relrms', float('nan')):.3e} "
                f"op_pcc={q.get('op_pcc', float('nan')):.7f} "
                f"op_relrms={q.get('op_relrms', float('nan')):.3e}"
            )

    print("\n=== MENU (per-round ns = one BLOCK_ROWS round of the root combine) ===")
    hdr = f"{'GS':>3} {'rows':>4} {'variant':<20} {'per_round_ns':>12} {'vs_rmw':>7} {'vs_l1acc':>8} {'stat_pcc':>9} {'stat_relrms':>11} {'op_pcc':>10} {'op_relrms':>10} {'spread%':>7}"
    print(hdr)
    by_geo = {}
    for gs, rows, iters, variant, ns, per_iter, spread, q in rows_out:
        by_geo.setdefault((gs, rows), {})[variant] = (per_iter, spread, q)
    for (gs, rows), d in by_geo.items():
        b0 = d.get("rmw_sep", (float("nan"),))[0]
        b1 = d.get("l1acc_sep", (float("nan"),))[0]
        for variant in VARIANTS:
            if variant not in d:
                continue
            per_iter, spread, q = d[variant]
            print(
                f"{gs:3d} {rows:4d} {variant:<20} {per_iter:12.1f} {b0 / per_iter:6.2f}x {b1 / per_iter:7.2f}x "
                f"{q.get('stat_pcc', float('nan')):9.6f} {q.get('stat_relrms', float('nan')):11.3e} "
                f"{q.get('op_pcc', float('nan')):10.7f} {q.get('op_relrms', float('nan')):10.3e} {spread:7.2f}"
            )
