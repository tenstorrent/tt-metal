# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off: ROTATE the rms_norm combine's ROOT DUTY across the group.

NOT the op.  The concept under test is a PLACEMENT, not a kernel primitive: the
combine's per-round fold + arrival wait + stat multicast is a fixed amount of
serial work, and today every round of it lands on the SAME core -- the one core
in each width group that also has a full member's share of pass A and pass B to
do.  Its group's other GROUP_SIZE-1 cores sit in a semaphore wait for the whole
time.  This bench moves round `blk`'s root duty to group member
`(blk + phase) % GROUP_SIZE`, so the fold spreads over `min(num_blocks,
GROUP_SIZE)` cores instead of piling on one.

A placement can only be measured against the imbalance it is supposed to remove,
so the bench keeps the op's real geometry, real per-stage costs and real
precision contract, and changes ONLY who does what.

How the isolation works
-----------------------
The bench asks the OP'S OWN descriptor for the program (so BLOCK_ROWS, the L1
solve, the CBs, the shard-backing and every other CT/RT arg are exactly the
op's), then rewrites four things on the returned ``ttnn.ProgramDescriptor``:

  * the three ``kernel_source`` paths -> ``kernels/bench_*.cpp``, VERBATIM copies
    of the op's kernels plus an ``RMS_ROT_VARIANT`` switch;
  * ``defines["RMS_ROT_VARIANT"]`` -> the variant bitmask;
  * the WRITER's runtime args: a ``rot_phase`` word is inserted at index 10 (so
    the mcast RT block starts at 11 in BOTH variants -- the baseline just never
    reads it), and for the rotating variants the mcast block itself is replaced by
    a ``rotating_sender`` Mcast1D/Mcast2D's own wire;
  * the COMPUTE kernel's runtime args: ``my_slot`` and ``rot_phase`` appended, so
    the fold's per-round root predicate is spelled identically on both sides.

Variant 0 is therefore the op's current approach -- the honest baseline -- and
every candidate differs from it by the switch plus that host wire.

NOTHING ELSE MOVES.  In particular the gather CB, the stat handoff and the mcast
landing CB are ALREADY allocated on every core of the program by the op's own
descriptor (`all_cores`), so rotation is L1-NEUTRAL to the byte; the bench asserts
that rather than claiming it.

Variants -- a BITMASK, so each lever is measurable alone and in composition
-------------------------------------------------------------------------
0   ``fixed``    the op today: one fixed root per group does every round.
1   ``rot``      root rotation, gather boot-zeroing at boot on every core that
                 will ever root.
2   ``zdefer``   (with ``rot``) a future root whose first turn is NOT round 0 does
                 its gather zeroing AFTER its round-0 ship, i.e. inside the mcast
                 wait where its BRISC is idle, instead of in front of that ship.
4   ``diag``     (with ``rot``) each width group's rotation is phase-shifted by its
                 grid row, so the several groups' roots are never all in one column.
8   ``nozero``   ABLATION ONLY -- strip the gather zeroing payload.  At
                 GATHER_FACES < 4 this is DELIBERATELY INCORRECT (undefined L1
                 faces reach the fold); it exists so the zeroing's contribution is
                 separable from rotation's.  Never a graduatable option.

Precision contract is FIXED and identical for every variant (it comes from the
caller's ComputeConfigDescriptor, untouched).  Every correct variant is expected
BIT-IDENTICAL to ``fixed``: rotation changes WHICH CORE folds, never the fold's
order, its DEST walk, its reconfig pair or its finalize.
"""

from __future__ import annotations

import csv
import os
import glob
from pathlib import Path


import ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as rpd


# `torch` is imported LAZILY here, on first attribute access, rather than at module scope.
# `ttnn/ttnn/operations/__init__.py` runs pkgutil.walk_packages over the operations tree, and
# the repo's `check-torch-imports-in-ttnn` pre-commit hook forbids a global torch import
# anywhere under ttnn/ for exactly that reason.  See perf_experiments/README.md.
class _LazyTorch:
    def __getattr__(self, name):
        import torch as _torch

        return getattr(_torch, name)


torch = _LazyTorch()


KERNEL_DIR = Path(__file__).parent / "kernels"

# Writer runtime-arg indices this bench depends on (the op's own layout).
W_RT_IS_ROOT = 4
W_RT_SLOT = 5
W_RT_MCAST_BASE = 10  # where the op puts the mcast block; the bench shifts it to 11

ROTATE = 1
ZDEFER = 2
DIAG = 4
NOZERO = 8
STILL = 16  # ATTRIBUTION: rotating machinery, placement held still at slot 0
# ATTRIBUTION, HOST-ONLY (uses the BASELINE kernel path): keep ONE fixed root, but move
# it to an INTERIOR slot.  Mcast1D's sender_rect_ special-cases slot 0 / slot span-1 into
# a COMPACT receivers-only rectangle; an interior sender gets the FULL-LINE rect and the
# SenderPipe's EXCLUDE-source mode -- which is exactly the rect any ROTATING sender must
# use, because a sender that moves is always inside its own line.  So this isolates "the
# cost of the rect a rotating root forces" from "the cost of moving the root", with the
# kernel byte-identical to the op's.
MIDROOT = 32

_BITNAMES = (
    (ROTATE, "rot"),
    (ZDEFER, "zdefer"),
    (DIAG, "diag"),
    (NOZERO, "nozero"),
    (STILL, "still"),
    (MIDROOT, "midroot"),
)


def variant_name(v):
    if v < 0:
        return "pure_op"
    if v == 0:
        return "fixed"
    return "+".join(n for b, n in _BITNAMES if v & b)


class _Variants(dict):
    def __missing__(self, v):  # any bitmask is a legal variant
        return variant_name(v)


VARIANTS = _Variants()


def _perf_config():
    """The focus case's pinned compute config -- NEVER a lever."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=False,
        math_approx_mode=False,
    )


def _cb_bytes(pd):
    """Total CB bytes the program allocates, summed over cores.  The L1-neutrality
    check: rotation must not move this by a single byte."""
    total = 0
    for cb in pd.cbs:
        total += int(cb.total_size) * len(rpd._cores_in(cb.core_ranges))
    return total


def _rotating_mcast(device, pd, cores, wargs, group_size):
    """A ``rotating_sender`` twin of the plan's own mcast family.

    The op builds ONE of two topologies and the bench must build the SAME one,
    rotating.  Which one is read off the wire rather than re-derived: the op marks
    each group's fixed root with writer runtime arg 4, so
      * several roots, all in one COLUMN -> a per-grid-row family (Mcast1D PerRow),
        one width group per grid row;
      * exactly one root              -> the PACKED single group (Mcast2D over the
        shard grid's bounding box, in-box/out-of-group cores INACTIVE).
    Returns (mcast, per_row) where `per_row` also decides whether DIAG is even
    meaningful (a single packed group has no second group to shift against, and a
    phase that varied WITHIN one group would make its cores disagree about who the
    round's root is -- a hang, not a slowdown).
    """
    roots = [c for c in cores if wargs[(c.x, c.y)][W_RT_IS_ROOT] == 1]
    assert roots, "no root on the wire: this bench only measures the COMBINE path"
    per_row = len(roots) > 1 and len({r.x for r in roots}) == 1
    cfg = ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=True, rotating_sender=True, base_sem_id=0)
    grid = pd.kernels[1].core_ranges
    if per_row:
        mc = ttnn.Mcast1D(device, grid, ttnn.Mcast1DShape.PerRow, 0, cfg)
    else:
        # num_active stays the op's explicit `group_size - 1`: the bounding box may hold
        # INACTIVE cores that receive the broadcast but never ack.
        mc = ttnn.Mcast2D(device, grid, ttnn.CoreCoord(roots[0].x, roots[0].y), cfg, group_size - 1)
    return mc, per_row


PUREOP = -1  # CALIBRATION variant: the op's descriptor handed back UNTOUCHED.


def build(device, x, out, gamma, *, variant, epsilon=1e-6, compute_config=None):
    """The op's program descriptor, repointed at the bench kernels + variant wire."""
    cfg = compute_config or _perf_config()
    pd = rpd.create_program_descriptor(x, out, gamma=gamma, epsilon=epsilon, compute_kernel_config=cfg)
    l1_before = _cb_bytes(pd)
    cct0 = list(pd.kernels[2].compile_time_args)
    if variant == PUREOP:
        # The op ITSELF, through the op's own kernels and wire.  This is the bench's
        # CALIBRATION reference: `fixed` must land on it, or the "baseline" is a
        # strawman and every delta measured against it is meaningless.
        max_rows0 = max(list(pd.kernels[2].runtime_args[c.x][c.y])[0] for c in rpd._cores_in(pd.kernels[2].core_ranges))
        return pd, dict(
            block_rows=cct0[3],
            group_size=cct0[13],
            num_blocks=-(-max_rows0 // cct0[3]),
            rows_per_core=max_rows0,
            extra_l1_bytes=0,
            rotating=0,
            diag_applied=0,
            per_row_groups=0,
            mid_root=0,
            root_cores=1,
        )

    # DEBUG/CALIBRATION knob (env RMS_ROT_BISECT, default "rwc"): which of the three
    # kernels is served from the bench COPY rather than the op's own file.  It exists
    # because `fixed` MUST land on `pure_op`; when it did not, this is what found the
    # kernel responsible.  The writer copy is what needs the shifted mcast RT base, so
    # the phase word is inserted only when the writer copy is in play.
    bisect = os.environ.get("RMS_ROT_BISECT", "rwc")
    for k, name, tag in zip(pd.kernels, ["bench_reader.cpp", "bench_writer.cpp", "bench_compute.cpp"], "rwc"):
        if tag in bisect:
            k.kernel_source = str(KERNEL_DIR / name)
            k.defines = [("RMS_ROT_VARIANT", str(variant))]

    # BLOCK_ROWS / GROUP_SIZE / COMBINE straight off the kernels' own CT args, so the
    # bench never restates a knob the descriptor owns.
    cct = list(pd.kernels[2].compile_time_args)
    block_rows, combine, group_size = cct[3], cct[12], cct[13]
    assert combine == 1, "this bench only measures the COMBINE path"

    cores = rpd._cores_in(pd.kernels[1].core_ranges)
    wrt, crt = pd.kernels[1].runtime_args, pd.kernels[2].runtime_args
    wargs = {(c.x, c.y): list(wrt[c.x][c.y]) for c in cores}
    cargs = {(c.x, c.y): list(crt[c.x][c.y]) for c in cores}

    rot = bool(variant & ROTATE)
    mcast, per_row = _rotating_mcast(device, pd, cores, wargs, group_size) if rot else (None, False)
    # MIDROOT (host only): one fixed root, moved to an interior slot -> the full-line
    # EXCLUDE-source rect a rotating sender is forced into.  Only meaningful on the
    # per-grid-row family; the PACKED single group's Mcast2D sender is already in its own
    # rect, so it already pays that rect and there is nothing to isolate.
    mid = 0
    if variant & MIDROOT and not rot:
        roots = [c for c in cores if wargs[(c.x, c.y)][W_RT_IS_ROOT] == 1]
        if len(roots) > 1 and len({r.x for r in roots}) == 1:
            mid = group_size // 2
            mcast = ttnn.Mcast1D(
                device,
                pd.kernels[1].core_ranges,
                ttnn.Mcast1DShape.PerRow,
                mid,
                ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=True, base_sem_id=0),
            )
    # DIAG needs one group PER GRID ROW to shift against, and the phase MUST be
    # line-uniform (see the kernel).  On the packed single group it is inexpressible, so
    # it silently degrades to phase 0 and the bench records that (`diag_applied`).
    diag = bool(variant & DIAG) and per_row

    new_w, new_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c in cores:
        a = list(wargs[(c.x, c.y)])
        cargs_c = list(cargs[(c.x, c.y)])
        if mid:
            # move the single fixed root to the interior slot, on both kernels' wires
            a[W_RT_IS_ROOT] = 1 if a[W_RT_SLOT] == mid else 0
            cargs_c[2] = a[W_RT_IS_ROOT]
        phase = (c.y % group_size) if diag else 0
        head, old_mc = a[:W_RT_MCAST_BASE], a[W_RT_MCAST_BASE:]
        tail = list(mcast.runtime_args(c)) if (rot or mid) else old_mc
        # The phase word (and the shifted mcast RT base it implies) exists ONLY in the
        # rotating build -- the baseline's writer is byte-identical to the op's, mcast RT
        # base included, so its wire must be too.
        new_w[c.x][c.y] = (head + [phase] + tail) if (rot and "w" in bisect) else (head + tail)
        new_c[c.x][c.y] = cargs_c + [a[W_RT_SLOT], phase]
    pd.kernels[1].runtime_args = new_w
    pd.kernels[2].runtime_args = new_c

    # rows-per-core off the compute kernel's OWN runtime args (arg 0 == row_count), so
    # num_blocks is the plan's, never re-derived from the shard shape.
    max_rows = max(v[0] for v in cargs.values())
    num_blocks = -(-max_rows // block_rows) if block_rows else 0

    info = dict(
        block_rows=block_rows,
        group_size=group_size,
        num_blocks=num_blocks,
        rows_per_core=max_rows,
        # ROTATION IS L1-NEUTRAL: cb_partials_gathered / cb_stat_handoff / cb_row_final
        # are already on `all_cores` in the op's own descriptor, so no core gains a page.
        extra_l1_bytes=_cb_bytes(pd) - l1_before,
        rotating=int(rot),
        diag_applied=int(diag),
        per_row_groups=int(per_row),
        mid_root=int(mid),
        # How many distinct cores per group ever take a root turn.
        root_cores=min(num_blocks, group_size) if (rot and not (variant & STILL)) else 1,
    )
    return pd, info


def run(device, x, out, gamma, *, variant, epsilon=1e-6, compute_config=None):
    pd, info = build(device, x, out, gamma, variant=variant, epsilon=epsilon, compute_config=compute_config)
    tensors = [x] + ([gamma] if gamma is not None else []) + [out]
    return ttnn.generic_op(tensors, pd), info


# ---------------------------------------------------------------------------
# reference + measurement
# ---------------------------------------------------------------------------


def torch_reference(x_t, g_t, eps):
    xf = x_t.to(torch.float32)
    denom = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = xf * denom
    if g_t is not None:
        y = y * g_t.to(torch.float32).reshape(1, 1, 1, -1)
    return y


def pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    if torch.allclose(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def rel_rms(a, b):
    """RMS(a-b) / RMS(b) -- the metric that catches a uniform scale error pcc hides
    (the op's own integration bug: pcc 0.9997 at a ~1000x scale error)."""
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    den = float(b.pow(2).mean().sqrt())
    return float((a - b).pow(2).mean().sqrt()) / (den if den else 1.0)


def device_kernel_ns(op_code_contains="GenericOp"):
    """The newest ops_perf_results CSV's DEVICE KERNEL DURATION [ns]."""
    reports = sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"))
    assert reports, "no profiler report found -- run under scripts/run_safe_pytest.sh --profile"
    out = []
    with open(reports[-1]) as f:
        for row in csv.DictReader(f):
            ns = row.get("DEVICE KERNEL DURATION [ns]")
            if ns and ns.strip():
                out.append((row.get("OP CODE", "").strip(), int(float(ns))))
    return reports[-1], out
