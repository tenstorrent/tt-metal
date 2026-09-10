# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off for idea `positional_work_skew` — PERF EXPERIMENT, not the
real op. Nothing here modifies a file of the op; `derive_plan` is wrapped at
runtime.

THE FINDING THIS COMES FROM (measured by the `block_to_core_mapping` sibling on
the same box, not re-derived here). Per-core `*-KERNEL` spans on the focus shape
`[1,1,32,16384]` form a monotone gradient in the PHYSICAL GRID ROW even though
per-core work is perfectly uniform, and the gradient is attached to the CORE, not
to the block: every block->core permutation is null. Cores start together
(~190 ns spread) and DRAIN in a fixed positional order over 5-7.6 us.

THE TWO MODELS THIS EXPERIMENT DECIDES BETWEEN.
  M1  the wall is `total_bytes / aggregate_bandwidth`; the gradient is only a
      service ORDER on a saturated shared resource. Redistributing work then
      only changes who waits last -> NULL, in both polarities.
  M2  the wall carries an end-of-kernel DEMAND COLLAPSE: the fast rows retire
      early, the surviving stragglers cannot by themselves keep DRAM busy, and
      the tail is served under-parallel. Flattening the finish times keeps
      demand high to the end and the wall falls toward the mean.

`active_cores.py` is the cheap discriminator and its OBSERVABLE looks like M2:
the last 20% of the baseline wall runs at 17/64 cores and the last decile at
8.5/64 (logs/active_cores.txt). THIS file is the causal test, because a demand
collapse can equally be a CONSEQUENCE of a positional service order rather than a
cause of the wall. If the effect is real, the SAME width multiset must behave
DIFFERENTLY depending on which physical rows carry the wide blocks — so every
skew has a `_rev` twin that issues byte-for-byte the same transactions on swapped
cores, which also makes the measurement immune to the read-size confound.

=========================== MEASURED RESULT: M1 ===========================
Wormhole B0 n150, 8x8 = 64/64 cores throughout, 1 GHz, 12 DRAM banks.
`run_safe_pytest.sh --profile`, DEVICE KERNEL DURATION [ns], all modes
INTERLEAVED in one process with the order rotated every rep. Every call of every
mode was bit-identical (`torch.equal`) and used 64/64 cores.

[1,1,32,16384] bf16 RM DRAM -> TILE DRAM, medians of 9 reps (logs/summary_focus3.txt):
    baseline       12660   1.000x     <- the op's current approach
    uniform8       12590   1.006x     <- byte-identical program: the NOISE FLOOR
    half97         12589   1.006x
    edge98         12680   0.998x     edge98_rev     12464  1.016x
    grad_brisc     13414   0.944x     grad_brisc_rev 12906  0.981x
and from the 7-rep menu (logs/summary_focus2.txt): half97 1.021x / half97_rev
1.032x, half106 0.999x / half106_rev 1.018x, grad_soft 1.004x / grad_soft_rev
1.026x, grad_full 0.950x / grad_full_rev 1.002x, checker97 0.990x.

NULL for every gentle skew (all inside the +-3% band `uniform8` sets), and a
REGRESSION for every skew strong enough to actually move the finish times.
THE POLARITY NEVER MATTERS: `_rev` tracks its twin everywhere, and where the two
differ it is the wide-to-SLOW control that is marginally FASTER — the opposite
sign to M2.

WHY (row_response.py, logs/row_response_focus3.txt). Regressing each grid row's
BRISC span on the width it was handed, across the modes, gives
`duration[y] ~ a[y] + b[y]*w[y]`:
    row      0     1     2     3      4     5     6      7
    a (ns) 7374  7353  8434  (fit)  3531  5327  5440  -3455
    b/tile  351   446   353  (fit)   782   586   527   1208
    a share  72%   67%   75%   -      36%   53%   56%     -
The slow rows are slow because they are WAITING, not working: 67-75% of their
span does not shrink when their work does. So the redistribution trade is priced
against itself — moving one tile of width from row 0 to row 7 SAVES 351 ns there
and COSTS 1208 ns here, a 3.4x losing exchange. That is M1 exactly: a positional
service order on a resource whose AGGREGATE is the constraint. The observed
demand collapse is the consequence of the drain order, not a separate slack the
schedule can reclaim.

Corroborated on the second expressible geometry [1,1,32,32768], 7 reps
(logs/summary_wide2.txt): baseline 23667, uniform8 23660 (1.000x), grad_brisc
0.989x, edge98 0.972x, half97 0.948x / half97_rev 0.954x — same null-to-
regression, same polarity indifference.
===========================================================================

HOW THE SKEW IS EXPRESSED. It is NOT a new mechanism: the op already carries
TWO CORE RANGES WITH DIFFERENT `block_width_tiles` (the RAGGED COLUMN TAIL of
Refinement 6 — `ColumnGroup`, `col_tile_offset`, the second `TilizeGroup`). All
this does is generalize "two groups, one of which happens to be the leftover" to
"N groups, whose widths are chosen". The kernels need no change at all: all three
derive `w_chunk = block_id % num_w_chunks` and
`col_base = col_tile_offset + w_chunk * block_width_tiles`, so a group with its
own `(block_width_tiles, num_w_chunks, col_tile_offset)` covers a contiguous
column range by construction, and the groups' ranges tile `C` exactly.

EXPRESSIBILITY (this is a real domain boundary, established by reading the plan,
not by benchmarking). A column-width skew needs each core to own its own column
extent, which requires `num_row_groups == 1` — i.e. ONE row group, `num_w_chunks
== num_cores`. At `num_row_groups > 1` every grid row spans the SAME set of
w_chunks, so each grid row necessarily does exactly `C` tiles' worth of columns
and a row-graded width is arithmetically impossible with this machinery. Those
geometries would need the other uneven-work form (uneven BLOCK COUNT per core,
which needs `num_blocks_total > num_cores`), which this bench does not build:
the focus-shape result below settles the question before it is worth building.
"""

from __future__ import annotations

import importlib
import math
import os

import ttnn

import ttnn.operations.tilize.tilize_program_descriptor as pd

tilize_mod = importlib.import_module("ttnn.operations.tilize.tilize")


# ---------------------------------------------------------------------------
# The per-grid-row width ladders. Every ladder sums to 64 = 8 rows x 8 (the
# uniform width the op picks on the focus shape), so `sum(8 * w[y]) == C == 512`
# and no variant can win by moving fewer bytes.
#
# `_rev` twins are the SAME MULTISET on the opposite polarity: identical read
# sizes, identical CB sizes, identical instruction counts, only swapped between
# fast and slow physical rows. They are the control that separates "positional
# redistribution helped" from "this width mix is just faster".
#
# Row 0 is the SLOW end of the measured gradient and row 7 the FAST end, so a
# `wide-to-fast` ladder is ASCENDING in y.
# ---------------------------------------------------------------------------
LADDERS = {
    # uniform: byte-identical to the op's current plan except that it is emitted
    # as 8 one-row groups instead of 1 eight-row group. The GROUPING CONTROL —
    # it prices the extra 7 core ranges / 21 kernel descriptors on their own.
    "uniform8": (8, 8, 8, 8, 8, 8, 8, 8),
    # gentle 9/7: 32 cores at 576 B, 32 at 448 B
    "half97": (7, 7, 7, 7, 9, 9, 9, 9),
    "half97_rev": (9, 9, 9, 9, 7, 7, 7, 7),
    # stronger 10/6: 640 B / 384 B
    "half106": (6, 6, 6, 6, 10, 10, 10, 10),
    "half106_rev": (10, 10, 10, 10, 6, 6, 6, 6),
    # per-row gradient, gentle
    "grad_soft": (6, 6, 7, 7, 9, 9, 10, 10),
    "grad_soft_rev": (10, 10, 9, 9, 7, 7, 6, 6),
    # per-row gradient sized to EQUALIZE the measured baseline NCRISC row means
    # (7271/7660/7250/6011/4788/3949/3608/2982 ns): w[y] ~ 1/duration[y],
    # renormalized to sum 64 and rounded. The "if M2 is exactly right, this is
    # the flat-finish plan" rung.
    "grad_full": (5, 5, 5, 6, 8, 10, 11, 14),
    "grad_full_rev": (14, 11, 10, 8, 6, 5, 5, 5),
    # Round 2. `grad_full` was sized off the NCRISC row means, but the WALL is
    # BRISC's (12.9 us vs 7.0), and BRISC's own row gradient is far flatter
    # (6969..11500 = 1.65x, against NCRISC's 2.7x). Sizing w[y] ~ 1/BRISC_mean[y]
    # from the baseline map (10233/11272/11500/10794/9289/9486/8874/6969) and
    # renormalizing to sum 64 gives 7.49/6.80/6.66/7.10/8.25/8.08/8.63/10.99,
    # i.e. the ladder below. This is the HONEST BEST ATTEMPT at M2's flat finish.
    "grad_brisc": (7, 7, 7, 7, 8, 8, 9, 11),
    "grad_brisc_rev": (11, 9, 8, 8, 7, 7, 7, 7),
    # The gentlest skew the integer width grid can express: only the two extreme
    # row-pairs move, and no read leaves the 448..576 B window. If even this is
    # null the null is not an artifact of over-shooting into small transactions.
    "edge98": (7, 7, 8, 8, 8, 8, 9, 9),
    "edge98_rev": (9, 9, 8, 8, 8, 8, 7, 7),
}
# Modes that are a width ladder over the grid rows.
LADDER_MODES = tuple(LADDERS)
# Non-ladder modes.
EXTRA_MODES = (
    "baseline",  # the op's current approach, untouched
    "checker97",  # 9/7 alternating WITHIN each grid row: same multiset, ZERO row correlation
)
MODES = EXTRA_MODES + LADDER_MODES


def parse_modes(env="TILIZE_SKEW_MODES", default="baseline"):
    modes = [m.strip() for m in os.environ.get(env, default).split(",") if m.strip()]
    for m in modes:
        assert m in MODES, f"unknown mode {m!r}; known: {MODES}"
    return modes


# ---------------------------------------------------------------------------
# Plan surgery
# ---------------------------------------------------------------------------


class SkewPlan(pd.TilizePlan):
    """A `TilizePlan` whose `groups` is an explicit list of `ColumnGroup`s.

    Subclassed rather than monkeypatching `TilizePlan.groups`, so the real op's
    class is never mutated even inside this process.
    """

    __slots__ = ("_groups",)

    @property
    def groups(self):
        return list(self._groups)


def _row_major_cores(plan):
    """The plan's cores in grid row-major order, from its own assignment."""
    return [a[0] for a in plan.assignment]


def _group_for(plan, cores, width, col_offset, elem_size, blocks_per_core=1):
    """One `ColumnGroup`: `blocks_per_core` blocks of `width` tiles per core.

    `assignment` hands core `i` the CONTIGUOUS block ids
    `[i*k, (i+1)*k)`; the kernels turn each into `w_chunk = block_id %
    num_w_chunks` and `col_base = col_tile_offset + w_chunk * block_width_tiles`,
    so a core's `k` blocks are `k` adjacent column chunks and the group as a
    whole covers `len(cores) * k * width` contiguous columns starting at
    `col_offset`. `k == 1` is the one-block-per-core case.
    """
    num_chunks = len(cores) * blocks_per_core
    row_bytes = width * pd.TILE_WIDTH * elem_size
    wrpb = max(1, math.ceil(pd.WRITE_BATCH_MIN_TILES / width))
    return pd.ColumnGroup(
        block_width_tiles=width,
        num_w_chunks=num_chunks,
        col_tile_offset=col_offset,
        block_row_bytes=row_bytes,
        write_rows_per_barrier=wrpb,
        num_blocks=num_chunks,
        cores=pd._cores_to_range_set(cores),
        assignment=[(c, i * blocks_per_core, blocks_per_core, 1) for i, c in enumerate(cores)],
        input_depth_rows=plan.input_depth_rows,
        output_depth_batches=plan.output_depth_batches,
        is_retile=plan.is_retile,
        pad_active=plan.pad_active,
        # Off by construction on every geometry this skew is expressible on: the
        # split-reader gate is `block_row_bytes <= 256` and the skew only ever
        # runs where the uniform width is already >= 512 B. Asserted below.
        split_reader=0,
    )


def widths_for(mode, cores, tensor_col_tiles, num_cores, blocks_per_core=1):
    """[(core, width), ...] in COLUMN order, or None if `mode` is the baseline.

    The list order is the order the column axis is walked, so consecutive equal
    widths become one `ColumnGroup`. Widths sum to `tensor_col_tiles` exactly —
    asserted by the caller, since a skew that moved different bytes would not be
    a comparison at all.
    """
    if mode == "baseline":
        return None
    xs = sorted({int(c.x) for c in cores})
    ys = sorted({int(c.y) for c in cores})
    if len(xs) * len(ys) != num_cores:
        return None  # not a full rectangle: no positional ladder to build
    by_row = {y: [c for c in cores if int(c.y) == y] for y in ys}

    # A ladder is written at the focus shape's scale (8 rows x 8 cores x width 8
    # = 64 tiles per row of the grid, `sum(ladder) == 64`). At another `C` the
    # SAME SHAPE of skew is the ladder scaled by `C / 512`, which keeps every
    # rung's ratio to the uniform width identical — so the second regime tests
    # the same intervention, not a differently-shaped one. A `C` that does not
    # scale to whole tiles has no equivalent ladder and is skipped.
    per_row_cores = num_cores // len(ys)
    uniform_total = sum(LADDERS["uniform8"]) * per_row_cores * blocks_per_core
    if tensor_col_tiles % uniform_total:
        return None
    scale = tensor_col_tiles // uniform_total

    if mode == "checker97":
        # 9/7 alternating by x WITHIN each row -> identical multiset to half97,
        # but corr(width, grid_row) == 0. Isolates "two widths cost something"
        # from "the positional assignment of the two widths matters".
        out = []
        for y in ys:
            for i, c in enumerate(sorted(by_row[y], key=lambda c: int(c.x))):
                out.append((c, scale * (9 if i % 2 == 0 else 7)))
        return out

    ladder = LADDERS[mode]
    assert len(ladder) == len(ys), f"ladder {mode} has {len(ladder)} rungs for {len(ys)} grid rows"
    out = []
    for i, y in enumerate(ys):
        for c in sorted(by_row[y], key=lambda c: int(c.x)):
            out.append((c, scale * ladder[i]))
    return out


def build_skew_plan(plan, mode: str, elem_size: int):
    """`SkewPlan` for `mode`, or `plan` unchanged when the skew is not expressible.

    Returns `(plan_or_skewplan, reason)`; `reason` is None when the skew applied.
    """
    if mode == "baseline":
        return plan, None
    # --- expressibility preconditions (see the module docstring) -----------
    if plan.tail_group is not None:
        return plan, "plan already has a ragged tail group"
    if plan.is_retile or plan.pad_active or plan.input_native or plan.output_native:
        return plan, "retile / padded / native-shard leg"
    if plan.input_pages_per_row > 1:
        return plan, "sub-row-paged source: the width faces its own constraint"
    if plan.num_row_groups != 1:
        return plan, (
            f"num_row_groups == {plan.num_row_groups} > 1: every grid row spans the same "
            "w_chunks, so a row-graded column width is arithmetically impossible"
        )
    cores = _row_major_cores(plan)
    num_cores = len(cores)
    if plan.num_blocks_total % num_cores or plan.num_w_chunks != plan.num_blocks_total:
        return plan, f"blocks ({plan.num_blocks_total}) do not divide evenly over {num_cores} cores"
    blocks_per_core = plan.num_blocks_total // num_cores

    spec = widths_for(mode, cores, plan.tensor_col_tiles, num_cores, blocks_per_core)
    if spec is None:
        return plan, "grid is not a full rectangle, or C has no equivalent ladder at this scale"
    total = sum(w for _, w in spec)
    assert (
        total * blocks_per_core == plan.tensor_col_tiles
    ), f"{mode}: widths sum to {total} x {blocks_per_core} blocks/core, C is {plan.tensor_col_tiles}"
    assert len(spec) == num_cores, f"{mode}: {len(spec)} widths for {num_cores} cores"

    # Runs of equal width -> one ColumnGroup each, offsets accumulating.
    groups = []
    offset = 0
    i = 0
    while i < len(spec):
        j = i
        while j < len(spec) and spec[j][1] == spec[i][1]:
            j += 1
        width = spec[i][1]
        run = [c for c, _ in spec[i:j]]
        assert (
            width * pd.TILE_WIDTH * elem_size > pd.SPLIT_READER_MAX_ROW_BYTES
        ), f"{mode}: width {width} would arm the split reader; this bench keeps it off"
        groups.append(_group_for(plan, run, width, offset, elem_size, blocks_per_core))
        offset += width * len(run) * blocks_per_core
        i = j
    assert offset == plan.tensor_col_tiles

    skew = SkewPlan(**{s: getattr(plan, s) for s in pd.TilizePlan.__slots__})
    skew._groups = groups
    return skew, None


# ---------------------------------------------------------------------------
# The runtime wrap
# ---------------------------------------------------------------------------

_ORIG_DERIVE_PLAN = pd.derive_plan
_INSTALLED = False
_CACHE: dict = {}

CURRENT_MODE = "baseline"
LAST_PLANS: list = []
LAST_REASON: dict = {}


def set_mode(mode: str):
    global CURRENT_MODE
    assert mode in MODES, mode
    CURRENT_MODE = mode


def _patched_derive_plan(input_tensor, output_tensor, **kw):
    base = _ORIG_DERIVE_PLAN(input_tensor, output_tensor, **kw)
    key = (CURRENT_MODE, id(base))
    got = _CACHE.get(key)
    if got is None:
        got = build_skew_plan(base, CURRENT_MODE, int(input_tensor.element_size()))
        _CACHE[key] = got
    plan, reason = got
    LAST_REASON[CURRENT_MODE] = reason
    LAST_PLANS.append(plan)
    return plan


def install():
    global _INSTALLED
    if not _INSTALLED:
        pd.derive_plan = _patched_derive_plan
        pd._PLAN_CACHE.clear()
        _INSTALLED = True


def plan_summary(plan):
    """[(width, num_cores, col_offset, read_bytes), ...] + the core total."""
    rows = []
    for g in plan.groups:
        rows.append(
            {
                "width": int(g.block_width_tiles),
                "cores": len(g.assignment),
                "col_offset": int(g.col_tile_offset),
                "row_bytes": int(g.block_row_bytes),
                "core_xy": sorted((int(c.x), int(c.y)) for c, *_ in g.assignment),
            }
        )
    return rows


def cores_used(plan):
    return sum(len(g.assignment) for g in plan.groups)
