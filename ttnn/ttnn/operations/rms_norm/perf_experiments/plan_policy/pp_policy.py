# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""The CANDIDATE plan policy, applied as a process-local patch (the op is untouched).

WHAT THE MEASUREMENT SAYS (see the bench tables in the round report):

  * The G policy (`_choose_group_size` + `_split_cost`) picks the measured optimum,
    or a value inside the +-1% run-to-run band of it, on 13 of the 14 guard-set
    cells.  Re-fitting `_COMBINE_FIXED_TILES` / `_COMBINE_PER_CORE_TILES` has
    nothing to correct there.
  * The one material miss is NOT on the G axis at all.  It is the ROW-BLOCK axis:
    `_solve` caps `max_block_ht` at `ceil(Rt / row_parallel_units)` — "never
    coarser than the block that still keeps every parallel unit busy" — and on
    `prefill_1024` @ bfloat8_b that cap (BLOCK_HT = 2, 128 of 130 cores busy) is
    1.108x slower than the block L1 can actually afford at unchanged CB depths
    (BLOCK_HT = 4, 64 cores busy).  Controls: the same 64-core grid forced back to
    BLOCK_HT = 2 measures the same as the 128-core plan (58.1 / 58.7 / 58.1 vs
    58.2 / 58.6 / 58.0 us), so the win is the coarser block and not the smaller
    core count; and BLOCK_HT = 5, which is affordable only by dropping
    IN_BUF_DEPTH 4 -> 3, gives the win straight back (57.3-57.9 us).

THE PROPOSAL — one line, in `_solve`:

    max_block_ht = max(1, _div_up(Rt, max(1, row_parallel_units)))   # current
    max_block_ht = Rt                                               # proposed

i.e. stop capping the row block by the core count and let the EXISTING growth
loop stop it, which it does exactly where L1 stops affording the streaming depth
profile the ladder already chose ("coarseness is never worth a buffer
generation", the rule the W-chunk search already runs on).  Everything else —
the depth ladder, the G policy, the W-chunk search — is unchanged.

`row_parallel_units` is used for NOTHING else inside `_solve`, so the patch below
(pass 1, i.e. uncapped) is exactly the proposed policy and not an approximation
of it.
"""

from __future__ import annotations

import contextlib

from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd


@contextlib.contextmanager
def proposed_policy(min_blocks: int = 0):
    """Run under the candidate policy: the row block is not capped by core count.

    `min_blocks` is the optional guard rail — refuse to coarsen past the point
    where fewer than `min_blocks` row-blocks remain — so the "does a
    parallelism-starved shape regress?" question can be measured with the guard on
    AND off instead of argued.  0 = no guard (the pure one-line proposal).
    """
    orig = pd._solve

    def patched(*, Rt, row_parallel_units, **kw):
        solved = orig(Rt=Rt, row_parallel_units=1, **kw)
        if min_blocks and solved.num_row_blocks < min_blocks:
            # Walk back to the coarsest block that still leaves min_blocks blocks.
            cap = max(1, Rt // min_blocks)
            solved = orig(Rt=Rt, row_parallel_units=pd._div_up(Rt, cap), **kw)
        return solved

    pd._solve = patched
    try:
        yield
    finally:
        pd._solve = orig


@contextlib.contextmanager
def depth_preserving_policy(min_blocks: int = 0):
    """Candidate policy B — the one the measurements actually support.

    Same one-line site as `proposed_policy`, but the row block is grown by the
    rule the W-CHUNK search already runs on (and which round 1 measured on 14
    cases): take the COARSEST row block that still affords the streaming depth
    PROFILE (IN_DEPTH_CAP, out 2, rm 2, gamma as solved) — never one that has to
    be paid for out of a buffer generation.  The core-count cap
    (`ceil(Rt / row_parallel_units)`) is dropped; L1-at-full-depth is what stops
    the growth.

    Why the profile and not "coarsest that fits at all" (= `proposed_policy`):
    `row_major` and `prefill_1024` @ bfloat8_b both go COARSER under the naive
    rule by dropping IN_BUF_DEPTH, and both measure SLOWER for it (95.5 vs 91.8 us,
    59.4 vs 58.5 us).  The profile rule stops exactly where the depth does.
    """
    orig = pd._solve

    def patched(
        *,
        Rt,
        row_parallel_units,
        Wt_core,
        w_split_group,
        dest_limit,
        l1_cb_budget,
        gamma_cap_tiles,
        layout_common,
        levers,
        **kw,
    ):
        base = orig(
            Rt=Rt,
            row_parallel_units=row_parallel_units,
            Wt_core=Wt_core,
            w_split_group=w_split_group,
            dest_limit=dest_limit,
            l1_cb_budget=l1_cb_budget,
            gamma_cap_tiles=gamma_cap_tiles,
            layout_common=layout_common,
            levers=levers,
            **kw,
        )
        common = dict(layout_common, Wt_core=Wt_core, w_split_group=w_split_group)

        def ws(block_ht):
            return pd._working_set_bytes(
                regime=base.regime,
                block_ht=block_ht,
                in_depth=pd.IN_DEPTH_CAP,
                out_depth=2,
                rm_depth=2,
                gamma_depth=base.GAMMA_DEPTH,
                wr=base.WT_REDUCE_BLOCK,
                ws=base.WT_SCALE_BLOCK,
                gamma_ingest_block=base.GAMMA_INGEST_BLOCK,
                **common,
            )

        cap = min(Rt, dest_limit)
        best = base.BLOCK_HT
        for b in range(cap, base.BLOCK_HT, -1):
            if min_blocks and pd._div_up(Rt, b) < min_blocks:
                continue
            if ws(b) <= l1_cb_budget:
                best = b
                break
        if best == base.BLOCK_HT:
            return base
        return orig(
            Rt=Rt,
            row_parallel_units=pd._div_up(Rt, best),
            Wt_core=Wt_core,
            w_split_group=w_split_group,
            dest_limit=dest_limit,
            l1_cb_budget=l1_cb_budget,
            gamma_cap_tiles=gamma_cap_tiles,
            layout_common=layout_common,
            levers=levers,
            **kw,
        )

    pd._solve = patched
    try:
        yield
    finally:
        pd._solve = orig


# The measured DRAM-saturation width of this grid, in concurrently-reading cores.
# Round 1's A0 core sweep on (1,1,8192,1024): full grid 93,535 ns, 96 cores 95,983,
# 64 cores 94,470 (== full grid, within the band), 32 cores 110,136 (1.17x WORSE).
# So >= ~64 readers saturate DRAM and any core past that buys nothing; below it,
# parallelism is still the wall.  Re-measurable with `_levers=dict(active_cores=N)`.
MIN_SATURATING_BLOCKS = 64


@contextlib.contextmanager
def coarsen_after_g(min_blocks: int = MIN_SATURATING_BLOCKS):
    """Candidate policy C — the one that is right on every guard-set cell.

    Two changes from B, both of which the measurements forced:

      * the coarsening is applied to the CHOSEN plan only, never inside the G
        scoring.  `_split_cost` prices a candidate by its per-core TILE COUNT, so
        feeding it coarser blocks makes it prefer a plan that splits W instead
        (bfloat8_b prefill flips G=1 -> G=2 and lands on 54.2 us instead of the
        52.7 us optimum).  The G policy already picks the measured optimum on 13
        of 14 guard cells; it is left exactly as it is.
      * the coarsening stops while at least `min_blocks` row-blocks remain, i.e.
        while the grid is still DRAM-saturated.  Without it, `h_nonalign` (Rt = 4)
        coarsens 4 row-blocks onto 2 cores and measures 7.4 -> 11.5 us (0.64x).

    Result on the guard set: exactly ONE plan changes — `prefill_1024` @ bfloat8_b,
    BLOCK_HT 2 -> 4 (128 -> 64 cores), 58.0 -> 52.8 us (1.10x) — and every other
    cell's plan is byte-identical, focus case included.
    """
    orig = pd._solve
    depth = depth_preserving_policy(min_blocks)
    inner = None

    def patched(**kw):
        # Only the FINAL solve (the one blocking_plan makes for the chosen G) is
        # coarsened.  `_choose_group_size` sets `_scoring` while it enumerates.
        if patched.scoring:
            return orig(**kw)
        return inner(**kw)

    patched.scoring = False

    orig_choose = pd._choose_group_size

    def choose(**kw):
        patched.scoring = True
        try:
            return orig_choose(**kw)
        finally:
            patched.scoring = False

    with depth:
        inner = pd._solve  # the depth-preserving solve installed by `depth`
        pd._solve = patched
        pd._choose_group_size = choose
        try:
            yield
        finally:
            pd._solve = orig
            pd._choose_group_size = orig_choose


# --- The graduation, in the op's own terms -----------------------------------
# `coarsen_after_g` above is the measured policy applied as a process-local patch.
# Graduating it is this edit to rms_norm_program_descriptor.blocking_plan — the G
# policy, the depth ladder and the W-chunk search are all untouched.
GRADUATION_PATCH = '''
# module scope, beside _COMBINE_FIXED_TILES:

# Row-blocks below which the grid stops saturating DRAM.  MEASURED, not assumed:
# the A0 core sweep on (1,1,8192,1024) reads 93,535 ns at 130 cores, 94,470 at 64
# (the same, within the band) and 110,136 at 32 (1.17x worse) — so ~64 concurrent
# readers already hold the DRAM roof and every core past that buys nothing, while
# below it parallelism is still the wall.  Re-measure with
# `_levers=dict(active_cores=N)`.
MIN_SATURATING_BLOCKS = 64


def _coarsen_row_block(solved, *, Rt, dest_limit, l1_cb_budget, resolve):
    """Grow the CHOSEN plan's row block past the one-block-per-core cap.

    `_solve` caps BLOCK_HT at `ceil(Rt / row_parallel_units)` — the coarsest block
    that still keeps every parallel unit busy.  On a shape whose grid is already
    DRAM-saturated that cap is the wrong objective: the extra cores buy no
    bandwidth, and the finer row block costs per-block overhead.  Measured on
    (1,1,8192,1024) @ bfloat8_b: BLOCK_HT 2 on 128 cores 58.2 us vs BLOCK_HT 4 on
    64 cores 52.7 us (1.10x); the same 64 cores forced back to BLOCK_HT 2 measure
    58.1 us, so it is the block and not the core count.

    Two brakes, both measured:
      * the coarser block must still afford the streaming depth PROFILE
        (IN_DEPTH_CAP, out 2, rm 2) — the rule the W-chunk search already runs on.
        Without it the search takes BLOCK_HT 5 by dropping IN_BUF_DEPTH 4 -> 3 and
        gives the whole win back (57.3-57.9 us), and `row_major` goes 91.8 -> 95.5.
      * at least MIN_SATURATING_BLOCKS row-blocks must remain.  Without it
        `h_nonalign` (1,1,100,736) coarsens its 4 row-blocks onto 2 cores:
        7.4 -> 11.5 us (0.64x).

    Runs on the chosen plan ONLY, never inside `_choose_group_size`: `_split_cost`
    prices a candidate by per-core tile count, so scoring coarsened candidates
    flips bfloat8_b prefill to G=2 and lands on 54.2 us instead of 52.7.
    """
    for b in range(min(Rt, dest_limit), solved.BLOCK_HT, -1):
        if _div_up(Rt, b) < MIN_SATURATING_BLOCKS:
            continue
        if ws_bytes_at_profile(solved, block_ht=b) <= l1_cb_budget:
            return resolve(row_parallel_units=_div_up(Rt, b))
    return solved

# in blocking_plan, immediately after the existing `solved = _solve(...)`:
solved = _coarsen_row_block(solved, Rt=Rt, dest_limit=dest_limit,
                            l1_cb_budget=l1_cb_budget, resolve=<the same _solve call>)
'''
