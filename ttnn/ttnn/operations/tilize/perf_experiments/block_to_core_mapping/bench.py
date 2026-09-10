# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off for idea `block_to_core_mapping` — PERF EXPERIMENT, not the
real op.

MECHANISM UNDER TEST. The op's plan hands block id `b` to the `b`-th core in the
grid's ROW-MAJOR order (`derive_plan` -> `split_work_to_cores(..., row_wise=True)`
-> `corerange_to_cores(group, None, True)`), so `block_id == core_index`. Every
address in all three kernels is derived from the block's OWN id, and the issue
rotations in `tilize_stick_read.hpp` / `tilize_writer.cpp` are keyed on
`block_id` too — so permuting WHICH core owns WHICH block is bit-identical by
construction and changes only (a) the physical position of the core doing a
given transfer, i.e. its NoC route to the bank, and (b) which core's rotation
phase is in flight next to which.

HOW IT IS APPLIED. `pd.derive_plan` is wrapped (a runtime monkeypatch of the
module attribute — no file of the op is touched) and the returned plan's
`assignment` list has its CORE COLUMN permuted: entry `i` keeps its
`(start_block_id, num_blocks, block_stride)` and is handed to `cores[perm[i]]`
instead of `cores[i]`. The core SET, the CoreRangeSet, the CB sizes, the
compile-time args and the block extents are all untouched, so the number of
cores used is invariant across every mode (checked in the test).

One mode per process (`TILIZE_PERM_MODE`), because `ttnn.generic_op` may reuse a
cached program across calls in one process and a second mode would then be
measured running the first mode's runtime args.

MEASURED RESULT (Wormhole B0 n150, 8x8 = 64/64 cores, 1 GHz, 12 DRAM banks;
`run_safe_pytest.sh --profile`, DEVICE KERNEL DURATION [ns]) — **NULL**.

Focus shape [1,1,32,16384] bf16 RM DRAM -> TILE DRAM, medians (9 fresh reps for
baseline and diag, 3 for the rest; every rep bit-identical, always 64/64 cores):

    baseline 12753   diag 12401   bitrev 12539   colmajor 12571
    snake    12709   reverse 12774  stride9 12909  stride5 13099

All inside the run-to-run band (baseline's own 9 reps span 12388..13751, 11%).
The `diag` -3% did not reproduce in the sweep run (baseline 12356 vs diag 12431
on the same shape), so it is noise.

WHY it is null — the per-core tail is attached to the CORE, not to the BLOCK.
`percore_map.py` on the same runs (last dispatch, per-core `*-KERNEL` spans):

  * `corr(NCRISC duration, grid_row)` = -0.94..-0.97 in EVERY mode, including
    `bitrev`/`stride5`/`stride9`, where `corr(duration, block_id)` collapses to
    ~0, and including `reverse`, where it FLIPS to +0.98. The gradient stays put
    while the blocks move.
  * NCRISC row means are the same curve in every mode: 7.2-7.6 us at y=0..2
    falling monotonically to 2.8-3.0 us at y=7. BRISC likewise (y=7 ~5-7 us,
    y=1..2 ~11-11.5 us) plus a mode-invariant COLUMN component (x=3..6 slow).
  * max/mean stays 1.40-1.52 (NCRISC) and 1.23-1.32 (BRISC) under every
    permutation, `bitrev` (maximal dispersion) included.
  * START spread is only ~190 ns against a 4.6-7.6 us END spread, so the tail is
    NOT dispatch go-signal skew either: cores start together and DRAIN in a
    fixed positional order.

The read side additionally cannot be permuted even in principle on this shape:
every core reads a 512 B slice of ALL 32 input pages, so every core's bank set
is identical whatever block it owns.
"""

import importlib
import math
import os

import ttnn

import ttnn.operations.tilize.tilize_program_descriptor as pd

tilize_mod = importlib.import_module("ttnn.operations.tilize.tilize")

MODES = (
    "baseline",  # identity: block_id == row-major core index (the op's current approach)
    "colmajor",  # block i -> the i-th core in COLUMN-major order
    "snake",  # row-major, but every odd grid row traversed right-to-left
    "reverse",  # block i -> core N-1-i
    "stride5",  # block i -> core (5*i) % N   (5 coprime with 64)
    "stride9",  # block i -> core (9*i) % N
    "bitrev",  # block i -> core bit_reverse(i) (maximal dispersion, N a power of 2)
    "diag",  # block i -> core (x = (i + row) % W, y = row): a per-row diagonal shift
)


def current_mode_from_env() -> str:
    mode = os.environ.get("TILIZE_PERM_MODE", "baseline").strip()
    assert mode in MODES, f"unknown TILIZE_PERM_MODE {mode!r}; known: {MODES}"
    return mode


def _perm(cores, mode):
    """`perm[i]` = index in `cores` of the core that should own entry `i`."""
    n = len(cores)
    idx = list(range(n))
    if n <= 1 or mode == "baseline":
        return idx
    xy = [(int(c.x), int(c.y)) for c in cores]
    by_coord = {c: i for i, c in enumerate(xy)}
    xs = sorted({x for x, _ in xy})
    width = len(xs)
    if mode == "colmajor":
        return sorted(idx, key=lambda i: (xy[i][0], xy[i][1]))
    if mode == "snake":
        return sorted(idx, key=lambda i: (xy[i][1], xy[i][0] if xy[i][1] % 2 == 0 else -xy[i][0]))
    if mode == "reverse":
        return list(reversed(idx))
    if mode in ("stride5", "stride9"):
        k = int(mode[6:])
        if math.gcd(k, n) != 1:
            return idx  # not a permutation at this core count -> identity
        return [(k * i) % n for i in idx]
    if mode == "bitrev":
        if n & (n - 1):
            return idx  # only defined on a power-of-two core count
        bits = n.bit_length() - 1
        return [int(format(i, f"0{bits}b")[::-1], 2) for i in idx]
    if mode == "diag":
        # Only well-defined on a full rectangle; fall back to identity otherwise.
        ys = sorted({y for _, y in xy})
        if width * len(ys) != n:
            return idx
        out = []
        for i in idx:
            row, col = divmod(i, width)
            out.append(by_coord[(xs[(col + row) % width], ys[row])])
        return out
    raise AssertionError(mode)


def _permute_assignment(assignment, mode):
    cores = [a[0] for a in assignment]
    perm = _perm(cores, mode)
    assert sorted(perm) == list(range(len(cores))), "not a permutation"
    return [(cores[perm[i]],) + tuple(assignment[i][1:]) for i in range(len(assignment))]


_ORIG_DERIVE_PLAN = pd.derive_plan
_BASE_ASSIGNMENTS: dict = {}
_INSTALLED = False


LAST_PLANS: list = []


def _patched_derive_plan(*args, **kwargs):
    plan = _ORIG_DERIVE_PLAN(*args, **kwargs)
    # The plan is MEMOIZED by the op, so the permutation must be applied to a
    # pristine base every time rather than to the already-permuted list.
    base = _BASE_ASSIGNMENTS.get(id(plan))
    if base is None:
        base = (
            list(plan.assignment),
            None if plan.tail_group is None else list(plan.tail_group.assignment),
        )
        _BASE_ASSIGNMENTS[id(plan)] = base
    mode = current_mode_from_env()
    plan.assignment = _permute_assignment(base[0], mode)
    if base[1] is not None:
        plan.tail_group.assignment = _permute_assignment(base[1], mode)
    LAST_PLANS.append(plan)
    return plan


def assignment_map(plan):
    """[(core_x, core_y, start_block_id, num_blocks), ...] over every group."""
    out = []
    for group in plan.groups:
        for core, start, n, _stride in group.assignment:
            out.append((int(core.x), int(core.y), int(start), int(n)))
    return out


def install():
    global _INSTALLED
    if not _INSTALLED:
        pd.derive_plan = _patched_derive_plan
        pd._PLAN_CACHE.clear()
        _INSTALLED = True


def plan_for(input_tensor, output_tensor, *, low_l1=False):
    grid_size = input_tensor.device().compute_with_storage_grid_size()
    return pd.derive_plan(input_tensor, output_tensor, low_l1=low_l1, grid=grid_size)
