# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""regime_b_resident lab: a MASKED resident-x plan for the shapes the W-split cannot take.

FORKED FROM THE CURRENT OP (`rms_norm_program_descriptor.py`, commit 034d34f)
so the BASELINE arm is the op as it ships TODAY - W-split, fused Regime B,
one-packet reads and all.  `assert_matches_op_plan()` gates that: with
`allow_c=0, allow_masked_resident=0` every field of this plan and its whole CB
layout must equal `opd.blocking_plan()`.

THE IDEA
--------
Regime A reads x from DRAM ONCE (resident in cb_input_tiles, reused by the scale
pass).  Regime B reads it TWICE.  Two things keep shapes in B:

  (i)  L1: Regime A needs the WHOLE per-core width resident for FIVE CBs
       (x + gamma + normed + out + ...), not just x.
  (ii) THE MASK: `maskless_w` (W % 32 == 0, or ROW_MAJOR).  A non-tile-aligned
       W on the TILE path has pad columns in the last W-tile that must never
       enter the sum, and Regime A's fused element-wise accumulator collapses
       all Wt columns onto ONE tile, whose 32 columns then carry a MIX of valid
       and pad contributions - there is no column position left to mask.

This lab removes BOTH:

  (i)  REGIME C - a ladder rung between A and B that keeps ONLY x resident and
       chunks the scale pass (gamma / normed / out sized per chunk).  Selected
       by a real property: does THIS CB set fit the L1 budget at this dtype and
       blocking.
  (ii) THE MASKED RESIDENT SUM-OF-SQUARES - the last W-tile gets its OWN
       accumulator (`sumsq_strided<Wt_core-1>(0)` then `sumsq_strided<1>(Wt-1)`,
       two `fold_partial_sum` calls into cb_sumsq), so the 32 columns of the
       second accumulator map 1:1 onto the last W-tile and the op's EXISTING
       masked fold - `partial_mask(W_PARTIAL, 1)` on AccumulateViaAdd,
       `last_tile_at(1)` on ReduceTile - zeroes precisely the pad columns.
       This is the same shape Regime B's `sumsq_chunk` third branch already
       runs; the only new thing is running it over the RESIDENT window
       (row stride Wt_core instead of WT_REDUCE_BLOCK).

COMPOSITION WITH THE SHIPPED W-SPLIT (load-bearing)
---------------------------------------------------
`_choose_group_size` is called with the ladder switched OFF, so G is chosen
BYTE-IDENTICALLY to the op.  The ladder is then applied only to the final
per-core solve, and only when the split already returned G == 1.  A shape the
split takes (the focus case) therefore cannot move at all.

MEASURED (Blackhole p150, 13x10 grid, DEVICE KERNEL DURATION ns, median of 4,
zones compiled out, bf16/HiFi2/fp32_dest_acc_en=False unless the case says
otherwise; BASE = the shipped op, LADDER = this module at its defaults)

  WINS - masked TILE, more than one row-block per core:
    (1,1, 8192,4095) bf16       588,825 -> 373,396   1.577x
    (1,1,16384,4095) bf16     1,203,270 -> 725,305   1.659x
    (1,1, 8192,6143) bf16       875,916 -> 556,424   1.574x
    (1,1, 8192,4095) bfloat8_b  335,299 -> 229,283   1.462x
    (1,1, 8192,4095) fp32 @HiFi4/fp32_dest  1,258,401 -> 803,502  1.566x
    (1,1, 8192,4095) bf16 @fp32_dest_acc_en=True (ReduceTile masked datapath)
                                590,104 -> 371,761   1.587x
  NO CHANGE (the gate returns Regime B / Regime A; plan byte-identical):
    focus (1,1,32,7168) 8,700 -> 8,610 (1.010x)  prefill_7168 1.006x
    wide_32768 1.003x  grid_starved 1.002x  row_major 1.003x  rm_nonalign 1.004x
    prefill_1024 1.006x  decode_1024 1.021x  bf8b 0.998x  fp32 0.995x
  WHY THE GATE (measured counterfactual, C FORCED at one row-block per core):
    (1,1,32,4095) 1.000x | (1,1,32,4127) 0.994x | (1,1,64,6143) 0.986x
    (1,1,32,1057) 1.014x | (32,17) 1.015x  -> flat
    (1,1,32,4095) bfloat8_b 0.971x | float32 0.908x -> measured regression

KNOBS (0 / "" everywhere == behave exactly like the op)
    allow_c                  0 -> Regime C is never selected
    allow_masked_resident    0 -> the maskless_w requirement stays (op behaviour)
    force_regime_lab         "A" | "B" | "C" -> pin the per-core plan
    c_ws / c_in_depth / c_out_depth / c_gamma_depth / c_normed_depth
    c_resident_gamma         -1 AUTO | 0 chunked | 1 resident
    no_zones                 1 -> compile the kernels with the zones no-op'd

Original op docstring follows.
---------------------------------------------------------------------------
Program descriptor for rms_norm (op_design.md "Blocking Model" / "Work Distribution").

Everything blocking-related is decided in exactly ONE place — `blocking_plan()` —
and every downstream quantity (CB page counts, kernel CT/RT args, loop trip
counts, grid sizing) reads a field off the returned frozen dataclass.  Turning a
knob is a one-line change inside `blocking_plan`.

Axes:
    Rt  (tile-rows)     INDEPENDENT  -> split across combine groups, block
                                        factor BLOCK_HT.
    Wt  (tile-columns)  DEPENDENT    -> split across the G cores of a combine
                                        group (`_choose_group_size`); block
                                        factors WT_REDUCE_BLOCK /
                                        WT_SCALE_BLOCK bound L1 in Regime B.
    gamma[W]            REUSE-SHARED -> replicated per group; each core reads
                                        only its OWN column slice.

Two compute regimes, selected by a pinned host predicate:
    A  RESIDENT-FUSED    single DRAM read, fused sum-of-squares, no mask.
    B  STREAMING-MASKED  two DRAM reads, chunked fused sum-of-squares +
                         accumulating reduce, with a partial scaler that zeroes
                         the pad columns of the last W-tile.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# The lab is a FORK, but it still reads the op module for the honest-baseline gate.
from ttnn.operations.rms_norm import rms_norm_program_descriptor as opd  # noqa: E402
from ttnn.operations.rms_norm.rms_norm import validate  # noqa: E402

TILE_DIM = 32

# Single source of truth for the non-CB L1 reservation (op_design.md).
L1_RESERVED_BYTES = 256 * 1024

# Core-count knob.  None == the full compute grid (Phase 0 default, lever A0).
ACTIVE_CORE_CAP: Optional[int] = None

# --- perf levers -------------------------------------------------------------
# Every entry is a live knob whose DEFAULT is the applied (fast) setting; the
# value shown in the comment is the counterfactual "off" arm that
# `_bench_rms_norm.py` measures against.  Threaded in as `levers=dict(...)`, so a
# lever can be re-measured on a new shape without editing a kernel.
LEVER_DEFAULTS = {
    "double_buffer": 1,  # C16 - 0: force every streaming CB to depth 1
    "barrier_per_block": 1,  # B7  - 0: one noc barrier per transaction
    "noc_split": 1,  # B9  - 0: reader and writer both on NOC_0
    "row_wise": 1,  # A1  - 0: split_work_to_cores column-wise
    "active_cores": 0,  # A0  - 0: full grid; N: cap the active core count
    "block_ht": 0,  # compute_block_size - 0: solver; N: force BLOCK_HT
    "dest_block": 0,  # compute_block_size - 0: solver; N: force DEST_BLOCK
    "coarse_chunk": 1,  # block-size fidelity - 0: force the W-chunk to 1 tile
    "wt_block": 0,  # 0: solver; N: cap WT_REDUCE_BLOCK / WT_SCALE_BLOCK at N
    # Regime B reduce datapath.  1 = AccumulateViaAdd (pairwise FPU accumulate +
    # ONE SFPU within-tile finalize), 0 = the Phase-0 ReduceTile datapath.  The
    # applied default is 1 because ReduceTile's long per-tile DEST accumulation
    # carries a systematic sum-of-squares overestimate at 16-bit DEST that grows
    # with the reduced width (+0.8% at Wt=32 -> +10.4% at Wt=224); see the
    # rationale block in kernels/rms_norm_compute.cpp.
    "reduce_via_add": 1,
    # --- F-group precision levers, measured through the same `_levers` hook ----
    # These two are the ONLY knobs that reach the ComputeConfigDescriptor, and
    # they exist so F24 / F25 have a re-runnable counterfactual arm instead of an
    # ad-hoc toggle.  At their defaults `_apply_precision_levers` returns the
    # caller's descriptor untouched, which is what F23 requires.
    "dest_acc": 1,  # F25 - 0: force fp32_dest_acc_en=False (the cheap DEST width)
    "pack_precise": 0,  # F24 - 1: force the PRECISE bfloat8_b packer (applied = fast)
    "coalesce": 1,  # B5/B6 - 0: split each tile transfer into two half-tile ones
    # F-group (precision cost): 1 = the accumulator CBs carry the CHEAPEST format
    # that loses nothing (fp32 only when DEST actually accumulates in fp32);
    # 0 = the Phase-0 arm, fp32 accumulator CBs unconditionally.  Byte-identical
    # at fp32_dest_acc_en=True, where both arms pick fp32.
    "acc_narrow": 1,
    # --- /perf-measure ablation arms (payload stubbed, sync scaffolding kept) ---
    "stub_dm": 0,  # 1: reader/writer keep every CB op + barrier, issue no NoC transfer
    "stub_compute": 0,  # 1: eltwise chains keep their CB lifecycle, do no math
    # --- W-split work distribution (Perf 1 graduation) -----------------------
    # 1 (applied) = `_choose_group_size` PICKS the combine-group size G from real
    # properties; 0 = force G = 1, i.e. the pure row-parallel plan this op shipped
    # before.  G = 1 is a value of the same policy, not a second code path, so the
    # off-arm is a genuine counterfactual and not a different program shape.
    "w_split": 1,
    # D20 - 0 (applied): the pinned host predicate picks Regime A (one DRAM read of
    # x, whole per-core width resident) wherever it fits; 1: force Regime B, the
    # streaming 2-pass, so the fast path's value is a measured number rather than a
    # byte-model claim.  Measured 1.32x on (1,1,8192,1024) and 1.47x on its
    # ROW_MAJOR twin.
    "force_regime": 0,
    # 0 = let the policy choose; N = pin G = N.  This is what makes the group-size
    # calibration in `_choose_group_size` re-MEASURABLE instead of asserted: a new
    # box or a new shape can be swept without editing the policy.
    "w_group": 0,
    # --- LAB KNOBS (regime_b_resident) --------------------------------------
    # Every one of these is OFF in the baseline arm, where this module must
    # reproduce the op's plan field for field (`assert_matches_op_plan`).
    "allow_c": 1,  # Regime C (x resident, scale pass chunked)
    "allow_masked_resident": 1,  # drop `maskless_w` from the resident predicate
    "force_regime_lab": "",  # "A" | "B" | "C" - pin the per-core plan
    "c_ws": 0,
    "c_in_depth": 0,
    "c_out_depth": 0,
    "c_gamma_depth": 0,
    "c_normed_depth": 0,
    "c_resident_gamma": -1,  # -1 AUTO, 0 chunked, 1 resident
    "no_zones": 1,
    # POSITIVE CONTROL for the pad-poison gate.  1 = run the masked resident fold
    # with the mask REMOVED (everything else identical).  It MUST fail the poison
    # cases; if it passes, the poison never landed and every "correct" number on
    # those shapes is vacuous.  Never a shippable arm.
    "resident_no_mask": 0,
}

# CT arg 1 is a REGIME CODE now, not a bool: three plans, not two.
REGIME_CODE = {"B": 0, "A": 1, "C": 2}


def _lever(levers, name):
    return LEVER_DEFAULTS[name] if levers is None else levers.get(name, LEVER_DEFAULTS[name])


# Upper bound on the ROW_MAJOR-gamma staging CB (cb_gamma_rm).  The staging CB
# is pure boot/ingest scaffolding, so it gets a small slice of L1; the ingest is
# chunked at GAMMA_INGEST_BLOCK to stay inside it.
GAMMA_STAGE_MAX_BYTES = 64 * 1024

# --- Circular buffer slots (semantic names; the number is just the slot) ------
CB_INPUT_TILES = 0
CB_GAMMA_TILES = 1
CB_REDUCE_SCALER = 2
# Slot 3 is RETIRED.  It held cb_squared, the full-block x^2 intermediate of the
# old Regime B `square -> reduce` pair; the fused sum-of-squares accumulates x*x
# in DEST instead, so there is nothing to park.  The slot is left unused rather
# than renumbered: the reader addresses cb_sumsq / cb_sumsq_acc by these same
# indices for the W-split combine.
CB_SUMSQ = 4
CB_RMS_RECIP = 5
CB_NORMED = 6
CB_OUTPUT_TILES = 7
CB_RM_IN = 8
CB_RM_OUT = 9
CB_SUMSQ_ACC = 10
CB_GAMMA_RM = 11
# --- W-split combine (only created when the plan picks G > 1) -----------------
CB_PARTIAL_GATHER = 12  # reader (remote NoC writes) -> compute, group root only
CB_SUMSQ_BCAST = 13  # compute (root) -> reader, the multicast source

# Semaphores the W-split combine owns.  `mcast_pipe.hpp` requires `consumer_ready`
# to be host-initialised to 0 (remote receivers increment it with no
# happens-before to the sender's ctor), so all three are host-owned.
SEM_DATA_READY = 0
SEM_CONSUMER_READY = 1
SEM_GATHER = 2


def _div_up(a: int, b: int) -> int:
    """Ceiling division (ttnn exposes no div_up binding in this tree)."""
    return (a + b - 1) // b


# Cap on cb_input_tiles' generations.  Named because TWO places must agree on it:
# the W-chunk search's affordability profile and the depth ladder that actually
# grows the depth (see `_solve`).  If they drift, the search buys a chunk it can
# only pay for by giving the input CB back its generations.
IN_DEPTH_CAP = 4


def _largest_divisor_at_most(value: int, cap: int) -> int:
    """Largest divisor of `value` that is <= `cap` (>= 1)."""
    for cand in range(min(value, max(1, cap)), 0, -1):
        if value % cand == 0:
            return cand
    return 1


# --- Numerical format policy (ONE place; see /numeric-formats-metal) ---------
# Block-float (bfp*) is a DRAM/L1 TRANSPORT format, never a compute intermediate:
# 16 datums share one exponent, so parking x^2 or the normalized activations there
# would re-quantise a value the very next phase reads back.  A compute-only
# intermediate therefore promotes to bfloat16, which is also exactly what the
# FPU's srcA/srcB carry.
BLOCK_FLOAT_DTYPES = (ttnn.bfloat8_b, ttnn.bfloat4_b)


def _interm_dtype(input_dtype):
    """Format for a compute-only intermediate CB (cb_normed).

    Identical to the input dtype for every non-block format, so this is
    byte-identical on the float32 / bfloat16 paths.
    """
    return ttnn.bfloat16 if input_dtype in BLOCK_FLOAT_DTYPES else input_dtype


def _acc_dtype(compute_kernel_config, interm_dtype, narrow: bool):
    """Format for a CB that parks a running reduction across phases.

    The rule (/numeric-formats-metal S4): an accumulator CB must be Float32 when
    the accumulation crosses the CB *in fp32* - i.e. when DEST itself accumulates
    in fp32.  With fp32_dest_acc_en=False the value packed out of DEST has already
    been rounded to the narrow DEST datum, so a Float32 CB buys no precision while
    costing 2x the L1 pages; narrowing hands that L1 back to BLOCK_HT /
    IN_BUF_DEPTH.  `narrow=False` is the measurable Phase-0 counterfactual arm.
    """
    if bool(getattr(compute_kernel_config, "fp32_dest_acc_en", True)) or not narrow:
        return ttnn.float32
    return interm_dtype


def _apply_precision_levers(compute_kernel_config, levers):
    """Bench-only overrides of the two precision fields levers F24 / F25 measure.

    F23 boundary: with `levers` at its defaults - i.e. EVERY real call, since
    `_levers` is an internal bench hook that no public argument reaches - this
    returns the caller's descriptor object untouched, so the op still cannot
    downgrade a caller-supplied precision knob.  Only an explicit bench arm
    rebuilds it, and then only to measure the counterfactual.
    """
    force_dest_off = not _lever(levers, "dest_acc")
    force_precise = bool(_lever(levers, "pack_precise"))
    if not (force_dest_off or force_precise):
        return compute_kernel_config

    out = ttnn.ComputeConfigDescriptor()
    out.math_fidelity = compute_kernel_config.math_fidelity
    out.math_approx_mode = bool(getattr(compute_kernel_config, "math_approx_mode", False))
    out.dst_full_sync_en = bool(getattr(compute_kernel_config, "dst_full_sync_en", False))
    out.fp32_dest_acc_en = bool(getattr(compute_kernel_config, "fp32_dest_acc_en", True)) and not force_dest_off
    out.bfp8_pack_precise = bool(getattr(compute_kernel_config, "bfp8_pack_precise", False)) or force_precise
    return out


def _reduce_via_add(
    regime, compute_kernel_config, acc_dtype, W_partial: int, lever_on: bool, is_row_major=False
) -> int:
    """Pick Regime B's reduce datapath: 1 = AccumulateViaAdd, 0 = ReduceTile.

    Regime A never accumulates across reduce() calls - its reduce is the single
    within-tile finalize of sum_of_squares' DEST accumulator - so the knob only
    has meaning in Regime B.

    AccumulateViaAdd is selected only where it is BOTH needed and correct.

    NEEDED - the DEST datum is 16 bit.  There, ReduceTile's long per-tile DEST
    accumulation carries a systematic sum-of-squares OVERESTIMATE that grows with
    the reduced width (measured +0.84% at Wt=32 -> +10.4% at Wt=224).  With an
    fp32 DEST accumulator the same datapath is flat in W (~0.7% rms), so there is
    nothing to buy and the Phase-0 datapath stays.

    CORRECT - AccumulateViaAdd's PARTIAL (non-tile-aligned) path folds the 0/1
    mask tile straight out of the scaler CB with NO data-format reconfig, so a
    masked reduce is only correct when the scaler CB - mandatorily bfloat16 -
    already matches the reduce INPUT CB's format.  That input CB is `cb_sumsq_acc`
    (the fused sum-of-squares' per-row accumulator), so the predicate reads
    `acc_dtype`, not the intermediate format: a float32 accumulator CB makes the
    datapath unpack the bfloat16 mask as float32 (measured rms 0.59 at W=17 / 0.11
    at W=50 versus 0.0011 on the ReduceTile datapath).  With `acc_narrow` the
    accumulator is already bfloat16 for bfloat16 and bfloat8_b activations
    whenever DEST is 16-bit, which is exactly when this datapath is NEEDED.  A
    tile-aligned reduce uses no mask at all, so the constraint does not apply
    there.
    """
    # LAB: the MASKED resident fold is exactly the same shape - two accumulating
    # reduce calls with a mask on the second - so it takes the same datapath
    # decision.  Unmasked A / C still do a single 1-tile finalize with nothing to
    # accumulate, so they keep the op's ReduceTile (0).  In the BASELINE arm
    # `masked_resident` is unreachable (W_partial > 0 forces regime B).
    masked_resident = regime in ("A", "C") and W_partial > 0 and not is_row_major
    if not lever_on:
        return 0
    if regime != "B" and not masked_resident:
        return 0
    needed = not bool(getattr(compute_kernel_config, "fp32_dest_acc_en", True))
    correct = (W_partial == 0) or (acc_dtype == ttnn.bfloat16)
    return 1 if (needed and correct) else 0


def _elem_size(dtype) -> int:
    """Bytes per datum, or 0 for a block format (which has no per-datum size).

    Only the ROW_MAJOR path consumes this (stick byte offsets), and a block-float
    tensor cannot BE row-major - ttnn refuses to construct one ("Layout must be
    Layout::TILE for bfloat8_b or bfloat4_b"); `blocking_plan` asserts it.
    Derived from the tile size rather than Tensor.element_size(), which raises
    outright for bfp2/bfp4/bfp8.
    """
    if dtype in BLOCK_FLOAT_DTYPES:
        return 0
    return ttnn.tile_size(dtype) // (TILE_DIM * TILE_DIM)


def _f32_bits(x: float) -> int:
    return struct.unpack("I", struct.pack("f", float(x)))[0]


def _dest_limit(compute_kernel_config) -> int:
    """Host-side mirror of kernel_lib's DEST_AUTO_LIMIT (dest_helpers.hpp).

    The kernel itself always clamps against the real constant; this is only used
    to cap the host-chosen knobs so the plan is self-consistent.
    """
    full_sync = bool(getattr(compute_kernel_config, "dst_full_sync_en", False))
    base = 16 if full_sync else 8
    return base // 2 if bool(getattr(compute_kernel_config, "fp32_dest_acc_en", True)) else base


# ---------------------------------------------------------------------------
# Alignment-aware tile geometry — ceil, PER IMAGE (never floor / //).
# ---------------------------------------------------------------------------


def tile_geometry(shape, is_row_major: bool):
    """Return (Rt, Wt, W_true, W_partial, num_rows).

    TILE layout: each (..., H, W) image is tile-padded independently, so
    Rt = batch * ceil(H / 32).  Writing (batch*H)//32 is wrong for every
    multi-batch h_non_aligned shape.

    ROW_MAJOR layout: the buffer is a flat list of `batch * H` sticks and the
    per-row reduction has no cross-row term, so tile-rows are just groups of 32
    consecutive sticks -> Rt = ceil(batch * H / 32).
    """
    W_true = shape[-1]
    H = shape[-2]
    batch = 1
    for d in shape[:-2]:
        batch *= d
    Wt = _div_up(W_true, TILE_DIM)
    num_rows = batch * H
    if is_row_major:
        Rt = _div_up(num_rows, TILE_DIM)
    else:
        Rt = batch * _div_up(H, TILE_DIM)
    return Rt, Wt, W_true, W_true % TILE_DIM, num_rows


@dataclass(frozen=True)
class BlockingPlan:
    # --- geometry -----------------------------------------------------------
    Rt: int
    Wt: int
    Wt_core: int
    W_true: int
    W_partial: int
    num_rows: int
    is_row_major: bool
    has_gamma: bool
    gamma_is_row_major: bool
    tile_out: bool
    # --- byte geometry ------------------------------------------------------
    elem_size: int
    gamma_elem_size: int
    in_tile_bytes: int
    gamma_tile_bytes: int
    bf16_tile_bytes: int
    # --- numerical formats (see _interm_dtype / _acc_dtype) -----------------
    interm_dtype: object
    acc_dtype: object
    interm_tile_bytes: int
    acc_tile_bytes: int
    row_bytes: int
    gamma_row_bytes: int
    # --- knobs (every one of these is a tunable, never an inlined literal) ---
    BLOCK_HT: int
    WT_REDUCE_BLOCK: int
    WT_SCALE_BLOCK: int
    DEST_BLOCK: int
    GAMMA_INGEST_BLOCK: int
    IN_BUF_DEPTH: int
    OUT_BUF_DEPTH: int
    RM_BUF_DEPTH: int
    GAMMA_DEPTH: int
    # --- derived ------------------------------------------------------------
    regime: str
    reduce_via_add: int
    num_row_blocks: int
    NORMED_DEPTH: int
    RESIDENT_GAMMA: int
    # --- W-split work distribution (group_size == 1 IS the row-parallel plan) --
    group_size: int
    group_x: int
    group_y: int
    num_groups: int  # groups the gx*gy tiling yields on this grid
    groups_used: int  # groups that actually get row-blocks (<= num_groups)
    l1_cb_budget: int
    # The frozen CB set this plan implies: ((cb_index, num_pages, page_bytes,
    # kind), ...) straight out of _cb_layout().  create_program_descriptor
    # instantiates exactly this — it never re-derives a page count.
    cb_layout: tuple

    def working_set_bytes(self) -> int:
        return sum(pages * page_bytes for _, pages, page_bytes, _ in self.cb_layout)


# --- The ONE description of the CB set ---------------------------------------
# `_cb_layout` is the single source of truth for every circular buffer this op
# creates: its page count, its page size and which tensor's data format it
# carries.  The L1 budget solver SUMS this list, and create_program_descriptor
# INSTANTIATES the same list — so the plan can never disagree with the
# descriptors it produced (a drift that would silently mis-size the budget and
# either OOM or under-use L1 the moment a block knob is turned).
#
# `kind` names the data format, and each one resolves in exactly ONE place
# (`fmt_of_kind` in create_program_descriptor, off fields of the plan):
#   "in"     the input tensor's dtype      (carries the tensor itself)
#   "out"    the output tensor's dtype     (carries the tensor itself)
#   "gamma"  the gamma tensor's dtype      (carries the tensor itself)
#   "interm" plan.interm_dtype  - compute-only intermediate, never block-float
#   "acc"    plan.acc_dtype     - parks a running reduction across phases
#   "bf16"   mandatory bfloat16 (the reduce scaler tile)


def _cb_layout(
    *,
    regime: str,
    block_ht: int,
    in_depth: int,
    out_depth: int,
    rm_depth: int,
    gamma_depth: int,
    wr: int,
    ws: int,
    Wt_core: int,
    has_gamma: bool,
    gamma_is_row_major: bool,
    is_row_major: bool,
    tile_out: bool,
    W_partial: int,
    gamma_ingest_block: int,
    T_in: int,
    T_g: int,
    T_interm: int,
    T_acc: int,
    T_bf16: int,
    w_split_group: int = 0,
    normed_depth: int = 1,
    resident_gamma: int = 0,
):
    """Return [(cb_index, num_pages, page_bytes, kind)] for this knob assignment."""
    wmax = max(wr, ws)
    if regime == "A":
        # Regime A is single-chunk by construction: one block spans the whole
        # per-core width.
        wr = ws = wmax = Wt_core
        resident_gamma = 1

    # x is RESIDENT in A and C: the input CB (and, on the RM path, the stick
    # staging CB) spans the WHOLE per-core width regardless of the scale chunk.
    # In B it spans the coarsest chunk.  This one expression is the whole
    # "one DRAM read instead of two" cost.
    w_in = Wt_core if regime in ("A", "C") else wmax
    # A CB that carries a running accumulation ACROSS reduce calls needs a second
    # generation live.  Two things produce that shape: Regime B's per-W-chunk
    # accumulate, and (NEW) the masked resident fold, which is always exactly two
    # calls - the aligned head, then the masked last W-tile.  In the BASELINE the
    # second disjunct is unreachable (W_partial > 0 forced regime B), so this is
    # byte-identical to the op there.
    # THE MASK IS A TILE-PATH PROPERTY, not a W property: on ROW_MAJOR input the
    # reader zero-fills every stick's pad tail, so the pad is exactly 0 and a
    # resident plan needs no mask at all (that is why `maskless_w` includes RM).
    masked_fold = (W_partial > 0) and not is_row_major
    accum_gens = 2 if (regime == "B" or masked_fold) else 1

    layout = [
        (CB_INPUT_TILES, in_depth * block_ht * w_in, T_in, "in"),
        (CB_SUMSQ, accum_gens * block_ht, T_acc, "acc"),
        (CB_RMS_RECIP, block_ht, T_acc, "acc"),
        # bfloat16 is mandatory for the reduce scaler.  Every regime needs the
        # within-tile REDUCE_ROW finalize; a non-tile-aligned W also needs the
        # PARTIAL tile that zeroes the pad columns of the last W-tile - now on the
        # resident plans too, which is what makes them expressible there at all.
        (CB_REDUCE_SCALER, 2 if (W_partial and (regime == "B" or masked_fold)) else 1, T_bf16, "bf16"),
    ]
    # `sum_of_squares`' element-wise tile accumulator: ONE tile per tile-row,
    # collapsed along W by the finalize reduce.  BOTH regimes run that shape now,
    # so this is the only x^2 storage the op has - Regime B's old full-block
    # `cb_squared` (block_ht * wr pages) is gone.  Regime B gets a second
    # generation so the finalize reduce's unpack can overlap the next W-chunk's
    # pack; Regime A writes it once per row-block.
    layout.append((CB_SUMSQ_ACC, accum_gens * block_ht, T_acc, "acc"))
    if has_gamma:
        # `gamma_depth` generations of the scale pass' gamma chunk.  In Regime B
        # this CB is STREAMED (one chunk pushed and popped per W-chunk), so a
        # second generation lets the reader fetch chunk k+1 while the compute is
        # still scaling chunk k; in Regime A gamma is RESIDENT (pushed once, never
        # popped), so there is only ever one generation to hold and `_solve`
        # leaves the depth at 1.
        # RESIDENT gamma (A always, C by knob) costs the full per-core width but is
        # read ONCE per core instead of once per row-block.
        layout.append((CB_GAMMA_TILES, Wt_core if resident_gamma else gamma_depth * ws, T_g, "gamma"))
        if gamma_is_row_major:
            layout.append((CB_GAMMA_RM, gamma_ingest_block, T_g, "gamma"))
        layout.append((CB_NORMED, normed_depth * block_ht * ws, T_interm, "interm"))
    # Streamed to the writer on the TILE path, but feeds the sequential untilize
    # helper on the RM path (must then hold the full block).
    layout.append((CB_OUTPUT_TILES, (out_depth if tile_out else 1) * block_ht * ws, T_in, "out"))
    if is_row_major:
        layout.append((CB_RM_IN, rm_depth * w_in, T_in, "in"))
        layout.append((CB_RM_OUT, rm_depth * ws, T_in, "out"))
    if w_split_group:
        # The combine's landing buffer, allocated on EVERY core of the group (not
        # just the root) so `get_write_ptr(cb_partial_gather)` resolves to the SAME
        # L1 address on every core.  That address identity is what lets a non-root
        # core compute the root's landing slot with no runtime address table.
        # Sized to EXACTLY one generation (group_size * BLOCK_HT pages) so the fifo
        # pointer wraps back to the CB base after every push/pop pair - if it did
        # not, the identity above would break after the first row-block.
        layout.append((CB_PARTIAL_GATHER, w_split_group * block_ht, T_acc, "acc"))
        layout.append((CB_SUMSQ_BCAST, block_ht, T_acc, "acc"))
    return layout


def _working_set_bytes(**kwargs) -> int:
    """Total L1 bytes the CB set costs for this knob assignment.

    All CBs are statically allocated for the whole program, so this is a SUM over
    every CB the configuration creates (not a per-phase max).
    """
    return sum(pages * page_bytes for _, pages, page_bytes, _ in _cb_layout(**kwargs))


# ---------------------------------------------------------------------------
# The L1 / blocking solver, parameterised by the per-core width.
# ---------------------------------------------------------------------------
# Factored out of `blocking_plan` so the W-split policy can SCORE a candidate
# group size with the very same solver that will later produce the shipped plan -
# the policy never guesses a regime or a block factor, it asks.


@dataclass(frozen=True)
class _Solved:
    regime: str
    BLOCK_HT: int
    WT_REDUCE_BLOCK: int
    WT_SCALE_BLOCK: int
    IN_BUF_DEPTH: int
    OUT_BUF_DEPTH: int
    RM_BUF_DEPTH: int
    GAMMA_DEPTH: int
    GAMMA_INGEST_BLOCK: int
    num_row_blocks: int
    NORMED_DEPTH: int = 1
    RESIDENT_GAMMA: int = 1


def _solve(
    *,
    Wt_core: int,
    w_split_group: int,
    row_parallel_units: int,
    Rt: int,
    maskless_w: bool,
    dest_limit: int,
    l1_cb_budget: int,
    gamma_cap_tiles: int,
    layout_common: dict,
    levers,
) -> _Solved:
    """Regime + every block factor / buffer depth for ONE per-core width."""
    common = dict(layout_common, Wt_core=Wt_core, w_split_group=w_split_group)

    def ws_bytes(regime, block_ht, in_depth, out_depth, rm_depth, wr, wsc, gamma_depth, nd=1, rg=0):
        # The gamma staging chunk must divide every ingest count the kernel uses,
        # so tilize<GAMMA_INGEST_BLOCK> never over-produces gamma tiles.
        return _working_set_bytes(
            regime=regime,
            block_ht=block_ht,
            in_depth=in_depth,
            out_depth=out_depth,
            rm_depth=rm_depth,
            gamma_depth=gamma_depth,
            wr=wr,
            ws=wsc,
            gamma_ingest_block=_largest_divisor_at_most(wsc, gamma_cap_tiles),
            normed_depth=nd,
            resident_gamma=rg,
            **common,
        )

    # --- Regime selection (pinned predicate, op_design.md) ------------------
    #  (1) can the reduce see the padded columns without a mask?
    #      RM   -> the reader zero-fills every stick's pad tail: pad is exactly 0.
    #      TILE -> the pad lives in DRAM and may be poisoned: mask mandatory.
    #  (2) does the MINIMAL resident working set fit the CB budget?  MINIMAL is
    #      deliberate: depth 1 on everything, gamma included.  It keeps the A/B
    #      boundary — and with it the W-split policy's P2 property, which rejects
    #      any group size whose slice does not solve to Regime A — invariant under
    #      the depth/chunk allocation below, so a change to how L1 is SPENT can
    #      never move which regime (or which G) is CHOSEN.
    fits = ws_bytes("A", 1, 1, 1, 1, Wt_core, Wt_core, 1) <= l1_cb_budget
    # LAB RUNG: does X ALONE fit resident, with the scale pass at its finest
    # (one-tile) chunk?  MINIMAL for the same reason `fits` is: the boundary must
    # not move when the depth/chunk allocation below spends L1 differently.
    fits_c = ws_bytes("C", 1, 1, 1, 1, Wt_core, 1, 1) <= l1_cb_budget
    # THE LADDER HAS EXACTLY ONE NEW RUNG, AND IT IS C - NOT A.  Two measurements
    # killed the "extend Regime A to the masked case" form of this idea:
    #   * A is single-chunk by construction, so its whole-width normed/out CBs
    #     leave nothing for a second generation.  On (1,1,32,4095) masked A fits
    #     only at depth 1: 19,344 -> 24,460 ns (0.79x).
    #   * A also holds gamma RESIDENT by construction, and a resident gamma is a
    #     SERIALIZED PROLOGUE - the reader must land all Wt_core gamma pages before
    #     the first x tile.  It only pays back when a core owns more than one
    #     row-block.  On (1,1,32,1057) (Wt_core 34, ONE row-block) masked A is
    #     double-buffered and STILL 8,207 -> 9,370 ns (0.88x), and the same single
    #     read through C with streamed gamma is 8,093 ns (1.01x).
    # Regime C SUBSUMES A anyway: C with ws == Wt_core and resident gamma IS A.
    # So the ladder leaves every shape that already solves to A exactly alone and
    # only ever converts a Regime-B shape to C.
    # THE SELECTION PROPERTY, both halves of it:
    #   L1        - does THIS CB set fit at this dtype / DEST width / layout?
    #   MASK      - can the reduce see the pad columns?  The op required
    #               `maskless_w` for every resident plan; the lab implements the
    #               masked resident fold, so the requirement becomes a knob.
    resident_ok = maskless_w or bool(_lever(levers, "allow_masked_resident"))
    # THE THIRD HALF OF THE PROPERTY, and the one the measurement forced.
    # Making x resident deletes the SECOND DRAM read of it - but that read is only
    # on the critical path when it is NOT already hidden.  With exactly ONE
    # row-block per core the kernel is a single pipeline fill: Regime B's scale
    # pass re-reads chunk c+1 while the compute is still scaling chunk c, so the
    # second read costs ~nothing, while the resident plan pays a real fill latency
    # (the whole per-core width must land before the reduce can start).  With more
    # than one row-block the pipe is saturated and DRAM bytes become the wall.
    # MEASURED, one row-block per core, best C arm vs Regime B:
    #   bf16 (1,1,32,4095) 1.000x | (1,1,32,4127) 0.994x | (1,1,64,6143) 0.986x
    #   bf16 (1,1,32,1057) 1.014x | (32,17) 1.015x            -> FLAT
    #   bfloat8_b (1,1,32,4095) 0.971x | float32 (1,1,32,4095) 0.908x -> REGRESSION
    # MEASURED, more than one row-block per core:
    #   (1,1,8192,4095) 1.577x | (1,1,16384,4095) 1.659x | (1,1,8192,6143) 1.574x
    # So the rung is gated on the per-core row-block count, which is a
    # work-distribution property the plan already computes - not a width or a
    # dtype list.
    blocks_per_unit = _div_up(Rt, max(1, row_parallel_units))
    # Lever D20's counterfactual arm: force the STREAMING 2-pass plan on a shape
    # that fits the resident single-read one, so the single-read fast path can be
    # priced from both sides instead of only asserted.  Regime B is always a legal
    # plan (it is the correctness fallback), so this arm never produces a wrong
    # answer - it just moves 1.5x the DRAM bytes.  Default 0 = the solver decides.
    allow_c = bool(_lever(levers, "allow_c"))
    forced_lab = _lever(levers, "force_regime_lab")
    if _lever(levers, "force_regime"):
        regime = "B"
    elif forced_lab:
        regime = forced_lab
        if regime == "A" and not (resident_ok and fits):
            raise ValueError(f"force_regime_lab=A inexpressible (resident_ok={resident_ok} fits={fits})")
        if regime == "C" and not (resident_ok and fits_c):
            raise ValueError(f"force_regime_lab=C inexpressible (resident_ok={resident_ok} fits_c={fits_c})")
        # NB: `force_regime_lab` deliberately bypasses the blocks-per-core gate -
        # that is how the gate's own counterfactual (C at one row-block) is priced.
    elif maskless_w and fits:
        regime = "A"  # unchanged from the op
    elif resident_ok and fits_c and allow_c and blocks_per_unit > 1:
        regime = "C"  # THE NEW RUNG
    else:
        regime = "B"

    # Coarsest useful row-block: any coarser and some row-parallel unit (a core
    # without a W split, a combine GROUP with one) gets no work at all.
    max_block_ht = max(1, _div_up(Rt, max(1, row_parallel_units)))
    max_block_ht = min(max_block_ht, dest_limit)

    block_ht = 1
    in_depth = out_depth = rm_depth = 1

    # DEPTH OF cb_gamma_tiles.  A CB only has anything to gain from a second
    # generation if it is REFILLED while the kernel runs: in Regime B the scale
    # pass pushes and pops one gamma CHUNK per W-chunk, so generation k+1 can be
    # fetched while the compute still holds k; in Regime A gamma is RESIDENT
    # (pushed once, never popped — see the WT_SCALE_BLOCK assert in
    # blocking_plan), so there is only ever one generation and a second would be
    # dead L1.  That is a property of the CB's lifecycle, not of the shape.
    gamma_streamed = layout_common["has_gamma"] and regime != "A"
    gamma_depth = 2 if (gamma_streamed and _lever(levers, "double_buffer")) else 1
    # Same question for the input CB, and it is answered the same way in both
    # regimes: cb_input_tiles always streams.
    stream_depth = 2 if _lever(levers, "double_buffer") else 1

    normed_depth = 1
    resident_gamma = 1 if regime == "A" else 0
    if regime == "A":
        wr = wsc = Wt_core
    elif regime == "C":
        # DEPTHS FIRST, THEN THE CHUNK - the same ordering Regime B's search uses
        # and for the same measured reason: letting the reader run a chunk ahead
        # buys more than a coarser chunk does.  Then GAMMA RESIDENCY is preferred
        # where it fits, because a chunked gamma is re-read once per row-block
        # while a resident one is read once per core.
        f_in = _lever(levers, "c_in_depth")
        f_out = _lever(levers, "c_out_depth")
        f_g = _lever(levers, "c_gamma_depth")
        f_n = _lever(levers, "c_normed_depth")
        if f_in or f_out or f_g or f_n:
            depth_prefs = [(f_in or 1, f_out or 1, f_g or 1, f_n or 1)]
        elif _lever(levers, "double_buffer"):
            depth_prefs = [(1, 2, 2, 1), (1, 1, 1, 1)]
        else:
            depth_prefs = [(1, 1, 1, 1)]
        # GAMMA RESIDENCY, and it is a per-core WORK property, not a shape one.
        # Resident gamma is read once per CORE instead of once per row-block, so it
        # only buys anything when a core owns MORE THAN ONE row-block.  When it owns
        # exactly one it buys nothing and costs a serialized prologue: the reader
        # must land all Wt_core gamma pages before the first x tile, and nothing
        # overlaps it.  Measured on (1,1,32,4095) (one row-block, one busy core):
        # resident gamma 22,679 ns vs chunked 19,112 ns at the same chunk - a
        # 3,567 ns prologue, 19% of the wall.
        rg_lever = int(_lever(levers, "c_resident_gamma"))
        rg_prefs = ([1, 0] if blocks_per_unit > 1 else [0]) if rg_lever < 0 else [rg_lever]
        forced_ws = _lever(levers, "c_ws")
        # CB-WRAP: the scale chunk must DIVIDE Wt_core (the resident window is
        # accessed as a fixed BLOCK_HT x Wt_core block from an aligned pointer).
        divisors = [c for c in range(Wt_core, 0, -1) if Wt_core % c == 0]
        if forced_ws:
            assert Wt_core % forced_ws == 0, f"c_ws={forced_ws} must divide Wt_core={Wt_core}"
            divisors = [forced_ws]
        # CHUNK COARSENESS IS NOT FREE - IT IS L1 THAT BLOCK_HT WANTS.  The search
        # therefore asks for the coarsest divisor that still affords the FULL
        # BLOCK_HT this core could use, and only falls back to "coarsest at
        # BLOCK_HT 1" when no chunk affords it.  Measured on (1,1,8192,4095)
        # (max_block_ht 2): the coarsest-that-fits chunk (ws 64) lands BLOCK_HT 1
        # and 379,186 ns, while ws 32 lands BLOCK_HT 2 and 369,547 ns; ws 4 is
        # 362,556.  On (1,1,32,4095) (max_block_ht 1) the same rule keeps the
        # coarse chunk, which is what that shape wants: ws 64 -> 19,376 ns,
        # ws 2 -> 35,638 ns (0.54x).
        wsc = 0
        for rg in rg_prefs:
            for di, do, dg, dn in depth_prefs:
                # Descend through every BLOCK_HT, not just max -> 1: at Rt 512 on a
                # 130-core grid max_block_ht is 4 and no chunk affords it, but 2 is
                # affordable and is worth more than the coarsest chunk at 1.
                for target_bht in range(max_block_ht, 0, -1):
                    for cand in divisors:
                        if ws_bytes("C", target_bht, di, do, di, Wt_core, cand, dg, dn, rg) <= l1_cb_budget:
                            wsc = cand
                            block_ht = target_bht
                            resident_gamma = rg
                            in_depth = rm_depth = di
                            out_depth, gamma_depth, normed_depth = do, dg, dn
                            break
                    if wsc:
                        break
                if wsc:
                    break
            if wsc:
                break
        assert wsc, "Regime C: no (chunk, depth) assignment fits L1"
        # Regime C has no separate reduce chunk: the sum-of-squares is one (or, in
        # the masked case, two) call(s) over the whole resident window.
        wr = wsc
    else:
        # Coarsest chunk of the dependent axis at which the STREAMING CBs still
        # reach depth 2 — i.e. the W-chunk is the variable that BUYS the overlap,
        # not a target the overlap has to fit inside.  A coarser chunk that can
        # only be single-buffered serialises the reader against the compute
        # (measured: the reader's gamma reserve alone was 6.4us of a 43us wall on
        # (1,1,32,7168), and going one divisor finer for depth 2 was 1.36x AND
        # used 344 KB less L1).  c = 1 always fits, so the search cannot fail; the
        # depth ladder below then spends whatever is left over.
        #
        # THE PROFILE IS THE DEPTH LADDER'S TOP RUNG - (IN_DEPTH_CAP, out 2, rm 2,
        # gamma 2) - AND NOT A SUBSET OF IT.  A subset lets the search call a
        # coarser chunk "affordable" while paying for it out of a buffer generation
        # it never accounted for, and the ladder below cannot claw that back:
        # coarseness is not worth a generation.  Both failure modes are MEASURED,
        # on the Regime-B sweep in perf_experiments/fused_sumsq/graduation_ab.py
        # (arms `fused_p21` / `fused` / `fused_p42`), once the fused
        # sum-of-squares freed 94-258 KB into this very decision:
        #   profile (in 2, out 1): (1,1,32,3071) took wr=96/in=2/out=1 instead of
        #       wr=48/in=4/out=2 and went 16.5us -> 18.7us (-12%).
        #   profile (in 2, out 2): (1,1,32,4095) no-gamma / bfloat8_b took
        #       wr=128/in=2..3 instead of wr=64/in=4 and went 16.7 -> 18.0us and
        #       20.0 -> 23.2us (-8% / -14%).
        #   profile (in 4, out 2): >= every other arm on all 14 cases of that
        #       sweep, and >= the pre-graduation plan on all 14.
        # Regime A never runs this search, so this is a Regime-B-only decision.
        #
        # CB-WRAP CONSTRAINT (load-bearing): a multi-page cb_reserve_back /
        # cb_wait_front followed by a contiguous N-page access is only legal when
        # the CB's page count is a multiple of N and the fifo pointer is
        # N-aligned.  A short trailing chunk would leave the pointer off-grid and
        # the NEXT full chunk would run past the end of the CB into the
        # neighbouring one (silent, deterministic corruption).  So the chunk must
        # DIVIDE Wt_core exactly — the search is over divisors, not over every
        # width.
        #
        # WT_REDUCE_BLOCK and WT_SCALE_BLOCK stay separate knobs (separate CT
        # args, separate loops in every kernel), but the solver has to give them
        # the same value: they share cb_input_tiles / cb_rm_in, whose page count
        # must be a multiple of BOTH access granularities.
        wr = wsc = 1
        if _lever(levers, "coarse_chunk"):
            forced_wt = _lever(levers, "wt_block")
            chunk_cap = min(Wt_core, forced_wt) if forced_wt else Wt_core
            # The ladder's top rung, scaled by `stream_depth` so the
            # double_buffer=0 counterfactual still degenerates to depth 1.
            in_target = IN_DEPTH_CAP if stream_depth > 1 else 1
            for cand in range(chunk_cap, 0, -1):
                if Wt_core % cand != 0:
                    continue
                if ws_bytes("B", 1, in_target, stream_depth, stream_depth, cand, cand, gamma_depth) <= l1_cb_budget:
                    wr = wsc = cand
                    break

    # Allocation priority (movement-dominated op: overlap beats amortization):
    #   1. double-buffer the streaming CBs (lever C16, measured 2.78x) — in
    #      Regime B the (in, gamma) rung is affordable BY CONSTRUCTION, because
    #      the chunk above was chosen for it; the out/rm rung is what the ladder
    #      still decides.
    #   2. grow BLOCK_HT (per-block-overhead amortization)
    #   3. grow IN_BUF_DEPTH further
    # Regime C already picked its depths together with its chunk (they are one
    # decision there: the chunk is the coarsest divisor that still affords them).
    if _lever(levers, "double_buffer") and regime != "C":
        if ws_bytes(regime, block_ht, 2, 2, 2, wr, wsc, gamma_depth) <= l1_cb_budget:
            in_depth = out_depth = rm_depth = 2
        elif ws_bytes(regime, block_ht, 2, 1, 2, wr, wsc, gamma_depth) <= l1_cb_budget:
            in_depth = rm_depth = 2
        elif ws_bytes(regime, block_ht, 1, 1, 2, wr, wsc, gamma_depth) <= l1_cb_budget:
            rm_depth = 2

    forced_block_ht = _lever(levers, "block_ht")
    if forced_block_ht:
        max_block_ht = min(max_block_ht, forced_block_ht)

    while (
        block_ht < max_block_ht
        and ws_bytes(
            regime, block_ht + 1, in_depth, out_depth, rm_depth, wr, wsc, gamma_depth, normed_depth, resident_gamma
        )
        <= l1_cb_budget
    ):
        block_ht += 1

    if _lever(levers, "double_buffer") and regime != "C":
        while (
            in_depth < IN_DEPTH_CAP
            and ws_bytes(regime, block_ht, in_depth + 1, out_depth, rm_depth, wr, wsc, gamma_depth) <= l1_cb_budget
        ):
            in_depth += 1

    assert Wt_core % wr == 0 and Wt_core % wsc == 0, "W-chunk must divide Wt_core (CB-wrap constraint)"

    return _Solved(
        regime=regime,
        BLOCK_HT=block_ht,
        WT_REDUCE_BLOCK=wr,
        WT_SCALE_BLOCK=wsc,
        IN_BUF_DEPTH=in_depth,
        OUT_BUF_DEPTH=out_depth,
        RM_BUF_DEPTH=rm_depth,
        GAMMA_DEPTH=gamma_depth,
        GAMMA_INGEST_BLOCK=_largest_divisor_at_most(wsc, gamma_cap_tiles),
        num_row_blocks=_div_up(Rt, block_ht),
        NORMED_DEPTH=normed_depth,
        RESIDENT_GAMMA=resident_gamma,
    )


# ---------------------------------------------------------------------------
# W-SPLIT: the work-distribution POLICY (op_design.md "Unlocked scheme L1")
# ---------------------------------------------------------------------------
# The INDEPENDENT axis (Rt) is spread over combine GROUPS of G cores; inside a
# group the DEPENDENT axis (Wt) is spread over the G cores, each reducing Wt/G
# columns.  The group root sums the G partial sums-of-squares cross-core and
# broadcasts the total back, so every core can scale its own column slice.
#
# G IS CHOSEN, NEVER FIXED, AND G == 1 IS THE ROW-PARALLEL PLAN.  There is one
# work-distribution path; "fall back to row-parallel" is the policy returning the
# value 1, not a second branch.  A fixed G is measurably wrong: the same G that
# gives 4.72x on (1,1,32,7168) gives 0.31x on (1,1,8192,1024).
#
# WHY IT PAYS - two INDEPENDENT mechanisms (both measured on Blackhole p150):
#   (1) PARALLELISM.  The per-core RISC-issue + TRISC cost divides by G.  Only
#       pays while the row axis leaves cores idle.
#   (2) REGIME + GAMMA.  A narrow per-core slice makes the RESIDENT-FUSED Regime A
#       fit L1, so x is read from DRAM ONCE instead of twice, and each core reads
#       only its OWN gamma slice instead of the whole row.  Pays at every Rt.
# WHAT IT COSTS - the combine, ~linear in G.  Measured on the focus case against a
# combine-removed arm: 1,666 ns @ G=14, 2,310 @ G=28, 4,001 @ G=56, i.e.
# ~890 + 56*G ns per row-block.  That is exactly why the focus optimum is G=32 and
# not the largest legal G=56, and why the policy has to WEIGH rather than maximize.
#
# MEASURED SWEEP the calibration below reproduces (baseline = G 1; DEVICE KERNEL
# DURATION ns, bf16 / HiFi2 / fp32_dest_acc_en=False, 13x10 grid):
#   (1,1,32,7168)   Rt   1  Wt 224      44,314 -> G 32     9,386   4.72x
#   (1,1,32,1024)   Rt   1  Wt  32       9,260 -> G 8..16  5,561   1.67x
#   (1,1,32,32768)  Rt   1  Wt 1024    187,764 -> G 64    26,693   7.03x
#   (1,1,64,12288)  Rt   2  Wt 384      68,821 -> G 32    19,165   3.59x
#   (1,1,8192,7168) Rt 256  Wt 224   1,229,243 -> G 4..16 621,319  1.98x
#   (1,1,8192,1024) Rt 256  Wt  32     117,456 -> G 2    100,336   1.17x
#
# HARD PROPERTIES a candidate G must satisfy.  Each is a correctness or
# expressibility FACT about the split, never a list of the shapes that were
# benchmarked - which is why the set shrinks as the kernels grow, instead of
# having to be widened by hand:
#
#   P1  G divides Wt.  A short trailing slice needs a per-core RUNTIME width (the
#       kernels take Wt_core as a compile-time arg) plus a partial mask on the
#       split boundary.  Not implemented -> inexpressible, not slow.
#   P2  the per-core slice must SOLVE to Regime A.  The combine consumes the
#       PRE-collapse sum-of-squares accumulator tile (`cb_sumsq_acc`), which only
#       Regime A produces; Regime B collapses inside its accumulating reduce and
#       has no such tile.  This property is ALSO the whole carve-out for a
#       non-tile-aligned W on the TILE path: there `maskless_w` is False, so every
#       candidate solves to Regime B and the policy returns G=1 BY CONSTRUCTION -
#       no `if W % 32` predicate anywhere.  (On the ROW_MAJOR path the reader
#       zero-fills each stick's pad tail per core, so `maskless_w` holds and a
#       non-aligned W CAN split.)
#   P3  the group must tile the grid as a rectangle whose VIRTUAL-coordinate
#       bounding box has multicast area exactly G.  The 1->N leg is one NoC
#       multicast and its fan-out is `McastRect::area()` (mcast_pipe.hpp:130-138),
#       so a box that is not dense in virtual coords would make the handshake
#       counts wrong - a hang or a premature release, not a slowdown.
#   P5  the reader and the writer must be on DIFFERENT NoCs (lever `noc_split`, the
#       applied default).  `ncrisc_noc_nonposted_writes_flushed`
#       (blackhole/noc_nonblocking_api.h:662-664) compares the PER-NOC hardware
#       register NIU_MST_WR_ACK_RECEIVED - which counts acks from EVERY RISC on that
#       NoC - against a PER-RISC software counter.  The combine makes the READER
#       issue non-posted writes; if the writer is on the same NoC its
#       `noc_async_write_barrier` sees the register run ahead of its own counter and
#       spins forever (measured: hang on (1,1,32,7168) with `noc_split=0`, writer
#       BRISC stuck in NWBW on the group root).  Under the applied default the
#       reader owns NOC_0 and the writer NOC_1, so this holds; only the lever's
#       off-arm - not any user-reachable cell - falls back to G=1.
#   P4  ROW_MAJOR INPUT ONLY: the per-core stick chunk must stay >= 1024 B.  On the
#       TILE path the read unit is a whole tile PAGE, so a split never shrinks a
#       transaction.  On the RM path the read unit is a STICK and the split cuts it
#       to Wt_core*32*elem bytes.  Measured: (1,1,8192,1024) RM 0.91x @1024 B/core,
#       0.59x @512 B, 0.43x @256 B; (1,1,1024,7168) RM still WINS 1.26x @1792 B.
#       Stated as a stick-byte property, not as "ROW_MAJOR is excluded" - an
#       outright RM exclusion would throw the 1.26x away.
#
# Everything that survives P1..P4 is SCORED by one cost model and the cheapest
# wins, with G=1 competing on exactly the same footing.

# The per-core cost model, in units of "one tile-page read by one core".  The two
# combine coefficients are the measured ~890 + 56*G ns above divided by the
# measured 66 ns/tile of the same sweep, so the model is a calibration and stays
# re-measurable: `_levers=dict(w_group=N)` pins G, so a new box or a new shape can
# be swept and these two numbers re-fit without touching the policy's structure.
_COMBINE_FIXED_TILES = 13.5
_COMBINE_PER_CORE_TILES = 0.85
# Candidates within this band of the best score are a measurement TIE (the device
# noise band is ~2-3%), so the tie-break decides - and it prefers the arm that
# puts more of the grid to work.  This is the "keep num_groups(G)*G near
# grid_cores" criterion, applied as a tie-break rather than as a filter: as a
# filter it would reject the focus optimum (G=32 leaves 66 of 130 cores idle and
# is still 4.72x).
_SCORE_TIE_BAND = 0.03
# P4: the measured floor on a per-core ROW_MAJOR stick chunk.
RM_MIN_CORE_STICK_BYTES = 1024


def group_rect(group_size: int, grid):
    """The (gx, gy) core rectangle one combine group occupies, or None.

    A combine group must be a RECTANGLE because the 1->N leg is a single NoC
    multicast, whose destination set is a bounding box (`McastRect`).  Pick the
    WIDEST legal factorization (gx as large as the grid allows) so the group spans
    the DRAM-facing axis first.
    """
    for gx in range(min(group_size, grid.x), 0, -1):
        if group_size % gx == 0 and group_size // gx <= grid.y:
            return gx, group_size // gx
    return None


def _virt(device, x, y):
    c = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    return c.x, c.y


def _group_tiling(device, grid, gx: int, gy: int, group_size: int):
    """Every group of a gx*gy tiling of `grid`, or None if P3 fails for any of them.

    Each entry is (logical_rect, root_virtual, bbox_virtual, members).  The bounding
    box is taken over EVERY member's VIRTUAL coord because the logical->virtual
    worker map is not monotonic in x on every arch, and it is the virtual box the
    multicast actually addresses (mcast_host.hpp::noc_ordered_bbox does the same).
    """
    # Virtual x columns that carry a worker core.  `McastRect::area()` discounts
    # the non-worker columns a box spans (mcast_pipe.hpp:130-138, hard-coded 8/9 on
    # Blackhole); deriving the worker set from the device mirrors that arch-neutrally.
    worker_vx = {_virt(device, x, 0)[0] for x in range(grid.x)}
    groups_x, groups_y = grid.x // gx, grid.y // gy
    out = []
    for g in range(groups_x * groups_y):
        g_ox, g_oy = (g % groups_x) * gx, (g // groups_x) * gy
        members = [ttnn.CoreCoord(g_ox + i, g_oy + j) for j in range(gy) for i in range(gx)]
        vs = [_virt(device, m.x, m.y) for m in members]
        xlo, xhi = min(v[0] for v in vs), max(v[0] for v in vs)
        ylo, yhi = min(v[1] for v in vs), max(v[1] for v in vs)
        width = sum(1 for x in range(xlo, xhi + 1) if x in worker_vx)
        if width * (yhi - ylo + 1) != group_size:
            return None
        out.append(
            ((g_ox, g_oy, g_ox + gx - 1, g_oy + gy - 1), _virt(device, g_ox, g_oy), (xlo, ylo, xhi, yhi), members)
        )
    return out


def _split_cost(solved, *, Wt_core, group_size, num_groups, has_gamma, T_in, T_g):
    """Per-core cost of one candidate, in tile-page-read equivalents.

    Only the READ side is modelled: the writer runs on the other NoC and the other
    RISC, so the wall is max(read, write) and the read side is the larger one in
    both regimes (Regime B reads x twice).  Byte-weighted so a wide gamma dtype is
    not counted as if it were the input's.
    """
    groups_used = max(1, min(num_groups, solved.num_row_blocks))
    blocks_per_group = _div_up(solved.num_row_blocks, groups_used)
    passes = 1 if solved.regime in ("A", "C") else 2
    read = blocks_per_group * solved.BLOCK_HT * Wt_core * passes
    # Regime A holds gamma resident for the whole kernel (one read per core);
    # Regime B re-reads the slice once per row-block.
    gamma = 0 if not has_gamma else (Wt_core if solved.RESIDENT_GAMMA else blocks_per_group * Wt_core)
    combine = (
        0.0 if group_size == 1 else blocks_per_group * (_COMBINE_FIXED_TILES + _COMBINE_PER_CORE_TILES * group_size)
    )
    return read + gamma * (T_g / T_in) + combine, groups_used


def _choose_group_size(
    *,
    device,
    grid,
    grid_cores: int,
    Wt: int,
    Rt: int,
    maskless_w: bool,
    is_row_major: bool,
    elem_size: int,
    has_gamma: bool,
    T_in: int,
    T_g: int,
    dest_limit: int,
    l1_cb_budget: int,
    gamma_cap_tiles: int,
    layout_common: dict,
    levers,
):
    """Pick (group_size, gx, gy, num_groups, tiling).  1 == the row-parallel plan."""

    # G = 1 is a candidate like any other, scored with the same model, so the
    # row-parallel plan wins by MEASUREMENT wherever the split does not pay.
    def evaluate(g):
        if Wt % g:  # P1
            return None
        if g == 1:
            gx = gy = 1
            num_groups, tiling = grid_cores, None
        else:
            rect = group_rect(g, grid)
            if rect is None:  # P3
                return None
            gx, gy = rect
            tiling = _group_tiling(device, grid, gx, gy, g)
            if tiling is None:  # P3
                return None
            num_groups = len(tiling)
        Wt_core = Wt // g
        if g > 1 and is_row_major and Wt_core * TILE_DIM * elem_size < RM_MIN_CORE_STICK_BYTES:  # P4
            return None
        solved = _solve(
            Wt_core=Wt_core,
            w_split_group=0 if g == 1 else g,
            row_parallel_units=num_groups,
            Rt=Rt,
            maskless_w=maskless_w,
            dest_limit=dest_limit,
            l1_cb_budget=l1_cb_budget,
            gamma_cap_tiles=gamma_cap_tiles,
            layout_common=layout_common,
            levers=levers,
        )
        if g > 1 and solved.regime != "A":  # P2
            return None
        cost, groups_used = _split_cost(
            solved,
            Wt_core=Wt_core,
            group_size=g,
            num_groups=num_groups,
            has_gamma=has_gamma,
            T_in=T_in,
            T_g=T_g,
        )
        return {
            "g": g,
            "gx": gx,
            "gy": gy,
            "num_groups": num_groups,
            "tiling": tiling,
            "cost": cost,
            "active": groups_used * g,
        }

    row_parallel = evaluate(1)
    assert row_parallel is not None, "G=1 must always be a legal plan"

    # P5: the combine makes the reader a WRITER, so it must not share a NoC with
    # the output writer.  Checked before `w_group` so a forcing arm cannot ask for a
    # configuration that hangs.
    if not _lever(levers, "noc_split"):
        return 1, 1, 1, grid_cores, None

    forced = _lever(levers, "w_group")
    if forced:
        cand = evaluate(forced)
        assert cand is not None, (
            f"_levers w_group={forced} is not a legal group size for Wt={Wt} on this grid "
            "(fails one of the P1..P4 properties in _choose_group_size)"
        )
        return cand["g"], cand["gx"], cand["gy"], cand["num_groups"], cand["tiling"]
    if not _lever(levers, "w_split"):
        return 1, 1, 1, grid_cores, None

    cands = [row_parallel]
    for g in range(2, min(Wt, grid_cores) + 1):
        cand = evaluate(g)
        if cand is not None:
            cands.append(cand)

    best = min(c["cost"] for c in cands)
    # Within the noise band, prefer the arm that keeps more of the grid busy; then
    # the smaller G (a smaller combine is the safer of two equal bets).
    band = [c for c in cands if c["cost"] <= best * (1.0 + _SCORE_TIE_BAND)]
    pick = min(band, key=lambda c: (-c["active"], c["g"]))
    return pick["g"], pick["gx"], pick["gy"], pick["num_groups"], pick["tiling"]


def blocking_plan(input_tensor, gamma, output_tensor, device, compute_kernel_config, levers=None) -> BlockingPlan:
    """The ONLY place block factors, buffer depths, the regime and G are decided."""
    shape = list(input_tensor.shape)
    is_row_major = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    Rt, Wt, W_true, W_partial, num_rows = tile_geometry(shape, is_row_major)

    has_gamma = gamma is not None
    gamma_is_row_major = bool(has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT)

    elem_size = _elem_size(input_tensor.dtype)
    gamma_elem_size = _elem_size(gamma.dtype) if has_gamma else elem_size
    # A block-float tensor cannot be row-major (ttnn refuses to build one), so the
    # stick-addressing path never sees a zero elem_size.
    assert not (is_row_major and elem_size == 0), "block-float dtype on the ROW_MAJOR path"

    # The two derived formats, decided HERE and nowhere else; _cb_layout consumes
    # the tile sizes and create_program_descriptor the dtypes.
    interm_dtype = _interm_dtype(input_tensor.dtype)
    acc_dtype = _acc_dtype(compute_kernel_config, interm_dtype, bool(_lever(levers, "acc_narrow")))

    T_in = ttnn.tile_size(input_tensor.dtype)
    T_g = ttnn.tile_size(gamma.dtype) if has_gamma else T_in
    T_interm = ttnn.tile_size(interm_dtype)
    T_acc = ttnn.tile_size(acc_dtype)
    T_bf16 = ttnn.tile_size(ttnn.bfloat16)

    tile_out = not is_row_major
    l1_cb_budget = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVED_BYTES

    dest_limit = _dest_limit(compute_kernel_config)
    forced_dest = _lever(levers, "dest_block")
    if forced_dest:
        dest_limit = min(dest_limit, forced_dest)

    #  (1) of the regime predicate: can the reduce see the padded columns without a
    #      mask?  RM -> the reader zero-fills every stick's pad tail: pad is exactly
    #      0.  TILE -> the pad lives in DRAM and may be poisoned: mask mandatory.
    maskless_w = is_row_major or (W_partial == 0)

    layout_common = dict(
        has_gamma=has_gamma,
        gamma_is_row_major=gamma_is_row_major,
        is_row_major=is_row_major,
        tile_out=tile_out,
        W_partial=W_partial,
        T_in=T_in,
        T_g=T_g,
        T_interm=T_interm,
        T_acc=T_acc,
        T_bf16=T_bf16,
    )
    gamma_cap_tiles = max(1, GAMMA_STAGE_MAX_BYTES // T_g)

    # --- Grid / core count --------------------------------------------------
    grid = device.compute_with_storage_grid_size()
    core_cap = _lever(levers, "active_cores") or ACTIVE_CORE_CAP
    if core_cap:
        # Truncate the grid row-wise so the cap keeps the DRAM-facing spread.  The
        # SAME truncation create_program_descriptor applies, so the plan's group
        # tiling and the descriptor's core ranges cannot disagree.
        grid = ttnn.CoreCoord(grid.x, min(grid.y, max(1, _div_up(core_cap, grid.x))))
    grid_cores = grid.x * grid.y

    # --- W-split policy: choose G (1 == row-parallel) ------------------------
    # COMPOSITION WITH THE SHIPPED SPLIT: the policy is asked with the lab ladder
    # SWITCHED OFF, so G is chosen byte-identically to the op.  The ladder is
    # applied only to the final per-core solve, and only where the split already
    # returned G == 1 - a shape the split takes cannot move.
    split_levers = dict(LEVER_DEFAULTS if levers is None else levers)
    split_levers.update(allow_c=0, allow_masked_resident=0, force_regime_lab="")
    group_size, gx, gy, num_groups, _tiling = _choose_group_size(
        device=device,
        grid=grid,
        grid_cores=grid_cores,
        Wt=Wt,
        Rt=Rt,
        maskless_w=maskless_w,
        is_row_major=is_row_major,
        elem_size=elem_size,
        has_gamma=has_gamma,
        T_in=T_in,
        T_g=T_g,
        dest_limit=dest_limit,
        l1_cb_budget=l1_cb_budget,
        gamma_cap_tiles=gamma_cap_tiles,
        layout_common=layout_common,
        levers=split_levers,
    )
    Wt_core = Wt // group_size
    w_split_group = 0 if group_size == 1 else group_size

    solved = _solve(
        Wt_core=Wt_core,
        w_split_group=w_split_group,
        row_parallel_units=num_groups,
        Rt=Rt,
        maskless_w=maskless_w,
        dest_limit=dest_limit,
        l1_cb_budget=l1_cb_budget,
        gamma_cap_tiles=gamma_cap_tiles,
        layout_common=layout_common,
        levers=(levers if group_size == 1 else split_levers),
    )
    regime = solved.regime
    groups_used = max(1, min(num_groups, solved.num_row_blocks))

    # R5: cb_gamma_tiles is never popped in Regime A, so one pass-B call must
    # span every gamma column from the CB front.
    if regime == "A":
        assert solved.WT_SCALE_BLOCK == Wt_core, "Regime A requires WT_SCALE_BLOCK == Wt_core (gamma is never popped)"
    if regime == "C":
        # Regime C RENEGOTIATES that constraint: with chunked gamma the reader
        # re-pushes each slice and the compute pops it (Regime B's protocol); with
        # RESIDENT gamma the scale pass addresses this chunk's columns through a
        # TileOffset base instead of consuming from the CB front.  Either way the
        # scale chunk must still DIVIDE Wt_core (CB-wrap).
        assert Wt_core % solved.WT_SCALE_BLOCK == 0, "Regime C: scale chunk must divide Wt_core"
        assert not w_split_group, "Regime C is only reachable at G == 1"

    common = dict(layout_common, Wt_core=Wt_core, w_split_group=w_split_group)
    cb_layout = tuple(
        _cb_layout(
            regime=regime,
            block_ht=solved.BLOCK_HT,
            in_depth=solved.IN_BUF_DEPTH,
            out_depth=solved.OUT_BUF_DEPTH,
            rm_depth=solved.RM_BUF_DEPTH,
            gamma_depth=solved.GAMMA_DEPTH,
            wr=solved.WT_REDUCE_BLOCK,
            ws=solved.WT_SCALE_BLOCK,
            gamma_ingest_block=solved.GAMMA_INGEST_BLOCK,
            normed_depth=solved.NORMED_DEPTH,
            resident_gamma=solved.RESIDENT_GAMMA,
            **common,
        )
    )

    if w_split_group:
        # P2, re-asserted on the shipped plan: only Regime A produces the
        # pre-collapse accumulator tile the combine sums.
        assert regime == "A", f"W-split needs Regime A per core, got B at Wt_core={Wt_core}"
        # Both combine CBs must hold EXACTLY one generation so their fifo pointer is
        # always the CB base - the cross-core address identity the gather relies on.
        sumsq_pages = [n for i, n, _, _ in cb_layout if i == CB_SUMSQ]
        assert sumsq_pages == [solved.BLOCK_HT], f"W-split: cb_sumsq must be exactly BLOCK_HT pages, got {sumsq_pages}"

    return BlockingPlan(
        Rt=Rt,
        Wt=Wt,
        Wt_core=Wt_core,
        W_true=W_true,
        W_partial=W_partial,
        num_rows=num_rows,
        is_row_major=is_row_major,
        has_gamma=has_gamma,
        gamma_is_row_major=gamma_is_row_major,
        tile_out=tile_out,
        elem_size=elem_size,
        gamma_elem_size=gamma_elem_size,
        in_tile_bytes=T_in,
        gamma_tile_bytes=T_g,
        interm_dtype=interm_dtype,
        acc_dtype=acc_dtype,
        interm_tile_bytes=T_interm,
        acc_tile_bytes=T_acc,
        bf16_tile_bytes=T_bf16,
        row_bytes=W_true * elem_size,
        gamma_row_bytes=W_true * gamma_elem_size,
        BLOCK_HT=solved.BLOCK_HT,
        WT_REDUCE_BLOCK=solved.WT_REDUCE_BLOCK,
        WT_SCALE_BLOCK=solved.WT_SCALE_BLOCK,
        DEST_BLOCK=dest_limit,
        GAMMA_INGEST_BLOCK=solved.GAMMA_INGEST_BLOCK,
        IN_BUF_DEPTH=solved.IN_BUF_DEPTH,
        OUT_BUF_DEPTH=solved.OUT_BUF_DEPTH,
        RM_BUF_DEPTH=solved.RM_BUF_DEPTH,
        GAMMA_DEPTH=solved.GAMMA_DEPTH,
        regime=regime,
        reduce_via_add=_reduce_via_add(
            regime, compute_kernel_config, acc_dtype, W_partial, bool(_lever(levers, "reduce_via_add")), is_row_major
        ),
        num_row_blocks=solved.num_row_blocks,
        NORMED_DEPTH=solved.NORMED_DEPTH,
        RESIDENT_GAMMA=solved.RESIDENT_GAMMA,
        group_size=group_size,
        group_x=gx,
        group_y=gy,
        num_groups=num_groups,
        groups_used=groups_used,
        l1_cb_budget=l1_cb_budget,
        cb_layout=cb_layout,
    )


# The shared geometry compile-time prefix is CT_ACCESSOR_BASE args wide; every
# kernel spells its TensorAccessorArgs<CT_ACCESSOR_BASE>, so the two cannot drift
# (create_program_descriptor asserts the prefix length).
CT_ACCESSOR_BASE = 32  # LAB: +2 (CT 30 RESIDENT_GAMMA, CT 31 the no-mask control)


def _even_split(total, buckets):
    """[count] per bucket — the same shape split_work_to_cores produces."""
    base, extra = divmod(total, buckets)
    return [base + (1 if i < extra else 0) for i in range(buckets)]


def _cb(index, num_pages, page_size, data_format, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def create_program_descriptor(
    input_tensor,
    gamma,
    output_tensor,
    *,
    epsilon: float,
    compute_kernel_config,
    levers=None,
    out_plan=None,
) -> "ttnn.ProgramDescriptor":
    device = input_tensor.device()
    # Applied FIRST: the plan reads fp32_dest_acc_en for the DEST limit, the
    # accumulator-CB format and the reduce datapath, so the arm must be in effect
    # before blocking_plan sees the config.  A no-op at the lever defaults.
    compute_kernel_config = _apply_precision_levers(compute_kernel_config, levers)
    plan = blocking_plan(input_tensor, gamma, output_tensor, device, compute_kernel_config, levers)
    if out_plan is not None:
        out_plan.append(plan)
    # LAB: the per-stage zones are compiled OUT by default so a chunk/depth sweep
    # is not paying a marker cost that DIFFERS between arms (a resident plan runs
    # a different NUMBER of zone executions than a streaming one, which would leak
    # straight into the ns delta being attributed to the idea).
    zone_def = [("RMSN_NO_ZONES", "1")] if _lever(levers, "no_zones") else []

    # ---------------- work distribution -------------------------------------
    # ONE assignment builder for both values of plan.group_size: a list of
    # (core, start_row_block, blocks_here, w_index, group).  With G == 1 the group
    # is None and w_index is always 0, which is exactly the row-parallel plan.
    grid = device.compute_with_storage_grid_size()
    core_cap = _lever(levers, "active_cores") or ACTIVE_CORE_CAP
    if core_cap:
        # Truncate the grid row-wise so the cap keeps the DRAM-facing spread.
        rows = max(1, _div_up(core_cap, grid.x))
        grid = ttnn.CoreCoord(grid.x, min(grid.y, rows))

    semaphores = []
    if plan.group_size == 1:
        row_wise = bool(_lever(levers, "row_wise"))  # lever A1: spread along the DRAM-facing axis
        (
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            blocks_per_core_g1,
            blocks_per_core_g2,
        ) = ttnn.split_work_to_cores(grid, plan.num_row_blocks, row_wise)

        cores = ttnn.grid_to_cores(num_cores, grid.x, grid.y, row_wise)

        def _blocks_of(core):
            if core_group_1.contains(core):
                return blocks_per_core_g1
            if core_group_2.contains(core):
                return blocks_per_core_g2
            return 0

        assignment = []
        start = 0
        for core in cores:
            n = _blocks_of(core)
            assignment.append((core, start, n, 0, None))
            start += n
    else:
        # Row-blocks go to combine GROUPS; inside a group the W axis is split.  An
        # idle group would still have to take part in its own combine handshake, so
        # only as many groups are instantiated as there are row-blocks to hand out.
        tiling = _group_tiling(device, grid, plan.group_x, plan.group_y, plan.group_size)
        assert tiling is not None and len(tiling) == plan.num_groups, "group tiling disagrees with the plan"
        groups_used = plan.groups_used
        per_group = _even_split(plan.num_row_blocks, groups_used)

        assignment = []
        core_ranges_used = []
        start = 0
        for g in range(groups_used):
            logical_rect, root_v, rect_v, members = tiling[g]
            core_ranges_used.append(logical_rect)
            group = {"root": root_v, "rect": rect_v}
            for w_index, core in enumerate(members):
                assignment.append((core, start, per_group[g], w_index, group))
            start += per_group[g]
        all_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(a, b), ttnn.CoreCoord(c, d)) for a, b, c, d in core_ranges_used]
        )
        semaphores = [
            ttnn.SemaphoreDescriptor(id=SEM_DATA_READY, core_ranges=all_cores, initial_value=0),
            # mcast_pipe.hpp: `consumer_ready` is incremented by REMOTE receivers with
            # no happens-before to the sender's ctor, so its initial 0 MUST be host-owned.
            ttnn.SemaphoreDescriptor(id=SEM_CONSUMER_READY, core_ranges=all_cores, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_GATHER, core_ranges=all_cores, initial_value=0),
        ]

    # ---------------- circular buffers ---------------------------------------
    # Instantiated straight off plan.cb_layout — the SAME list the L1 budget
    # solver summed.  No page count is re-derived here.
    fmt_of_kind = {
        "in": input_tensor.dtype,
        "out": output_tensor.dtype,
        "gamma": gamma.dtype if plan.has_gamma else input_tensor.dtype,
        # Derived numerical formats — decided once, in blocking_plan.
        "interm": plan.interm_dtype,
        "acc": plan.acc_dtype,
        "bf16": ttnn.bfloat16,
    }
    cbs = [
        _cb(index, num_pages, page_bytes, fmt_of_kind[kind], all_cores)
        for index, num_pages, page_bytes, kind in plan.cb_layout
    ]

    # ---------------- compile-time args --------------------------------------
    # One shared geometry prefix so reader / writer / compute cannot drift.
    geometry_ct_args = [
        1 if plan.is_row_major else 0,  # 0  IS_ROW_MAJOR
        REGIME_CODE[plan.regime],  # 1  REGIME  (0=B streaming, 1=A resident, 2=C resident-x)
        1 if plan.has_gamma else 0,  # 2  HAS_GAMMA
        1 if plan.gamma_is_row_major else 0,  # 3  GAMMA_IS_ROW_MAJOR
        plan.Wt_core,  # 4
        plan.W_partial,  # 5
        plan.BLOCK_HT,  # 6
        plan.WT_REDUCE_BLOCK,  # 7
        plan.WT_SCALE_BLOCK,  # 8
        plan.Rt,  # 9
        plan.num_rows,  # 10
        plan.row_bytes,  # 11
        plan.elem_size,  # 12
        plan.gamma_elem_size,  # 13
        plan.gamma_row_bytes,  # 14
        plan.DEST_BLOCK,  # 15
        plan.gamma_tile_bytes,  # 16
        plan.in_tile_bytes,  # 17
        plan.GAMMA_INGEST_BLOCK,  # 18
        _lever(levers, "barrier_per_block"),  # 19 (lever B7 off-arm)
        _lever(levers, "stub_dm"),  # 20 (ablation arm)
        _lever(levers, "coalesce"),  # 21 (lever B5/B6 off-arm)
        plan.reduce_via_add,  # 22 (Regime B reduce datapath)
        # --- W-split combine (all zero / inert when the policy picked G == 1) ---
        1 if plan.group_size > 1 else 0,  # 23 W_SPLIT
        plan.group_size,  # 24 GROUP_SIZE
        plan.Wt,  # 25 WT_TOTAL (the DRAM tile-row stride when a core owns a slice)
        plan.acc_tile_bytes,  # 26 ACC_TILE_BYTES (the combine payload unit)
        SEM_DATA_READY,  # 27
        SEM_CONSUMER_READY,  # 28
        SEM_GATHER,  # 29
        plan.RESIDENT_GAMMA,  # 30 LAB: gamma resident (A always; C by the solver)
        _lever(levers, "resident_no_mask"),  # 31 LAB: pad-poison positive control
    ]
    assert len(geometry_ct_args) == CT_ACCESSOR_BASE, "geometry prefix must match TensorAccessorArgs<CT_ACCESSOR_BASE>"

    # ---------------- reader --------------------------------------------------
    reader_ct_args = list(geometry_ct_args)
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if gamma is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    writer_ct_args = list(geometry_ct_args)
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct_args = list(geometry_ct_args)

    inv_w_bits = _f32_bits(1.0 / float(plan.W_true))
    eps_bits = _f32_bits(epsilon)

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0

    # Reader RT layout (fixed width, so the kernel indexes it unconditionally):
    #   0 in_addr, 1 gamma_addr, 2 start_row_block, 3 blocks_here,
    #   4 w_offset (tiles), 5 is_root, 6..7 group root (virtual),
    #   8..11 group multicast rect corners (virtual).
    # With G == 1 slots 4..11 are zero and the kernel's W_SPLIT branch is compiled out.
    for core, start, blocks_here, w_index, group in assignment:
        w_offset = w_index * plan.Wt_core
        if group is None:
            reader_rt[core.x][core.y] = [in_addr, gamma_addr, start, blocks_here, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            reader_rt[core.x][core.y] = [
                in_addr,
                gamma_addr,
                start,
                blocks_here,
                w_offset,
                1 if w_index == 0 else 0,
                group["root"][0],
                group["root"][1],
                *group["rect"],
            ]
        writer_rt[core.x][core.y] = [out_addr, start, blocks_here, w_offset]
        compute_rt[core.x][core.y] = [inv_w_bits, eps_bits, start, blocks_here, 1 if w_index == 0 else 0]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        defines=list(zone_def),
        runtime_args=reader_rt,
        config=(
            ttnn.ReaderConfigDescriptor()
            if _lever(levers, "noc_split")
            else ttnn.DataMovementConfigDescriptor(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0)
        ),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        defines=list(zone_def),
        runtime_args=writer_rt,
        config=(
            ttnn.WriterConfigDescriptor()
            if _lever(levers, "noc_split")
            else ttnn.DataMovementConfigDescriptor(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0)
        ),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        defines=(
            list(zone_def) + ([("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1")] if _lever(levers, "stub_compute") else [])
        ),
        runtime_args=compute_rt,
        # Pass-through: the caller's descriptor is handed over verbatim.
        config=compute_kernel_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=cbs,
    )


# ===========================================================================
# LAB SURFACE
# ===========================================================================
def plan_summary(p: BlockingPlan) -> str:
    return (
        f"regime={p.regime} G={p.group_size} Wt_core={p.Wt_core} Wpart={p.W_partial} "
        f"BLOCK_HT={p.BLOCK_HT} wr={p.WT_REDUCE_BLOCK} ws={p.WT_SCALE_BLOCK} "
        f"din={p.IN_BUF_DEPTH} dout={p.OUT_BUF_DEPTH} dg={p.GAMMA_DEPTH} dn={p.NORMED_DEPTH} "
        f"rg={p.RESIDENT_GAMMA} dest={p.DEST_BLOCK} rva={p.reduce_via_add} "
        f"blocks={p.num_row_blocks} L1={p.working_set_bytes()}/{p.l1_cb_budget}"
    )


BASELINE_LEVERS = dict(allow_c=0, allow_masked_resident=0, force_regime_lab="")

_PLAN_FIELDS = (
    "regime",
    "Wt_core",
    "BLOCK_HT",
    "WT_REDUCE_BLOCK",
    "WT_SCALE_BLOCK",
    "DEST_BLOCK",
    "IN_BUF_DEPTH",
    "OUT_BUF_DEPTH",
    "RM_BUF_DEPTH",
    "GAMMA_DEPTH",
    "GAMMA_INGEST_BLOCK",
    "reduce_via_add",
    "num_row_blocks",
    "group_size",
    "num_groups",
    "groups_used",
)


def assert_matches_op_plan(input_tensor, gamma, output_tensor, device, compute_kernel_config, levers=None):
    """HONEST-BASELINE GATE: at the baseline levers this fork == the shipped op.

    Everything the lab measures is a delta against THIS, so if it ever drifts the
    whole comparison is against a strawman.
    """
    lev = dict(levers or {})
    lev.update(BASELINE_LEVERS)
    lab = blocking_plan(input_tensor, gamma, output_tensor, device, compute_kernel_config, lev)
    ref = opd.blocking_plan(input_tensor, gamma, output_tensor, device, compute_kernel_config, levers)
    for field in _PLAN_FIELDS:
        assert getattr(lab, field) == getattr(
            ref, field
        ), f"baseline drift on {field}: lab {getattr(lab, field)} != op {getattr(ref, field)}"
    assert lab.cb_layout == ref.cb_layout, f"CB layout drift:\n lab {lab.cb_layout}\n op  {ref.cb_layout}"
    return ref


def lab_rms_norm(input_tensor, *, gamma=None, epsilon=1e-6, compute_kernel_config=None, levers=None, out_plan=None):
    """The op's entry point, pointed at the lab plan + the lab's kernel copies."""
    validate(
        input_tensor,
        gamma=gamma,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
        memory_config=None,
    )
    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        input_tensor.memory_config(),
    )
    pd = create_program_descriptor(
        input_tensor,
        gamma,
        output_tensor,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
        levers=levers,
        out_plan=out_plan,
    )
    tensors = [input_tensor] if gamma is None else [input_tensor, gamma]
    tensors.append(output_tensor)
    return ttnn.generic_op(tensors, pd)
