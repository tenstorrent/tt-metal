# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm ProgramDescriptor — the Blocking Model of op_design.md §1 made concrete.

Everything below derives from ONE knob, ``L1_BLOCK_BUDGET_BYTES``:

    BLOCK_CB_UNITS    = CB pages consumed per block tile      (per layout/gamma cell)
    TILE_BLOCK_BUDGET = L1_BLOCK_BUDGET_BYTES // (BLOCK_CB_UNITS * unit_bytes)
    WT_CHUNK          = coarsest chunk of the reduced W axis that fits the budget
    NW                = Wt // WT_CHUNK                        (chunks per row)
    HT_BLOCK          = tile-rows per compute block           (1 when NW > 1)

Every CB page count, every kernel loop trip count and the core grid are
functions of those parameters — never of a whole-op dimension (Wt, W, H) and
never a magic literal.  The two ``Wt``-sized CBs (``cb_input_tiles`` when
``X_RESIDENT``, ``cb_gamma`` when ``GAMMA_RESIDENT``) are predicate-guarded
residents with a bounded streaming fallback (§1.3).

Work distribution (§5): the *independent* tile-row axis is spread over the
whole compute grid via ``ttnn.split_work_to_cores(..., row_wise=True)``; each
core owns whole rows and reduces the *dependent* W axis sequentially in-core.

--------------------------------------------------------------------------
Chunk-uniformity invariant (implementation constraint, documented deviation)
--------------------------------------------------------------------------
op_design.md §1.2 sets ``WT_CHUNK = min(Wt, TILE_BLOCK_BUDGET)`` and allows a
short last chunk (``WT_LAST``).  A short last chunk is NOT expressible here:

  * a circular buffer's write/read pointer may never straddle ``fifo_limit``
    (dataflow_api.h:216-222) — with a mixed push sequence
    ``[C, C, ..., L]`` on a CB of capacity ``k*C`` the pointer lands
    mid-buffer and the next full push runs past the limit; and
  * ``cb_input_rm`` / ``cb_output_rm`` page sizes ARE the chunk width, so the
    tilize/untilize LLK row stride and
    ``write_sticks_after_untilize``'s ``round_up(row_bytes, tile_row_bytes)``
    pop count cannot both be satisfied by two different chunk widths.

So ``WT_CHUNK`` is picked as the **coarsest divisor of ``Wt`` that fits the
budget**, making ``WT_LAST == WT_CHUNK`` by construction.  ``WT_LAST`` stays a
separate emitted knob so the general case remains expressible.  The knob is
still "coarsest that fits L1" for every ``Wt`` with a divisor near the budget;
only a prime ``Wt`` larger than the budget degrades (correct, just more
chunks) — a follow-up could recover it with a padded tail on the TILE path.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

# ---------------------------------------------------------------------------
# The knob family (§1.2). One definition each; everything else derives.
# ---------------------------------------------------------------------------

# Block factor budget: bytes of L1 a single compute block may occupy across all
# block-scaled CBs. Halved automatically if the final CB total misses
# L1_CB_BUDGET_BYTES (documented fallback, same single source).
L1_BLOCK_BUDGET_BYTES = 512 * 1024

# Total per-core CB budget. Worker L1 is 1.5 MB; the remainder is
# firmware/stack/kernel-args headroom.
L1_CB_BUDGET_BYTES = 1_100_000

# ---------------------------------------------------------------------------
# Cross-core W-split knobs (Refinement 2; op_design.md Lamp L1 / §4.2)
# ---------------------------------------------------------------------------
#
# The reduced W axis is *dependent*, so splitting it across cores costs a
# partial-sum combine + a 1/rms broadcast per row-block. Three single-source
# knobs govern when that trade is taken and how wide it goes.

# Engage the W-split only when the INDEPENDENT row axis cannot even fill one
# core per grid row -- i.e. the row split alone leaves >= (grid.x - 1)/grid.x of
# the grid idle. Structural discriminator, not a tuned threshold.
W_SPLIT_MAX_HT_FOR_SPLIT = None  # None => grid.y (derived at call time)

# ... and only when the reduced axis is wide enough that a core still gets a
# real chunk after the split. 8 tiles is the measured read-batch plateau
# (examples/double_buffer): below it the combine handshake dominates the work.
W_SPLIT_MIN_WT = 8

# L1 the ROOT core may spend on the gathered partial-sum tiles
# (CW * HT_BLOCK fp32 tiles). Caps how wide an *interleaved* W-split goes; a
# sharded input's CW is pinned by its shard grid and is instead absorbed by the
# halve-and-re-derive loop in _derive_blocking.
L1_GATHER_BUDGET_BYTES = 256 * 1024

# Semaphore ids. Disjoint combine groups reuse the SAME ids -- a semaphore id
# resolves to a per-core L1 cell, so group {A,B} bumping id 0 on B is a
# different cell from group {C,D} bumping id 0 on D
# (references/cross_core_reduction_design.md §5).
SEM_GATHER = 0  # workers -> group root: "my partial has landed" counter
SEM_MCAST_BASE = 1  # Mcast2D takes SEM_MCAST_BASE (data_ready) and +1 (consumer_ready)

# Buffer depths (§1.2). Phase-1 minimal = 2 (double buffer).
X_DEPTH = 2
OUT_DEPTH = 2
GAMMA_DEPTH = 2

# Depth of the RESIDENT input row-strip (§1.3's fast path). At depth 1 the strip
# is single-buffered, so the reader (TILE) / tilize (RM) cannot begin row-block
# hb+1 until compute has drained hb — read and compute serialize across
# row-blocks. Depth 2 overlaps them. Predicate-guarded: the derivation walks
# down from this value to 1 and then to the streaming path, taking the deepest
# strip that fits L1_CB_BUDGET_BYTES.
X_RESIDENT_DEPTH = 2

# The tilize LLK always consumes a whole 32-row block from its row-page CB, so
# the row-page CBs must be physically able to hold 32 rows even when only one
# row carries data (the single-stick gamma tilize).
TILIZE_ROWS = TILE_DIM

# dtypes the reduce scaler / partial-W mask dataflow helpers can actually fill.
# reduce_helpers_dataflow.inl static_asserts on exactly these two.
_MASKABLE_DTYPES = (ttnn.float32, ttnn.bfloat16)

# Block-float dtypes: 16 datums share one exponent, so there is no per-element
# size. Two consequences are used below.
_BLOCK_FLOAT_DTYPES = tuple(
    dt for dt in (getattr(ttnn, name, None) for name in ("bfloat8_b", "bfloat4_b")) if dt is not None
)


def _row_elem_bytes(tensor):
    """Bytes per element along a ROW_MAJOR stick.

    ``Tensor.element_size()`` *raises* for a block-float dtype ("datum for
    bfp2, bfp4, bfp8 is invalid", tt_backend_api_types.hpp) because a
    block-float datum has no standalone width. That is not a gap to work
    around: a block-quantized tensor equally cannot BE row-major (no blocks in
    a stick), so every consumer of this value — the ``cb_input_rm`` /
    ``cb_gamma_rm`` page sizes and the four ``*_ROW_BYTES`` compile-time args —
    sits behind an ``is_rm`` / ``is_rm_gamma`` predicate the dtype can never
    satisfy. Return the unpacked-datum width as a structurally-unused
    placeholder instead of letting the exception escape a branch that discards
    the result anyway.
    """
    if tensor.dtype in _BLOCK_FLOAT_DTYPES:
        return 2  # unreachable placeholder — see docstring
    return tensor.element_size()


# ---------------------------------------------------------------------------
# CB indices (semantic names; the number is just the buffer slot)
# ---------------------------------------------------------------------------

CB_INPUT_TILES = 0
CB_GAMMA = 1
CB_SCALER = 2
CB_INPUT_RM = 3
CB_GAMMA_RM = 4
# --- cross-core W-split only (Refinement 2) ---
CB_ONES = 5  # Float32 scaler for the combine reduce (root cores only)
CB_GROUP_PARTIALS = 6  # gathered raw sum(x^2), CW slots per tile-row (root only)
CB_RMS_MEAN = 7  # root's combined mean(x^2)  compute -> reader (mcast source)
CB_PARTIAL_OUT = 8  # this core's raw sum(x^2)  compute -> writer (gather source)
CB_OUTPUT_TILES = 16
CB_OUTPUT_RM = 17
CB_X_SQUARED = 24
CB_PARTIALS = 25
CB_RMS_SUM = 26
CB_RMS_RECIP = 27
CB_SCALED = 28


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _prod(values) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


def _coarsest_divisor(n: int, cap: int) -> int:
    """Largest d with ``d | n`` and ``d <= cap`` (cap >= 1, n >= 1)."""
    cap = max(1, min(cap, n))
    for d in range(cap, 0, -1):
        if n % d == 0:
            return d
    return 1


class _Blocking:
    """The derived blocking + CB plan for one (tensor, gamma, budget) triple.

    ``wt_core`` is the W-tile extent a SINGLE core owns. Without a W-split it is
    the whole ``Wt``; under the cross-core W-split (Refinement 2) it is the
    per-core slice, and every knob below (``WT_CHUNK``/``NW``/``HT_BLOCK``, the
    residency predicates, every CB page count) derives from it exactly as
    before — the split changes the *value* of the axis extent, not the model.
    """

    def __init__(
        self,
        input_tensor,
        gamma,
        l1_block_budget_bytes,
        grid_cores,
        wt_core=None,
        rows_core_max=None,
        cw=1,
        sharded_in=False,
        sharded_out=False,
    ):
        self.sharded_in = bool(sharded_in)
        self.sharded_out = bool(sharded_out)
        self.is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
        self.has_gamma = gamma is not None
        self.is_rm_gamma = self.has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT

        shape = list(input_tensor.shape)
        self.W = int(shape[-1])
        self.wt_global = _ceil_div(self.W, TILE_DIM)
        self.w_valid_last = self.W - (self.wt_global - 1) * TILE_DIM  # in [1, 32]
        self.has_partial_w = (self.W % TILE_DIM) != 0
        # Per-core W extent — the axis the block knobs are derived against.
        self.Wt = self.wt_global if wt_core is None else int(wt_core)
        self.cw = int(cw)
        self.w_split = self.cw > 1

        # §5.1 tile geometry — alignment-aware, per image, ceil everywhere.
        if self.is_rm:
            self.total_sticks = _prod(shape[:-1])
            self.ht_total = _ceil_div(self.total_sticks, TILE_DIM)
        else:
            self.total_sticks = 0
            self.ht_total = _prod(shape[:-2]) * _ceil_div(int(shape[-2]), TILE_DIM)
        self._rows_core_max = self.ht_total if rows_core_max is None else int(rows_core_max)

        self.tile_bytes = ttnn.tile_size(input_tensor.dtype)
        self.elem_bytes = _row_elem_bytes(input_tensor)
        self.gamma_tile_bytes = ttnn.tile_size(gamma.dtype) if self.has_gamma else self.tile_bytes
        self.gamma_elem_bytes = _row_elem_bytes(gamma) if self.has_gamma else self.elem_bytes
        self.fp32_tile_bytes = ttnn.tile_size(ttnn.float32)
        # --- the reduce-datapath format, and the mask that must match it -----
        #
        # cb_scaler carries the partial-W 0/1 MASK tile, which
        # reduce_accumulate_via_add's fold_partial_last reads through srcB via
        # llk_unpack_AB<ROW>(input_dfb, scaler_dfb) WITHOUT reconfiguring srcB.
        # Reduce entry already ran reconfig_data_format(input_dfb, input_dfb),
        # so srcB is configured at the format of the reduce's INPUT CB — which
        # is `cb_x_squared`, not the input tensor. Phase 0 wrote "the input
        # dtype" for both because the two were always the same value; the
        # load-bearing quantity is cb_x_squared's format, and separating them
        # is what makes a third input dtype expressible. (op_design.md R4's
        # fixed Float16_b is right for the ReduceTile datapath, wrong for this
        # one: a Float16_b mask under an fp32 reduce reads as fp32.)
        #
        # prepare_reduce_mask / calculate_and_prepare_reduce_scaler static_assert
        # on {Float16_b, Float32} (reduce_helpers_dataflow.inl), so the reduce
        # datapath has to be one of those two. bfloat8_b is not: a bf8 tile's
        # leading bytes are the shared-exponent header, so a Float16_b mask read
        # through a Bfp8_b-configured srcB decodes as all-zeros — measured, the
        # whole last reduce-dim tile then contributes 0 (probe_005: all-ones
        # W=49 summed to 32, not 49). PCC hides it completely, because dropping
        # elements only rescales each row and PCC is scale-invariant.
        #
        # So block-float inputs square into a bfloat16 cb_x_squared: the reduce
        # then programs srcA/srcB at Float16_b, the mask matches, and partial-W
        # is correct instead of quietly wrong. It is also the more accurate
        # accumulator input — x^2 spans twice the exponent range of x, which is
        # the worst case for 16-datum shared-exponent blocks.
        self.x_squared_dtype = input_tensor.dtype if input_tensor.dtype in _MASKABLE_DTYPES else ttnn.bfloat16
        self.x_squared_tile_bytes = ttnn.tile_size(self.x_squared_dtype)
        self.scaler_dtype = self.x_squared_dtype
        self.scaler_tile_bytes = self.x_squared_tile_bytes

        # --- BLOCK_CB_UNITS: CB pages charged per block tile (§6.3) ----------
        units = 0
        units += 1 if self.is_rm else X_DEPTH  # cb_input_tiles
        units += X_DEPTH if self.is_rm else 0  # cb_input_rm  (row pages == 1 tile/row-block)
        units += 1  # cb_x_squared
        units += 1 if self.has_gamma else 0  # cb_scaled
        units += 1 if self.is_rm else OUT_DEPTH  # cb_output_tiles
        units += OUT_DEPTH if self.is_rm else 0  # cb_output_rm
        units += GAMMA_DEPTH if self.has_gamma else 0  # cb_gamma (streaming)
        units += 1 if self.is_rm_gamma else 0  # cb_gamma_rm (conservative unit)
        self.block_cb_units = units
        # Widest block-scaled page, so TILE_BLOCK_BUDGET stays an upper bound on
        # the real per-block footprint. cb_x_squared is included because it can
        # be wider than the input tile (block-float input -> bfloat16 square).
        self.unit_bytes = max(self.tile_bytes, self.gamma_tile_bytes, self.x_squared_tile_bytes)

        # --- the block factors -----------------------------------------------
        self.tile_block_budget = max(1, l1_block_budget_bytes // (self.block_cb_units * self.unit_bytes))
        self.wt_chunk = _coarsest_divisor(self.Wt, self.tile_block_budget)
        self.nw = self.Wt // self.wt_chunk
        self.wt_last = self.Wt - (self.nw - 1) * self.wt_chunk  # == wt_chunk by construction
        if self.nw > 1:
            # R7: TileOffset::Set(wc*WT_CHUNK) into a resident row-strip is only
            # correct when a resident block is one flat Wt-tile strip.
            self.ht_block = 1
        else:
            self.ht_block = max(1, min(self.tile_block_budget // self.wt_chunk, self._rows_core_max))

        assert self.wt_last == self.wt_chunk, "chunk-uniformity invariant broken"
        assert not (self.nw > 1 and self.ht_block > 1), "R7: NW > 1 requires HT_BLOCK == 1"

        # --- residency fast-path predicates (§1.3) ---------------------------
        #
        # L1 is spent in strict order of how much DRAM TRAFFIC each step removes;
        # a step that only buys pipelining comes last:
        #
        #   1. X_RESIDENT at depth 1  — removes an entire second read pass over x
        #                               (ht_per_core * Wt tile-reads). Biggest win.
        #   2. GAMMA_RESIDENT         — removes NH_core * Wt gamma re-reads.
        #   3. extra resident depth   — removes NO traffic; it only lets the
        #                               producer run a row-block ahead.
        #
        # §1.3 fixes the 1-before-2 order; 3 must come last for the same reason,
        # and getting that wrong is expensive: buying depth 2 before gamma cost
        # 1.20x on (1,1,8192,5120) (482_655 -> 577_752 ns) purely by evicting
        # gamma from L1.
        #
        # Step 3 is additionally gated on there being latency to hide at all. With
        # the grid full every core is queued on DRAM, so running the reader a
        # row-block ahead only front-loads contention: measured 1.04x SLOWER on
        # (1,1,8192,1024) (103_238 -> 107_592 ns). Same structural discriminator
        # as the read-batch knob (see _x_read_chunks).
        num_cores = min(grid_cores, self.ht_total * self.cw)
        self.grid_full = num_cores >= grid_cores
        # Row-blocks a single core loops over. Both "hold it across row-blocks"
        # levers below are worth nothing when this is 1.
        self.nh_core_max = _ceil_div(self._rows_core_max, self.ht_block)
        base = self._cb_total(x_res_depth=0, gamma_resident=False)

        # A sharded input is resident BY PLACEMENT — the shard is already in this
        # core's L1, so the residency predicate is not a choice here.
        self.x_res_depth = 1 if (self.sharded_in or self._cb_total(1, False) <= L1_CB_BUDGET_BYTES) else 0
        self.x_resident = self.x_res_depth > 0

        # Gamma residency removes (NH_core - 1) * Wt gamma re-reads per core — so
        # at NH_core == 1 it removes NOTHING: gamma is read exactly once either
        # way, and holding it resident only converts an overlappable per-chunk
        # read in pass B into a serial prologue that must complete before the
        # first input tile is even requested. Measured on (1,1,32,5120), one
        # core: resident gamma 47_571 ns vs streamed gamma 42_407 ns (1.12x).
        self.gamma_resident = False
        if self.has_gamma and self.nh_core_max > 1:
            self.gamma_resident = self._cb_total(self.x_res_depth, True) <= L1_CB_BUDGET_BYTES

        # Extra resident depth overlaps row-block hb+1's read with hb's compute,
        # so it likewise needs NH_core > 1 — and needs spare latency to hide
        # (see the grid_full note above).
        #
        # It is also meaningless on the RM path: there cb_input_tiles is produced
        # by compute's own tilize and consumed by compute's square/mul, so its
        # producer and consumer are the SAME RISC. A second strip cannot be
        # filled ahead — it would only cost L1 (and can evict gamma).
        if self.x_resident and not self.is_rm and not self.grid_full and self.nh_core_max > 1:
            for depth in range(X_RESIDENT_DEPTH, self.x_res_depth, -1):
                if self._cb_total(depth, self.gamma_resident) <= L1_CB_BUDGET_BYTES:
                    self.x_res_depth = depth
                    break

        self.cb_total_bytes = self._cb_total(self.x_res_depth, self.gamma_resident)
        self.base_cb_total_bytes = base

    # -- CB plan ------------------------------------------------------------
    def cb_plan(self, x_res_depth=None, gamma_resident=None):
        """[(name, index, page_size, num_pages)] for every CB.

        `x_res_depth` is the resident-strip buffer depth (0 = streaming).

        Unused CBs are still declared (1 page) so that the compile-time CB
        descriptors the helpers read (tile size / tile dims / format) are valid
        even in `if constexpr`-discarded branches.
        """
        x_depth = self.x_res_depth if x_res_depth is None else x_res_depth
        g_res = self.gamma_resident if gamma_resident is None else gamma_resident

        H = self.ht_block
        C = self.wt_chunk
        B = H * C
        Wt = self.Wt

        if self.sharded_in:
            # Zero-copy: the CB *is* the resident shard, so its extent is the
            # whole shard, not a block multiple.
            input_tile_pages = self._rows_core_max * Wt
        elif x_depth > 0:
            input_tile_pages = x_depth * H * Wt
        elif self.is_rm:
            input_tile_pages = B  # compute->compute (tilize -> square), one block
        else:
            input_tile_pages = X_DEPTH * B

        gamma_pages = 0
        if self.has_gamma:
            gamma_pages = Wt if g_res else GAMMA_DEPTH * C

        plan = [
            ("cb_input_tiles", CB_INPUT_TILES, self.tile_bytes, input_tile_pages),
            ("cb_gamma", CB_GAMMA, self.gamma_tile_bytes, gamma_pages),
            ("cb_scaler", CB_SCALER, self.scaler_tile_bytes, 1),
            # One row-page per stick; a tile-row of the block is TILIZE_ROWS sticks.
            (
                "cb_input_rm",
                CB_INPUT_RM,
                C * TILE_DIM * self.elem_bytes,
                (X_DEPTH * H * TILIZE_ROWS) if self.is_rm else 0,
            ),
            (
                "cb_gamma_rm",
                CB_GAMMA_RM,
                C * TILE_DIM * self.gamma_elem_bytes,
                TILIZE_ROWS if self.is_rm_gamma else 0,
            ),
            (
                "cb_output_tiles",
                CB_OUTPUT_TILES,
                self.tile_bytes,
                (self._rows_core_max * Wt) if self.sharded_out else (B if self.is_rm else OUT_DEPTH * B),
            ),
            ("cb_output_rm", CB_OUTPUT_RM, self.tile_bytes, (OUT_DEPTH * B) if self.is_rm else 0),
            ("cb_x_squared", CB_X_SQUARED, self.x_squared_tile_bytes, B),
            # --- cross-core W-split (Refinement 2). Every page count is a
            # function of the SAME knobs (HT_BLOCK and the group width CW),
            # never of a whole-op dimension. Sized 1 (dummy) when CW == 1.
            ("cb_ones", CB_ONES, self.fp32_tile_bytes, 1),
            (
                "cb_group_partials",
                CB_GROUP_PARTIALS,
                self.fp32_tile_bytes,
                (H * self.cw) if self.w_split else 0,
            ),
            ("cb_rms_mean", CB_RMS_MEAN, self.fp32_tile_bytes, H if self.w_split else 0),
            ("cb_partial_out", CB_PARTIAL_OUT, self.fp32_tile_bytes, H if self.w_split else 0),
            ("cb_partials", CB_PARTIALS, self.fp32_tile_bytes, 2 * H),
            ("cb_rms_sum", CB_RMS_SUM, self.fp32_tile_bytes, H),
            ("cb_rms_recip", CB_RMS_RECIP, self.fp32_tile_bytes, H),
            ("cb_scaled", CB_SCALED, self.tile_bytes, B if self.has_gamma else 0),
        ]
        return [(n, i, ps, max(1, np)) for (n, i, ps, np) in plan]

    def _cb_total(self, x_res_depth, gamma_resident):
        return sum(ps * np for (_, _, ps, np) in self.cb_plan(x_res_depth, gamma_resident))


def _tile_geometry(input_tensor):
    """(ht_total, wt_global) — the whole-op tile grid, before any core split."""
    shape = list(input_tensor.shape)
    wt_global = _ceil_div(int(shape[-1]), TILE_DIM)
    if input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT:
        ht_total = _ceil_div(_prod(shape[:-1]), TILE_DIM)
    else:
        ht_total = _prod(shape[:-2]) * _ceil_div(int(shape[-2]), TILE_DIM)
    return ht_total, wt_global


def _derive_blocking(input_tensor, gamma, grid_cores, placement, sharded_in=False, sharded_out=False):
    """Derive the blocking, halving the block budget until the CB total fits.

    The halving loop also absorbs the ROOT core's gather buffer
    (``HT_BLOCK * CW`` fp32 tiles): it is charged into ``cb_total_bytes`` for
    every core, so an over-wide combine shrinks ``HT_BLOCK`` rather than
    silently overflowing L1 on the roots.
    """
    kwargs = dict(
        wt_core=placement.wt_core,
        rows_core_max=placement.rows_core_max,
        cw=placement.cw,
        sharded_in=sharded_in,
        sharded_out=sharded_out,
    )
    budget = L1_BLOCK_BUDGET_BYTES
    blk = _Blocking(input_tensor, gamma, budget, grid_cores, **kwargs)
    while blk.cb_total_bytes > L1_CB_BUDGET_BYTES and budget > blk.unit_bytes:
        budget //= 2
        blk = _Blocking(input_tensor, gamma, budget, grid_cores, **kwargs)
    assert blk.cb_total_bytes <= L1_CB_BUDGET_BYTES, (
        f"rms_norm: per-core CB total {blk.cb_total_bytes} B exceeds "
        f"L1_CB_BUDGET_BYTES={L1_CB_BUDGET_BYTES} even at the minimum block size"
    )
    return blk


def _x_read_chunks(blk):
    """How many W-chunks the reader coalesces into ONE reserve/barrier/push.

    This is the read-granularity knob, and it is measured, not guessed. The two
    regimes want opposite values (blackhole_p150b, 110 cores, bf16 TILE gamma,
    device kernel ns):

      shape              cores   1 chunk/barrier   NW chunks/barrier
      (1,1,32,5120)          1            42_407              51_434
      (1,1,32,7168)          1            57_396              63_313
      (1,1,8192,5120)      110           575_867             482_655
      (1,1,8192,2304)      110           227_673             214_633

    With only a few cores busy the op is LATENCY-bound: one chunk per barrier
    lets compute start on chunk 0 while the NoC fetches chunk 1, worth up to
    1.21x. With the grid full every core is already queued on DRAM, so the op is
    THROUGHPUT-bound: there is no latency left to hide and each extra barrier
    just exposes the (now much longer) NoC tail, costing up to 1.19x.

    The discriminator is therefore structural, not a tuned threshold: has the
    independent row axis already filled the grid? Only the resident TILE path is
    affected — the streaming path's input CB is sized X_DEPTH*B and physically
    cannot hold a multi-chunk batch, and the RM path's cb_input_tiles is filled
    by compute's tilize, not by the reader.
    """
    if blk.sharded_in or not (blk.x_resident and not blk.is_rm):
        return 1
    return blk.nw if blk.grid_full else 1


class _CoreWork:
    """What one core owns: a tile-row range x a W-tile slice, plus its combine role."""

    __slots__ = ("core", "start_row", "num_rows", "wt_start", "wt_real", "slot", "group")

    def __init__(self, core, start_row, num_rows, wt_start, wt_real, slot, group):
        self.core = core
        self.start_row = start_row
        self.num_rows = num_rows
        self.wt_start = wt_start
        self.wt_real = wt_real  # W-tiles that actually exist (rest is shard padding)
        self.slot = slot  # index inside the combine group
        self.group = group  # index into _Placement.groups (-1 == no group)


class _Placement:
    """Core assignment + cross-core combine topology (op_design.md §5 and §4.2).

    Two axes are placed, not one:
      * the INDEPENDENT tile-row axis, split across *groups* (zero traffic), and
      * the DEPENDENT W axis, split across the ``cw`` cores INSIDE each group,
        which therefore need a partial-sum combine + a 1/rms broadcast.

    ``cw == 1`` degenerates to the phase-0 row-only split, and every W-split
    structure below collapses out.

    Invariant relied on by the broadcast: **every group is a rectangle**, so one
    ``Mcast2D`` per group serves it. Groups the data placement hands us that are
    ragged are padded up to their bounding box with zero-work filler cores.
    """

    def __init__(self, all_cores, works, groups, cw, wt_core, rows_core_max):
        self.all_cores = all_cores
        self.works = works
        self.groups = groups  # [(root_core, rect_CoreRangeSet, num_receivers)]
        self.cw = cw
        self.wt_core = wt_core
        self.rows_core_max = rows_core_max

    @property
    def w_split(self):
        return self.cw > 1

    @property
    def num_cores(self):
        return len(self.works)


def _rect(x0, y0, x1, y1):
    return ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))


def _rows_split(total, n):
    """Split `total` tile-rows over `n` groups, biggest-first (like split_work_to_cores)."""
    q, r = divmod(total, n)
    return [q + 1 if i < r else q for i in range(n)]


def _placement_rows(grid, ht_total):
    """Phase-0 placement: the independent tile-row axis over the whole grid."""
    (
        _num,
        all_cores,
        group_1,
        group_2,
        rows_g1,
        rows_g2,
    ) = ttnn.split_work_to_cores(grid, ht_total, row_wise=True)

    works = []
    start = 0
    for group, per_core in ((group_1, rows_g1), (group_2, rows_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            works.append(_CoreWork(core, start, per_core, 0, None, 0, -1))
            start += per_core
    assert start == ht_total, f"work split covered {start} of {ht_total} tile-rows"
    rows_max = max((w.num_rows for w in works), default=1)
    return _Placement(all_cores, works, [], 1, None, rows_max)


def _w_split_rectangle(grid, band_h, wt_global, gather_tile_bytes):
    """Widest core rectangle (cw_x <= grid.x, cw_y <= band_h) that DIVIDES `wt_global`.

    Requiring the group size to divide `Wt` is what keeps every core's slice the
    same width, so one compile-time `WT` / `WT_CHUNK` / `NW` serves the whole
    grid (and the last core's last tile is still the tensor's last tile, so the
    partial-W mask lands where it belongs).
    """
    cap = max(1, L1_GATHER_BUDGET_BYTES // gather_tile_bytes)
    best = (1, 1)
    for cy in range(1, band_h + 1):
        for cx in range(1, grid.x + 1):
            n = cx * cy
            if n > cap or wt_global % n != 0:
                continue
            if n > best[0] * best[1]:
                best = (cx, cy)
    return best


def _placement_wsplit(grid, ht_total, wt_global, gather_tile_bytes):
    """Cross-core W-split for an INTERLEAVED input whose row axis under-fills the grid.

    Groups are stacked along the grid's y axis (one band of `cw_y` grid rows
    each) and are rectangles by construction. Returns None when no split wider
    than one core is available.
    """
    num_groups = min(ht_total, grid.y)
    band_h = grid.y // num_groups
    cw_x, cw_y = _w_split_rectangle(grid, band_h, wt_global, gather_tile_bytes)
    cw = cw_x * cw_y
    if cw < 2:
        return None

    wt_core = wt_global // cw
    rows = _rows_split(ht_total, num_groups)

    works = []
    groups = []
    ranges = []
    start_row = 0
    for g in range(num_groups):
        y0 = g * band_h
        rect = _rect(0, y0, cw_x - 1, y0 + cw_y - 1)
        ranges.append(rect)
        root = ttnn.CoreCoord(0, y0)
        groups.append((root, ttnn.CoreRangeSet([rect]), cw - 1))
        slot = 0
        for yy in range(y0, y0 + cw_y):
            for xx in range(cw_x):
                works.append(
                    _CoreWork(
                        ttnn.CoreCoord(xx, yy),
                        start_row,
                        rows[g],
                        slot * wt_core,
                        wt_core,
                        slot,
                        g,
                    )
                )
                slot += 1
        start_row += rows[g]
    assert start_row == ht_total
    return _Placement(ttnn.CoreRangeSet(ranges), works, groups, cw, wt_core, max(rows))


def _shard_geometry(tensor):
    """(shard_ht_tiles, shard_wt_tiles, cores_row_major, grid) for a sharded tensor."""
    spec = tensor.memory_config().shard_spec
    sh, sw = int(spec.shape[0]), int(spec.shape[1])
    grid = spec.grid
    cores = ttnn.corerange_to_cores(grid, None, True)
    return _ceil_div(sh, TILE_DIM), _ceil_div(sw, TILE_DIM), cores, grid


def _bbox_cores(core_range_set):
    bb = core_range_set.bounding_box()
    return bb.start.x, bb.start.y, bb.end.x, bb.end.y


def _placement_sharded(input_tensor, ht_total, wt_global):
    """Placement pinned by the input's shard grid (WIDTH / BLOCK sharded).

    WIDTH: one group over every shard core (each owns a W slice of ALL rows).
    BLOCK: one group per grid row (each row owns a tile-row band, split by W).

    A ragged shard grid (auto_shard_config emits full rows + a partial last row)
    is padded up to its bounding box with zero-work filler cores so the group
    stays the single rectangle a multicast can address.
    """
    layout = input_tensor.memory_config().memory_layout
    ht_s, wt_s, cores, grid = _shard_geometry(input_tensor)
    x0, y0, x1, y1 = _bbox_cores(grid)

    works = []
    groups = []
    if layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        nx = x1 - x0 + 1
        ny = y1 - y0 + 1
        assert len(cores) == nx * ny, "BLOCK_SHARDED grid must be a full rectangle"
        for gy in range(ny):
            y = y0 + gy
            groups.append((ttnn.CoreCoord(x0, y), ttnn.CoreRangeSet([_rect(x0, y, x1, y)]), nx - 1))
            for gx in range(nx):
                works.append(
                    _CoreWork(
                        ttnn.CoreCoord(x0 + gx, y),
                        gy * ht_s,
                        ht_s,
                        gx * wt_s,
                        max(0, min(wt_s, wt_global - gx * wt_s)),
                        gx,
                        gy,
                    )
                )
        cw = nx
        rows_max = ht_s
    else:  # WIDTH_SHARDED — one group, every shard core owns a W slice of all rows
        rect_set = ttnn.CoreRangeSet([_rect(x0, y0, x1, y1)])
        cw = len(cores)  # combine width == real W slices (fillers do not contribute)
        groups.append((cores[0], rect_set, cw - 1))
        owned = {(int(c.x), int(c.y)) for c in cores}
        for slot, core in enumerate(cores):
            works.append(_CoreWork(core, 0, ht_total, slot * wt_s, max(0, min(wt_s, wt_global - slot * wt_s)), slot, 0))
        # filler cores: inside the multicast rectangle, but outside the shard grid.
        # Zero work, no gather contribution, no readiness ack — they only need
        # the broadcast to land somewhere legal on them.
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                if (xx, yy) not in owned:
                    works.append(_CoreWork(ttnn.CoreCoord(xx, yy), 0, 0, 0, 0, 0, 0))
        rows_max = ht_total

    all_cores = ttnn.CoreRangeSet([_rect(x0, y0, x1, y1)])
    return _Placement(all_cores, works, groups, cw, wt_s, rows_max)


def create_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    gamma=None,
    epsilon: float = 1e-6,
    compute_kernel_config=None,
    device=None,
) -> "ttnn.ProgramDescriptor":
    grid = (
        device.compute_with_storage_grid_size()
        if device is not None
        else input_tensor.device().compute_with_storage_grid_size()
    )
    ht_total, wt_global = _tile_geometry(input_tensor)
    in_sharded = input_tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    out_sharded = output_tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED

    placement = _select_placement(grid, input_tensor, ht_total, wt_global, in_sharded)
    blk = _derive_blocking(
        input_tensor, gamma, grid.x * grid.y, placement, sharded_in=in_sharded, sharded_out=out_sharded
    )

    all_cores = placement.all_cores
    x_read_chunks = _x_read_chunks(blk)

    # ---------- circular buffers ----------
    # A sharded tensor's shard IS the per-core block and it is ALREADY in this
    # core's L1, so its CB is placed straight on the buffer (zero-copy, no NoC
    # read at all). Everything else is a normal program-allocated CB.
    #
    # Every non-sharded CB is declared over the WHOLE core set, at identical
    # sizes. That uniformity is load-bearing for the combine: a worker derives
    # the root's landing address for its partial from `get_write_ptr` on its OWN
    # copy of cb_group_partials, which is only valid because the CB sits at the
    # same L1 offset on every core.
    cbs = []
    for name, index, page_size, num_pages in blk.cb_plan():
        if in_sharded and name == "cb_input_tiles":
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(index, input_tensor))
            continue
        if out_sharded and name == "cb_output_tiles":
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(index, output_tensor))
            continue
        cbs.append(
            ttnn.CBDescriptor(
                total_size=page_size * num_pages,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=index,
                        data_format=_cb_format(name, blk, input_tensor, gamma),
                        page_size=page_size,
                    )
                ],
            )
        )

    # ---------- cross-core combine wiring (Refinement 2) ----------
    # One Mcast2D per combine group: the root broadcasts the finalized
    # mean(x^2) tile back over its rectangle. Disjoint groups reuse the same
    # semaphore ids, so the ids are created ONCE over the whole grid and every
    # group adopts them.
    semaphores = []
    mcasts = {}  # group index -> Mcast2D
    if placement.w_split:
        semaphores = [
            ttnn.SemaphoreDescriptor(id=SEM_GATHER, core_ranges=all_cores, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_MCAST_BASE, core_ranges=all_cores, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_MCAST_BASE + 1, core_ranges=all_cores, initial_value=0),
        ]
        cfg = ttnn.McastConfig(sem_ids=[SEM_MCAST_BASE, SEM_MCAST_BASE + 1])
        for g, (root, rect_set, num_ack) in enumerate(placement.groups):
            mcasts[g] = ttnn.Mcast2D(device, rect_set, root, cfg, num_ack)

    # ---------- shared compile-time knobs ----------
    chunk_row_bytes = blk.wt_chunk * TILE_DIM * blk.elem_bytes
    # Real bytes in the FINAL chunk of the core that owns the tensor's last
    # W-tile (every other core's final chunk is a full one).
    last_elems = blk.W - (wt_global - blk.wt_chunk) * TILE_DIM
    last_row_bytes = last_elems * blk.elem_bytes
    g_chunk_row_bytes = blk.wt_chunk * TILE_DIM * blk.gamma_elem_bytes
    g_last_row_bytes = last_elems * blk.gamma_elem_bytes

    regime = [
        1 if blk.is_rm else 0,
        1 if blk.has_gamma else 0,
        1 if blk.is_rm_gamma else 0,
        1 if blk.x_resident else 0,
        1 if blk.gamma_resident else 0,
        1 if blk.has_partial_w else 0,
    ]
    knobs = [blk.Wt, blk.wt_chunk, blk.wt_last, blk.nw, blk.ht_block, x_read_chunks]

    # The reader and the writer share ONE compile-time-arg layout (indices 0..23,
    # the mcast block at 24..28, TensorAccessorArgs from 29). Built once here so
    # a knob added to either kernel cannot drift between the two — the CT index
    # each kernel reads is then guaranteed to name the same quantity.
    mcast_ct = list(mcasts[0].compile_time_args()) if mcasts else [0, 0, 0, 0, 0]
    dataflow_ct_args = (
        regime
        + knobs
        + [
            blk.w_valid_last,
            chunk_row_bytes,
            last_row_bytes,
            g_chunk_row_bytes,
            g_last_row_bytes,
            blk.total_sticks,
            1 if placement.w_split else 0,
            placement.cw,
            wt_global,
            1 if in_sharded else 0,
            1 if out_sharded else 0,
            SEM_GATHER,
        ]
        + mcast_ct
    )
    DATAFLOW_ACCESSOR_ARG_BASE = len(dataflow_ct_args)  # kernels read TensorAccessorArgs<29>
    assert DATAFLOW_ACCESSOR_ARG_BASE == 29, (
        "rms_norm: reader/writer read TensorAccessorArgs<29>; the shared CT block "
        f"is now {DATAFLOW_ACCESSOR_ARG_BASE} long — update both kernels together"
    )

    # ---------- reader (NCRISC / NoC0) ----------
    reader_ct_args = list(dataflow_ct_args)
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if gamma is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    src_addr = input_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0
    dst_addr = output_tensor.buffer_address()
    eps_bits = struct.unpack("<I", struct.pack("<f", float(epsilon)))[0]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for w in placement.works:
        core = w.core
        wt_real = blk.Wt if w.wt_real is None else min(w.wt_real, blk.Wt)
        is_root = 0
        root_v = (0, 0)
        mcast_rt = [0, 0, 0, 0]
        if placement.w_split:
            root = placement.groups[w.group][0]
            is_root = 1 if (int(core.x) == int(root.x) and int(core.y) == int(root.y)) else 0
            rv = device.worker_core_from_logical_core(root)
            root_v = (rv.x, rv.y)
            mcast_rt = list(mcasts[w.group].runtime_args(core))
        # The core whose slice ENDS on the tensor's last W-tile is the one that
        # must mask the tile-padded columns of that tile.
        is_last_w = 1 if (w.wt_start + wt_real == wt_global) else 0

        reader_rt[core.x][core.y] = [
            src_addr,
            gamma_addr,
            w.start_row,
            w.num_rows,
            w.wt_start,
            w.slot,
            is_root,
            is_last_w,
            wt_real,
        ] + mcast_rt
        writer_rt[core.x][core.y] = [
            dst_addr,
            w.start_row,
            w.num_rows,
            w.wt_start,
            w.slot,
            root_v[0],
            root_v[1],
        ]
        compute_rt[core.x][core.y] = [w.num_rows, eps_bits, is_root, is_last_w]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---------- writer (BRISC / NoC1) ----------
    writer_ct_args = list(dataflow_ct_args)
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---------- compute ----------
    compute_ct_args = (
        regime
        + knobs
        + [
            blk.w_valid_last,
            blk.W,
            1 if placement.w_split else 0,
            placement.cw,
        ]
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=cbs,
    )


def _select_placement(grid, input_tensor, ht_total, wt_global, in_sharded):
    """Pick the core assignment: data-pinned when sharded, else the §5 row split
    optionally widened by the cross-core W-split (Lamp L1).

    The W-split is engaged when the INDEPENDENT row axis under-fills the grid —
    it cannot even put one core on every grid row — and the reduced axis is wide
    enough to still hand each core a real chunk after the split. Both gates are
    named constants at the top of this module, never inlined here.
    """
    if in_sharded:
        return _placement_sharded(input_tensor, ht_total, wt_global)

    ht_cap = grid.y if W_SPLIT_MAX_HT_FOR_SPLIT is None else W_SPLIT_MAX_HT_FOR_SPLIT
    env = os.environ.get("RMS_NORM_W_SPLIT")
    want_split = (ht_total <= ht_cap) and (wt_global >= W_SPLIT_MIN_WT)
    if env is not None:
        want_split = env != "0"
    if want_split:
        p = _placement_wsplit(grid, ht_total, wt_global, ttnn.tile_size(ttnn.float32))
        if p is not None:
            return p
    return _placement_rows(grid, ht_total)


def _cb_format(name, blk, input_tensor, gamma):
    if name in ("cb_gamma", "cb_gamma_rm"):
        return gamma.dtype if gamma is not None else input_tensor.dtype
    if name == "cb_x_squared":
        # The reduce's input CB — its format IS the reduce datapath. See
        # _Blocking.x_squared_dtype.
        return blk.x_squared_dtype
    if name == "cb_scaler":
        # Must match the format srcB is configured at inside the reduce, i.e.
        # cb_x_squared's; see _Blocking.x_squared_dtype for the full rationale.
        return blk.scaler_dtype
    if name in (
        "cb_partials",
        "cb_rms_sum",
        "cb_rms_recip",
        "cb_ones",
        "cb_group_partials",
        "cb_rms_mean",
        "cb_partial_out",
    ):
        return ttnn.float32
    return input_tensor.dtype
