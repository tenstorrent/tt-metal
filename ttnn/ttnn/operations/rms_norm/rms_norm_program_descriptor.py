# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for rms_norm.

THIS FILE IS THE SINGLE SOURCE OF TRUTH FOR EVERY BLOCK / DEPTH / GRID KNOB
(op_design.md section 1.4).  Kernels receive the knobs as compile-time /
runtime args and never re-derive them; no block or chunk count is restated as a
second literal anywhere.

Knob map (all tunable parameters, none inlined):

  primary (the only hand-set numbers)
    L1_SAFETY_FRACTION      fraction of usable per-core L1 the CBs may take
    CB_RM_STAGE_DEPTH       depth of the ROW_MAJOR stick staging CBs
    CB_DEPTH_CANDIDATES     ordered depths the regime search may give the two
                            cross-processor CBs (see D4)
    GRID_W                  cores along `width` (Lamp L1 lives at its trivial 1)
    REDUCE_BULK             reduce input policy (BulkWaitBulkPop vs per-tile)
    CB_ROW_STAT_DEPTH       ring depth of cb_row_stat, in units of BLOCK_ROWS.
                            NOT a perf knob -- >= 2 is a CORRECTNESS floor for
                            the partial final row-block (see D6)
    REDUCE_ACC_VIA_ADD_MIN_WT
                            smallest WT_CHUNK at which the reduce runs on
                            ReduceAlgorithm::AccumulateViaAdd instead of
                            ReduceTile (see D7)

  derived buffer depths
    CB_X_DEPTH / CB_OUT_DEPTH   the depth the regime search settled on; forced
                            to 1 on the ROW_MAJOR path, where the producer /
                            consumer is a sequential compute helper (tilize /
                            untilize) and depth buys no overlap

  derived helpers (one source of truth each; both L1 solves call them)
    _cb_block_mult()        which CBs scale with BLOCK_ROWS * WT_CHUNK, and at what
                            depth -- never re-spelled inline
    scaler_pages            page count of cb_scaler (2 when PARTIAL_W, else 1)
    reduce_acc_via_add      the chosen reduce datapath (D7); also decides what the
                            reader fills cb_scaler with
    scaler_tiles            tiles the reader actually pushes into cb_scaler and the
                            compute pops (<= scaler_pages)

  derived block factors
    BLOCK_ROWS   tile-rows per compute block = min(per-core assignment,
                 the coarsest chunk that fits the L1 budget)
    WT_CHUNK     width tiles per compute block = Wt in the RESIDENT regime;
                 the coarsest DIVISOR of Wt that fits L1 in STREAM
    NUM_W_CHUNKS = Wt // WT_CHUNK      (X_RESIDENT == GAMMA_RESIDENT == (== 1))

Deviations from op_design.md section 1.4 (advisory: CB sizing / knob selection;
the scheme, topology, work split and helper mapping are unchanged):

  D1  WT_CHUNK is constrained to a DIVISOR of Wt, so every width chunk is the
      same size.  Three mechanisms in the chosen helper set require a uniform
      chunk and would otherwise need a ragged-tail special case:
        * compute_kernel_lib::tilize / untilize take `block_width_tiles` as a
          COMPILE-TIME template parameter (tilize_helpers.hpp:188);
        * reduce()'s BulkWaitBulkPop asserts
          `num_pages(cb_in) % cols == 0` (reduce_helpers_compute.inl:698-699);
        * a multi-page cb_reserve_back / get_write_ptr batch must not straddle
          the CB ring, i.e. the ring size must be a multiple of the push unit.
      WT_CHUNK is still the coarsest value the L1 budget allows (largest
      admissible divisor), so the knob is not collapsed.
  D2  The STREAM chunk-size solve counts the ROW_MAJOR staging CBs at
      WT_CHUNK tiles (what is actually allocated), not at Wt.
  D3  RESOLVED by Refinement 1b.  accumulate_reduce_block() used not to expose
      reduce()'s ReduceFp32Mode / ReduceAlgorithm template slots; both are now
      forwarded (streaming_reduce_helpers.hpp), which is what made D7 possible.
      The op still passes ReduceFp32Mode::Fast: Accurate only routes *Float32*
      SUM through the SFPU, and the wide-W precision cell this op cares about is
      bfloat16 -- D7 is the lever that reaches it.  fp32 DEST accumulation still
      comes from fp32_dest_acc_en=True.
  D4  The regime predicate SEARCHES the depth knob rather than fixing it: it
      walks CB_DEPTH_CANDIDATES coarsest-first and takes RESIDENT at the first
      depth whose whole-row working set fits, dropping to STREAM only when no
      candidate fits.  With the shipped CB_DEPTH_CANDIDATES = (2,) this is
      BYTE-IDENTICAL to the design's fixed-depth predicate; the search exists so
      the depth is a live knob instead of an inlined constant.  Still a pure
      function of the same inputs as the design's predicate, so section 4.2's
      device-independent reproducibility property holds.

      MEASURED, and the reason the second candidate is NOT shipped
      (blackhole p150b, 110-core grid, ~1.35 GHz, bf16 + gamma, one fresh-cache
      run per variant):

        shape                     depth=(2,)   depth=(2,1)
        (1,1,32,4032) g=TILE        38399 ns     46136 ns   0.83x  REGRESSION
        (1,1,32,3072) g=RM          32712 ns     33076 ns   0.99x
        (1,1,32,4096) g=RM          42442 ns     42404 ns   1.00x  (outside band)
        (1,1,8192,1024) g=RM        88354 ns     88247 ns   1.00x  (outside band)

      Only widths in the band between the depth-2 and depth-1 residency
      thresholds (Wt in [91,126] for TILE gamma, [80,105] for ROW_MAJOR gamma)
      can move at all; test_rms_norm_perf.py::test_rms_norm_perf_depth_band
      pins two of them.  Inside the band, depth 1 does halve the DRAM bytes
      (x read once instead of twice) and still LOSES: at Rt = 1 the core has a
      single row-block, so depth 1 serializes reader -> compute -> writer for
      that block, and the lost overlap costs more than the saved bytes.
      Complementary step before depth 1 is worth offering: Lamp L5 (row-resident
      W-chunked third regime) removes STREAM's pass-B re-read WITHOUT giving up
      depth 2 -- strictly better than this trade -- and Lamp L1 (cross-core
      width split) gives the core many blocks again so depth 1 would no longer
      serialize.  Recorded as a follow-up, not a finished win.
  D5  Refinement 1 (precision surface) needed NO new format machinery: every CB
      already declares `data_format` = the dtype of the tensor it carries and
      `page_size` = ttnn.tile_size() of that same dtype, so bfloat8_b rides the
      existing path.  Per-CB roles, unchanged:
        cb_input_{sticks,tiles} / cb_x_squared / cb_normalized   input dtype
        cb_output_{tiles,sticks}                                 output dtype
        cb_gamma_{sticks,tiles}                                  GAMMA dtype
        cb_scaler                                                bfloat16 (1.0)
        cb_row_stat                                              float32, ALWAYS
      cb_row_stat stays fp32 in BOTH fp32_dest_acc_en modes: it is the
      cross-chunk accumulator that reduce()'s Accumulate::at reloads, so an
      fp32 CB keeps the STREAM reload lossless even when DEST itself is bf16.
      Demoting it to the input dtype would erase exactly the precision this op
      cares about (op_requirements.md Refinement 1, lever 1).

      Two consequences worth spelling out, both DELIBERATE non-changes:
        * `unpack_to_dest_mode` is left entirely at Default -- NO CB qualifies
          for UnpackToDestMode::UnpackToDestFp32.  The only fp32 CB is
          cb_row_stat, and while its reduce reload (AccumulateReloadMode::
          CopySeedPairs, the default) and the transform_in_place finalize are
          both copy_tile-into-DEST and would be compatible, pass B consumes it
          as operand B of an FPU broadcast multiply (mul<BroadcastDim::Col>).
          An UnpackToDestFp32 CB may never be an FPU operand
          (reduce_helpers_compute.inl:127-137) -- tagging it would corrupt
          silently.  Tagging cb_input_sticks is separately forbidden by
          tilize's Fp32Mode::Fast static_assert (op_design.md R16).
        * `Tensor.element_size()` is NOT defined for a block-float dtype, so the
          ROW_MAJOR stick byte math goes through _stick_elem_bytes(); see its
          docstring for why 0 is correct there rather than a fudge.
  D6  cb_row_stat is CB_ROW_STAT_DEPTH (= 2) * BLOCK_ROWS pages, not BLOCK_ROWS.
      This is a CORRECTNESS requirement, found by the resilience loose cases that
      Refinement 1 made reachable -- it is NOT a perf/overlap depth.

      transform_in_place ROTATES its CB: it pops one page then reserves one page
      (streaming_reduce_helpers.inl:88-95), so running it `rows` times advances
      cb_row_stat's front by `rows`.  With a ring of exactly BLOCK_ROWS that is
      harmless while rows == BLOCK_ROWS (the advance is a whole revolution, so
      the finalized block lands back on pages 0..BLOCK_ROWS-1, contiguous), but
      the LAST row-block of a core is PARTIAL whenever BLOCK_ROWS does not
      divide its assignment.  Then the advance is `rows mod BLOCK_ROWS != 0`, the
      finalized tiles STRADDLE the ring wrap, and pass B's
      `mul<..., OperandKind::Col>` -- a bulk cb_wait_front(rows) plus LINEAR tile
      indexing off the read pointer -- reads past the end of the ring for every
      index after the wrap.  Symptom: the 2nd..last row of each partial block is
      garbage while every full block is correct; catastrophic (PCC 0.55-0.93),
      not a precision drift, and invisible to Phase 0 because every Phase-0
      golden cell had Rt <= 64 < the 110-core grid, hence BLOCK_ROWS == 1.

      Doubling the ring restores contiguity for ANY rows <= BLOCK_ROWS: a block
      starts with front == 0 (a full block pushes B, rotates B, pops B == 2B == 0
      mod 2B), the rotation leaves the finalized tiles on pages
      [rows, 2*rows) which is within the ring since 2*rows <= 2*BLOCK_ROWS.
      Both L1 solves count this depth through CB_ROW_STAT_DEPTH -- one source of
      truth, so raising it cannot drift from the budget.
  D7  The reduce runs on ReduceAlgorithm::AccumulateViaAdd once
      WT_CHUNK >= REDUCE_ACC_VIA_ADD_MIN_WT, and on the default ReduceTile below
      that.  Refinement 1b's precision lever; also a measured perf win.

      WHY.  ReduceTile is the FPU matmul-with-ones: each input tile's 32-column
      row sum lands in ONE DEST word, so a row of Wt tiles drives WT_CHUNK*32
      all-positive addends through a single accumulator -- 16-bit at
      fp32_dest_acc_en=False.  That is precisely the wide-W error Refinement 1
      diagnosed (reduce output +12.4 % at W=11008, bit-invariant across chunk
      count / REDUCE_BULK / math_fidelity, so unreachable by any chunking knob).
      AccumulateViaAdd instead sums the width tiles ELEMENTWISE into DST with
      pairwise add_tiles and finishes the within-tile 32-column sum on the SFPU
      (fp32 LREGs, one rounding at the store).  DEST-resident accumulation depth
      drops from WT_CHUNK*32 serial adds to WT_CHUNK/2 pairwise ones -- and the
      cross-chunk carry still goes through the fp32 cb_row_stat, so the depth is
      bounded by WT_CHUNK rather than by Wt.

      COUPLED, not a one-word swap.  AccumulateViaAdd + cross-chunk Accumulate is
      BulkWaitBulkPop-only (so it is gated on REDUCE_BULK == 1), and its
      non-tile-aligned mechanism is a 0/1 MASK tile
      (dataflow_kernel_lib::prepare_reduce_mask + ReducePartialScaler::
      partial_mask) instead of ReduceTile's [full, partial] SCALER pair -- hence
      `scaler_tiles`, which the reader fills to and the compute pops.  Both
      mechanisms zero the pad lanes by an exact multiply-by-0, so the reader's
      pad-lane invariant is unchanged.

      THRESHOLD, not unconditional: AccumulateViaAdd is a LOSS at 1-2 reduce-dim
      tiles (0.67x / 0.94x) and a win from 4 up (1.40x .. 5.35x at 32) --
      examples/reduce_block/report_reduced_sweep.md, dim=row.  Narrow rows also
      have no precision problem to fix (Wt=1 is 32 addends).  So the knob is a
      crossover, and BOTH datapaths stay live and covered: the pad-poison shapes
      alone span Wt = 2, 3 (ReduceTile) and 5, 7 (AccumulateViaAdd).

      MEASURED on the WHOLE OP (blackhole p150b, 110-core grid, ~1.35 GHz,
      bf16 + TILE gamma + HiFi2 + fp32_dest_acc_en=False -- the `_perf_case`
      config; one fresh-cache profiled run per variant, A/B by flipping this
      knob between 4 and 10**9;
      test_rms_norm_perf.py::test_rms_norm_perf_reduce_datapath):

        shape                  Wt    ReduceTile   AccViaAdd   speedup
        (1,1,32,7168)         224      44690 ns    42253 ns    1.06x
        (1,1,224,3072)         96      23758 ns    22544 ns    1.05x
        (1,1,32,1024)          32      11132 ns    10881 ns    1.02x
        (1,1,8192,5120)       160     754579 ns   752410 ns    1.00x

      So the datapath is a small, uniform win here -- NOT the 2.87-5.35x the
      isolated reduce bake-off shows, and that gap is the finding: rms_norm is
      dataflow-bound at these widths (x is read twice in STREAM, and pass B plus
      the writer move the same bytes again), so shaving reduce MATH cycles moves
      the total by only a few percent.  A perf phase that budgets against the
      reduce-block micro-benchmark will over-predict; the levers with headroom
      are the byte-count / occupancy ones (Lamp L1 / L5), not this one.
      Precision, not speed, is why the knob ships.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

# ---------------------------------------------------------------------------
# Primary knobs — the only hand-set numbers in this op.
# ---------------------------------------------------------------------------

# Fraction of usable per-core L1 the CBs may occupy.  Lower it if CB-OOM shows
# up.  Everything about block size derives from the resulting byte budget.
L1_SAFETY_FRACTION = 0.85

# Depth of the ROW_MAJOR stick staging CBs (reader <-> tilize overlap).
CB_RM_STAGE_DEPTH = 2

# Ordered depth candidates for the two cross-processor CBs (cb_input_tiles,
# cb_output_tiles), COARSEST FIRST.  The regime search (D4) walks them and takes
# the RESIDENT regime at the first depth whose whole-row working set fits L1.
#
# Parked at the single value 2 -- byte-identical to the design's fixed-depth
# predicate -- because (2, 1) was MEASURED to be a net loss today; see D4 for
# the numbers and for the complementary step (Lamp L5 / L1) that would make a
# shallower depth worth offering.  This stays a live knob: appending 1 is the
# one-line change a later refinement flips once that step lands.
CB_DEPTH_CANDIDATES = (2,)

# Cores along the `width` axis.  Trivial value 1 in Phase 0 = one core owns the
# whole width of every row it owns; Lamp L1 turns this knob up and adds the
# cross-core partial-sum combine.
GRID_W = 1

# reduce() input policy knob: 1 = BulkWaitBulkPop (bulk wait/indexed/bulk pop),
# 0 = WaitAndPopPerTile.  Bulk is the coarse default (op_design.md section 1.4).
REDUCE_BULK = 1

# Reduce-datapath crossover knob (D7): the smallest WT_CHUNK (reduce-dim tiles
# per reduce() call) at which ReduceAlgorithm::AccumulateViaAdd is preferred over
# the default ReduceTile.  4 is the MEASURED crossover for REDUCE_ROW on this
# helper (ttnn/ttnn/operations/examples/reduce_block/report_reduced_sweep.md:
# R=1 0.67x, R=2 0.94x, R=4 1.40x, R=8 2.21x, R=16 3.54x, R=32 5.35x), and it is
# also where the precision motive starts: AccumulateViaAdd's DEST-resident
# accumulation depth is WT_CHUNK/2 pairwise adds instead of ReduceTile's
# WT_CHUNK*32 serial ones.  Raise it to 10**9 to pin the op back to ReduceTile
# everywhere; lower it to 1 to force AccumulateViaAdd everywhere.
REDUCE_ACC_VIA_ADD_MIN_WT = 4

# Ring depth of cb_row_stat, in units of BLOCK_ROWS.  MUST be >= 2 -- this is a
# correctness constant, not a perf knob.  See D6.
CB_ROW_STAT_DEPTH = 2


# ---------------------------------------------------------------------------
# Small host helpers (ttnn exposes no div_up / round_up binding).
# ---------------------------------------------------------------------------


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def _prod(xs) -> int:
    n = 1
    for x in xs:
        n *= x
    return n


def _largest_divisor_at_most(n: int, cap: int) -> int:
    """Coarsest d with n % d == 0 and d <= cap (D1)."""
    cap = max(1, min(cap, n))
    for d in range(cap, 0, -1):
        if n % d == 0:
            return d
    return 1


def _cb_block_mult(depth_x: int, depth_out: int, has_gamma: bool) -> int:
    """Tiles-per-block-tile summed over the BLOCK-SCOPED CBs (op_design.md 1.4).

    ONE source of truth for "which CBs scale with BLOCK_ROWS * WT_CHUNK, and at
    what depth":

        cb_input_tiles (depth_x) + cb_x_squared (1)
        + cb_normalized (1, gamma only) + cb_output_tiles (depth_out)

    Both the L1 fit predicate and the STREAM chunk-size solve call this, so a new
    block-scoped CB (or a depth change) is a one-line edit that cannot drift
    between the two solves.
    """
    return depth_x + 1 + (1 if has_gamma else 0) + depth_out


def _f32_bits(v: float) -> int:
    return struct.unpack("I", struct.pack("f", float(v)))[0]


# Block-float dtypes: 16 data values share one 8-bit exponent, so there is no
# such thing as "bytes per element" for them.
BLOCK_FLOAT_DTYPES = (ttnn.bfloat8_b, ttnn.bfloat4_b)


def _stick_elem_bytes(tensor) -> int:
    """Bytes per element, for the ROW_MAJOR stick byte math ONLY (D5).

    `Tensor.element_size()` raises "datum for bfp2, bfp4, bfp8 is invalid" on a
    block-float dtype, and rightly so.  The number is only ever consumed by the
    ROW_MAJOR stick path (CHUNK_ROW_BYTES in the reader / writer), and a
    block-float tensor cannot be ROW_MAJOR -- it has no sticks, only exponent
    blocks -- so the CT arg is dead on exactly the dtypes that cannot answer.
    Report 0 there rather than teaching every caller the special case.
    """
    if tensor is None:
        return 0
    if tensor.dtype in BLOCK_FLOAT_DTYPES:
        assert tensor.layout == ttnn.TILE_LAYOUT, (
            f"rms_norm: {tensor.dtype} is a block-float format and cannot be ROW_MAJOR " f"(got layout {tensor.layout})"
        )
        return 0
    return tensor.element_size()


# ---------------------------------------------------------------------------
# Circular-buffer slots (semantic names; the number is just the slot).
# ---------------------------------------------------------------------------

CB_INPUT_STICKS = 0  # ROW_MAJOR only: padded row-major staging of x
CB_INPUT_TILES = 1  # x tiles (reader for TILE, tilize for ROW_MAJOR)
CB_X_SQUARED = 2  # x^2 tiles, pass A
CB_SCALER = 3  # reduce scaler (bf16, value 1.0) + partial scaler
CB_ROW_STAT = 4  # fp32: sum(x^2) accumulator -> in-place 1/rms
CB_GAMMA_STICKS = 5  # ROW_MAJOR gamma only
CB_GAMMA_TILES = 6  # gamma tiles (row 0 valid)
CB_NORMALIZED = 7  # x * (1/rms), only when gamma is present
CB_OUTPUT_TILES = 8  # output tiles
CB_OUTPUT_STICKS = 9  # ROW_MAJOR only: untilized row-major staging of out


def _core_range_set_full_grid(device):
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])


def _cores_in(core_range_set):
    # row_wise=True must match split_work_to_cores' row_wise=True, or the
    # (row_start, row_count) prefix sum below would be assigned to the wrong
    # cores.  Never swallow a failure here: an empty list would surface as a
    # confusing work-split assertion instead of the real error.
    return list(ttnn.corerange_to_cores(core_range_set, None, True))


def _cb(index, page_size, num_pages, data_format, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def create_program_descriptor(
    input_tensor: "ttnn.Tensor",
    output_tensor: "ttnn.Tensor",
    *,
    gamma: "ttnn.Tensor" = None,
    epsilon: float = 1e-6,
    compute_kernel_config: "ttnn.ComputeConfigDescriptor" = None,
) -> "ttnn.ProgramDescriptor":
    # GRID_W is a live knob, but turning it past 1 is a SCHEME-CHANGE (Lamp L1:
    # cross-core partial-sum combine + mcast), not just a wider grid -- the
    # dependent `width` axis cannot be split without it.  Fail loudly rather than
    # silently computing every core's slice as if it were the whole row.
    if GRID_W != 1:
        raise NotImplementedError(
            f"rms_norm: GRID_W={GRID_W} requires the cross-core partial-sum combine "
            f"(op_design.md Lamp L1); Phase 0 only implements GRID_W == 1"
        )

    device = input_tensor.device()
    shape = list(input_tensor.shape)

    is_tile = input_tensor.layout == ttnn.TILE_LAYOUT
    has_gamma = gamma is not None
    gamma_is_rm = has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT

    # ---- geometry (alignment-aware: ceil everywhere, per image) ------------
    W = shape[-1]
    Wt = _div_up(W, TILE_DIM)
    partial_w = W % TILE_DIM  # 0 => tile-aligned width

    if is_tile:
        # Every (..., H, W) image is tile-padded independently.
        Rt = _prod(shape[:-2]) * _div_up(shape[-2], TILE_DIM)
        R_rm = 0  # unused
    else:
        # ROW_MAJOR has no implicit H padding: all images' rows are contiguous.
        R_rm = _prod(shape[:-1])
        Rt = _div_up(R_rm, TILE_DIM)

    elem_bytes = _stick_elem_bytes(input_tensor)
    gamma_elem_bytes = _stick_elem_bytes(gamma) if has_gamma else 0

    # ---- L1 byte budget (derived from the device, never a literal) --------
    l1_budget = int(ttnn.get_max_worker_l1_unreserved_size() * L1_SAFETY_FRACTION)

    bt = ttnn.tile_size(input_tensor.dtype)  # x / x^2 / normalized / out
    gt = ttnn.tile_size(gamma.dtype) if has_gamma else 0
    st = ttnn.tile_size(ttnn.bfloat16)  # scaler CB (R4: value exactly 1.0)
    ft = ttnn.tile_size(ttnn.float32)  # cb_row_stat

    # Scaler CB page count: one source of truth for the budget term, the CB
    # allocation and (via PARTIAL_W) the compute kernel's final pop.
    scaler_pages = 2 if partial_w else 1
    scaler_bytes = st * scaler_pages

    # Depth > 1 only buys overlap when the producer/consumer pair spans two
    # processors.  On the ROW_MAJOR path cb_input_tiles is produced by the
    # `tilize` compute helper and cb_output_tiles is consumed by `untilize`;
    # sequential compute helpers own all three TRISCs and cannot pipeline, so
    # those CBs drop to depth 1 (and must instead hold a whole block — R5), and
    # the depth search below collapses to its single trivial candidate.
    depth_candidates = CB_DEPTH_CANDIDATES if is_tile else (1,)

    def _resident_fit(depth):
        """(block_rows_l1_max, per-block CB multiplier) for a candidate depth."""
        mult = _cb_block_mult(depth, depth, has_gamma)
        fixed = (
            (Wt * gt)  # cb_gamma_tiles
            + (Wt * gt if gamma_is_rm else 0)  # cb_gamma_sticks
            + (2 * CB_RM_STAGE_DEPTH * Wt * bt if not is_tile else 0)  # stick staging
            + scaler_bytes
        )
        # + cb_row_stat's pages per tile-row (CB_ROW_STAT_DEPTH of them -- D6)
        per_tilerow = Wt * bt * mult + CB_ROW_STAT_DEPTH * ft
        return max(0, (l1_budget - fixed) // per_tilerow), mult

    # ---- cross-core split of the independent `row` axis: SIZE and COUNT ----
    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        rows_per_core_g1,
        rows_per_core_g2,
    ) = ttnn.split_work_to_cores(
        _core_range_set_full_grid(device), Rt, True
    )  # row_wise=True is mandatory
    max_rows_per_core = max(rows_per_core_g1, rows_per_core_g2)

    # ---- regime selection (op_design.md section 4.2, extended by D4) -------
    # Walk the depth candidates coarsest-first and take RESIDENT at the first
    # depth whose whole-row working set fits.  Only if no depth fits do we drop
    # to STREAM, which is the genuinely expensive fallback (x re-read in pass B
    # => ~2x the DRAM bytes).  Still a pure function of
    # (layout, dtypes, gamma format, Wt, budget) — device-independent, so the
    # regime-pinned tests stay reproducible.
    resident_depth = None
    for depth in depth_candidates:
        brmax, mult = _resident_fit(depth)
        if brmax >= 1:
            resident_depth, block_rows_l1_max, cb_block_mult = depth, brmax, mult
            break

    if resident_depth is not None:
        # RESIDENT: the whole row is resident; take the coarsest row block that
        # fits, i.e. the entire per-core assignment whenever it fits.
        cb_x_depth = cb_out_depth = resident_depth
        block_rows = min(max_rows_per_core, block_rows_l1_max)
        wt_chunk = Wt
        num_w_chunks = 1
    else:
        # STREAM: one tile-row's width does not fit at ANY depth -> chunk
        # `width` (an L1 fallback, not a parallelization).  The chunk size
        # adapts to L1, so keep the preferred (coarsest) depth here: overlap is
        # affordable and the byte count is already paid.
        cb_x_depth = cb_out_depth = depth_candidates[0]
        cb_block_mult = _cb_block_mult(cb_x_depth, cb_out_depth, has_gamma)
        block_rows = 1
        per_chunk_tile_bytes = (
            bt * cb_block_mult
            + (gt * (2 if gamma_is_rm else 1) if has_gamma else 0)
            + (2 * CB_RM_STAGE_DEPTH * bt if not is_tile else 0)  # D2
        )
        fixed_stream = scaler_bytes + CB_ROW_STAT_DEPTH * ft  # block_rows == 1 here
        wt_chunk_l1_max = max(1, (l1_budget - fixed_stream) // per_chunk_tile_bytes)
        wt_chunk = _largest_divisor_at_most(Wt, wt_chunk_l1_max)  # D1
        num_w_chunks = Wt // wt_chunk

    # X_RESIDENT == GAMMA_RESIDENT == (NUM_W_CHUNKS == 1) is derived in the
    # kernels from the NUM_W_CHUNKS CT arg -- one source of truth, so it is
    # deliberately NOT passed as a second flag.

    # ---- reduce datapath (D7) ---------------------------------------------
    # WT_CHUNK is the reduce-dim tile count of ONE reduce() call, so it -- not Wt
    # -- is what the crossover is measured against.  AccumulateViaAdd's cross-chunk
    # Accumulate indexes a resident block, hence BulkWaitBulkPop only: the two
    # knobs are coupled here (one place), and the compute kernel static_asserts it.
    reduce_acc_via_add = REDUCE_BULK == 1 and wt_chunk >= REDUCE_ACC_VIA_ADD_MIN_WT
    # Tiles the reader actually pushes into cb_scaler, and the compute pops.
    # AccumulateViaAdd takes ONE: the 0/1 mask (partial W) or an unused 1.0 scaler.
    # ReduceTile takes the [full, partial] pair when W is not tile-aligned.
    scaler_tiles = 1 if reduce_acc_via_add else scaler_pages
    assert scaler_tiles <= scaler_pages, "rms_norm: cb_scaler is sized below the tiles the reader pushes"

    # ---- per-core assignment: (row_start, row_count) prefix sum -----------
    cores_g1 = _cores_in(core_group_1)
    cores_g2 = _cores_in(core_group_2)
    assignment = []  # [(core, row_start, row_count)]
    row_cursor = 0
    for core in cores_g1:
        assignment.append((core, row_cursor, rows_per_core_g1))
        row_cursor += rows_per_core_g1
    for core in cores_g2:
        assignment.append((core, row_cursor, rows_per_core_g2))
        row_cursor += rows_per_core_g2
    assert row_cursor == Rt, f"rms_norm: work split covers {row_cursor} of {Rt} tile-rows"
    assert len(assignment) == num_cores, f"rms_norm: {len(assignment)} cores assigned, expected {num_cores}"

    # ---- circular buffers -------------------------------------------------
    # Every page count below is a function of the block/depth knobs only.
    # cb_gamma_* is the one place Wt appears; it *is* the gamma tensor's extent
    # and is bounded by the same L1 predicate through `fixed_resident`
    # (collapsing to WT_CHUNK in the STREAM regime).
    cbs = []
    if not is_tile:
        cbs.append(_cb(CB_INPUT_STICKS, bt, CB_RM_STAGE_DEPTH * wt_chunk, input_tensor.dtype, all_cores))
        cbs.append(_cb(CB_OUTPUT_STICKS, bt, CB_RM_STAGE_DEPTH * wt_chunk, output_tensor.dtype, all_cores))
    cbs.append(_cb(CB_INPUT_TILES, bt, cb_x_depth * block_rows * wt_chunk, input_tensor.dtype, all_cores))
    cbs.append(_cb(CB_X_SQUARED, bt, block_rows * wt_chunk, input_tensor.dtype, all_cores))
    cbs.append(_cb(CB_SCALER, st, scaler_pages, ttnn.bfloat16, all_cores))
    # D6: CB_ROW_STAT_DEPTH * block_rows, so transform_in_place's rotation leaves
    # a PARTIAL final block's stat tiles contiguous for pass B's indexed read.
    cbs.append(_cb(CB_ROW_STAT, ft, CB_ROW_STAT_DEPTH * block_rows, ttnn.float32, all_cores))
    if has_gamma:
        if gamma_is_rm:
            cbs.append(_cb(CB_GAMMA_STICKS, gt, wt_chunk, gamma.dtype, all_cores))
        cbs.append(_cb(CB_GAMMA_TILES, gt, wt_chunk, gamma.dtype, all_cores))
        cbs.append(_cb(CB_NORMALIZED, bt, block_rows * wt_chunk, input_tensor.dtype, all_cores))
    cbs.append(_cb(CB_OUTPUT_TILES, bt, cb_out_depth * block_rows * wt_chunk, output_tensor.dtype, all_cores))

    # ---- reader -----------------------------------------------------------
    reader_ct_args = [
        1 if is_tile else 0,  # 0  IS_TILE
        Wt,  # 1  WT
        wt_chunk,  # 2  WT_CHUNK
        num_w_chunks,  # 3  NUM_W_CHUNKS
        block_rows,  # 4  BLOCK_ROWS
        partial_w,  # 5  PARTIAL_W (0 => aligned)
        1 if has_gamma else 0,  # 6  HAS_GAMMA
        1 if gamma_is_rm else 0,  # 7  GAMMA_IS_RM
        elem_bytes,  # 8  input element bytes
        gamma_elem_bytes,  # 9  gamma element bytes
        R_rm,  # 10 total ROW_MAJOR sticks (0 for TILE)
        W,  # 11 logical width (elements)
        1 if reduce_acc_via_add else 0,  # 12 REDUCE_ACC_VIA_ADD (picks mask vs scaler pair)
    ]
    # The kernel reads its accessor args at TensorAccessorArgs<N>() -- N must equal
    # the scalar CT-arg count above.  Assert it here so adding a scalar arg fails
    # in Python instead of mis-parsing on device.
    n_reader_scalars = len(reader_ct_args)
    assert n_reader_scalars == 13, "rms_norm_reader.cpp expects TensorAccessorArgs<13>()"
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    # ---- writer -----------------------------------------------------------
    writer_ct_args = [
        1 if is_tile else 0,  # 0  IS_TILE
        Wt,  # 1  WT
        wt_chunk,  # 2  WT_CHUNK
        num_w_chunks,  # 3  NUM_W_CHUNKS
        block_rows,  # 4  BLOCK_ROWS
        elem_bytes,  # 5  output element bytes
        R_rm,  # 6  total ROW_MAJOR sticks (0 for TILE)
        W,  # 7  logical width (elements)
    ]
    assert len(writer_ct_args) == 8, "rms_norm_writer.cpp expects TensorAccessorArgs<8>()"
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # ---- compute ----------------------------------------------------------
    compute_ct_args = [
        1 if is_tile else 0,  # 0  IS_TILE
        wt_chunk,  # 1  WT_CHUNK
        num_w_chunks,  # 2  NUM_W_CHUNKS
        block_rows,  # 3  BLOCK_ROWS
        partial_w,  # 4  PARTIAL_W
        1 if has_gamma else 0,  # 5  HAS_GAMMA
        1 if gamma_is_rm else 0,  # 6  GAMMA_IS_RM
        _f32_bits(1.0 / float(W)),  # 7  INV_W (raw fp32 bits) -- R1/R4: logical W
        _f32_bits(epsilon),  # 8  EPS (raw fp32 bits)
        REDUCE_BULK,  # 9  reduce input policy knob
        1 if reduce_acc_via_add else 0,  # 10 REDUCE_ACC_VIA_ADD (reduce datapath, D7)
        scaler_tiles,  # 11 tiles the reader pushed into cb_scaler
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    x_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    g_addr = gamma.buffer_address() if has_gamma else 0
    for core, row_start, row_count in assignment:
        reader_rt[core.x][core.y] = [x_addr, g_addr, row_start, row_count]
        writer_rt[core.x][core.y] = [out_addr, row_start, row_count]
        compute_rt[core.x][core.y] = [row_count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),  # reads on NoC0
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),  # writes on NoC1
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=compute_kernel_config,  # passed through unmodified
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
