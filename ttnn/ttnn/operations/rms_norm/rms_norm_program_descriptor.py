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
    GRID_W                  cores along `width` (Lamp L1).  0 = AUTO (the
                            policy below); >= 1 forces the group size (1 = off)
    WIDTH_SPLIT_MIN_WT_PER_CORE
    WIDTH_SPLIT_MAX_GROUP_CORES
    WIDTH_SPLIT_MIN_GAIN    the three AUTO-policy knobs (see D11)
    (no new knob in Refinement 2b -- the BAND scheme's extents are all read off
     the shard spec; see D10)
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
  D8  Refinement 2 refines D7's crossover INPUT: it is now measured against this
      core's WHOLE reduce dim (`wt_per_core`) rather than against WT_CHUNK.  In
      the RESIDENT regime the two are equal, so this is byte-identical to
      Refinement 1b; they differ only when L1 forces a chunked width.

      WHY.  D7's threshold of 4 came from a PERF bake-off measured per reduce()
      call, but the precision motive scales with the TOTAL reduce dim: Refinement
      1 established the error is bit-invariant across chunk count (it is the FPU
      matmul reduce's within-tile 32-column sum, which chunking cannot reach).
      Refinement 2 made that gap reachable -- a HEIGHT-sharded (1,1,160,11008)
      holds a 344-tile shard in L1, which squeezed WT_CHUNK to 2, dropped below
      the threshold, and brought rms 0.127 back on exactly the cells 1b closed.
      Gating on the total keeps AccumulateViaAdd wherever the row is wide enough
      to need it, at the cost of running it at 1-2 tiles per call (0.67x-0.94x on
      the reduce step alone) for those L1-squeezed builds -- a trade this op
      should always take, since rms_norm is dataflow-bound at those widths (see
      D7's whole-op A/B: the datapath is worth 1.00x-1.06x either way).

      Narrow rows are unaffected and BOTH datapaths stay covered: the pad-poison
      shapes span Wt = 2, 3 (partial-SCALER pair) and 5, 7 (0/1 MASK tile), and
      those are RESIDENT, so wt_per_core == WT_CHUNK == Wt there.
  D10 Refinement 2b's ROW_MAJOR BAND scheme (_plan_band).  An RM shard that cuts
      the WIDTH axis has a sub-tile edge -- `eval.sharding` rounds it to
      (1 stick x L1_align/elem_size elements), 8 for bf16 and 4 for fp32 -- so its
      PAGE is a row SEGMENT and no core holds a whole width TILE.  Refinement 2
      read that as structural and excluded the two cells.  It is not: the
      section-3.4 combine sums the group's per-row PARTIALS **elementwise**, so a
      partial may cover ANY contiguous element range of the row.  Each core
      therefore reduces the BAND it already holds, staged out of its OWN L1, and
      the combine, the compute kernel and every knob are untouched.

      Two things are NOT free, and both are why this is a descriptor note:

        * THE STAGING FRAME IS THE TENSOR'S GLOBAL TILE GRID, not the band's own
          byte offset: the band's first element is placed at lane
          (w_off_elems % 32) of the staged stick.  A DRAM read whose SOURCE offset
          is not 64-byte aligned is silently TRUNCATED down to the alignment --
          measured, with an 8-element shard: bands 1, 2 and 3 all received
          gamma[0..8), a whole-tensor PCC of 0.32 that a spot check of band 0 sees
          as perfect.  Staging in the global tile frame keeps every gamma fetch on
          a tile column (a multiple of 64 bytes for every dtype, x's and gamma's
          independently), which is why gamma works at BOTH layouts here and why
          this refinement added no EXCLUSIONS.  The shard granule is itself a
          multiple of L1_align/elem_size, so the shifted L1 destination stays
          16-byte aligned, matching the local source.
        * PARTIAL_W IS PASSED TO THE KERNELS AS 0 (`kernel_partial_w`).  A band
          boundary is per-core and cannot be one program-wide PARTIAL_W, and it
          does not need to be: the staging ring is zeroed once at boot and only
          the band's own bytes are ever written into it, so every lane outside
          [delta, delta + band) contributes an exact 0 to sum(x^2).  Zero-staging
          REPLACES the reduce mask on this path; the finalize's 1/W is the LOGICAL
          width as always.  STAGE_ZERO (which used to be spelled PARTIAL_W != 0 in
          the reader) is now the explicit "some staged stick is narrower than the
          ring's padded row" flag that both cases share.

      WT_CHUNK is the WIDEST global tile span any core's band touches (it is a
      compile-time template on tilize / untilize, so it must cover every core); a
      core whose band spans fewer tiles stages an all-zero pad tile column, which
      the reduce adds 0 for and the writer never writes back.

      MEASURED (blackhole p150b, 110-core grid, ~1.35 GHz, bf16 / HiFi2 /
      fp32_dest_acc_en=False, one fresh-cache profiled run per variant,
      test_rms_norm_perf.py::test_rms_norm_perf_row_major_band):

        (1,1,224,3072)  WIDTH   TILE shard 133224 ns   RM BAND 136856 ns  +2.7 %
        (1,1,224,3072)  BLOCK   TILE shard  10373 ns   RM BAND  12509 ns  +21 %
        (1,1,256,512)   WIDTH   TILE shard  29004 ns   RM BAND  96479 ns  +233 %

      The band's OWN overhead is the +2.7 % / +21 %: the sub-tile case stages one
      local read per stick instead of one per tile-row (the reader takes the single
      wide transfer only when the band fills its tile columns and the shard stride
      matches -- true for (1,1,224,3072) WIDTH, whose shard is exactly 32
      elements).  The +233 % is NOT the band: an RM shard's 8-element granule makes
      `auto_shard_config` cut W=512 into 64 slices where the TILE granule cuts it
      into 16, so the same tensor gets a 4x larger combine GROUP -- a placement
      cost the caller chose, and the same cost the TILE path pays at the same group
      size (see the WIDTH row, where both are 96 cores and the two agree to 3 %).
  D11 Refinement 3 (perf) turns GRID_W -- Lamp L1, the cross-core width split on
      an INTERLEAVED input -- from its parked 1 to an AUTO policy
      (_auto_width_split).  No new dataflow: the combine, both kernels and every
      other knob are Refinement 2's, unchanged; what is new is (a) the policy that
      decides (gw, gh) from the shape and the live grid, and (b) a PACKED
      single-group topology for a group wider than one grid row (Mcast2D over its
      bounding box, in-box/out-of-group cores INACTIVE -- the same shape a
      row-major-packed WIDTH shard grid already had).

      WHY.  The row split can only ever use min(Rt, num_cores) cores, so a decode
      profile (Rt = 1) runs an arbitrarily wide tensor through ONE core: measured
      41779 ns on (1,1,32,7168), i.e. 1.34 MB at ~32 GB/s, which is one core's NoC
      and nothing else.  Splitting `width` is the only axis left to parallelize.

      MEASURED (blackhole p150b, 110-core 11x10 grid, ~1.35 GHz, bf16 / TILE /
      HiFi2 / fp32_dest_acc_en=False -- the `_perf_case` config; one fresh-cache
      profiled run per variant; test_rms_norm_perf.py::
      test_rms_norm_perf_width_split, A/B by the GRID_W override):

        group size (gw)        1        8       16       32       56
        (1,1,32,7168)      41779    13926    12876    14224    19338 ns
        (1,1,32,1024)      11207     7149*   8305                     ns
          * gw = 4 -> 7296 ns, so the narrow control's optimum is 8.

      Two opposing terms set the optimum, which is why the ceiling knob exists:
      per-core BYTES fall as 1/gw, while the root's GATHER cost rises with gw --
      every member ships a full fp32 TILE (4096 B) per row-block into the root's
      cb_partials_gathered, and that CB's L1 footprint is GROUP_SIZE * BLOCK_ROWS
      pages.  Past ~16 members the gather is the whole story (56 members is
      1.5x SLOWER than 16).  Hence WIDTH_SPLIT_MAX_GROUP_CORES = 16 and
      WIDTH_SPLIT_MIN_WT_PER_CORE = 4 (below 4 tiles per core the bytes saved no
      longer pay for a member; measured on the narrow control).

      Whole-op, at the shipped AUTO policy:

        shape             row split   width split   speedup   cores
        (1,1,32,7168)       41779        12876       3.24x     1 -> 16
        (1,1,32,8192)       47216        13644       3.46x     1 -> 16
        (1,1,32,5120)       32699        11181       2.92x     1 -> 16
        (1,1,32,4096)       26925        10415       2.59x     1 -> 16
        (1,1,32,2304)       20932         8620       2.43x     1 -> 12
        (1,1,32,1024)       11207         7223       1.55x     1 ->  8
        (1,1,128,4096)      27074        15763       1.72x     4 -> 32
        (1,1,224,3072)      22863        17770       1.29x     7 -> 56
        (1,1,224,1000)      12113         9427       1.29x     7 -> 56
        (1,1,512,4096)      45793        38224       1.20x    16 -> 80
        (1024,1024)         20960        20960       1.00x    32 (NO split)
        (1,1,2048,256)      10944        10944       1.00x    64 (NO split)
        (1,1,8192,1024)    105435       105435       1.00x   110 (NO split)

      WIDTH_SPLIT_MIN_GAIN = 4 is what keeps the last three byte-identical, and it
      is not cosmetic: at MIN_GAIN = 2 the (1024,1024) case DID split (32 -> 80
      cores) and measured 21560 -> 23315 ns, a 0.92x REGRESSION -- 2.5x more cores
      cannot pay for a combine round when the row split already has 32 cores well
      fed.  Every shape that splits under MIN_GAIN = 4 has >= 4x more cores at
      work and every one of them measured faster.

      PRECISION IS SAFE BY CONSTRUCTION, and that is worth spelling out because
      D8 records the trap: a smaller per-core reduce dim can switch Refinement
      1b's AccumulateViaAdd fix OFF (a resident shard did exactly that).  Here it
      cannot -- gw <= Wt // WIDTH_SPLIT_MIN_WT_PER_CORE means wt_per_core >= 4 ==
      REDUCE_ACC_VIA_ADD_MIN_WT, so every split build keeps the datapath 1b
      needed.  Measured on the shapes that split: rel RMS 0.0087 on (1,1,32,7168)
      and 0.0102 on (1,1,32,16384) at bf16 / fp32_dest_acc_en=False (gate 0.04),
      i.e. unchanged from 1b's single-core numbers -- and the cross-core sum is
      itself an fp32 elementwise add, so a split row accumulates LESS DEST-resident
      depth than an unsplit one, never more.

      GRANULARITY.  gw is clamped to a DIVISOR of Wt (_width_group_cores), so a
      prime Wt does not split at all -- the same D1 limit the STREAM chunk lives
      under, and for the same reason on this path: an interleaved core has no pad
      storage, so a ragged tail would make the reader read x tiles it does not own
      (the NATIVE_IN path can zero a resident shard's pad tiles instead, which is
      why the SHARDED width schemes DO take a ragged tail).

      REMAINING HEADROOM, measured rather than guessed.  A one-core minimal
      program is 3456 ns of fixed launch/dispatch floor, and (1,1,32,7168) at
      gw = 16 moves only 56 kB per core (~1.8 us at the measured 32 GB/s), so the
      ~7 us balance of its 12876 ns is the COMBINE round trip: gather (16 x 4 kB
      into one root), the root's sum + finalize, and the stat multicast, none of
      which overlaps anything when Rt = 1 gives a core a single row-block.  The two
      levers that follow from that -- (a) a hierarchical/two-stage gather, which
      examples/tensix_all_reduce measures at 1.45-1.60x over a flat root on 2-D
      groups and which would also RAISE the useful group ceiling, and (b) a
      compact partial handoff (a member's REDUCE_ROW partial is a column vector:
      32 floats carried in a 4096-byte tile) -- are both changes to the combine's
      topology / data format, not knob turns, so they are recorded here rather
      than half-built.
  D9  Refinement 2's placement layer.  The three SCHEME_* values, the zero-copy
      (shard-backed) cb_input_tiles / cb_output_tiles, the cross-core width
      combine's four CBs and L1_CB_ARENA_BASE_RESERVE are all documented at
      their definitions above; the one thing worth stating here is what stayed
      the same.  The COMPUTE kernel's phase sequence, every helper call, the
      block/depth knob set and the L1 predicate are unchanged -- a sharded build
      differs only in (a) who fills cb_input_tiles, (b) which CB pass B reads the
      stat from, and (c) whether the finalize runs locally or on a group root.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import NamedTuple

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

# ---------------------------------------------------------------------------
# Primary knobs — the only hand-set numbers in this op.
# ---------------------------------------------------------------------------

# Fraction of usable per-core L1 the CBs may occupy.  Lower it if CB-OOM shows
# up.  Everything about block size derives from the resulting byte budget.
L1_SAFETY_FRACTION = 0.85

# Bytes between the worker-L1 UNRESERVED base and the first circular-buffer
# address (kernel binaries / launch messages / semaphores live in between).
# get_max_worker_l1_unreserved_size() does NOT subtract it, and metal's own check
# is absolute: `static circular buffer region ends` must stay below the LOWEST L1
# buffer address.  Without an L1-resident tensor there is no such buffer and the
# term is irrelevant (the CB arena may run to the top of L1) -- which is why the
# solve only subtracts it when a shard is resident, keeping every interleaved
# build BYTE-IDENTICAL to Phase 0.  With one, L1_SAFETY_FRACTION alone is a
# PROPORTIONAL margin and cannot cover a fixed offset: a shard pair holding
# ~1.38 MB of the 1.53 MB leaves 52 kB of real headroom while 0.85 of the
# nominal remainder claims 105 kB, and the program then fails to launch.
#
# MEASURED on blackhole p150b (from the clash message's own numbers: buffer at
# 163840 with the CB region ending at 179072 for a 67584-byte CB set, so the
# arena base is 111488 while the unreserved base is 40832).  Over-reserving here
# only costs a little block size; under-reserving is a hard launch failure, so
# round up rather than down on a new part.
L1_CB_ARENA_BASE_RESERVE = 70656

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

# Cores along the `width` axis (Lamp L1: the cross-core width split on an
# INTERLEAVED input).  Phase 0 pinned this at the trivial 1 (one core owns the
# whole width of every row it owns); Refinement 2 built the combine the knob
# needs; Refinement 3 (D11) turns it up.
#
#   0   AUTO -- _auto_width_split() picks (cores per width group, row groups)
#        from the shape and the live grid, so a shape whose row split leaves the
#        grid idle gets the width split and one that already fills it does not.
#   >=1 OVERRIDE -- force this many cores per width group (1 == no split at all,
#        byte-identical to Phase 0).  This is the A/B handle the perf probes use.
GRID_W = 0

# AUTO-policy knobs (all three feed _auto_width_split; none is inlined).
#
# Smallest number of width TILES a core may be left with.  The width split trades
# per-core bytes (which fall as 1/gw) against the combine's cost (which RISES with
# gw: every member ships a full fp32 tile per row-block into the root's gather CB,
# and the gather CB's L1 footprint is GROUP_SIZE * BLOCK_ROWS pages).  Splitting
# past a few tiles per core buys almost no bytes and pays the whole handshake.
#
# MEASURED on (1,1,32,7168) / bf16 / HiFi2 / fp32_dest_acc_en=False -- see D11.
WIDTH_SPLIT_MIN_WT_PER_CORE = 4
# Hard ceiling on cores per width group, i.e. on the gather fan-in.  See D11 for
# the sweep: the curve is flat-to-worse past ~32 because the root's serialized
# gather grows linearly while the per-core byte count is already small.
WIDTH_SPLIT_MAX_GROUP_CORES = 16
# Only take the split when it puts at least this many times as many cores to work
# as the plain row split would.  A split that merely re-arranges the same core
# count adds the combine for nothing, so 2 keeps grid-filling shapes (prefill:
# Rt >= num_cores) on the untouched Phase-0 path.
WIDTH_SPLIT_MIN_GAIN = 4

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

# Per-CALL floor on the same datapath (D8).  REDUCE_ACC_VIA_ADD_MIN_WT is measured
# against the core's whole reduce dim, but AccumulateViaAdd degenerates at ONE tile
# per call: its within-tile SFPU finalize runs on the last chunk only
# (Accumulate::at_last), so a 1-tile chunk carries the running total through the
# cross-chunk reload for every one of Wt steps.  MEASURED at HiFi4 on the prime-Wt
# shapes D1 collapses to WT_CHUNK == 1 (one fresh run per variant, rel RMS):
#
#   shape             fp32_dest_acc_en   AccViaAdd   ReduceTile
#   (1,1,32,4064) RM        True           0.0609      0.0103   <- 6x WORSE
#   (1,1,32,4064) RM        False          0.0039      0.0263
#   (1,1,32,2848) RM        True           0.0455      0.0079   <- 6x WORSE
#   (1,1,32,2848) RM        False          0.0037      0.0174
#
# So the datapath needs BOTH a wide total (the error it fixes) and a chunk with
# something to pair up (2 tiles is the smallest that has). Raise to 10**9 to pin
# the op back to ReduceTile everywhere.
REDUCE_ACC_VIA_ADD_MIN_CHUNK_WT = 2

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
# CT-arg indices the acceptance tests assert on -- named HERE so the index has one
# source of truth (the lists below assert against these, so a reordering fails in
# Python rather than silently re-pointing a test).
READER_CT_BAND = 15
WRITER_CT_OUT_SHARD_ROW_BYTES = 14

# --- Refinement 2, WIDTH/BLOCK sharded only (cross-core width combine) -------
CB_SUM_HANDOFF = 10  # fp32: this core's raw partial sum(x^2), compute -> writer
CB_PARTIALS_GATHERED = 11  # fp32: root's per-sender landing slots, writer -> compute
CB_STAT_HANDOFF = 12  # fp32: root's finalized 1/rms tiles, compute -> writer (mcast src)
CB_ROW_FINAL = 13  # fp32: mcast landing of the finalized 1/rms, writer -> compute


# ---------------------------------------------------------------------------
# Placement (`memory_layout`) — Refinement 2
# ---------------------------------------------------------------------------
#
# op_design.md section 5.3 maps each TARGET memory_layout onto this op's axes:
#
#   INTERLEAVED     -> no placement-imposed split; the op splits `row` itself.
#   HEIGHT_SHARDED  -> cuts the INDEPENDENT `row` axis  => knob-turn (Lamp L3).
#                      Each core already holds whole rows, so the reduction is
#                      LOCAL and the shard IS the per-core block: cb_input_tiles
#                      / cb_output_tiles are backed directly on the shard
#                      (zero-copy, no NoC read for x at all).
#   WIDTH_SHARDED   -> cuts the DEPENDENT `width` axis   => scheme-change (L4/L1).
#   BLOCK_SHARDED      Per-core partial sum(x^2) must be combined across the
#                      group of cores that share a row range: gather to the
#                      group root, finalize there, multicast the stat back
#                      (op_design.md section 3.4).
#
# Three internal SCHEMES realize that mapping.  The scheme is a pure function of
# (layout, placement, shard geometry, L1 budget) -- see _plan_placement.
SCHEME_ROWS = "rows"  # split `row` over the full grid, TensorAccessor dataflow
SCHEME_SHARD_H = "shard_h"  # rows come from the shard, zero-copy CBs, local reduce
SCHEME_SHARD_W = "shard_w"  # width comes from the shard, cross-core combine

_INTERLEAVED = ttnn.TensorMemoryLayout.INTERLEAVED
_HEIGHT_SHARDED = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
_WIDTH_SHARDED = ttnn.TensorMemoryLayout.WIDTH_SHARDED
_BLOCK_SHARDED = ttnn.TensorMemoryLayout.BLOCK_SHARDED
_SHARDED = (_HEIGHT_SHARDED, _WIDTH_SHARDED, _BLOCK_SHARDED)


def _shard_shape(tensor):
    """Per-core shard extent in ELEMENTS, or None for an interleaved tensor."""
    mc = tensor.memory_config()
    if mc.memory_layout == _INTERLEAVED or mc.shard_spec is None:
        return None
    return (int(mc.shard_spec.shape[0]), int(mc.shard_spec.shape[1]))


def _shard_l1_bytes(tensor):
    """Per-core L1 bytes this tensor's shard occupies (0 when not L1-sharded).

    The CB arena and the allocator's L1 buffers share one L1 region, so a
    resident shard is budget the CBs may NOT spend.  Counting it here is what
    keeps a sharded build from CB-OOM'ing (an op-charged failure) on exactly the
    geometries sharding exists to serve.
    """
    sh = _shard_shape(tensor)
    if sh is None or tensor.memory_config().buffer_type != ttnn.BufferType.L1:
        return 0
    if tensor.layout == ttnn.TILE_LAYOUT:
        return (sh[0] // TILE_DIM) * (sh[1] // TILE_DIM) * ttnn.tile_size(tensor.dtype)
    align = int(ttnn._ttnn.device.get_l1_alignment())
    row_bytes = sh[1] * _stick_elem_bytes(tensor)
    return sh[0] * (((row_bytes + align - 1) // align) * align)


def _shard_tile_extent(tensor):
    """(shard_h_tiles, shard_w_tiles) for a TILE-layout sharded tensor."""
    sh = _shard_shape(tensor)
    assert sh is not None, "rms_norm: _shard_tile_extent on an interleaved tensor"
    assert sh[0] % TILE_DIM == 0 and sh[1] % TILE_DIM == 0, f"rms_norm: TILE shard {sh} is not tile-aligned"
    return sh[0] // TILE_DIM, sh[1] // TILE_DIM


def _same_shard_spec(a, b):
    """Two tensors carry the identical placement (layout + geometry + grid)."""
    ma, mb = a.memory_config(), b.memory_config()
    if ma.memory_layout != mb.memory_layout:
        return False
    if ma.shard_spec is None or mb.shard_spec is None:
        return ma.shard_spec is None and mb.shard_spec is None
    return (
        _shard_shape(a) == _shard_shape(b)
        and ma.shard_spec.grid == mb.shard_spec.grid
        and ma.shard_spec.orientation == mb.shard_spec.orientation
    )


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


class _Work(NamedTuple):
    """One core's slice of the work.  ONE record type for all three schemes.

    The TILE schemes think in tile-rows / width TILES; the ROW_MAJOR ones think
    in sticks / width ELEMENTS.  Both views live here and are derived from each
    other by the two builders below, so the kernels never re-derive one from the
    other and the RT-arg loop is a single unpack.

    An INACTIVE core (row_count == 0) joined the program only so the width
    combine's stat multicast lands in a CB this program owns.
    """

    core: object
    row_start: int  # first TILE-row this core owns
    row_count: int  # tile-rows owned (0 => inactive core)
    w_start: int  # first width TILE owned
    w_real: int  # REAL width tiles owned (<= wt_per_core)
    is_root: bool  # group root: gathers the partials, finalizes, multicasts
    slot: int  # index within the width group
    stick_base: int  # first ROW_MAJOR stick owned
    stick_count: int  # ROW_MAJOR sticks owned
    w_off_elems: int  # first width ELEMENT owned
    w_real_elems: int  # REAL width elements owned


def _work_tile_axis(core, row_start, row_count, w_start, w_real, is_root, slot, *, W, R_rm):
    """A _Work whose primary extents are tile-aligned (SCHEME_ROWS / SHARD_H /
    the TILE SHARD_W path).  The stick / element view is DERIVED, so the two can
    never disagree."""
    stick_base = row_start * TILE_DIM
    sticks = row_count * TILE_DIM
    if R_rm:  # ROW_MAJOR build: the last tile-row of the tensor is short
        sticks = max(0, min(sticks, R_rm - stick_base))
    w_off = w_start * TILE_DIM
    return _Work(
        core=core,
        row_start=row_start,
        row_count=row_count,
        w_start=w_start,
        w_real=w_real,
        is_root=is_root,
        slot=slot,
        stick_base=stick_base,
        stick_count=sticks,
        w_off_elems=w_off,
        w_real_elems=max(0, min(w_real * TILE_DIM, W - w_off)),
    )


def _band_tile_span(w_off, w_real_elems):
    """Width TILES the band [w_off, w_off + w_real_elems) touches in the tensor's
    GLOBAL tile grid -- the frame the band is staged in (see _plan_band)."""
    return _div_up((w_off % TILE_DIM) + w_real_elems, TILE_DIM)


def _work_band(core, *, stick_base, stick_count, w_off, band_elems, W, is_root, slot):
    """A _Work for the ROW_MAJOR BAND scheme (Refinement 2b), whose primary
    extents are a stick range x an ELEMENT range and need not be tile-aligned on
    either axis.  `row_start` is unused there (the dataflow is addressed in
    sticks); `w_start` IS meaningful -- it is the first GLOBAL width tile the band
    touches, which is the frame the band is staged in and hence the tile index
    gamma is fetched at."""
    w_real_elems = max(0, min(band_elems, W - w_off))
    return _Work(
        core=core,
        row_start=0,
        row_count=_div_up(stick_count, TILE_DIM),
        w_start=w_off // TILE_DIM,
        w_real=_band_tile_span(w_off, w_real_elems),
        is_root=is_root,
        slot=slot,
        stick_base=stick_base,
        stick_count=stick_count,
        w_off_elems=w_off,
        w_real_elems=w_real_elems,
    )


_INACTIVE = dict(row_start=0, row_count=0, w_start=0, w_real=0, is_root=False, slot=0)


def _work_inactive(core):
    return _Work(core=core, stick_base=0, stick_count=0, w_off_elems=0, w_real_elems=0, **_INACTIVE)


class _Plan:
    """The resolved placement plan: which scheme, which cores own what.

    One object so the scheme decision has a single source of truth that the L1
    solve, the CB table, all three CT-arg lists and the RT-arg loop all read
    from -- nothing re-derives "who owns which width tiles" a second time.
    """

    __slots__ = (
        "scheme",
        "assignment",  # [_Work]
        "all_cores",  # CoreRangeSet the program (kernels + arena CBs) covers
        "native_in",  # cb_input_tiles is backed on the input shard (zero-copy)
        "native_out",  # cb_output_tiles is backed on the output shard (zero-copy)
        "wt_per_core",  # width tiles a core owns == the shard row stride
        "combine",  # cross-core width combine active (GRID_W > 1)
        "group_size",  # cores per width group (GRID_W)
        "mcast",  # ttnn.Mcast1D / Mcast2D, or None
        "gather_sem_id",  # arrival semaphore for the partial gather, or None
        "l1_reserved",  # per-core L1 bytes the resident shards already hold
        # --- Refinement 2b: the ROW_MAJOR BAND scheme -----------------------
        "band",  # x/out are staged from/to THIS core's resident RM shard
        "band_out_local",  # the out shard matches the in shard => local write-back
        "shard_row_bytes",  # L1 stride of one stick inside the input shard
        "out_shard_row_bytes",  # ... inside the output shard (0 when not local)
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))
        self.band = bool(kw.get("band", False))
        self.band_out_local = bool(kw.get("band_out_local", False))
        self.shard_row_bytes = int(kw.get("shard_row_bytes") or 0)
        self.out_shard_row_bytes = int(kw.get("out_shard_row_bytes") or 0)


def _width_group_cores(Wt, cap):
    """Cores per width group: the coarsest split of `Wt` tiles into `<= cap` cores
    that leaves every core the SAME number of real width tiles.

    A DIVISOR, not `ceil`, and that is load-bearing on the interleaved path: a
    ragged tail leaves the last core's block ending in whole PAD tiles, and unlike
    a resident shard (whose pad tiles the reader can zero once -- see NATIVE_X)
    an interleaved core has no pad storage at all, so the reader would have to
    read x tiles it does not own.  A prime `Wt` therefore does not split; that is
    the same D1 granularity limit the STREAM chunk size lives under.
    """
    return _largest_divisor_at_most(Wt, max(1, cap))


def _auto_width_split(device, Rt, Wt):
    """(cores per width group, row groups) for GRID_W == 0 -- the AUTO policy (D11).

    The row split can only ever use `min(Rt, num_cores)` cores, so a decode
    profile (`Rt = 1`) runs the whole tensor through ONE core no matter how wide
    it is.  This picks the (gw, gh) rectangle-or-line that puts the most cores to
    work, subject to the three policy knobs, and returns the trivial (1, 1) when
    the row split already fills the grid.

    `gh` is the number of ROW groups (each owns a row range and combines within
    itself); `gw` is the group size.  Two topologies are expressible with the
    combine Refinement 2 built, and the loop only ever proposes those:
      * gh == 1  -> ONE group of up to num_cores cores, packed row-major over the
                    grid (its bounding box may be ragged; the few in-box cores
                    outside the group join INACTIVE).  This is the decode case.
      * gh >  1  -> a gw x gh RECTANGLE, one group per grid row, so no group's
                    multicast rectangle can overlap another group's cores.
    """
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    row_cores = max(1, min(Rt, num_cores))
    best = (1, 1, row_cores)  # (gw, gh, total cores at work)
    for gh in range(1, min(Rt, grid.y) + 1):
        cap = min(
            num_cores // gh,  # the grid
            Wt // WIDTH_SPLIT_MIN_WT_PER_CORE,  # leave every core real work
            WIDTH_SPLIT_MAX_GROUP_CORES,  # bound the gather fan-in
        )
        if gh > 1:
            cap = min(cap, grid.x)  # a multi-group split must be a rectangle
        gw = _width_group_cores(Wt, cap)
        total = gw * gh
        # Prefer more cores at work; on a tie prefer the SMALLER group (a cheaper
        # gather), which the strict `>` gives us since gh ascends.
        if gw >= 2 and total > best[2]:
            best = (gw, gh, total)
    gw, gh, total = best
    if total < WIDTH_SPLIT_MIN_GAIN * row_cores:
        return 1, 1
    return gw, gh


def _resolve_width_split(device, Rt, Wt):
    """(gw, gh) for the interleaved width split -- the GRID_W knob, resolved.

    GRID_W == 0 is the AUTO policy; >= 1 forces the group size (1 => no split).
    """
    if GRID_W == 0:
        return _auto_width_split(device, Rt, Wt)
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    cap = min(GRID_W, num_cores)
    gh = 1 if cap > grid.x else max(1, min(Rt, grid.y))
    gw = _width_group_cores(Wt, min(cap, num_cores // gh))
    return gw, gh


def _plan_interleaved_width_split(device, input_tensor, output_tensor, Rt, Wt, W, R_rm, gw, gh):
    """The cross-core width split on an interleaved input: Lamp L1.

    The dependent `width` axis is cut across `gw` cores per row group and the
    partials are combined with the SAME topology the sharded schemes use -- only
    `native_in` differs (x still arrives through a TensorAccessor, because an
    interleaved tensor has no resident per-core slice).  Refinement 3 turns this
    on by default through _resolve_width_split; at GRID_W == 1 it is not reached.

    `gw` divides `Wt` (see _width_group_cores), so every core owns the same
    number of real width tiles -- the uniformity the sharded path gets from a
    non-ragged shard.  Two topologies, chosen by whether the group fits one
    physical grid row:

      gw <= grid.x   a gw x gh RECTANGLE, one group per grid row (Mcast1D PerRow).
      gw >  grid.x   ONE group (gh == 1) PACKED row-major across the grid, with
                     the stat multicast over its bounding box and the few in-box
                     cores outside the group joining INACTIVE -- exactly what a
                     row-major-packed WIDTH shard grid already does.
    """
    grid = device.compute_with_storage_grid_size()
    wt_per_core = Wt // gw
    mc_cfg = ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=True, base_sem_id=0)
    assignment = []
    if gw <= grid.x:
        gh = max(1, min(gh, grid.y))
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gw - 1, gh - 1))])
        base, extra = divmod(Rt, gh)
        for core in _cores_in(crs):
            y = core.y
            rows = base + (1 if y < extra else 0)
            row_start = y * base + min(y, extra)
            w_start = core.x * wt_per_core
            assignment.append(
                _work_tile_axis(core, row_start, rows, w_start, wt_per_core, core.x == 0, core.x, W=W, R_rm=R_rm)
            )
        mcast = ttnn.Mcast1D(device, crs, ttnn.Mcast1DShape.PerRow, 0, mc_cfg) if gw > 1 else None
    else:
        # PACKED single group.  Its bounding box is the first ceil(gw / grid.x)
        # whole grid rows; the (gw % grid.x) trailing cores of the last row are in
        # the box but not in the group, so they join INACTIVE (row_count == 0) --
        # they exist only so the stat multicast lands in a cb_row_final this
        # program owns.  _cores_in is row-major (row_wise=True), which is the same
        # order the packed slice is taken in.
        assert gh == 1, "rms_norm: a width group wider than the grid must be the only group"
        rows_used = _div_up(gw, grid.x)
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, rows_used - 1))])
        for i, core in enumerate(_cores_in(crs)):
            if i >= gw:
                assignment.append(_work_inactive(core))
                continue
            assignment.append(_work_tile_axis(core, 0, Rt, i * wt_per_core, wt_per_core, i == 0, i, W=W, R_rm=R_rm))
        root = assignment[0].core
        mcast = ttnn.Mcast2D(device, crs, ttnn.CoreCoord(root.x, root.y), mc_cfg, gw - 1)
    return _Plan(
        scheme=SCHEME_SHARD_W,
        assignment=assignment,
        all_cores=crs,
        native_in=False,
        native_out=False,
        wt_per_core=wt_per_core,
        combine=mcast is not None,
        group_size=gw,
        mcast=mcast,
        gather_sem_id=(mcast.next_base_sem_id() if mcast is not None else None),
        l1_reserved=_shard_l1_bytes(input_tensor) + _shard_l1_bytes(output_tensor),
    )


def _plan_rows(device, input_tensor, output_tensor, Rt, Wt, W, R_rm, *, allow_width_split=True):
    """SCHEME_ROWS: split the independent `row` axis over the FULL grid.

    Phase 0's scheme, and the universal fallback: every tensor is reached through
    a TensorAccessor, which resolves an interleaved *or* a sharded buffer, so it
    is correct for any placement.  Used for INTERLEAVED, for every ROW_MAJOR
    sharded build (see _plan_placement's note on the RM shard granule) and as the
    L1 bail-out for a tiled shard whose per-core block does not fit.

    `allow_width_split=False` is that L1 bail-out: the caller has already tried a
    width-split plan and its per-core block did not fit, so re-proposing one would
    loop.  It is also what keeps the width split off the ROW_MAJOR interleaved
    path (an RM row is addressed by STICK, and the accessor reads a whole page, so
    a width slice of an interleaved RM row is not a page -- the BAND scheme
    reaches those only because a shard makes the segment a page of its own).
    """
    if allow_width_split and Wt > 1 and input_tensor.layout == ttnn.TILE_LAYOUT:
        gw, gh = _resolve_width_split(device, Rt, Wt)
        if gw > 1:
            return _plan_interleaved_width_split(device, input_tensor, output_tensor, Rt, Wt, W, R_rm, gw, gh)
    num_cores, all_cores, g1, g2, rpc1, rpc2 = ttnn.split_work_to_cores(_core_range_set_full_grid(device), Rt, True)
    assignment = []
    row_cursor = 0
    for group_cores, rpc in ((_cores_in(g1), rpc1), (_cores_in(g2), rpc2)):
        for core in group_cores:
            assignment.append(_work_tile_axis(core, row_cursor, rpc, 0, Wt, True, 0, W=W, R_rm=R_rm))
            row_cursor += rpc
    assert row_cursor == Rt, f"rms_norm: work split covers {row_cursor} of {Rt} tile-rows"
    assert len(assignment) == num_cores, f"rms_norm: {len(assignment)} cores assigned, expected {num_cores}"
    return _Plan(
        scheme=SCHEME_ROWS,
        assignment=assignment,
        all_cores=all_cores,
        native_in=False,
        native_out=False,
        wt_per_core=Wt,
        combine=False,
        group_size=1,
        mcast=None,
        gather_sem_id=None,
        l1_reserved=_shard_l1_bytes(input_tensor) + _shard_l1_bytes(output_tensor),
    )


def _plan_band(device, input_tensor, output_tensor, *, Rt, Wt, W, R_rm):
    """SCHEME_SHARD_W on a ROW_MAJOR shard that cuts the `width` axis (Ref. 2b).

    THE INSIGHT.  An RM shard edge rounds to (1 stick x L1_align/elem_size
    elements) -- 8 for bf16, 4 for fp32 -- and a shard may not hold a partial
    page, so the tensor's PAGE is the shard's row SEGMENT and NO core holds a
    whole width TILE.  That is fatal only if the width split has to be
    tile-granular.  It does not: the cross-core combine sums the group's per-row
    PARTIALS **elementwise**, and a partial may cover ANY contiguous element
    range of the row -- Sum(x^2) over the row is the sum over the bands however
    the bands are cut.  Nothing downstream ever has to reassemble a row either:
    pass B scales, gamma-multiplies and writes back entirely inside the band.

    So each core takes its own resident shard as its `band` -- all of its sticks
    x `shard_w` elements -- and:
      * STAGES it from its OWN L1 (x_addr + local_stick * shard_row_bytes) into
        the tilize ring at the ring's padded stride.  There is no NoC read of x
        from anywhere but this core's own L1: no DRAM traffic, and no accessor on
        a local shard.  One transaction per tile-row when the band fills its tile
        columns exactly, one per stick otherwise.
      * tilizes it into `ceil(shard_w / 32)` tile columns whose trailing lanes
        are the staging ring's BOOT ZEROS.  Those lanes contribute an exact 0 to
        Sum(x^2), which is why the band scheme needs no partial scaler / mask at
        all (PARTIAL_W is passed to the kernels as 0 here; the finalize's 1/W is
        the LOGICAL width, as always).
      * joins the SAME section-3.4 combine the TILE shards use, unchanged.
      * reads gamma at the band's BYTE offset (gamma is placement-independent),
        and writes the result back into the band's own L1.

    This subsumes both levers op_requirements.md Refinement 2b listed: lever 2
    ("native band tilize when shard_w % 32 == 0") is the contiguous fast path
    here rather than the whole scheme, and lever 1's ceil(W / shard_w) reads per
    stick are never paid, because a core only ever reads its OWN segment.
    """
    in_ml = input_tensor.memory_config().memory_layout
    shard_h, shard_w = _shard_shape(input_tensor)
    # The stick stride inside the shard is the buffer's own aligned page size --
    # read it off the buffer rather than re-deriving align_up(shard_w * elem).
    shard_row_bytes = int(input_tensor.buffer_aligned_page_size())
    shard_grid = input_tensor.memory_config().shard_spec.grid
    shard_cores = _cores_in(shard_grid)
    l1_reserved = _shard_l1_bytes(input_tensor) + _shard_l1_bytes(output_tensor)

    # Write-back target.  Identical placement => core i's band IS its own output
    # shard, so the write is local L1.  Otherwise the output must be addressable
    # by STICK through the accessor, which is true exactly when its page is a
    # whole row (interleaved, or height-sharded: shard_w == the full padded W).
    band_out_local = output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT and _same_shard_spec(input_tensor, output_tensor)
    out_ml = output_tensor.memory_config().memory_layout
    if not band_out_local and out_ml not in (_INTERLEAVED, _HEIGHT_SHARDED):
        raise NotImplementedError(
            f"rms_norm: a ROW_MAJOR {in_ml} input needs an output that is either the SAME "
            f"shard spec (written in place) or stick-paged (INTERLEAVED / HEIGHT_SHARDED); "
            f"got {out_ml} with a different geometry"
        )
    out_shard_row_bytes = int(output_tensor.buffer_aligned_page_size()) if band_out_local else 0

    bbox = shard_grid.bounding_box()
    if in_ml == _WIDTH_SHARDED:
        # Every core holds the full row range and one band, so the whole shard
        # grid is ONE group.  That grid is row-major-PACKED and need not be a
        # rectangle, so the multicast runs over its bounding box and the in-box /
        # out-of-shard cores join as INACTIVE -- exactly as on the TILE path.
        bbox_crs = ttnn.CoreRangeSet([ttnn.CoreRange(bbox.start, bbox.end)])
        group_size = len(shard_cores)
        root = shard_cores[0]
        owned = {(c.x, c.y): i for i, c in enumerate(shard_cores)}
        assignment = []
        for core in _cores_in(bbox_crs):
            i = owned.get((core.x, core.y))
            if i is None:
                assignment.append(_work_inactive(core))
                continue
            assignment.append(
                _work_band(
                    core,
                    stick_base=0,
                    stick_count=R_rm,
                    w_off=i * shard_w,
                    band_elems=shard_w,
                    W=W,
                    is_root=(i == 0),
                    slot=i,
                )
            )
        all_cores = bbox_crs
        mcast = (
            ttnn.Mcast2D(
                device,
                bbox_crs,
                ttnn.CoreCoord(root.x, root.y),
                ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=True, base_sem_id=0),
                group_size - 1,
            )
            if group_size > 1
            else None
        )
    else:  # _BLOCK_SHARDED -- a true rectangle; each grid ROW is a width group
        nx = bbox.end.x - bbox.start.x + 1
        ny = bbox.end.y - bbox.start.y + 1
        assert shard_grid.num_cores() == nx * ny, f"rms_norm: BLOCK shard grid {shard_grid} is not a full rectangle"
        group_size = nx
        assignment = []
        for core in _cores_in(shard_grid):
            x, y = core.x - bbox.start.x, core.y - bbox.start.y
            assignment.append(
                _work_band(
                    core,
                    stick_base=min(y * shard_h, R_rm),
                    stick_count=max(0, min(shard_h, R_rm - y * shard_h)),
                    w_off=x * shard_w,
                    band_elems=shard_w,
                    W=W,
                    is_root=(x == 0),
                    slot=x,
                )
            )
        all_cores = shard_grid
        mcast = (
            ttnn.Mcast1D(
                device,
                shard_grid,
                ttnn.Mcast1DShape.PerRow,
                0,
                ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=True, base_sem_id=0),
            )
            if group_size > 1
            else None
        )

    # WT_CHUNK is compile-time and shared, so the staged block must cover the
    # WIDEST tile span any core's band touches; a core whose band spans fewer tiles
    # just stages an all-zero pad tile column, which adds exactly 0 to sum(x^2).
    wt_band = max((w.w_real for w in assignment if w.row_count), default=1) or 1

    return _Plan(
        scheme=SCHEME_SHARD_W,
        assignment=assignment,
        all_cores=all_cores,
        native_in=False,  # the band is staged (tilized), not aliased -- but from LOCAL L1
        native_out=False,
        wt_per_core=wt_band,
        combine=mcast is not None,
        group_size=group_size,
        mcast=mcast,
        gather_sem_id=(mcast.next_base_sem_id() if mcast is not None else None),
        l1_reserved=l1_reserved,
        band=True,
        band_out_local=band_out_local,
        shard_row_bytes=shard_row_bytes,
        out_shard_row_bytes=out_shard_row_bytes,
    )


def _plan_placement(device, input_tensor, output_tensor, *, is_tile, Rt, Wt, W, R_rm, partial_w, force_rows=False):
    """Resolve `memory_layout` into one of the internal schemes.

    A pure function of (layout, placement, shard geometry, alignment) -- no
    device-grid or work-distribution input beyond the grid the shard already
    names, so the scheme is reproducible for a fixed input.

    ROW_MAJOR + HEIGHT_SHARDED deliberately takes SCHEME_ROWS: a height shard
    spans the FULL padded width, so the tensor's page IS the stick and the
    accessor addresses rows exactly (measured PCC 1.000000).  ROW_MAJOR +
    WIDTH/BLOCK_SHARDED takes the BAND scheme -- there the page is a row SEGMENT,
    so the accessor cannot reach a row at all (see _plan_band).
    """
    in_ml = input_tensor.memory_config().memory_layout
    if not force_rows and not is_tile and in_ml in (_WIDTH_SHARDED, _BLOCK_SHARDED):
        return _plan_band(device, input_tensor, output_tensor, Rt=Rt, Wt=Wt, W=W, R_rm=R_rm)
    if force_rows or not is_tile or in_ml == _INTERLEAVED:
        # force_rows is the L1 bail-out from a width-split plan, so it must not
        # propose one again (see _plan_rows).
        return _plan_rows(device, input_tensor, output_tensor, Rt, Wt, W, R_rm, allow_width_split=not force_rows)

    shard_h_t, shard_w_t = _shard_tile_extent(input_tensor)
    shard_grid = input_tensor.memory_config().shard_spec.grid
    shard_cores = _cores_in(shard_grid)
    l1_reserved = _shard_l1_bytes(input_tensor) + _shard_l1_bytes(output_tensor)
    # A zero-copy OUTPUT CB is only meaningful when the output shard is laid out
    # exactly like the input's -- same grid, same per-core extent, same order --
    # because then core `i`'s output block is the block it just computed.
    native_out = output_tensor.layout == ttnn.TILE_LAYOUT and _same_shard_spec(input_tensor, output_tensor)

    if in_ml == _HEIGHT_SHARDED:
        # Lamp L3 knob-turn: the shard cuts `row`, so the shard IS the per-core
        # block and the reduce stays local.  A legal HEIGHT shard spans the full
        # padded width by construction; refuse to guess if it somehow does not.
        if shard_w_t != Wt:
            # The input's own placement already picked the cores; don't overlay a
            # second width split on top of it (byte-identical to Refinement 2).
            return _plan_rows(device, input_tensor, output_tensor, Rt, Wt, W, R_rm, allow_width_split=False)
        assignment = []
        for i, core in enumerate(shard_cores):
            row_start = i * shard_h_t
            row_count = max(0, min(shard_h_t, Rt - row_start))
            assignment.append(_work_tile_axis(core, min(row_start, Rt), row_count, 0, Wt, True, 0, W=W, R_rm=R_rm))
        return _Plan(
            scheme=SCHEME_SHARD_H,
            assignment=assignment,
            all_cores=shard_grid,
            native_in=True,
            native_out=native_out,
            wt_per_core=Wt,
            combine=False,
            group_size=1,
            mcast=None,
            gather_sem_id=None,
            l1_reserved=l1_reserved,
        )

    # ---- WIDTH / BLOCK: the shard cuts the DEPENDENT `width` axis ------------
    # Lamp L4 (superset of L1).  Each core owns a width SLICE of a row, so its
    # sum(x^2) is a PARTIAL and must be combined across the group of cores that
    # share the row range.
    #
    # Bail-out: a RAGGED width tail (Wt not a multiple of the shard's tile width)
    # leaves the last core's block ending in whole PAD tiles.  Zeroing those pad
    # tiles keeps a tile-aligned W exact, but when W is ALSO non-tile-aligned the
    # partial-W mask would have to land on the last REAL width tile rather than
    # the block's last page, which no ReducePartialScaler can express.  That
    # combination takes SCHEME_ROWS (correct, just not width-split).
    ragged_w = (Wt % shard_w_t) != 0
    if ragged_w and partial_w:
        # As above: the shard's own geometry chose the cores, so this fallback stays
        # the plain row split it was in Refinement 2.
        return _plan_rows(device, input_tensor, output_tensor, Rt, Wt, W, R_rm, allow_width_split=False)

    bbox = shard_grid.bounding_box()
    nx = bbox.end.x - bbox.start.x + 1
    ny = bbox.end.y - bbox.start.y + 1

    if in_ml == _WIDTH_SHARDED:
        # Every core holds the full row range and one width slice, so the whole
        # shard grid is ONE group.  That grid is row-major-packed and need not be
        # a rectangle (64 cores on an 11-wide grid = 5 full rows + 9), so the
        # multicast runs over its BOUNDING BOX and the few cores inside the box
        # but outside the shard join the program as INACTIVE (row_count == 0).
        # They exist only so the stat multicast lands in a cb_row_final this
        # program owns instead of in whatever else holds that L1.
        group_size = len(shard_cores)
        root = shard_cores[0]
        assignment = []
        owned = {(c.x, c.y): i for i, c in enumerate(shard_cores)}
        for core in _cores_in(bbox_crs := ttnn.CoreRangeSet([ttnn.CoreRange(bbox.start, bbox.end)])):
            i = owned.get((core.x, core.y))
            if i is None:
                assignment.append(_work_inactive(core))  # inactive padding core
                continue
            w_start = i * shard_w_t
            assignment.append(
                _work_tile_axis(core, 0, Rt, w_start, min(shard_w_t, Wt - w_start), i == 0, i, W=W, R_rm=R_rm)
            )
        all_cores = bbox_crs
        mcast = (
            ttnn.Mcast2D(
                device,
                bbox_crs,
                ttnn.CoreCoord(root.x, root.y),
                ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=True, base_sem_id=0),
                group_size - 1,
            )
            if group_size > 1
            else None
        )
    else:  # _BLOCK_SHARDED
        # A rectangle: grid column x owns width slice x, grid row y owns row
        # block y (eval.sharding maps rows->y, cols->x).  Each grid ROW is a
        # width group with its root at column 0 -- exactly Mcast1D's PerRow
        # line family, one CT block covering all ny groups.
        assert shard_grid.num_cores() == nx * ny, f"rms_norm: BLOCK shard grid {shard_grid} is not a full rectangle"
        group_size = nx
        assignment = []
        for core in _cores_in(shard_grid):
            x, y = core.x - bbox.start.x, core.y - bbox.start.y
            row_start = y * shard_h_t
            w_start = x * shard_w_t
            assignment.append(
                _work_tile_axis(
                    core,
                    min(row_start, Rt),
                    max(0, min(shard_h_t, Rt - row_start)),
                    w_start,
                    min(shard_w_t, Wt - w_start),
                    x == 0,
                    x,
                    W=W,
                    R_rm=R_rm,
                )
            )
        all_cores = shard_grid
        mcast = (
            ttnn.Mcast1D(
                device,
                shard_grid,
                ttnn.Mcast1DShape.PerRow,
                0,
                ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=True, base_sem_id=0),
            )
            if group_size > 1
            else None
        )

    return _Plan(
        scheme=SCHEME_SHARD_W,
        assignment=assignment,
        all_cores=all_cores,
        native_in=True,
        native_out=native_out,
        wt_per_core=shard_w_t,
        combine=mcast is not None,
        group_size=group_size,
        mcast=mcast,
        gather_sem_id=(mcast.next_base_sem_id() if mcast is not None else None),
        l1_reserved=l1_reserved,
    )


def create_program_descriptor(
    input_tensor: "ttnn.Tensor",
    output_tensor: "ttnn.Tensor",
    *,
    gamma: "ttnn.Tensor" = None,
    epsilon: float = 1e-6,
    compute_kernel_config: "ttnn.ComputeConfigDescriptor" = None,
) -> "ttnn.ProgramDescriptor":
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

    bt = ttnn.tile_size(input_tensor.dtype)  # x / x^2 / normalized / out
    gt = ttnn.tile_size(gamma.dtype) if has_gamma else 0
    st = ttnn.tile_size(ttnn.bfloat16)  # scaler CB (R4: value exactly 1.0)
    ft = ttnn.tile_size(ttnn.float32)  # cb_row_stat & the combine CBs

    # ---- placement -> scheme, cores, per-core (row, width) extents ---------
    # Refinement 2.  The scheme decides which axis the cores cut, whether the
    # x / out CBs alias a resident shard, and whether a cross-core combine runs;
    # everything below reads it from the plan rather than re-deriving it.
    plan = _plan_placement(
        device, input_tensor, output_tensor, is_tile=is_tile, Rt=Rt, Wt=Wt, W=W, R_rm=R_rm, partial_w=partial_w
    )

    # ---- the reduce's partial-W mechanism (Refinement 2b) ------------------
    # PARTIAL_W as the KERNELS see it.  The BAND scheme stages each core's band at
    # its REAL element width into a ring whose trailing bytes are zero, so its pad
    # lanes contribute an exact 0 to sum(x^2) with no scaler / mask tile at all --
    # and a per-core band boundary is not expressible as one program-wide
    # PARTIAL_W anyway.  Every other scheme is unchanged (kernel_partial_w ==
    # partial_w), so this is byte-identical off the band path.
    kernel_partial_w = 0 if plan.band else partial_w

    # Scaler CB page count: one source of truth for the budget term, the CB
    # allocation and (via PARTIAL_W) the compute kernel's final pop.
    scaler_pages = 2 if kernel_partial_w else 1
    scaler_bytes = st * scaler_pages

    def _solve_blocking(plan):
        """(block_rows, wt_chunk, num_w_chunks, cb_x_depth, cb_out_depth) or None.

        None => this plan's per-core block does not fit L1 at all, and the caller
        must fall back to SCHEME_ROWS (which can chunk `width`).
        """
        wt_core = plan.wt_per_core
        # The resident shards share L1 with the CB arena, so they are budget the
        # CBs may not spend (_shard_l1_bytes) -- and once ANY L1 buffer exists the
        # CB region's absolute ceiling is that buffer's address, so the arena's
        # fixed base offset has to come off too (L1_CB_ARENA_BASE_RESERVE).
        avail = ttnn.get_max_worker_l1_unreserved_size()
        if plan.l1_reserved:
            avail -= plan.l1_reserved + L1_CB_ARENA_BASE_RESERVE
        budget = int(max(0, avail) * L1_SAFETY_FRACTION)
        max_rows = max((a.row_count for a in plan.assignment), default=1) or 1
        # A CB backed on the shard costs ZERO arena bytes (it aliases the tensor's
        # own L1), so its depth term drops out of the block multiplier -- and with
        # no NoC read to overlap, depth buys nothing there either.
        dx0 = 0 if plan.native_in else None
        do0 = 0 if plan.native_out else None
        # Depth > 1 only buys overlap when the producer/consumer pair spans two
        # processors.  On the ROW_MAJOR path cb_input_tiles is produced by the
        # `tilize` compute helper and cb_output_tiles is consumed by `untilize`;
        # sequential compute helpers own all three TRISCs and cannot pipeline, so
        # those CBs drop to depth 1 (and must instead hold a whole block — R5).
        depth_candidates = CB_DEPTH_CANDIDATES if is_tile else (1,)
        # fp32 pages per tile-row of a block: cb_row_stat, plus the combine's
        # handoff / gather / mcast CBs when the width split is active.
        f32_pages = CB_ROW_STAT_DEPTH + ((1 + plan.group_size + 1 + CB_ROW_STAT_DEPTH) if plan.combine else 0)

        def _resident_fit(depth):
            mult = _cb_block_mult(depth if dx0 is None else dx0, depth if do0 is None else do0, has_gamma)
            fixed = (
                (wt_core * gt)  # cb_gamma_tiles
                + (wt_core * gt if gamma_is_rm else 0)  # cb_gamma_sticks
                + (2 * CB_RM_STAGE_DEPTH * wt_core * bt if not is_tile else 0)  # stick staging
                + scaler_bytes
            )
            per_tilerow = wt_core * bt * mult + f32_pages * ft
            return max(0, (budget - fixed) // max(1, per_tilerow)), mult

        for depth in depth_candidates:
            brmax, _ = _resident_fit(depth)
            if brmax >= 1:
                # RESIDENT: the whole per-core row slice is resident; take the
                # coarsest row block that fits, i.e. the entire assignment when it does.
                return min(max_rows, brmax), wt_core, 1, depth, depth

        if plan.band:
            # The BAND scheme has no fallback: its width is shard-derived (so not
            # chunkable) and SCHEME_ROWS cannot address an RM width shard at all
            # (the page is a row SEGMENT).  Take the finest block -- ONE tile-row --
            # and let metal's own CB-region check be the arbiter rather than
            # pre-refusing on a proportional safety margin.
            return 1, wt_core, 1, depth_candidates[0], depth_candidates[0]
        if plan.scheme != SCHEME_ROWS:
            return None  # a shard-derived width cannot be chunked -> caller falls back

        # STREAM: one tile-row's width does not fit at ANY depth -> chunk `width`
        # (an L1 fallback, not a parallelization).  The chunk size adapts to L1,
        # so keep the preferred (coarsest) depth: overlap is affordable and the
        # byte count is already paid.
        depth = depth_candidates[0]
        mult = _cb_block_mult(depth, depth, has_gamma)
        per_chunk_tile_bytes = (
            bt * mult
            + (gt * (2 if gamma_is_rm else 1) if has_gamma else 0)
            + (2 * CB_RM_STAGE_DEPTH * bt if not is_tile else 0)  # D2
        )
        fixed_stream = scaler_bytes + f32_pages * ft  # block_rows == 1 here
        wt_chunk_l1_max = max(1, (budget - fixed_stream) // per_chunk_tile_bytes)
        wtc = _largest_divisor_at_most(wt_core, wt_chunk_l1_max)  # D1
        return 1, wtc, wt_core // wtc, depth, depth

    solved = _solve_blocking(plan)
    if solved is None:
        # L1 bail-out: the tiled shard's per-core block does not fit even at
        # BLOCK_ROWS == 1, and a shard-derived width is not chunkable.  Re-plan on
        # SCHEME_ROWS, which reads x through the accessor and CAN chunk `width`.
        plan = _plan_placement(
            device,
            input_tensor,
            output_tensor,
            is_tile=is_tile,
            Rt=Rt,
            Wt=Wt,
            W=W,
            R_rm=R_rm,
            partial_w=partial_w,
            force_rows=True,
        )
        solved = _solve_blocking(plan)
        assert solved is not None, "rms_norm: no admissible blocking even on SCHEME_ROWS"
    block_rows, wt_chunk, num_w_chunks, cb_x_depth, cb_out_depth = solved
    all_cores = plan.all_cores
    assignment = plan.assignment
    wt_per_core = plan.wt_per_core
    combine = plan.combine

    # X_RESIDENT == GAMMA_RESIDENT == (NUM_W_CHUNKS == 1) is derived in the
    # kernels from the NUM_W_CHUNKS CT arg -- one source of truth, so it is
    # deliberately NOT passed as a second flag.

    # ---- reduce datapath (D7, refined by D8) ------------------------------
    # The crossover is measured against THIS CORE'S WHOLE reduce dim
    # (`wt_per_core`), not against WT_CHUNK.  In the RESIDENT regime the two are
    # equal, so this is byte-identical to Refinement 1b; they differ only when L1
    # forces a chunked width, and there it is the total that decides -- see D8.
    # AccumulateViaAdd's cross-chunk Accumulate indexes a resident block, hence
    # BulkWaitBulkPop only: the two knobs are coupled here (one place), and the
    # compute kernel static_asserts it.
    reduce_acc_via_add = (
        REDUCE_BULK == 1 and wt_per_core >= REDUCE_ACC_VIA_ADD_MIN_WT and wt_chunk >= REDUCE_ACC_VIA_ADD_MIN_CHUNK_WT
    )
    # Tiles the reader actually pushes into cb_scaler, and the compute pops.
    # AccumulateViaAdd takes ONE: the 0/1 mask (partial W) or an unused 1.0 scaler.
    # ReduceTile takes the [full, partial] pair when W is not tile-aligned.
    scaler_tiles = 1 if reduce_acc_via_add else scaler_pages
    assert scaler_tiles <= scaler_pages, "rms_norm: cb_scaler is sized below the tiles the reader pushes"

    # ---- circular buffers -------------------------------------------------
    # Every page count below is a function of the block/depth knobs only.
    # cb_gamma_* is the one place a whole-op width appears; it *is* the gamma
    # tensor's per-core extent and is bounded by the same L1 predicate.
    #
    # cb_input_tiles / cb_output_tiles are ZERO-COPY when the plan says native:
    # cb_descriptor_from_sharded_tensor aliases the tensor's own resident L1, so
    # there is no NoC read for x and no arena allocation at all.
    cbs = []
    if not is_tile:
        cbs.append(_cb(CB_INPUT_STICKS, bt, CB_RM_STAGE_DEPTH * wt_chunk, input_tensor.dtype, all_cores))
        cbs.append(_cb(CB_OUTPUT_STICKS, bt, CB_RM_STAGE_DEPTH * wt_chunk, output_tensor.dtype, all_cores))
    in_shard_pages = 0
    out_shard_pages = 0
    if plan.native_in:
        sh_t, sw_t = _shard_tile_extent(input_tensor)
        in_shard_pages = sh_t * sw_t
        assert sw_t == wt_chunk, f"rms_norm: native x CB row stride {sw_t} != WT_CHUNK {wt_chunk}"
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_TILES, input_tensor))
    else:
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
    if plan.native_out:
        osh_t, osw_t = _shard_tile_extent(output_tensor)
        out_shard_pages = osh_t * osw_t
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT_TILES, output_tensor))
    else:
        cbs.append(_cb(CB_OUTPUT_TILES, bt, cb_out_depth * block_rows * wt_chunk, output_tensor.dtype, all_cores))
    if combine:
        # op_design.md section 3.4.  cb_sum_handoff is DEDICATED to the cross-kernel
        # handoff and deliberately distinct from cb_row_stat: a dataflow reader on
        # the compute accumulator would be a second consumer.
        # cb_partials_gathered lands on EVERY core (not just the roots) so its L1
        # address is identical everywhere -- that is how a sender computes the
        # root's landing address without the host having to know a CB address.
        cbs.append(_cb(CB_SUM_HANDOFF, ft, block_rows, ttnn.float32, all_cores))
        cbs.append(_cb(CB_PARTIALS_GATHERED, ft, plan.group_size * block_rows, ttnn.float32, all_cores))
        cbs.append(_cb(CB_STAT_HANDOFF, ft, block_rows, ttnn.float32, all_cores))
        cbs.append(_cb(CB_ROW_FINAL, ft, CB_ROW_STAT_DEPTH * block_rows, ttnn.float32, all_cores))

    # ---- ROW_MAJOR staging-ring zero (R3, generalized by Refinement 2b) ----
    # The pad bytes of a staged stick are never written by a read, so whatever L1
    # garbage was there would survive into the reduce (and inf*0 / nan*0 = NaN
    # would poison the whole row).  Zeroing the ring ONCE at boot establishes
    # "every pad byte is either zero or real tensor data".  It is needed exactly
    # when some staged stick is NARROWER than the ring's padded row: the W tile
    # padding on the whole-row schemes, or a band that does not fill its tile
    # columns on the BAND scheme -- where it also REPLACES the reduce mask.
    stage_pad_bytes = wt_chunk * TILE_DIM * elem_bytes
    stage_zero = (not is_tile) and (
        any(
            (a.w_off_elems % TILE_DIM) != 0 or a.w_real_elems * elem_bytes != stage_pad_bytes
            for a in assignment
            if a.row_count
        )
        if plan.band
        else partial_w != 0
    )

    # ---- reader -----------------------------------------------------------
    reader_ct_args = [
        1 if is_tile else 0,  # 0  IS_TILE
        Wt,  # 1  WT (whole-row width tiles: gamma / x tile ids)
        wt_chunk,  # 2  WT_CHUNK
        num_w_chunks,  # 3  NUM_W_CHUNKS
        block_rows,  # 4  BLOCK_ROWS
        kernel_partial_w,  # 5  PARTIAL_W (0 => aligned, or the BAND scheme)
        1 if has_gamma else 0,  # 6  HAS_GAMMA
        1 if gamma_is_rm else 0,  # 7  GAMMA_IS_RM
        elem_bytes,  # 8  input element bytes
        gamma_elem_bytes,  # 9  gamma element bytes
        R_rm,  # 10 total ROW_MAJOR sticks (0 for TILE)
        W,  # 11 logical width (elements)
        1 if reduce_acc_via_add else 0,  # 12 REDUCE_ACC_VIA_ADD (picks mask vs scaler pair)
        1 if plan.native_in else 0,  # 13 NATIVE_IN: cb_input_tiles aliases the shard
        in_shard_pages,  # 14 pages of the resident x shard to publish (native only)
        1 if plan.band else 0,  # 15 BAND: stage x from THIS core's own RM shard
        plan.shard_row_bytes,  # 16 L1 stick stride inside that shard (0 if !BAND)
        1 if stage_zero else 0,  # 17 zero the RM staging ring at boot
    ]
    # The kernel reads its accessor args at TensorAccessorArgs<N>() -- N must equal
    # the scalar CT-arg count above.  Assert it here so adding a scalar arg fails
    # in Python instead of mis-parsing on device.
    assert len(reader_ct_args) == 18, "rms_norm_reader.cpp expects TensorAccessorArgs<18>()"
    assert reader_ct_args[READER_CT_BAND] == (1 if plan.band else 0), "READER_CT_BAND index drifted"
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    # ---- writer -----------------------------------------------------------
    # The writer owns the whole cross-core combine (gather -> root -> mcast back):
    # it runs on NoC1, which is idle through pass A, so the reader's NoC0 keeps
    # streaming x / gamma while the combine handshake runs.
    writer_ct_args = [
        1 if is_tile else 0,  # 0  IS_TILE
        Wt,  # 1  WT
        wt_chunk,  # 2  WT_CHUNK
        num_w_chunks,  # 3  NUM_W_CHUNKS
        block_rows,  # 4  BLOCK_ROWS
        elem_bytes,  # 5  output element bytes
        R_rm,  # 6  total ROW_MAJOR sticks (0 for TILE)
        W,  # 7  logical width (elements)
        1 if plan.native_out else 0,  # 8  NATIVE_OUT: cb_output_tiles aliases the shard
        1 if combine else 0,  # 9  COMBINE: cross-core width combine active
        (plan.gather_sem_id if combine else 0),  # 10 gather arrival semaphore id
        plan.group_size,  # 11 cores per width group (GRID_W)
        out_shard_pages,  # 12 pages of the resident out shard (native only)
        1 if plan.band else 0,  # 13 BAND: write the band back stick-by-stick
        plan.out_shard_row_bytes,  # 14 L1 stick stride inside the out shard (0 => accessor)
    ]
    assert len(writer_ct_args) == 15, "rms_norm_writer.cpp expects McastArgs<15, 10>()"
    assert (
        writer_ct_args[WRITER_CT_OUT_SHARD_ROW_BYTES] == plan.out_shard_row_bytes
    ), "WRITER_CT_OUT_SHARD_ROW_BYTES index drifted"
    writer_ct_args.extend(plan.mcast.compile_time_args() if combine else [0] * 5)
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # ---- compute ----------------------------------------------------------
    compute_ct_args = [
        1 if is_tile else 0,  # 0  IS_TILE
        wt_chunk,  # 1  WT_CHUNK
        num_w_chunks,  # 2  NUM_W_CHUNKS
        block_rows,  # 3  BLOCK_ROWS
        kernel_partial_w,  # 4  PARTIAL_W (0 on the BAND scheme -- see kernel_partial_w)
        1 if has_gamma else 0,  # 5  HAS_GAMMA
        1 if gamma_is_rm else 0,  # 6  GAMMA_IS_RM
        _f32_bits(1.0 / float(W)),  # 7  INV_W (raw fp32 bits) -- R1/R4: logical W
        _f32_bits(epsilon),  # 8  EPS (raw fp32 bits)
        REDUCE_BULK,  # 9  reduce input policy knob
        1 if reduce_acc_via_add else 0,  # 10 REDUCE_ACC_VIA_ADD (reduce datapath, D7)
        scaler_tiles,  # 11 tiles the reader pushed into cb_scaler
        1 if combine else 0,  # 12 COMBINE: partial sum -> gather -> mcast-back
        plan.group_size,  # 13 cores per width group (GRID_W)
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    x_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    g_addr = gamma.buffer_address() if has_gamma else 0
    for w in assignment:
        core = w.core
        # owns_last_w: only the core holding the row's LAST width tile applies the
        # partial-W scaler / mask.  On the whole-row schemes that is every core; on
        # the BAND scheme PARTIAL_W is 0, so the flag is inert either way.
        owns_last_w = 1 if (w.w_start + w.w_real >= Wt) else 0
        # The stick / element extents are what the ROW_MAJOR dataflow addresses in;
        # on the tile-axis schemes they are the DERIVED view of the same slice
        # (_work_tile_axis), so passing both cannot drift.
        band_rt = [w.stick_base, w.stick_count, w.w_off_elems, w.w_real_elems]
        reader_rt[core.x][core.y] = [x_addr, g_addr, w.row_start, w.row_count, w.w_start, w.w_real] + band_rt
        writer_rt[core.x][core.y] = (
            [out_addr, w.row_start, w.row_count, w.w_start, 1 if w.is_root else 0, w.slot]
            + band_rt
            + (list(plan.mcast.runtime_args(core)) if combine else [])
        )
        compute_rt[core.x][core.y] = [w.row_count, owns_last_w, 1 if w.is_root else 0]

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

    # The mcast helper owns its two handshake semaphores; the partial-gather needs
    # one more (the root's arrival counter), taken from the next free id so the
    # two families can never collide.
    semaphores = []
    if combine:
        semaphores = list(plan.mcast.owned_semaphores())
        semaphores.append(ttnn.SemaphoreDescriptor(id=plan.gather_sem_id, core_ranges=all_cores, initial_value=0))

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=cbs,
    )
