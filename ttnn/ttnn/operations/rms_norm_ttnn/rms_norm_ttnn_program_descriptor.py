# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for rms_norm_ttnn.

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
    DEST_ACC_SQUARE_MAX_WT  largest WT_CHUNK at which pass A's square folds the
                            width tiles into DEST instead of packing every x^2
                            tile out to cb_x_squared (Lamp L6d; see D12)
    GATHER_FACES            faces per fp32 partial tile the cross-core combine's
                            GATHER ships member -> root on the BLOCK_ROWS == 1
                            branch (see D13, scoped by D27)
    CB_COMBINE_FLAT_DEPTH   page depth of the combine CBs that carry ONE compact
                            tile per round and are therefore FLAT in BLOCK_ROWS
                            (see D27)
    COMBINE_TREE_F0_MIN / _MAX
                            the level-0 fan-in BAND the slot tree's arity is
                            derived in -- f0 is the largest DIVISOR of
                            GROUP_SIZE in it that both tree gates admit
                            (Refinement 1 lever 1; measured table at the
                            constants)
    COMBINE_TREE_MIN_DELETED_FOLD_TILES
                            the tree's cost gate, on root fold-tiles deleted
                            per round (see D28)
    COMBINE_NOC_RESIDENT / COMBINE_NOC_STREAMED
                            which NoC carries the combine, chosen per plan on
                            `native_in` -- NOC_0 (with a reader/writer SWAP)
                            when x is a resident shard, NOC_1 when the reader
                            still streams x from DRAM (Refinement 1 lever 2)
    CB_R_DEPTH              ring depth of a STREAMED cb_residual_tiles.
                            0 = follow CB_X_DEPTH, which is byte-identical to
                            Phase 0 and is where it is parked (Refinement 1
                            lever 4; measured null)
    ROW_RESIDENT_MIN_ROWS_PER_CORE
                            tile-rows a core must own before the ROW_RESIDENT
                            regime is taken at a SHALLOWER depth than STREAM
                            would use (Lamp L5; see D14)

  derived buffer depths
    CB_X_DEPTH / CB_OUT_DEPTH   the depth the regime search settled on; forced
                            to 1 on the ROW_MAJOR path, where the producer /
                            consumer is a sequential compute helper (tilize /
                            untilize) and depth buys no overlap

  derived helpers (one source of truth each; both L1 solves call them)
    _cb_block_mult()        which CBs scale with BLOCK_ROWS * WT_CHUNK, and at what
                            depth -- never re-spelled inline
    _residual_depth()       the residual ring's depth (CB_R_DEPTH or CB_X_DEPTH),
                            read by both L1 solves AND the CB table
    _combine_tree_arity()   (f0, f1) or None -- THE one place the tree is decided
    _combine_noc()          the combine's NoC, read by the host mcast wire AND by
                            both data-movement kernel configs
    scaler_pages            page count of cb_scaler (2 when PARTIAL_W, else 1)
    reduce_acc_via_add      the chosen reduce datapath (D7); also decides what the
                            reader fills cb_scaler with
    scaler_tiles            tiles the reader actually pushes into cb_scaler and the
                            compute pops (<= scaler_pages)

  derived block factors
    BLOCK_ROWS   tile-rows per compute block = min(per-core assignment,
                 the coarsest chunk that fits the L1 budget)
    WT_CHUNK     width tiles per compute block = Wt in the RESIDENT regime; the
                 coarsest chunk that fits L1 in ROW_RESIDENT / STREAM -- a DIVISOR
                 of Wt (D1) unless the ragged/padded split is coarser (D33)
    NUM_W_CHUNKS = ceil(Wt / WT_CHUNK).  NUM_W_CHUNKS * WT_CHUNK is the PADDED
                 width; the excess (WT_PAD) is pad tiles the core does not own
    WT_PAD       = NUM_W_CHUNKS * WT_CHUNK - wt_per_core (D33).  0 on every
                 divisor build.  The reader zeroes those tiles so they add exactly
                 0 to sum(t^2); the writer never ships them
    X_RESIDENT   whether cb_input_tiles / cb_gamma_tiles are HELD across pass A
                 and pass B.  Since Refinement 4 this is DECOUPLED from
                 NUM_W_CHUNKS == 1 (that decoupling is the third regime -- D14)
    x_hold_wt    width tiles the two held CBs span: the PADDED per-core width
                 (NUM_W_CHUNKS * WT_CHUNK) when X_RESIDENT, else WT_CHUNK.  ONE
                 source of truth for both CB sizes and the kernels' final pops

Deviations from op_design.md section 1.4 (advisory: CB sizing / knob selection;
the scheme, topology, work split and helper mapping are unchanged).  D1..D28 are
the SEED's, preserved verbatim; D29..D33 at the end are this op's own, and each
is priced in l1_ledger.md's "Deviations" table with its measurement:

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

      SUPERSEDED BY D33 for tile-aligned widths.  The divisor clamp is a
      GRANULARITY CLIFF at a prime Wt -- 127 has no divisor between 1 and 127, so
      any cap below the whole row collapses the chunk to ONE tile.  D33 keeps all
      three mechanisms above (the chunk stays UNIFORM) by padding the last chunk
      instead of raggedizing it.  D1 still governs a non-tile-aligned width, where
      the reduce's partial scaler / 0-1 mask is aimed at the last tile of the block
      and a pad tile there would silently drop the mask.
  D2  The STREAM chunk-size solve counts the ROW_MAJOR staging CBs at
      WT_CHUNK tiles (what is actually allocated), not at Wt.
  D3  RESOLVED by Refinement 1b.  accumulate_reduce_block() used not to expose
      reduce()'s ReduceFp32Mode / ReduceAlgorithm template slots; both are now
      forwarded (streaming_reduce_helpers.hpp), which is what made D7 possible.
      The op still passes ReduceFp32Mode::Fast: Accurate only routes *Float32*
      SUM through the SFPU, and the wide-W precision cell this op cares about is
      bfloat16 -- D7 is the lever that reaches it.  fp32 DEST accumulation still
      comes from fp32_dest_acc_en=True.

      MEASURED by the Phase-0 verifier, and the result is stronger than the
      argument above -- Accurate is not merely unhelpful here, it is UNSAFE at
      this op's own default.  A/B by flipping the single ReduceFp32Mode template
      argument at the accumulate_reduce_block() call site, float32 input, TILE,
      HiFi4/approx, rel-RMS against an fp32 torch reference:

        distribution / shape          Fast dest16  Acc dest16   Fast dest32  Acc dest32
        positive_only (1,1,32,64)       0.00459      **inf**      0.00068     0.00068
        positive_only (1,1,128,256)     0.00537      **inf**      0.00076     0.00076
        negative_only (1,1,32,64)       0.00454      **inf**      0.00068     0.00068
        randn         (1,1,128,256)     0.00424      **inf**      0.00076     0.00076
        bfloat16 control                0.00357     unchanged (the flag only touches Float32)

      Two facts, and each on its own settles it:
        * at fp32_dest_acc_en=False -- the op's OWN DEFAULT (A7) -- the SFPU
          reduce path returns inf/NaN (pcc nan).  The path needs a 32-bit DEST;
          the 16-bit one corrupts it.  So Accurate could never be unconditional
          here, only predicated on fp32_dest_acc_en.
        * at fp32_dest_acc_en=True, where it DOES work, it is bit-for-bit as
          accurate as Fast to 5 significant figures.  The residual error there
          (~5.3e-4 ratio std) is tf32's 2^-11 = 4.9e-4 mantissa step, but
          removing the FPU's tf32 truncation buys nothing because the reduce's
          *accumulation* -- not its input rounding -- sets the error.
      So there is no predicate under which Accurate is worth taking, and Fast is
      not a default left unexamined.  Do not re-propose it without re-running the
      A/B above; the inf at dest16 is silent (no assert, no hang).
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
      isolated reduce bake-off shows, and that gap is the finding: rms_norm_ttnn is
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
      should always take, since rms_norm_ttnn is dataflow-bound at those widths (see
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
  D12 Refinement 4 (perf), Lamp L6d.  Pass A's `square` folds the chunk's width
      tiles straight into DEST (DestAccumulation::PerRow) instead of packing every
      x^2 tile out to cb_x_squared for the reduce to read back, once
      WT_CHUNK <= DEST_ACC_SQUARE_MAX_WT.  cb_x_squared then holds ONE tile per
      tile-row (`x_squared_wt`) and the reduce's per-call width is 1, so the pass
      loses WT_CHUNK-1 packs and the matching unpacks per tile-row.

      A CEILING, not a floor, and that is the whole subtlety: the fold accumulates
      SERIALLY over the chunk's width tiles inside a DEST register that is 16-bit at
      fp32_dest_acc_en=False, where the reduce's own AccumulateViaAdd datapath
      accumulates PAIRWISE (Refinement 1b).  Bounding WT_CHUNK bounds that depth; the
      cross-chunk carry still goes through the fp32 cb_row_stat either way.  The
      shipped 8 covers every sharded `_perf_case` geometry (WT_CHUNK 4..8) and leaves
      the prefill profiles (32..80) on the packed path.

      Gated on PARTIAL_W == 0: the fold folds the row's last width tile INCLUDING its
      pad lanes before the reduce runs, so the reduce's partial scaler / 0-1 mask can
      no longer reach them.  The BAND scheme keeps the fold because it passes
      kernel_partial_w == 0 and zeroes its staging ring, so its pad lanes are an
      exact 0 (D10).

      MEASURED alone (blackhole p150b, bf16 / HiFi2 / fp32_dest_acc_en=False):
      (1,1,8192,1024) BLOCK-sharded 64c 102173 -> 98989 ns (1.03x), the four pinned
      WIDTH-shard geometries 1.02x-1.03x.  Small, because those geometries turned out
      NOT to be compute-throughput-bound the way the tile-op count suggested -- see
      D15 for what actually dominated them.
  D13 SCOPED by Perf 3 / D27, not retired.  GATHER_FACES selects how many of a partial
      tile's four 16x16 faces the member -> root gather ships (2 = the pair that can hold
      a REDUCE_ROW column vector, half the bytes).  D27 replaces the layout wherever a
      block has more than one tile-row -- a member PERMUTES its partials into COLUMNS of
      ONE tile and ships that tile WHOLE, because the receiver's un-permute matmul turns
      an un-written column into NaN -- but at block_rows == 1 the permute is the identity
      and the face-run gather survives, MEASURED: whole-tile ship there regressed the four
      pinned WIDTH-shard geometries 0.86x-0.98x, monotone in group_size.  Two D13 findings
      remain invariants either way and are recorded where they bind
      (rms_norm_ttnn_writer.cpp): zeroing the whole gather CB at boot is a RACE against an
      already-landed member (measured pcc 0.87-0.99), and the gather was never
      byte-bound at these group sizes -- halving its bytes moved the 64-core BLOCK
      shard ~5%, which is why D27 goes after the TRANSACTION COUNT and the root's
      per-row serialization instead.

  D27 Perf 3 (perf) -- THE COMPACT PARTIAL TRANSPOSE.  The combine's per-round unit
      stops being "BLOCK_ROWS column-shaped tiles" and becomes "ONE tile whose columns
      0..BLOCK_ROWS-1 are the block's stats".  Every core matmul-permutes its own
      partials into that shape before shipping and un-permutes the multicast stat back
      afterwards (rms_norm_ttnn_compute.cpp), against a one-hot bank the READER synthesizes
      in L1 (rms_norm_ttnn_reader.cpp `reader_bank_boot`).  What it changes here:
        * cb_partials_gathered  GATHER_SLOTS * block_rows pages -> GATHER_SLOTS.  The
          GROUP_SIZE x BLOCK_ROWS term LEAVES the L1-bound block_rows solve, which is
          what makes a coarse block (and at the focus geometry ONE round) expressible
          at all -- the flat ring is 1152 kB at block_rows 32 / group_size 8.
        * cb_stat_handoff / cb_compact_handoff / cb_mcast_in are CB_COMBINE_FLAT_DEPTH
          pages each, flat in block_rows.
        * cb_bank is a NEW bf16 CB of block_rows pages, device-generated.
        * cb_row_stat is NOT ALLOCATED on the combine path.  D22 already left it
          strictly dead there (the fused fold accumulates in DEST and packs
          cb_stat_handoff; CB_REDUCE_ACC is cb_sum_handoff and CB_STAT_B is
          cb_row_final), so its CB_ROW_STAT_DEPTH * block_rows fp32 pages were pure
          waste -- 256 kB/core at block_rows 32.
      The measured justification lives at the kernel sites; the L1 arithmetic is at
      `_solve_blocking`'s f32 page terms.
  D28 Perf 3 (perf) -- THE COMBINE'S SLOT TREE, at WIDE GROUPS ONLY.  The flat combine
      has exactly ONE gatherer: the other group_size - 1 members all write into the
      root's L1 and the root folds every page itself, so BOTH per-group_size terms land
      on one core while the rest of the group idles.  The tree interposes one level:
      contiguous runs of f0 (the measured `COMBINE_TREE_F0_MIN.._MAX` band) slots are folded IN PARALLEL by
      f1 = ceil(group_size / f0) different cores, which forward only their RAW sums (no
      finalize -- only the last level rsqrts), and the root folds f1 of those.  Root fold
      and root ingress both drop from group_size to max(f0, f1); the price is ONE extra
      NoC hop per round.  Two levels, full stop -- 3 and 4 levels lost at 6 of 7 cells in
      the isolated bench.  What it changes here: CB_PARTIALS_GATHERED becomes the level-0
      ring (f0 pages, on every core), plus a new CB_GATHER_L1 (f1 pages, the root's) and
      CB_NODE_OUT; ONE ARRIVAL SEMAPHORE PER LEVEL (a level-1 sender can legally arrive
      before one of the root's own level-0 members, so a single cumulative counter would
      satisfy the level-0 wait with a slot that has not landed); and ONE extra runtime
      arg, the level-0 parent's virtual coords (level 1's parent is slot 0, which IS the
      multicast sender, so the mcast helper already carries it).  L1 goes DOWN -- 32 -> 14
      fp32 pages at group_size 32 -- so it never competes with the block_rows solve.
      The multicast is untouched (f0 * f1 >= group_size makes slot 0 the unique last-level
      gatherer).  GATED on `_combine_tree_arity`, which is a threshold on the DERIVED
      quantity the mechanism is about -- the fold tiles the tree deletes from the root's
      critical path -- and is blind to shape, dtype and placement; the whole-op A/B that
      brackets it (deleted 17 = 0.972x, deleted 18 = 1.028x) is at the constants.
  D14 Refinement 4 (perf), Lamp L5 -- the op's THIRD compute regime, ROW_RESIDENT.
      X_RESIDENT is now an EXPLICIT flag instead of `num_w_chunks == 1`, and that
      decoupling IS the regime:

        RESIDENT      X_RESIDENT, NUM_W_CHUNKS == 1   whole row in one chunk
        ROW_RESIDENT  X_RESIDENT, NUM_W_CHUNKS >  1   ONE whole tile-row of x and
                                                      the whole row of gamma held,
                                                      only the DERIVED CBs chunked
        STREAM        !X_RESIDENT                     x re-read in pass B

      WHY IT IS THE PREFILL LEVER.  STREAM pays TWICE over: x is read once per pass
      (R10), and gamma -- chunked and not held -- is re-read for every pass-B chunk of
      every row-block, which on a prefill profile is as many DRAM bytes as x itself.
      On (1,1,8192,7168) that is 117 MB (x, pass A) + 117 (x, pass B) + 118 (gamma) +
      117 (out) = 470 MB, measured at 1043918 ns == 450 GB/s aggregate, i.e. already
      at the part's DRAM roofline: the only thing left to improve was the byte count.
      ROW_RESIDENT moves 117 + 50 + 117 = 285 MB.

      HOW, with no second code path.  Every helper call still works on one WT_CHUNK;
      the two held CBs are simply indexed at a TILE OFFSET (`TileOffset::Set`,
      eltwise_chain.hpp:311 -- base = c * WT_CHUNK, folded away to 0 when the offset
      mode is Unset) and popped ONCE per row-block by an explicit cb_pop_front, the
      same sanctioned pattern cb_row_stat / cb_scaler already use.  `x_hold_wt` is
      the one source of truth for their width.  BLOCK_ROWS is 1 here, which is why a
      flat `Set` base suffices rather than a `Strided` range.

      MEASURED (blackhole p150b, 110-core grid, bf16 / TILE / HiFi2 /
      fp32_dest_acc_en=False -- the `_perf_case` config; one fresh-cache profiled run
      per variant; test_rms_norm_perf.py::test_rms_norm_perf_r4target):

        shape              STREAM      ROW_RESIDENT   speedup   WT_CHUNK
        (1,1,8192,5120)    753345 ns     468487 ns     1.61x     80 -> 32 (2->5 chunks)
        (1,1,8192,7168)   1043918 ns     655687 ns     1.59x     56 -> 56 (4 chunks)

      GATED ON THE DEPTH SACRIFICE, not on L5.  Holding a whole tile-row of x can
      leave no L1 for the depth the two cross-processor CBs were spending on
      movement<->compute overlap.  At the SAME depth L5 is pure profit; only when it
      forces a shallower one does it cost anything, and then only when the core has a
      single row-block -- nothing left to overlap.  Both halves are measured:
      (1,1,32,7168) at GRID_W=1 (one core, depth 2 -> 1) went 41779 -> 50598 ns
      (0.83x) and is now correctly declined, while ROW_MAJOR (1,1,32,4096) -- already
      depth 1 in BOTH regimes, so no sacrifice -- takes L5 even at one row-block and
      goes 52197 -> 47144 ns (1.11x).  Hence ROW_RESIDENT_MIN_ROWS_PER_CORE, applied
      only to the depth-sacrificing case.

      Not reachable with a cross-core width split active: that plan is SCHEME_SHARD_W,
      which never enters the chunked branches (the writer static_asserts
      NUM_W_CHUNKS == 1 for a combine, and the descriptor asserts it too).

      WHAT IT DOES NOT FIX: the prime-`Wt` cliff.  D1 still forces WT_CHUNK | Wt, so
      Wt = 127 collapses to one tile per chunk; L5 removes that shape's pass-B
      RE-READ but not its 127 one-tile compute phases.  A ragged-tail chunk (runtime
      wt_c) is still the lever there.
  D20-D25 PERF 2 (a fan-out perf tournament; SUPPORTED and EXCLUSIONS untouched).  Each
      is documented in full, with its measured authorisation, at its own site -- listed
      here only so the set is discoverable.  Focus shape (1,1,8192,1024) BLOCK_SHARDED
      64c: 64677 -> 34438 ns, 1.878x, and no supported cell got slower.

      D20 the reduce datapath's THIRD floor, on the reduce's actual PER-CALL width
          (`REDUCE_ACC_VIA_ADD_MIN_CALL_WT` vs `x_squared_wt`).  Spelled as a narrow
          carve-out BELOW D7/D8's two floors, both of which stay load-bearing -- each
          direction of getting that wrong is a measured rel-RMS regression, recorded at
          the predicate.  Pass A 11551 -> 6079 ns (1.90x) at BETTER rel-RMS.
      D21 pass B's DEST-LANE BLOCK SIZE (`PASS_B_BLK`, a divisor of WT_CHUNK capped by
          `DEST_AUTO_LIMIT`) plus the PerChunk pack lifecycle it requires.  14050 ->
          8860 ns (1.59x), BITWISE identical.  The assigned fusion of pass B's two
          multiplies was measured a REGRESSION (0.84x) and is NOT here.
      D22 the FUSED ROOT CHAIN: the group fold accumulates PAIRWISE IN DEST and the
          finalize runs in that same DEST window, one pack.  Replaces D16's
          packer-L1-accumulation AND D19's separate finalize pass; `ROOT_FOLD_OUT` and
          every COMBINE-path use of cb_row_stat are deleted.  Stage pair 5874 -> 2698 ns
          (2.18x) and MORE accurate than the chain it replaces (rel-RMS 2.42e-3 vs
          3.38e-3), which refutes D16's recorded reasoning.  Needs `GATHER_SLOTS`
          (group_size rounded up to even) so the pairwise walk is universal.
      D23 TILE-gamma read granularity (`gamma_trim`): fetch only the face-rows pass B's
          BroadcastDim::Row consumer reads.  Interleaved prefill 1.107-1.182x, BIT-EXACT.
          bfloat8_b demotes to a half-page read (272-byte face, not 64-B aligned) -- a
          format fact, not a shape guard.
      D24 the root publishes its OWN stat copy BEFORE the broadcast, so its pass B no
          longer waits out its own multicast (-2643 ns on the root).
      D25 the COMBINE PIPELINE: block blk+1's pass A is issued before block blk's
          combine, filling the root's measured ~3400 ns/round arrival idle, plus
          cb_sum_handoff at depth 2.  Carved out to `native_in` -- on a reader-fed input
          ring the pipeline is INCORRECT (a tile offset cannot cross a ring wrap) and,
          once made correct, still 0.894x.  The gather ring stays at ONE round: deepening
          it was a measured regression twice over.
  D15 Refinement 4 (perf), Lamp L6b.  The finalize's rsqrt is scoped to the tile
      faces pass B reads back (RSQRT_COL_SCOPE).  cb_row_stat is a REDUCE_ROW result
      whose ONLY consumer is mul<BroadcastDim::Col>, i.e. tile column 0, which in the
      2x2-faces-of-16x16 layout lives in faces 0 and 2 == `VectorMode::C` -- the same
      two faces D13 independently established are the only ones the gather has to
      ship.  So the SFPU's 8-iteration rsqrt walks half the datums and the other half
      keeps the (never-read) pre-rsqrt value.

      This is the one raw-LLK addition: `rsqrt_tile` hard-codes VectorMode::RC and
      exposes no seam (see the substitution note at the head of rms_norm_ttnn_compute.cpp).

      PERF 1 RETIRED THE `RSQRT_COL_SCOPE` KNOB.  D17 extends the scope to the WHOLE
      finalize chain and makes it the only path, so the whole-tile spelling and its
      selector are gone (compute CT arg 16 with them).  Nothing needed the unscoped
      form: it was the slowest cell measured at every geometry, and an isolated bench
      ran pass B's real consumer over a stat tile with columns 1..31 poisoned five
      orders of magnitude wrong and still passed the PCC gate -- the lanes the whole-tile
      path bothered to finalize are provably never read.  The A/B table below is kept as
      the measurement that justified the scope in the first place.

      MEASURED (same config, A/B on RSQRT_COL_SCOPE):

        shape                          RC (whole)   C (scoped)   speedup
        (1,1,8192,1024) BLOCK 64c        94297 ns     82474 ns    1.14x
        (1,1,32,1024)   WIDTH  8c         5863 ns      5470 ns    1.07x
        (1,1,32,2304)   WIDTH  9c         6759 ns      6350 ns    1.06x
        (1,1,32,5120)   WIDTH 32c        11460 ns     11042 ns    1.04x
        (1,1,32,7168)   WIDTH 28c        10897 ns     10581 ns    1.03x
        prefill (interleaved)                      within noise

      THIS is what the sharded geometries were actually spending their time on, and
      it is the finding D12/D13 were looking for: the combine's ROOT runs one
      transform_in_place per tile-row per round (32 of them on the 64-core BLOCK
      shard) and every member waits on it, so the finalize sits on the critical path
      with a GROUP_SIZE-wide fan-out of waiters behind it.  A per-tile SFPU cost that
      is invisible in a tile-op count dominated the geometry.

  D9  Refinement 2's placement layer.  The three SCHEME_* values, the zero-copy
      (shard-backed) cb_input_tiles / cb_output_tiles, the cross-core width
      combine's four CBs and L1_CB_ARENA_BASE_RESERVE are all documented at
      their definitions above; the one thing worth stating here is what stayed
      the same.  The COMPUTE kernel's phase sequence, every helper call, the
      block/depth knob set and the L1 predicate are unchanged -- a sharded build
      differs only in (a) who fills cb_input_tiles, (b) which CB pass B reads the
      stat from, and (c) whether the finalize runs locally or on a group root.
  D29 cb_normalized is allocated whenever HAS_GAMMA | HAS_BIAS -- NOT "only when
      !HAS_RESIDUAL" as op_design.md's CB table says.  That row wants pass B to
      pack back into cb_x_sum; cb_x_sum is pass B's HELD Upfront/None srcA, and
      an in-place chain requires an incrementally-POPPING input
      (chain.inl:82-85, inplace_chain.cpp:5-21), so the alias would DEADLOCK
      rather than return a wrong answer.  The design's own Key-Risks row states
      that cb_x_sum is never aliased, so the two statements contradict each other
      and this follows the risk row.
      Its DEPTH (_norm_cb_depth) is 2 only when BOTH post-normalize stages are
      present AND BLOCK_ROWS > 1: the in-place scale rotates the ring by
      rows*WT_CHUNK, a whole revolution for a FULL block but not for the partial
      final one, and the bias stage's bulk cb_wait_front + linear tile indexing
      would then read past the ring end.  That is D6's hazard on a different CB
      and it takes D6's fix.  At BLOCK_ROWS == 1 no block is ever partial, so
      depth 1 is exactly correct -- which is every ROW_RESIDENT / STREAM / BAND /
      width-shard build, i.e. the L1-tightest regimes get the cheaper ring.
  D30 a ROW_MAJOR per-channel operand's STAGING ring may be CB_RM_STAGE_DEPTH
      pages instead of WT_CHUNK, consumed as tilize<1>(WT_CHUNK) rather than
      tilize<WT_CHUNK>(1).  Same tiles BIT-FOR-BIT -- tile column j of a wide
      block and block j of a one-wide walk are the same 32 elements laid out the
      same way -- at 1/WT_CHUNK of the L1, for WT_CHUNK LLK block calls instead
      of one (paid once per core in the resident regimes).
      MEASURED NEED, not a tidy-up: (128, 8192) fp32 ROW_MAJOR BLOCK_SHARDED
      with gamma_bias_residual (a 13x748 shard on an 11x10 grid, WT_CHUNK = 25)
      built 1 406 976 B of CBs against a 1 344 512 B ceiling -- a hard launch
      failure on 2 golden cells.  A per-channel operand is ONE stick, and the
      wide ring reserves 25 whole fp32 tiles (100 kB) to carry 3 200 B of it,
      twice over.  Taken only when the budget asks, so every build that already
      fit is byte-identical.
  D31 CB_RM_STAGE_DEPTH is SEARCHED (2, 1) on the BAND scheme, ordered BEFORE
      D30 because a band's activation reads come from the core's OWN L1 -- the
      cheapest overlap in the op to sacrifice.  Same cause as D30:
      op_design.md gives the band no L1 fallback at all ("take the finest block
      and let metal's own CB-region check be the arbiter"), which held at the
      seed's two operands and does not hold with three activations plus
      cb_x_sum plus a second per-channel operand.
  D32 the D25 combine pipeline (PIPE_A) is gated OFF when HAS_RESIDUAL.  The
      hoisted pass A for block blk+1 would write cb_x_sum, whose ring is ONE
      block deep and whose front block blk's pass B still owns -- so the hoist
      would either overwrite live data or self-deadlock on the reserve.  Making
      it legal needs cb_x_sum at depth 2 AND a runtime tile base on the PACK,
      and output(...) carries no tile base, so it is not expressible without a
      new chain seam.  Recorded as a follow-up with the measurement to take (
      cb_x_sum at depth 2 with a pack-side base, against the serial order),
      never as a finished trade.  Every build without a residual is unchanged.
  D33 Refinement 4 -- THE RAGGED (PADDED) WIDTH CHUNK, which removes D1's prime-Wt
      granularity cliff.  `_width_chunk` takes the coarsest BALANCED chunk
      `ceil(wt_core / ceil(wt_core / cap))` and pads the last one out to it, so
      every chunk -- and therefore every CB ring, every helper block width and
      every batched NoC group -- stays UNIFORM and all three D1 mechanisms hold
      unchanged.  The compute kernel is byte-for-byte untouched: `X_HOLD_WT` was
      already `WT_CHUNK * NUM_W_CHUNKS`.

      WHO OWNS THE PAD.  The reader: pad tiles are never read from the tensor and
      pass A sums x^2 over the whole padded chunk, so they must be exactly ZERO,
      not merely finite.  It zeroes them with the device zero API on the pages it
      just reserved -- the same mechanism as `publish_native_shard`'s ragged-SHARD
      pad and the RM rings' boot zero.  The writer skips them with the predicate it
      already had (`wt < WT`) on the TILE path, and writes the tail chunk's real
      bytes at the PADDED stride on the ROW_MAJOR one.  The pad is bounded by
      construction: pad < NUM_W_CHUNKS, i.e. under 1/cap of the width (ONE tile in
      128 at Wt = 127).

      GATED on `kernel_partial_w == 0` (see D1's supersession note) and, on the L5
      path, RE-CAPPED against its own pad: the held CBs span the padded row, so the
      first candidate at Wt = 127 can miss by one tile of L1 -- and the divisor
      below it is 1, the very cliff -- so the solve shrinks the cap and retries
      rather than falling straight back.  `RAGGED_WIDTH_CHUNK = 0` restores D1
      exactly.

      MEASURED (blackhole p150b 1350 MHz, ragged vs the divisor clamp, min over 2
      reps of median-of-5):
        shape                       layout  mode                 divisor   ragged
        (1,1,32,4064)   Wt=127      RM      gamma_bias_residual   590592    56486  10.46x
        (1,1,32,4064)   Wt=127      RM      gamma                 376319    37950   9.92x
        (1,1,3104,4064) Wt=127      RM      gamma_bias_residual  1991000   216590   9.19x
        (1,1,3104,2848) Wt=89       RM      gamma                 805205    98090   8.21x
        (1,1,3104,4064) Wt=127      RM      gamma                1145067   138309   8.28x
        (1,1,32,4064)   Wt=127      TILE    no_gamma              101279    16616   6.10x
        (1,1,32,4064)   Wt=127      TILE    gamma                 148493    33278   4.46x
        (1,1,3104,4064) Wt=127      TILE    gamma                 213058   150576   1.42x
        (1,1,3104,4064) Wt=127      TILE    gamma_bias_residual   329830   246944   1.34x
        (1,1,1024,16384) Wt=512     TILE    gamma_bias_residual   525549   498670   1.05x
      The last row is a NON-prime width: 512's coarsest divisor under the cap (32)
      is not its coarsest FITTING chunk (57), so D33 pays there too.  pcc IMPROVES
      on every one of them (0.99987 -> 0.99999 at Wt=127): a coarse chunk clears
      D7/D8's reduce-datapath floors, which is the precision lever Refinement 1
      built.  Fourteen guard cases 0.99-1.01x.
  D34 Refinement 4b -- THE PER-STAGE DEVICE ZONES ARE OPT-IN (`STAGE_ZONES` ->
      the `RMS_STAGE_ZONES` kernel define, default OFF).  NOT a perf trade: the
      profiler keys a zone by a 16-BIT hash of "<name>,<abs path>,<line>", so the
      41 zone sites this op carries are 41 entries in a 65 536-slot table that
      EVERY graded run populates -- because the eval runner sets
      TT_METAL_DEVICE_PROFILER=1 to read the op-level duration off the firmware
      markers, which compiles user zones in as a side effect.  Two colliding
      entries are a hard TT_THROW on every profiler read and a `terminate` at the
      next device open; D33's kernel edits moved `writer_tree_forward` onto
      writer.cpp:662, which collides with `compute_scale`@compute.cpp:1680 at
      0x0773, and that -- not the ragged chunk -- is what zeroed the Refinement 4
      golden run.  Gating the zones off by default takes the op's contribution to
      that table to ZERO on the graded path while leaving every zone in the source
      and one env var away.  `test_rms_norm_ttnn_zone_hashes.py` pins the ON build.
  D35 Perf 1 (perf) -- THE READER PUBLISHES THE RESIDENT SHARD FIRST.  The whole
      change is a statement move in rms_norm_ttnn_reader.cpp: the `NATIVE_X`
      `publish_native_shard` block now sits at the TOP of the kernel, above
      `reader_scaler_boot` / `reader_bank_boot` / `stage_per_channel`, instead of
      below all three.  It is a pure CB hand-off of data already resident in this
      core's L1 and it is what `compute_square`'s `cb_wait_front` blocks on, so
      running it last made the compute kernel idle through a ~1 us per-channel
      DRAM read that nothing needs until pass B.  MEASURED on the pinned
      `(1,1,32,7168)` WIDTH `[32,256]` `(7,4)` 28-core case: reader_native_publish
      END 2358 -> 616 ns, compute_reduce END 2740 -> 1289, writer_gather_ship END
      3166 -> 1936, whole op 5335 -> 4121 = 1.295x.  1.09x-1.46x across every
      other native-shard geometry, 1.02x-1.05x interleaved, flat where there is no
      shard to publish (STREAM / BAND / ragged-Wt).  NO precision knob is touched
      and the output is bit-identical.
      THE HAZARD IT EXPOSED, and the fix, because a later edit must not undo it:
      `cb_scaler` and `cb_bank` are reader-synthesized constants that the compute
      kernel reads with NO wait of its own -- the reduce helper's contract puts
      that on the caller (reduce_helpers_compute.hpp:37) and `matmul_tiles` has no
      CB lifecycle at all -- so their availability was ordered ONLY by the publish
      running last.  Hoisting it deleted that ordering and produced a
      NON-DETERMINISTIC corruption on the COMPACT combine path (`1x1x2048x256` and
      `4x1x512x512` BLOCK_SHARDED at pcc 0.08-0.12 / rel-RMS 1e5, a different set
      of cells on every run).  The op now waits both fronts EXPLICITLY, one-shot,
      AT FIRST USE -- see the `scaler_ready` / `bank_ready` flags in the compute
      kernel.  At first use and not in a joint prologue: a prologue makes
      `compute_square` (which needs neither) block on both and costs 7-8% on the
      COMPACT BLOCK shards (22001 vs 20453 ns on `(1,1,8192,1024)` BLOCK 64c).
  D36 Perf 1 (perf) -- THE GATHER SIGNALS ON *DEPARTED*, NOT *ACKNOWLEDGED*.
      Every REMOTE gather ship in rms_norm_ttnn_writer.cpp (the flat member ship,
      the tree's level-0 leaf ship and its level-1 forward) now calls
      `Noc::async_writes_flushed()` instead of `noc_async_write_barrier()` before
      bumping the gatherer's arrival semaphore.  `Semaphore::up` and
      `noc_async_write` share `NOC_UNICAST_WRITE_VC` and the same source ->
      destination path, and the NoC does not reorder same-VC traffic on one path,
      so the increment provably cannot overtake the bytes; the ACK round trip was
      dead time on the member's critical path, which IS the root's because
      `writer_gather_wait` is bounded by the slowest member.  LOCAL ships keep
      their barrier (nothing orders their visibility to the gatherer's own fold).
      MEASURED, bit-identical: isolated 28-core gather 1827 -> 1680 ns (1.087x);
      whole op 1.038x at G=28, 1.037x at G=8, 1.026x at G=9, 1.034x at G=32 (tree
      on), 1.024x at gbr/fp32_dest=True, 1.004x/1.009x on the BLOCK shards.
      Non-combine plans are byte-identical (`if constexpr (CROSS_CORE)`).
  D37 Perf 1 (perf) -- THE ROOT'S FOLD PACKS STRAIGHT INTO THE MULTICAST LANDING
      PAGE.  The last-level `combine_fold` now targets `CB_FOLD_OUT` (cb_mcast_in
      on the COMPACT path, cb_row_final on the identity one) instead of
      cb_stat_handoff, which the writer then copied 3 kB L1->L1 with an acked
      barrier purely to satisfy D24's "publish the root's own copy before the
      broadcast".  D24's REQUIREMENT is unchanged; it is now free, because the
      publish IS the pack that already existed.  `cb_stat_handoff` degenerates to a
      one-page READY TOKEN whose bytes are never read.  The writer names round
      blk's landing page from a BOOT SNAPSHOT of the ring base plus its own round
      counter -- NOT `get_read_ptr(CB_MCAST_LAND)`, which on the root is moved
      under it by the un-permute's pop once compute runs ahead (D25 pipelines
      exactly that): measured correct at 1-2 rounds and pcc 0.9106 at 3.  The
      address is derived identically on every core, so the send stays src == dst
      (the EXCLUDE-source path).  MEASURED, bit-identical: 1.013x-1.025x at G=28,
      1.024x at G=8, 1.020x at G=9, 1.017x at G=32, 1.006x/1.007x on the 2- and
      3-round BLOCK shards.  Carries `static_assert(!FIN_SPREAD)`: under the
      spread finalize the last level forwards a RAW sum, so its pack is not the
      stat.  COMBINE_FIN_SPREAD is measured off, so that guards a dead path.
  D38 Perf 1 (perf) -- A ONE-TILE-ROW BLOCK TAKES THE *SMALL* PASS-B DEST BLOCK.
      `PASS_B_AUTO` in the compute kernel: at BLOCK_ROWS == 1 the auto rule becomes
      `pass_b_blk_small` (the SMALLEST divisor of WT_CHUNK >= 2), and D21's
      largest-divisor rule is kept as a CARVE-OUT for BLOCK_ROWS > 1.  D21
      amortizes pass B's per-element init / reconfig / reserve over as many tiles
      as DEST holds, which is right when a core has many tile-rows and pass B is
      throughput-bound; at BLOCK_ROWS == 1 pass B is the TAIL after the combine's
      multicast and a smaller window lets the packer overlap the math.  The
      carve-out is earned by a MEASURED regression, not a hunch: the small block
      is 0.958x on `(1,1,8192,1024)` BLOCK 64c and 0.976x on `(1,1,7168,1024)`
      BLOCK gbr.  Where it applies it is 1.045x / 1.056x / 1.037x / 1.038x /
      1.021x on the (1,1,32,7168) / (1,1,32,2304) / (1,1,32,1024) WIDTH shards,
      the (1,1,2048,256) HEIGHT shard and the (1,1,32,7168) interleaved decode.
      Output is bit-identical everywhere (this changes only the DEST windowing).
      A caller's `subblock_w` still overrides both rules, unclamped, exactly as A5
      requires.  `>= 2` and not 1: block 1 measured 0.994x / 0.991x / 0.987x, so 1
      is a fallback for WT_CHUNK == 1 and never a choice.

  D39 Perf 2 (perf) -- THE COMPACT PER-CHANNEL HOLD, AND THE STREAM SHAPES IT
      LIFTS INTO ROW_RESIDENT.  A TILE per-channel operand is a (1,1,1,W) vector:
      31 of every 32 tile rows are PADDING, and D23's `TRIM == 2` already fetches
      only the two FACE-ROWS that carry row 0.  Those `2 * TILE_DIM * elem` bytes
      per width tile are the WHOLE of the operand's information -- 1/16 of its
      tiled size -- so the reader caches the entire row of every present operand
      in L1 once (`cb_gamma_compact` / `cb_bias_compact`) and re-materializes each
      chunk's tiles with a LOCAL L1 copy instead of a DRAM read.  Two consequences,
      and the second is the whole win:
        * STREAM stops re-reading the operands from DRAM per pass-B chunk of every
          row-block (that re-read is 230,305 ns of the target case's 1,580,377 ns);
        * `_row_resident_chunk` can price its per-channel HOLD at the compact size
          -- at W=7168 with weight+bias, 57 kB instead of 917 kB -- which is
          exactly the L1 that used to push a shape off the ROW_RESIDENT cliff into
          STREAM.  Landing in ROW_RESIDENT deletes pass B's re-read of x AND of
          the residual: 5 tensor-crossings become 3.
      MEASURED, blackhole p150b, perf case #15 `(1,1,8192,7168)` interleaved
      gamma_bias_residual fp32_dest_acc_en=True: `WT_CHUNK=56 x 4 X_RESIDENT=0`
      becomes `WT_CHUNK=32 x 7 X_RESIDENT=1 PC_COMPACT=1`, **1,578,655 -> 1,039,664
      ns, 1.517x**, pcc 0.999987.  Also 1.495x on `(1,1,8192,10240)` gbr and 1.110x
      on `(1,1,3520,16384)` gbr.
      THREE things scope it, each earned:
        * The compact hold is tried as a FALLBACK, after the tiled hold, at every
          depth -- coarsest-first on what each step costs, the same discipline the
          BAND search uses.  A shape that already fits ROW_RESIDENT tiled keeps its
          shipped program byte-identical, because taking the compact form there
          measured 0.84-0.94x (the per-chunk local expand sits on pass B's
          `Upfront` wait with no DRAM traffic to hide behind).
        * `ROW_RESIDENT_COMPACT_MIN_GRID_FRACTION` -- ROW_RESIDENT trades DRAM
          BYTES for reader/compute overlap, and that trade only pays when DRAM is
          the constraint.  Holding W, dtype, residual, rows-per-core AND the solved
          chunk fixed and moving only the active-core count: 32/110 cores 0.66x,
          64/110 0.97x, 96/110 1.09x, 110/110 1.11x.  Monotone in grid occupancy,
          so the gate is occupancy and the constant is no tighter than the
          58%-loses / 87%-wins bracket it was measured in.
        * `pc_compact_ok` requires the face-row form to be legal on EVERY present
          operand.  A block-float operand's 272-byte face is not 64-byte DRAM
          aligned (D23 already demotes it to the half page), so it has no compact
          form at all; a ROW_MAJOR operand already arrives compact -- it IS a stick
          -- and goes through `cb_*_sticks` + tilize.  Both keep the shipped path.
      The cache fill is LAZY (chunk c filled the first time chunk c is staged), not
      eager: an eager boot fill puts a device-wide burst of 110 x 896 tiny reads in
      front of the first x read (profiled at 290,385 ns/core mean, 653,014 ns max --
      pure arbitration spread) where the lazy fill issues the identical reads into
      the hole where the reader is already blocked on pass B.  1,101,601 -> 1,048,060 ns.

  D40 Perf 2 (perf) -- THE PER-CHANNEL BROADCAST.  Closes the Blocking Model's
      deferred `GAMMA_MCAST` regime row.  `weight` / `bias` do not vary along the
      ROW axis, so every core owning the same width slice reads the IDENTICAL DRAM
      bytes.  On the focus shape that is ALL 110 cores hitting the SAME 72 gamma
      pages -- 144 x 64-byte reads each (D23's TRIM) against a 72-page working set,
      a hot-bank storm that moves 1.0 MB of real traffic for 16,138 ns, 8.4% of the
      op, at ~63 GB/s effective (~7x off the op's own payload rate).
      `reader_read_gamma` profiles at 67,563 ns/core MEAN and 158,451 ns MAX -- a
      2.3x spread that is pure DRAM arbitration.  One injector per grid ROW now
      reads the block and multicasts it (`SenderPipe` / `ReceiverPipe` over
      `ttnn.Mcast1D(PerRow)`).  MEASURED, blackhole p150b, whole-op:
          FOCUS (1,1,8192,2304) INT gamma        192,455 -> 179,342 ns   1.073x
          (1,1,8192,2304) INT gamma+bias         213,759 -> 194,924      1.097x
          (1,1,8192,5120) INT gbr fp32_dest      649,237 -> 608,816      1.066x
          (1,1,8192,7168) INT gamma              575,877 -> 557,880      1.032x
          (1,1,8192,1024) INT gamma               86,541 ->  84,906      1.019x
      The focus number is 1.9% off the no-operand ABLATION (176,013 ns), i.e. this
      recovers ~73% of the operand's entire whole-op cost.
      FOUR decisions, each measured, none inherited:
        * ROW lines, not COLUMN lines.  Perf 1's ancestor used columns; under the
          deferred wait columns REGRESS (0.968-0.988x), and on the STREAM geometry
          a column engages 0 of 11 lines because it mixes 2-block and 3-block cores.
        * DEFER the receiver's wait past its first row-block of x (`PC_MCAST_LATE`).
          This is where the win lives, not in the transport: row eager is 1.005x,
          row deferred 1.059x.  The stage is a prologue with an empty pipeline
          behind it, so a receiver blocking on the broadcast idles its own x stream.
          Only expressible at NUM_W_CHUNKS == 1, so the host clears the bit
          otherwise and those plans take the eager form -- which still wins
          (1.097x / 1.066x above).
        * Keep D23's trim ON THE INJECTOR.  Whole-tile injector reads lose once the
          wait is deferred (182,700 vs 181,683 ns focus; 1,502k vs 1,450k STREAM).
        * Do NOT split the read across N injectors.  11 injectors is 204,884 vs
          192,455 ns, and `one` (1 injector reading all 72 tiles) TIES `col` (11
          injectors each reading all 72) -- so the injector's serial DRAM read is
          not the limiter, and N concurrent broadcasts only add N x the ack fan-out.
      ENGAGEMENT IS PER LINE, not per program: a line whose members disagree on
      (w_start, w_real, block count) takes the pre-D40 per-core read (role OPT_OUT)
      and costs the other lines nothing.  On the STREAM case 9 of 10 grid rows
      engage and 11 cores opt out, and that still yields 1.089x.
      WHERE IT IS INERT, and why that is not a carve-out: on WIDTH / BLOCK shards
      and on the one-tile-row interleaved decode a grid ROW's members own DISJOINT
      width slices -- there is no reuse to remove -- so no line engages and the
      program is the pre-D40 one, byte for byte.  Perf 1's measured BLOCK regression
      therefore does not reproduce and is NOT carved out: BLOCK's reuse runs along
      COLUMNS, and column geometry is the measured loser.  Leaving BLOCK on the
      table is the trade.  Two hard exclusions: ROW_MAJOR per-channel operands (a
      different staging-ring shape), and D39's COMPACT hold (its per-channel staging
      is a local L1 expand -- there is no DRAM read left to broadcast; the compact
      plan's boot FILL is broadcastable in principle and is left for a later round
      rather than claimed).
      THE OFF PATH IS PROVED IDENTICAL, not asserted -- and that proof is the whole
      reason this graduated now and not in Perf 1.  Perf 1 measured its ancestor
      regressing NON-ENGAGED plans 4-5% and could not attribute it.  The cause was
      not the idea: `ttnn/ttnn/operations/__init__.py` `walk_packages()`-executes
      every package under `operations/`, two `perf_experiments/` dirs set
      `RMS_STAGE_ZONES=1` at module scope, and the shipped descriptor is imported
      EARLIER in that walk than a forked one -- so the baseline compiled clean
      kernels and the candidate compiled zone-instrumented ones.  The `__init__.py`
      files are gone (see `perf_experiments/README.md`) and identity is now checked
      two ways: `check_off_identity.py` preprocesses the reader and diffs (4/4 files
      identical), and `check_off_descriptor.py` records every CB / semaphore /
      kernel / runtime-arg call from both factories and diffs (9/9 cases, the
      engaged mode landing on a non-engaged plan included).

  D41 Perf 2 (perf) -- RESIDENT PICKS ITS BLOCK ON ROW-BLOCKS PER CORE, AND MAY BUY
      A DEEPER RING TO GET ONE MORE.  The seed's RESIDENT search took the COARSEST
      BLOCK_ROWS that fit, which on a shape whose whole per-core assignment fits in
      one block leaves the core with a SINGLE row-block: it reads all of its work,
      then computes all of it, then writes all of it, with no cross-stage overlap at
      all.  `(1,1,8192,1024)` interleaved is exactly that -- BLOCK_ROWS == rows_max
      == 3 -- and it is the 2nd-worst cell of the `perf` group.  The search now picks
      the candidate that gives the core the MOST row-blocks, which lets a DEEPER
      `cb_input_tiles` / `cb_output_tiles` pay for itself by forcing a smaller block:
      BLOCK_ROWS 3 at depth 2 becomes BLOCK_ROWS 2 at depth 3, 84,906 -> 83,544 ns
      (1.016x; 1.023x on a second session).
      TWO things keep it from costing anything anywhere else, and both are the same
      principle -- a deeper ring is only worth what it buys in overlap:
        * THE TIE-BREAK IS THE SHALLOWEST.  Candidates are walked shallowest-first
          and only a STRICTLY larger block count displaces the incumbent, so where
          the deeper ring cannot raise the block count it is pure L1 and is refused.
          That covers every one-tile-row-per-core plan (all the WIDTH shards, the
          interleaved decode, the HEIGHT shard) and the BLOCK shard, whose block
          count the shallow ring already reaches.  All of them keep their pre-D41
          program byte for byte -- verified with `RMS_TRACE_BLOCKING` across the
          guard set, where `(1,1,8192,1024)` is the ONLY plan that moves.
        * THE DEEPER CANDIDATE IS OFFERED ONLY TO THE RESIDENT SEARCH
          (`CB_DEPTH_CANDIDATES_RESIDENT`, a second ladder).  What a deeper ring
          trades against is regime-dependent: RESIDENT has NUM_W_CHUNKS == 1 so it
          can only cost BLOCK_ROWS, which is the point; but ROW_RESIDENT and STREAM
          already sit at one tile-row, so there a deeper ring can only be bought with
          a FINER WIDTH CHUNK.  MEASURED, that trade LOSES: `(1,1,8192,5120)`
          gamma_bias_residual fp32_dest went WT_CHUNK 18x9 -> 11x15 and 608,816 ->
          621,203 ns (0.980x), and `(1,1,1024,16384)` gbr went 57x9 -> 43x12 and
          501,932 -> 509,953 (0.984x).  Before D39 freed the L1 that pays for depth
          3, neither shape could reach it and this loss was invisible -- which is why
          the isolated bench that found the win did not find the trap.

      SUPERSEDED BY D42, which reaches the same objective with the right lever and
      hands the depth-3 L1 back.  D41's ROW_RESIDENT/STREAM measurement above is the
      part that survives -- it is why the single ladder still offers no depth 3.

  D42 Perf 3 (perf) -- THE BLOCK IS PICKED, NOT INHERITED FROM WHAT FITS, AND THE
      RULE IS REGIME-SPLIT.  D41 raised a core's row-block count only by DEEPENING
      the ring; at each depth it still took `min(max_rows, brmax)`, the LARGEST block
      that fits.  A SMALLER block always fits, so the search never offered itself the
      finest split at all -- `(1,1,8192,1024)` interleaved shipped at BLOCK_ROWS == 2
      with 74 of 110 cores holding exactly two tile-rows, i.e. ONE row-block.

      WHY IT IS THE ROUND'S TARGET, measured rather than argued.  Perf 3's cumulative
      peel on that shape stubs every payload and the op STILL costs 14,225 ns (16.8%
      of an 84,586 ns wall); the residue is TRISC-bound (NCRISC marker span 2,493 ns
      against a TRISC span of 14,377, with `writer_write` 14,293 ns of pure WAIT); and
      it ADDS to the payload rather than hiding behind it -- payload 70,923 + floor
      14,225 = 85,148 = the wall.  The payload half is roofline-gated (473 GB/s of the
      494 GB/s best ever measured on this op), so the floor was the whole tournament.

      THE TWO REGIMES MEASURED OPPOSITE SIGNS, which is why one rule cannot serve
      both:
        * x READ OVER THE NoC (`not native_in`) -- the row-blocks ARE the pipeline.
          Take ONE tile-row.  `(1,1,8192,1024)` INT 83,997 -> 82,889 ns in the
          isolated bench (1.013x, ~6.7 sigma over 17-20 reads) and 83,777 -> 82,656
          on the integrated tree; the masked `(1,1,8192,1000)` 84,766 -> 83,512
          (1.015x); FLAT -- never worse -- where a core already holds many blocks
          (`(1,1,28160,1024)`, `(1,1,65536,1024)`).
        * x RESIDENT IN L1 (`native_in`, a zero-copy CB) -- there is NO read to
          overlap, so every extra block is pure per-block overhead.  br 16 -> 1 on
          the `(1,1,8192,1024)` BLOCK shard is 20,372 -> 40,618 ns (0.50x) and on
          `(1,1,7168,1024)` BLOCK gbr 28,880 -> 69,566 (0.42x).  That is the
          carve-out, and it is spelled as the narrow exception.  What a native shard
          CAN take is the BALANCED block at the SAME block count -- 32 rows in 2
          blocks is 20+12 today, 16+16 balanced -- 20,424 -> 19,538 ns (1.045x) on
          the integrated tree.  Only on an EXACT divisor: rebalancing 11 -> 10 on
          `(1,1,7168,1024)` measured 0.993x, so an inexact one is refused.

      DEPTH IS NOW PROVABLY INERT, which is what retires D41's second ladder.
      `br == 1` is the cheapest resident configuration in L1, so if it fits at any
      depth it fits at the shallowest; and it already yields `max_rows` blocks, the
      maximum, so no deeper candidate can strictly exceed it.  On a native shard the
      depth term drops out of the block multiplier entirely (a shard-backed CB costs
      zero arena bytes), so deeper is at best equal there too.  MEASURED on the focus
      shape under this rule at depths 2/3/4/5/6/8: 82,924 / 82,961 / 83,153 / 82,570 /
      82,713 / 82,673 ns -- a 0.7% band.

      MEASUREMENT DISCIPLINE this established, and it is worth more than the number:
      the FIRST variant measured in a device session reads 1.5-2.2% SLOW, proven on
      cases whose program does not change at all (4,381 vs 4,281 ns for the identical
      program).  Any A-then-B comparison without a burn read or ABBA counterbalancing
      silently favours B, and at this op's effect sizes that is the whole signal.

  D43 Perf 3 (perf) -- THE GROUPED SQUARE FOLD.  See `SQ_FOLD_GROUP` for the rule, the
      measurements and the precision evidence.  In one line: `DEST_ACC_SQUARE_MAX_WT`
      is a PRECISION ceiling on the fold's serial 16-bit accumulation depth, and because
      the fold was all-or-nothing that ceiling was also a PERF ceiling.  Folding in
      GROUPS decouples the two.  1.026x-1.053x on the TRISC-bound geometries, flat on
      the roofline-gated prefill band, byte-identical wherever the fold already shipped.

  D44 Perf 3 (perf) -- PASS B'S ORDER: GAMMA FIRST ON A CROSS-CORE PLAN.  Pass B's
      FIRST op is the one that needs the finalized stat, and on a `combine` plan that
      stat arrives by gather -> root fold -> multicast.  The gamma mul depends on x and
      gamma ONLY, so doing it first fills that wait with the traversal instead of idling
      through it.  MEASURED 1.011x-1.154x on all 7 combine=True plans of the `perf`
      group (the four WIDTH shards 1.089x-1.154x, both BLOCK shards 1.011x/1.025x, the
      three W-split interleaved decodes 1.037x-1.074x) and BIT-EXACT with the shipped
      order on all 6 combine=False plans.

      WHAT MAKES IT WORK IS THE TRAVERSAL, NOT THE MATH, and an ablation settles that
      rather than arguing it: deleting only the gamma MULTIPLY (a bare CopyTile over the
      same tiles, lifecycle and trip count kept) is 0.982-1.004x, while deleting the
      whole TRAVERSAL is 1.096-1.244x.  So the pass costs its traversal and the multiply
      is free -- which retires the entire "cut pass B's op count" idea class (fewer or
      cheaper muls, pre-combined broadcasts, matmul-by-diagonal) at every geometry, and
      also explains why Perf 1's DEST-reuse fusion LOST: it deleted packs, which were
      never the cost, and added per-face MOP restarts.  On the small-block combine plans
      the reorder measures AS FAST AS DELETING the traversal outright (1.095 vs 1.096;
      1.129 vs 1.113; 1.036 vs 1.034; 1.074 vs 1.075), which only latency-hiding
      explains.

      THE COST, recorded because it is real and because the round would be dishonest
      without it: the reordered intermediate is `x * gamma`, which is UN-normalized, so
      it can saturate the intermediate CB's dtype where the shipped intermediate
      (approximately 1) cannot.  The boundary is exactly |x * gamma| > dtype_max --
      measured, x=1e10 with gamma=1e29 gives 9.965e28 shipped against 3.373e28
      reordered.  This NARROWS THE OP'S DYNAMIC RANGE on combine-engaged plans.  Nothing
      in the op's tested universe reaches that band (all 19 perf cases and 31 structural
      LOOSE_CASES match the shipped order to 6 decimal places of pcc, and the op's own
      sum(x^2) already saturates by |x| ~ 1.8e19), so it is graduated -- but it is a
      price, not a free win, and reverting it is one predicate.

      TWO CARVE-OUTS, each earned, each spelled as the NARROW exception so it shrinks
      rather than has to be widened:
        * `!HAS_G` -- INFEASIBLE, there is no second mul to move.
        * `!CROSS_CORE` -- MEASURED REGRESSION.  With the stat computed locally there is
          no arrival to hide behind; the reorder cost a reproducible 0.983x and 0.989x
          in two independent sessions on `(1,1,8192,2304)`.  Every other combine=False
          case was flat, so the carve-out is the REGIME, not that one shape.

"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import NamedTuple

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ===========================================================================
# D40 (Perf 2) -- THE PER-CHANNEL BROADCAST.  See the D40 note in the module
# docstring for the measurements behind every constant below.
# ===========================================================================
# `weight` / `bias` do not vary along the ROW axis, so every core owning the same
# width slice reads the IDENTICAL DRAM bytes.  On the focus shape that is ALL 110
# cores against the SAME 72 gamma pages -- a hot-bank storm moving 1.0 MB of real
# traffic for 16,138 ns (8.4% of the op).  One injector per grid ROW reads it and
# broadcasts the staged block instead.
#
# `None` is the OFF build and it is what every plan that is NOT a reuse group gets:
# no extra CB, no extra semaphore, no extra runtime arg, no extra compile-time arg
# and no `RMS_PC_MCAST` define, so the reader preprocesses to the pre-D40 text and
# the descriptor is field-for-field the pre-D40 one.  Both halves of that are
# PROVED, not asserted -- `perf_experiments/per_channel_mcast_v2/check_off_identity.py`
# (preprocess-and-diff, 4/4 kernel files identical) and `check_off_descriptor.py`
# (record every CB / semaphore / kernel / RT-arg call from both factories and diff,
# 9/9 cases).  Perf 1 shipped this idea's ancestor un-graduated for exactly the lack
# of that proof.
#
#   "col"    Mcast1D PerColumn -- one injector per grid column (Perf 1's geometry;
#                                MEASURED LOSER under the deferred wait, 0.968-0.988x,
#                                and it engages 0 of 11 lines on the STREAM case
#                                because a column mixes 2-block and 3-block cores)
#   "row"    Mcast1D PerRow    -- one injector per grid row.  THE SHIPPED CHOICE.
#   "one"    Mcast2D           -- ONE injector for the whole rectangle (1.00-1.02x)
#   "split"  Mcast2D rotating  -- PC_MCAST_SPLIT injectors each reading 1/N and
#                                broadcasting its slice.  MEASURED LOSER: 11 injectors
#                                is 204,884 vs 192,455 ns.  `one` (1 injector reading
#                                all 72 tiles) TIES `col` (11 injectors each reading
#                                all 72), so the injector's serial DRAM read is not
#                                the limiter -- N concurrent broadcasts just add N x
#                                the atomic-ack fan-out.
PC_MCAST_MODE = "row"

# The receiver-readiness PRE-HANDSHAKE.  There is no landing buffer to protect here:
# the per-channel CBs are boot-filled once and never popped, so on every core the
# landing pointer is the ring base from program start.  With the handshake off the
# data-ready signal switches to the monotone Counter, whose `wait_min(n)` is correct
# no matter which side reaches the cell first -- so the elision needs no ordering
# argument at all.  The SPLIT mode REQUIRES this (its rounds complete out of order).
PC_MCAST_HANDSHAKE = False

# How many injectors split the chunk in "split" mode.
PC_MCAST_SPLIT = 1

# The INJECTOR's per-channel read granularity.  `None` keeps D23's trim; 0 makes the
# injector fetch WHOLE tiles.  See the kernel-side comment on `pc_issue_slice_reads`.
PC_MCAST_INJ_TRIM = None

# DEFER the receiver's broadcast wait past its first activation read.  The per-channel
# stage is a prologue with an empty pipeline behind it, so a receiver that blocks on the
# broadcast is idling its own x stream; with this on it reserves, reads its first
# row-block, and collects the broadcast afterwards.  Only expressible at
# NUM_W_CHUNKS == 1 (where "the first activation read" is unambiguous), so the host
# clears the bit otherwise rather than the kernel guessing.
# MEASURED: row eager 1.005x, row DEFERRED 1.059x -- the deferral is where the
# win actually lives, not the transport.
PC_MCAST_LATE = True

# Which core of a "col" / "row" LINE injects, and whether the choice STAGGERS from line
# to line.  Index 0 is the line's first core (leftmost / topmost); a middle index halves
# the worst-case broadcast distance; `Diagonal` advances the index per line so the
# injectors do not all sit in one grid column and start their DRAM reads together.
# The smallest LINE the broadcast is allowed to engage; see the carve-out note at the
# `live` filter for the measurement that set it.  2 is the measured loser (one receiver
# pays the whole fixed handshake); 3 is untested and included.
PC_MCAST_MIN_GROUP = 3

PC_MCAST_SENDER_INDEX = 0
PC_MCAST_DIAGONAL = True  # ~0.5% on the focus shape, neutral on STREAM

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

# ---- Refinement 3 / Lamp L-OPERAND-TRIM: per-channel read granularity ---------
# D23 derived ONE policy (two face-rows where the dtype's face offset is 64-byte
# DRAM aligned, else the half page) and the bias copied it.  The lamp's question is
# whether the TRANSACTION COUNT -- two trimmed reads per tile instead of one, and
# two operands instead of one -- is the thing that actually matters (D13 found it
# was), so each operand now carries its own measured override.
#
#   TRIM_DERIVED  take D23's derived answer (the shipped default: byte-identical)
#   2             two face-rows      (2 reads x 32*elem bytes per tile)
#   1             one half page      (1 read x tile/2 bytes per tile)
#   0             the whole tile     (1 read x tile bytes per tile)
#
# MEASURED -- see the changelog's Refinement 3 table.  One source of truth: the
# value is read ONCE, in `_trim_for`, and both the reader CT arg and the CB table
# derive from that.
TRIM_DERIVED = -1
PER_CHANNEL_TRIM_GAMMA = TRIM_DERIVED
PER_CHANNEL_TRIM_BIAS = TRIM_DERIVED

# ---- D39 / perf_experiments/stream_regime: THE COMPACT PER-CHANNEL HOLD ------
# A TILE per-channel operand is a (1,1,1,W) vector: 31 of every 32 tile rows are
# PADDING, and D23's TRIM == 2 already fetches only the two FACE-ROWS that carry
# row 0.  Those `2 * TILE_DIM * elem` bytes per width tile are the WHOLE of the
# operand's information -- 1/16 of its tiled size -- so the reader can cache the
# entire row of BOTH operands in L1 once and re-materialize each chunk's tiles
# LOCALLY (an L1->L1 copy, no NoC read).  Two things fall out:
#   * STREAM stops re-reading the operands from DRAM per pass-B chunk of every
#     row-block (measured 230,305 ns of a 1,580,377 ns wall on the target case);
#   * ROW_RESIDENT can price its per-channel HOLD at the compact size, which is
#     what lets a shape that used to fall off the L1 cliff into STREAM hold a
#     tile-row instead -- deleting pass B's re-read of x AND the residual.
PC_COMPACT_HOLD = True
# WHEN the cache is filled: False = EAGER, the whole row at boot; True = LAZY,
# chunk c the first time chunk c is staged.  Lazy issues the identical reads into
# the hole where the reader is already blocked on pass B instead of as a
# device-wide burst in front of the first x read (measured 1.16x on the target).
PC_COMPACT_LAZY = True
# Whether the ROW_RESIDENT L1 solve is allowed to use the compact price (and thus
# chunk the per-channel tile ring).  False keeps the shipped hold exactly.
ROW_RESIDENT_COMPACT_PC = True
# Chunks of head-room in the per-channel TILE ring.  1 = the reader's local expand
# of chunk c sits on pass B's critical path (compute's Upfront wait on the ring);
# 2 lets the reader run one chunk ahead, at one extra chunk of L1 -- which the
# chunk solve then has to give back.  A measured trade, not an obvious one.
PC_RING_CHUNKS = 1
# 0 = take the L1 solve's own cap.  A positive value CAPS the ROW_RESIDENT chunk,
# so a sweep can ask "is a finer chunk better than the coarsest that fits?".
ROW_RESIDENT_CAP_OVERRIDE = 0
# ---- THE COMPACT BRANCH'S ONE PRECONDITION: a SATURATED grid -----------------
# The compact hold makes ROW_RESIDENT reachable for shapes the shipped solve sent
# to STREAM.  ROW_RESIDENT trades DRAM BYTES (it deletes pass B's re-read of x and
# of the residual -- 2 of the 5 tensor-crossings a residual STREAM pays) for
# READER/COMPUTE OVERLAP (its pass B issues no read at all, so the reader has
# nothing to do there but run ahead into the next row-block).  That trade pays
# exactly when DRAM is the constraint, and the shape-blind proxy for "DRAM is the
# constraint" is how much of the compute grid is pulling on it: every active core
# is an independent NoC client.
#
# MEASURED, blackhole p150b (110-core grid), base(STREAM) -> compact ROW_RESIDENT.
# The middle block holds W, the residual, the dtype, rows-per-core AND the solved
# chunk (5) FIXED and moves only the number of active cores:
#     32 / 110 cores   (1,1,1024,16384) gbr    501,524 ->   763,861   0.66x
#     64 / 110 cores   (1,1,2048,16384) gbr    947,448 ->   975,876   0.97x
#     96 / 110 cores   (1,1,3072,16384) gbr  1,401,062 -> 1,280,538   1.09x
#    110 / 110 cores   (1,1,3520,16384) gbr  1,583,861 -> 1,429,391   1.11x
# and the two shapes the op's perf group actually cares about, both on the full
# grid, are where the trade is worth most because the chunk stays coarse:
#    110 / 110 cores   (1,1,8192, 7168) gbr fp32d  1,584,324 -> 1,048,060  1.51x
#    110 / 110 cores   (1,1,8192,10240) gbr fp32d  2,263,244 -> 1,517,643  1.49x
#
# It is monotone in grid occupancy and BLIND to everything else the candidates
# differ in -- the 0.66x row and the 1.11x row solve to the SAME WT_CHUNK (5) and
# the SAME one row-block per core, and a 1.02x row at 110 cores / 2 blocks
# ((1,1,4096,16384) gbr, 1,876,550 -> 1,836,015) rules the block count out too.
# The crossover is bracketed at 58% (loses) / 87% (wins) of the grid; 3/4 sits
# between the two measured points.  That bracket is the whole justification for
# the constant -- it is not tighter than the two shapes that bracket it.
ROW_RESIDENT_COMPACT_MIN_GRID_FRACTION = 0.75
# Sweep-only: 0 == take the gate off, which is what the 0.66x / 0.97x rows above
# were measured with.
#
# Sweep-only: a chunk-WIDTH floor on the compact branch.  1 == off, and off is what
# the measurements support -- the win is 1.11x at WT_CHUNK 5 and 1.51x at 32 on a
# saturated grid, and 0.66x at WT_CHUNK 5 and 0.83x at 18 on a third of one, so the
# chunk does not carry the sign.
ROW_RESIDENT_COMPACT_MIN_CHUNK_WT = 1
# Sweep-only: try the COMPACT hold BEFORE the tiled one.  This is the ordering the
# experiment's first cut had, kept so the sweep can show what it costs.
COMPACT_FIRST = False

# ---- Refinement 3 / lever 3: the data-movement TRANSACTION UNIT --------------
# op_design.md's block-schedule table states the TILE path's intent as "one NoC
# barrier per (block, chunk) per stream"; the shipped reader and writer issue one
# per TILE-ROW of the chunk (WT_CHUNK tiles).  `DM_TXN_ROWS_MAX` is the cap on how
# many tile-rows share one reserve / issue / barrier / push group:
#
#   1   the seed: one barrier per tile-row                        (the default)
#   n   up to n tile-rows per barrier
#   0   the whole row-block  (BLOCK_ROWS tile-rows -- the design's stated intent)
#
# The unit is always a DIVISOR of BLOCK_ROWS, which is what makes the multi-page
# group straddle-free: every block-scoped ring is `depth * BLOCK_ROWS * WT_CHUNK`
# pages, so a group of `TXN_ROWS * WT_CHUNK` pages starting at a block-aligned
# offset can never cross the ring wrap.  It is a genuine TRADE, not a free win --
# a coarser handoff costs reader <-> compute overlap INSIDE a block -- which is why
# the default is measured rather than assumed.  See the changelog's R3 table.
#
# ONE source of truth: `_dm_txn_rows` decides, `_pack_txn_rows` encodes, and BOTH
# dataflow kernels decode the same CT word (index 4, the BLOCK_ROWS slot).  The
# encoding stores `TXN_ROWS - 1` in the high half so the default is byte-identical
# to the seed's plain `block_rows`.
DM_TXN_ROWS_MAX = 1


def _dm_txn_rows(block_rows: int) -> int:
    """Tile-rows per NoC transaction group: the largest DIVISOR of block_rows <= the cap."""
    cap = block_rows if DM_TXN_ROWS_MAX == 0 else min(DM_TXN_ROWS_MAX, block_rows)
    cap = max(1, cap)
    for n in range(cap, 0, -1):
        if block_rows % n == 0:
            return n
    return 1


def _pack_txn_rows(block_rows: int, txn_rows: int) -> int:
    """Encode (BLOCK_ROWS, TXN_ROWS) into the CT word both dataflow kernels read at index 4."""
    assert 0 < block_rows < (1 << 16), f"rms_norm_ttnn: BLOCK_ROWS={block_rows} does not fit the packed CT word"
    assert txn_rows >= 1 and block_rows % txn_rows == 0, (
        f"rms_norm_ttnn: the NoC transaction unit ({txn_rows}) must divide BLOCK_ROWS ({block_rows}) -- "
        "otherwise a group can straddle the block-scoped ring's wrap"
    )
    return block_rows | ((txn_rows - 1) << 16)


# ---- Refinement 3 / levers 2 + the pass-A DEST block -------------------------
# PASS_A_SQ_BLOCK: give pass A's `square` the SAME DEST-lane block size pass B's
#   chains already take (D21, measured 1.28-1.66x there) instead of the per-tile
#   spelling.  0 == the seed's per-tile chain (byte-identical); 1 == blocked at
#   PASS_B_BLK.  Inert under the D12 DEST fold, whose accumulation already owns D0
#   for the whole tile-row.
# RES_FUSE (Lamp L-RES-FUSE): fuse pass A's `t = x + r` and its square into ONE
#   eltwise_chain -- Add -> Square(DEST) -> PackTile(cb_x_squared) -- so the square
#   takes the sum straight out of DEST and `t` is never packed at all in pass A.
#   STREAM-ONLY, and that boundary is measured rather than argued: in both RESIDENT
#   regimes cb_x_sum IS the tensor pass B normalizes, and a chain cannot publish an
#   INTERMEDIATE DEST value (pack is its own cohort, so every pack runs after every
#   compute element) -- the four-element form writes the square into cb_x_sum too and
#   measures pcc 0.260.  See the gate comment in the compute kernel.
#   0 == the seed's two chains.
PASS_A_SQ_BLOCK = 1
RES_FUSE = 0

# ---- Refinement 3 (folded in): price cb_x_squared at the width it ACTUALLY takes
# `_cb_block_mult` prices every block-scoped CB at the full chunk width, but under the
# D12 DEST fold cb_x_squared is `BLOCK_ROWS x 1` tiles, not `BLOCK_ROWS x WT_CHUNK`.
# The error is CONSERVATIVE -- it can only shrink BLOCK_ROWS, never overflow L1 -- and
# it bites only where the fold is on (WT_CHUNK <= DEST_ACC_SQUARE_MAX_WT), which is
# also where L1 is rarely the binding constraint.  `CB_SQ_EXACT = 1` charges the true
# width in the RESIDENT solve (the one place the chunk is already known to be
# `wt_core`); 0 keeps the seed's conservative price, so the default is byte-identical
# and `test_program_is_structurally_the_seeds` cannot move for this reason.  Measured:
# see the changelog's R3 table.
CB_SQ_EXACT = 0

# Ordered depth candidates for the two cross-processor CBs (cb_input_tiles,
# cb_output_tiles), COARSEST FIRST.  The regime search (D4) walks them and takes
# the RESIDENT regime at the first depth whose whole-row working set fits L1.
#
# (2, 1) was MEASURED a net loss (see D4), so a SHALLOWER depth is still not on
# offer.  ONE ladder, walked by every regime.
#
# D42 (Perf 3) RETIRED D41's second, DEEPER ladder (`CB_DEPTH_CANDIDATES_RESIDENT
# = (3, 2)`).  D41 bought depth 3 in order to force a SMALLER block and so raise a
# core's row-block count; D42 picks the block directly (`br = 1` wherever x is read
# over the NoC), which reaches the maximum block count at the SHALLOWEST depth and
# makes any deeper candidate provably unable to beat it -- see the proof and the
# measured depth sweep (82,570..83,153 ns across depths 2/3/4/5/6/8, a 0.7% band) at
# the RESIDENT search.  So the deeper ring is not merely unnecessary now, it is
# unreachable, and its L1 comes back.  D41's ROW_RESIDENT / STREAM loss (0.980x /
# 0.984x, from a deeper ring bought with a finer WIDTH CHUNK) is what made the second
# ladder necessary in the first place, and it stays closed for the same reason: those
# regimes are on this ladder, which offers no depth 3.
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

# Refinement 4b (D34) -- PER-STAGE DEVICE ZONES ARE OPT-IN.  The kernels' 41
# `MaybeDeviceZoneScope` sites compile only when this is on; it emits the
# `RMS_STAGE_ZONES` kernel DEFINE, which is the ONE switch `perf_instrumentation.hpp`
# reads.  Default OFF, from the env var of the same name, because a graded eval run
# sets `TT_METAL_DEVICE_PROFILER=1` (to read the op-level DEVICE KERNEL DURATION off
# the FIRMWARE markers) and that would otherwise register all 41 op zones in the
# profiler's 16-BIT source-location hash table -- where two colliding entries are a
# hard `TT_THROW` on every profiler read and a process `terminate` at the next device
# open.  See the header for the full argument and the measured collision.
#   RMS_STAGE_ZONES=1 scripts/run_safe_pytest.sh --profile <test>
STAGE_ZONES = os.environ.get("RMS_STAGE_ZONES", "0") not in ("", "0")


# Perf 3 -- THE ABLATION SWITCHES ARE DEFINES, NOT SOURCE EDITS.
#
# `RMS_ABLATE=READ_X,WRITE,COMPUTE,PER_CHANNEL,ROOT_SUM,ROOT_FINALIZE,RECONFIG,GATHER_ZERO`
# stubs each named stage's PAYLOAD while leaving its CB reserve/push/wait/pop, its loop
# trip counts and its zone in place -- the cumulative-peel instrument /perf-measure asks
# for.  Names match the `RMS_ABLATE_<NAME>` guards in the three kernels.
#
# WHY A DEFINE AND NOT THE COMMENTED-OUT `#define` AT EACH KERNEL HEAD (measured):
# the JIT kernel cache key does NOT include the kernel source's CONTENT, so editing a
# .cpp in place and re-running is a CACHE HIT on the previously compiled binary.  Perf 3
# lost two measurements to exactly that -- an "unablated baseline" reproduced twice at
# 56,090 ns with pcc=nan against a true 84,510 ns, because it was still running the
# all-stubbed build from the peel before it.  A DEFINE is part of the cache key, so every
# configuration gets its own entry and no cache purge is needed.  Ablating by source edit
# additionally re-rolls every zone's 16-bit hash (see perf_instrumentation.hpp).
_ABLATE_NAMES = (
    "READ_X",
    "WRITE",
    "COMPUTE",
    "PER_CHANNEL",
    "ROOT_SUM",
    "ROOT_FINALIZE",
    "RECONFIG",
    "GATHER_ZERO",
)
ABLATE = tuple(
    n for n in (t.strip().upper() for t in os.environ.get("RMS_ABLATE", "").split(",")) if n in _ABLATE_NAMES
)


def _kernel_defines():
    """Kernel `-D` defines shared by all three kernels.  ONE source of truth.

    Empty (the default) means the build key is exactly what it was before D34, so
    every non-profiled build stays byte-identical.
    """
    defines = [("RMS_STAGE_ZONES", "1")] if STAGE_ZONES else []
    defines += [(f"RMS_ABLATE_{n}", "1") for n in ABLATE]
    return defines


# Refinement 4 (D33) -- THE RAGGED (PADDED) WIDTH CHUNK.  1 = a chunked width may
# take the coarsest BALANCED chunk and pad the last one out to it; 0 = D1's divisor
# clamp, exactly as shipped through Refinement 3.  See `_width_chunk` for what the
# knob decides and `WT_PAD` for what the kernels do with the pad tiles.  Taken only
# where the divisor is strictly coarser-limited AND the width is tile-aligned
# (PARTIAL_W == 0), so every shape whose Wt already had a coarse divisor -- which is
# every INPUTS and every perf case -- builds a byte-identical program.
RAGGED_WIDTH_CHUNK = 1

# Refinement 4, the OTHER half of the same cliff.  The L5 / ROW_RESIDENT regime holds
# the whole tile-row of x (and of every per-channel operand) and chunks only the
# derived CBs -- which is pure profit while the chunk stays coarse, but at a chunk of
# ONE tile the hold has eaten so much L1 that the regime is paying 127 per-phase init
# / reconfig / pipeline fill-and-drain cycles to save a re-read.  This is the SMALLEST
# chunk at which L5 is still taken; below it the solve declines and falls through to
# STREAM, which holds nothing and can therefore afford a far coarser chunk (it pays a
# pass-B re-read of x and the residual instead).  1 == the pre-Refinement-4 behaviour
# (L5 at any chunk); see the changelog for the A/B that set it.
ROW_RESIDENT_MIN_CHUNK_WT = 1

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
#
# STILL LOAD-BEARING after Perf 2, and measured to be so: dropping this floor in
# favour of D20's regressed `test_sharded_row_major[ragged_width_tail_wt127]` --
# (1,1,32,4064) ROW_MAJOR, the very shape in the table above -- to rel-RMS 0.06093
# against its 0.04 bound.  D20 is layered as a carve-out BELOW this floor, never as a
# replacement for it.
REDUCE_ACC_VIA_ADD_MIN_CHUNK_WT = 2

# Per-CALL floor (Perf 2, D20).  A THIRD, independent quantity: the reduce's ACTUAL
# per-call reduce-dim width, which is `x_squared_wt` -- NOT `wt_chunk`.  D12's
# square-DEST-fold collapses the per-call width to 1 while leaving `wt_chunk` wide, so
# the two floors above cannot see that regime at all: they both read a wide number
# while every reduce call is one tile deep.
#
# MEASURED (blackhole p150b 1350 MHz, at the op's pinned config -- bf16 / HiFi2 /
# fp32_dest_acc_en=False, isolated bench perf_experiments/reduce_at_percall_width_1,
# ns for the reduce alone, one fresh-cache profiled run per point):
#
#   per-call width   rows=8: AccViaAdd / ReduceTile      relRMS AVA / RT
#      1               7672  /  2796   RT 2.75x          0.0034 / 0.0032   <- RT wins BOTH
#      2               7316  /  2843   RT 2.58x          (RT pcc starts sliding)
#      4               7770  /  4101   RT 1.90x
#      8               8691  /  6258   RT 1.39x          RT pcc 0.99907  <- below the gate
#     16              10648  / 10727   FLAT              RT pcc 0.99634
#     32              15297  / 20147   AVA 1.32x         RT pcc 0.98775
#
# Width 1 is the ONLY width where ReduceTile is both faster AND at least as accurate;
# ReduceTile's precision degrades monotonically from there.  Whole pass A
# (square+reduce) measures 11551 -> 6079 ns (1.90x) at the focus shape, rel-RMS
# 0.00491 -> 0.00459 (i.e. BETTER).  Raise to 1 to disable D20 entirely.
REDUCE_ACC_VIA_ADD_MIN_CALL_WT = 2

# Ring depth of cb_row_stat, in units of BLOCK_ROWS.  MUST be >= 2 -- this is a
# correctness constant, not a perf knob.  See D6.
CB_ROW_STAT_DEPTH = 2

# Ring depth of cb_residual_tiles when the residual is STREAMED (Refinement 1, Lamp
# L-RES-DEPTH).  0 means "follow cb_x_depth", which is what the residual has always done
# and is byte-identical to Phase 0; 1 or 2 pins it independently of the x stream.
#
# THE PREMISE THE LAMP WAS FILED ON IS FALSE ON ITS OWN TARGET, and that is a measured
# host-side fact, not an argument: on `(1,1,32,5120)` WIDTH [32,160] (8,4) with
# `gamma_bias_residual` -- the middle Refinement-1 case -- the residual carries the input's
# shard spec, so `native_residual` holds and cb_residual_tiles is
# `cb_descriptor_from_sharded_tensor`, an ALIAS of the resident shard.  Dumped from the
# built descriptor: cb 1 (x) 5 pages, cb 20 (residual) 5 pages, both exactly the shard, so
# neither costs a byte of the CB arena and NO depth is applied to either.  There is no
# "second double-buffered activation stream" to shrink there.
#
# The knob is kept anyway, at its byte-identical default, because it IS live on the
# STREAMED-residual plans (an interleaved input with a residual, where cb_residual_tiles is
# a real `cb_x_depth * BLOCK_ROWS * WT_CHUNK` ring).  MEASURED there -- see the changelog:
# depth 1 vs the tied depth on the interleaved gamma_bias_residual perf cases.
CB_R_DEPTH = 0


def _residual_depth(depth_x: int) -> int:
    """The residual ring's depth -- ONE definition, read by both L1 solves AND the CB table.

    They must agree page-for-page or BLOCK_ROWS is solved against a budget that does not
    exist, which is exactly the class of bug `_cb_block_mult` was factored out to prevent.
    """
    return CB_R_DEPTH or depth_x


# Largest WT_CHUNK at which pass A's `square` folds the width tiles straight into
# DEST (DestAccumulation::PerRow) instead of packing every x^2 tile out to
# cb_x_squared and having the reduce read them all back -- op_design.md Lamp L6(d).
# See D12.  It is a CEILING, not a floor, because the fold's accumulation runs
# SERIALLY over the chunk's width tiles inside a DEST register that is only 16-bit
# at fp32_dest_acc_en=False; bounding the chunk bounds that depth.  0 disables the
# fold everywhere (byte-identical to Refinement 3); 10**9 forces it wherever it is
# legal.
DEST_ACC_SQUARE_MAX_WT = 8

# D43 (Perf 3) -- THE GROUPED FOLD.  `DEST_ACC_SQUARE_MAX_WT` above is a PRECISION
# ceiling: it bounds how many x^2 tiles the fold accumulates SERIALLY inside a DEST
# register that is 16-bit at fp32_dest_acc_en=False.  Because the shipped fold folds
# the WHOLE chunk or nothing, that precision bound was also a PERF bound -- every
# prefill profile (WT_CHUNK 32..80) sat on the packed path, paying WT_CHUNK packs and
# WT_CHUNK unpacks per tile-row.
#
# The grouped fold DECOUPLES the two.  `SQ_FOLD_GROUP = G` folds in groups of at most
# G width tiles: the serial accumulation depth stays <= G, while cb_x_squared holds
# WT_CHUNK/G tiles per tile-row instead of WT_CHUNK, deleting (G-1)/G of the square's
# packs and of the reduce's unpacks.
#   0 or 1  the pre-D43 behaviour exactly: fold the whole chunk when it is <= the
#           ceiling, otherwise pack every tile.
#   G > 1   fold in groups of the LARGEST DIVISOR of WT_CHUNK that is <= G.  A divisor
#           is required because the group must tile the chunk exactly -- a ragged last
#           group needs a second iteration shape, which one `eltwise_chain` call cannot
#           express (that is the ONE inexpressible case; it simply gets no fold).
#
# MEASURED at G = 16 (bf16 / HiFi2 / fp32_dest_acc_en=False unless the case says
# otherwise; median of 9, against two identical-program controls that bound the noise
# band at +-0.6%):
#   TRISC-bound geometries WIN -- (1,1,32,5120) INT decode 1.033x, (1,1,32,7168) INT
#   decode 1.036x, (1,1,128,4096) ROW_MAJOR-weight fp32_dest 1.041x, (1,1,256,512)
#   WIDTH-auto ROW_MAJOR 1.061x, bfloat8_b (1,1,8192,1024) 1.006x / (1,1,8192,5120)
#   1.014x.
#   The ROOFLINE-GATED interleaved prefill band is FLAT and stays in the domain --
#   focus (1,1,8192,1024) 1.005x, 8192x2304 0.998, 8192x5120 1.005, 8192x7168 0.999,
#   the fp32_dest gbr pair 0.998/1.002, ROW_RESIDENT and STREAM 1.000.
#   Every geometry where the fold ALREADY shipped (WT_CHUNK 4..8: all four WIDTH
#   shards, both BLOCK shards) builds a program the grouped rule leaves BYTE-IDENTICAL,
#   because a chunk <= the ceiling still folds whole.
#
# WHERE THE WIN COMES FROM, measured not assumed: the fold deletes exactly what it
# promises -- `compute_reduce`'s summed-TRISC occupancy 3,710 -> 1,781 ns -- but on the
# focus shape the TRISC span is 44.7 us against the writer's 54.5 us BRISC span, so
# TRISC has ~10 us of slack and the deletion is invisible at the wall.  Stub only the
# write payload (`RMS_ABLATE=WRITE`) and it appears: 56,113 -> 55,068 ns (1.019x).  So
# the focus shape is an honest NULL for this idea and the win is real elsewhere.
#
# PRECISION, measured against a float64 reference on adversarial inputs (every summand
# identical and non-dyadic -- the textbook serial-sum worst case).  The ceiling's worry
# is REAL but it belongs to the FLAT fold only:
#   W=1024 (chunk 32): base / G=4 / G=8 / G=16 all rel-RMS 0.00737;
#                      an UNBOUNDED fold (depth 32) is 0.01862 -- 2.5x WORSE.
#   W=2304 (chunk 72): base 0.01849; G=8 and G=16 both 0.00717 -- 2.6x BETTER than
#                      what ships, because grouping bounds a depth the packed path
#                      pays in bf16 anyway.
# On randn inputs the fold is at least as accurate as base on every case measured
# (focus 0.00667 -> 0.00630; (1,1,8192,7168) 0.00924 -> 0.00615).  So G = 16 is the
# fastest option that meets the precision contract, and the UNBOUNDED fold -- which was
# the obvious way to raise the ceiling and is ~1.001x on the focus shape anyway -- is
# refused ON PRECISION, not on speed.  G = 8 is the conservative alternative (it pins
# the depth at exactly today's vetted 8) and costs 0.5-2 points of the win.
SQ_FOLD_GROUP = 16


def _x_squared_wt(wt_chunk: int, partial_w: int) -> int:
    """cb_x_squared's width tiles per tile-row == the reduce's per-call reduce-dim width.

    ONE definition, read by the RESIDENT L1 solve, the CB table and the compute kernel's
    CT arg -- they must agree page for page or the ring is sized against a layout that
    does not exist.  Returns `wt_chunk` (no fold), 1 (the flat fold), or a divisor of
    `wt_chunk` (D43's grouped fold).
    """
    if partial_w != 0:
        # UNCHANGED, and it is a CORRECTNESS gate, not a perf one: the fold folds the
        # row's last width tile INCLUDING its pad lanes before the reduce runs, so the
        # reduce's partial scaler / 0-1 mask can no longer reach them.
        return wt_chunk
    if wt_chunk <= DEST_ACC_SQUARE_MAX_WT:
        # Already inside the vetted serial depth: fold the whole chunk, exactly as
        # pre-D43.  Keeping this branch is what makes every currently-folding geometry
        # byte-identical for ANY setting of SQ_FOLD_GROUP.
        return 1
    g = max((d for d in range(2, min(int(SQ_FOLD_GROUP), wt_chunk) + 1) if wt_chunk % d == 0), default=1)
    return wt_chunk // g

# Faces of each fp32 partial tile that the cross-core width combine's GATHER ships from a
# member into the group root -- see D13.  A tile is 2x2 faces of 16x16; a REDUCE_ROW partial
# is a column vector, so only faces 0 and 2 can carry data.
#   2  COMPACT (shipped): half the bytes, two face-sized transfers per tile.
#   4  WHOLE: one whole-tile transfer -- Refinement 2/3's behaviour, byte-identical.
#
# Perf 3 / D27 CONFINES this to the BLOCK_ROWS == 1 branch (the compact branch must ship
# whole tiles, or the receiver's un-permute matmul turns an un-written column into NaN) and
# MEASURES that it has to stay there: shipping whole tiles at BLOCK_ROWS == 1 regressed the
# four pinned WIDTH-shard geometries 0.86x-0.98x, MONOTONE in group_size, which is the
# gather's fan-in multiplier showing itself.  So the knob is not dead -- it is scoped.
GATHER_FACES = 2

# Ring depth (in PAGES, not row-blocks) of the combine's BLOCK_ROWS-INDEPENDENT CBs:
# cb_stat_handoff, cb_compact_handoff and cb_mcast_in each carry exactly ONE tile per
# combine round under D27's compact layout, so their depth is a small constant instead
# of a multiple of BLOCK_ROWS.  2 for the same reason CB_ROW_STAT_DEPTH is 2: the
# producer must be able to fill round r+1's page while the consumer still holds round
# r's (D24 publishes the root's own stat copy before the broadcast and needs the other
# half of the ring to stay untouched; D25's pipeline packs block blk+1's compact
# partial while the writer is still shipping block blk's).
CB_COMBINE_FLAT_DEPTH = 2

# ---------------------------------------------------------------------------------------
# THE COMBINE'S SLOT TREE (Perf 3, descriptor D28) -- two knobs, both MEASURED.
# ---------------------------------------------------------------------------------------
# The FLAT combine has exactly ONE gatherer per group: every one of the other
# GROUP_SIZE - 1 members writes into the root's L1 and the root folds all GATHER_SLOTS
# pages itself.  Both of those terms are linear in GROUP_SIZE and both land on ONE core.
# The tree interposes a level of intermediate gatherers: level 0 folds contiguous runs of
# f0 slots on f1 = ceil(GROUP_SIZE / f0) different cores IN PARALLEL, and only
# those f1 raw sums travel to the root, which folds f1 of them.  So the root's ingress
# fan-in and its fold both drop from GROUP_SIZE to max(f0, f1), at the price of ONE extra
# NoC hop per round.
#
# COMBINE_TREE_F0_MIN / _MAX -- the level-0 fan-in BAND.  `f0` is not a constant: it is
# the LARGEST DIVISOR of GROUP_SIZE inside this band that both gates admit, with the cap
# itself as a ragged fallback for a group no divisor covers (see
# `_combine_tree_candidates`).  Refinement 1 lever 1 swept it on the real op, on top of
# the NOC_0 combine, one fresh in-process profiled measurement per cell (median of 5,
# whole-op DEVICE KERNEL DURATION, blackhole p150b 1350 MHz):
#
#   G   flat   f0=4   f0=5   f0=6   f0=8   f0=10  f0=15/16/32   best
#   28  5699   6004    --    6081   6018   5987      --         FLAT (every tree loses)
#   30  5058   5051   4981   4981   5056   4787   5264 (15)     f0=10, then 6/5
#   32  5253   4979    --    4991   4736   5133   5201 (16)     f0=8   (divisor)
#   40  5402   5064    --    4927   4941   4789      --         f0=10  (divisor)
#   64  6695   5498    --    5362   5047   5363   5754/6436     f0=8   (divisor)
#
# TWO effects, both visible in that table and both mechanical:
#  (a) A LARGER f0 UNLOADS THE ROOT.  The root's ingress is f1 - 1 remote writes and its
#      fold is f1 pages, so trading f1 down for f0 up moves work off the one core that is
#      also the finalizer and the multicast sender.  f0=4 loses to the 6..10 band at every
#      G >= 30.
#  (b) BUT AN EXACT DIVISOR BEATS A BIGGER RAGGED f0.  At G=32, f0=8 (8x4, exact) is 4736
#      while f0=10 (10x4, ragged 10+10+10+2) is 5133 -- SAME f1, 8.4% apart, so it is the
#      ragged run and not the fan-in.  A ragged run makes one gatherer boot-zero its tail
#      and gives the level-1 arrival a non-uniform set to wait on.
# The CAP is measured too, in the same table: f0=15/16/32 lose at every G -- past ~10 the
# level-0 gatherer becomes the serial bottleneck the tree was built to remove.  The FLOOR
# keeps a degenerate divisor (G=34 -> 2) from being preferred over the ragged cap; f0=2
# measured 5277/5384/6476 at G=32/40/64 on NOC_1, the worst arity in the whole sweep.
#
# ONE RECORDED MISS, deliberately left: at G=30 the best measured arity is f0=10 (4787),
# but 30 - 10 - 3 = 17 fold tiles deleted, one below the threshold below, so the rule takes
# f0=6 (4981) instead.  The threshold is NOT loosened to reach it: G=28 also deletes 17 and
# there EVERY tree is a ~5% LOSS against the flat root, so 17 is a point where the sign
# genuinely depends on the geometry and the conservative side is the one that protects a
# pinned target.
# TWO LEVELS, full stop: THREE and FOUR levels were measured at 7 cells and lost at 6 of
# them (GROUP_SIZE 32: 3 levels 3870-3991 vs 2 levels 3744; 4 levels 4461).  A deeper tree
# buys another fold division but pays another hop, and the hop is the expensive half.  So
# this file has no depth knob -- adding one would be re-proposing a refuted shape.
#
# COMBINE_TREE_MIN_DELETED_FOLD_TILES -- the crossover, and it is a threshold on the
# DERIVED quantity the mechanism is about, not on a shape/dtype/placement:
#
#     deleted = rows_per_round * (GROUP_SIZE - f0 - f1)
#
# i.e. the number of root fold-tiles the tree takes off the per-round critical path (the
# root folds f1 instead of GROUP_SIZE, and one of the f1 is its own level-0 result), against
# the ONE extra hop it pays for them.  `rows_per_round` is IDENTICALLY 1 in this op since
# D27 -- a sender's whole row-block travels as one compact tile -- so the term is written
# out at the call site and multiplies by one.  It is kept in the expression because it is
# the physical quantity: if a future layout ever ships more than one page per sender per
# round, the threshold moves with it automatically.
#
# 18 is BRACKETED ON THE REAL OP, not inherited from the isolated bench, and the two
# bracketing points are ADJACENT.  A/B on the WHOLE OP, one fresh-cache profiled run per
# cell, same build, tree forced OFF (this constant = 10**9) vs forced ON (= 0), at the
# `_perf_case` config -- the numbers are `DEVICE KERNEL DURATION` for the whole op, so a
# combine-only speedup shows up diluted by the ~3.5 us launch/dispatch floor:
#     GROUP_SIZE  8  deleted  2  (1,1,32,1024)  8c   3729 -> 4307   0.866x  REGRESSION
#     GROUP_SIZE  9  deleted  2  (1,1,32,2304)  9c   4481 -> 5040   0.889x  REGRESSION
#     GROUP_SIZE  8  deleted  2  (1,1,8192,1024) 64c BLOCK 24181 -> 25244  0.958x  REGRESSION
#     GROUP_SIZE 28  deleted 17  (1,1,32,7168) 28c   5717 -> 5881   0.972x  REGRESSION
#     GROUP_SIZE 30  deleted 18  (1,1,32,4800) 30c   5130 -> 4991   1.028x  WIN
#     GROUP_SIZE 32  deleted 20  (1,1,32,5120) 32c   5376 -> 5004   1.074x  WIN
# So the crossover is between deleted 17 and deleted 18 and there is nothing to
# extrapolate: 18 is the smallest MEASURED win and 17 is a measured loss.  (The GROUP_SIZE
# 30 cell is also the RAGGED-run proof: 30 % f0 == 2, so its last level-0 run holds 2 of 4
# slots and that gatherer boot-zeroes its own tail -- and it still wins.)
#
# WHY THE ISOLATED BENCH SAW 1.40x-1.45x AND THE OP SEES 1.03x-1.07x, stated plainly
# because the gap is the interesting part: (a) the bench's FLAT baseline still paid the
# gather's per-face boot zeroing, ~56 API calls at these group sizes, and Perf 3 / D26 has
# since DELETED that from the op -- so most of what the bench credited to the tree was
# already banked on the flat path; and (b) the combine is only ~1.5-2 us of a 5-5.4 us op
# that sits on a ~3.5 us one-core launch/dispatch floor, so even a 1.4x on the combine is
# worth a few hundred ns of wall, not 1.4x of it.  Both effects are arithmetic, not noise.
#
# It is NOT a smooth function of `deleted` alone -- above GROUP_SIZE 16 a SECOND mechanism
# starts dominating (the flat root's GROUP_SIZE - 1 remote writes serialise into one core's
# L1 ingress, which the tree caps at max(f0, f1)) -- so 18 is where the ingress term starts
# paying rather than a pure fold-cost crossover.  To re-check it, force the constant to 0
# and re-run the pinned WIDTH targets; the isolated bench's own `*_compact` cells agree on
# the sign at every point they overlap (GROUP_SIZE 8: 0.76x, 16: 0.94x, 32: 1.25x).
#
# `f1 >= 2` is a SEPARATE and independent gate, and it is why GROUP_SIZE <= 4 can never
# take the tree: the only legal tree at GROUP_SIZE 4 with f0 = 4 has f1 == 1, i.e. a level
# that gathers a single member -- it deletes ZERO fold tiles and pays a pure hop (measured
# 0.78x / 0.85x / 1.02x).  It falls out of `deleted >= 17` too; it is spelled separately
# because it is an EXPRESSIBILITY floor (a one-member level is not a fold), not a cost one.
COMBINE_TREE_F0_MIN = 4
COMBINE_TREE_F0_MAX = 10
COMBINE_TREE_MIN_DELETED_FOLD_TILES = 18

# ---------------------------------------------------------------------------------------
# LAMP L-FIN (Refinement 2) -- WHERE THE FINALIZE RUNS, and the transport it implies.
# ---------------------------------------------------------------------------------------
# `op_design.md`'s stall-shadow table has exactly one row marked NOT BUILT: "every core
# waits for the root's finalize ... nothing is independent of it, because pass B's first
# operand IS the finalized stat."  There are only two places the rsqrt can run, and this
# pair of knobs is the whole choice:
#
#   COMBINE_FIN_SPREAD = False  ROOT.    The root's last-level fold FUSES the finalize
#                                        into its own DEST window (D22) and the multicast
#                                        carries the FINALIZED stat.  One rsqrt per round
#                                        for the whole group, on the root, before the send.
#   COMBINE_FIN_SPREAD = True   SPREAD.  The root packs the RAW group sum, the multicast
#                                        carries THAT, and EVERY core finalizes its own
#                                        copy in parallel -- the literal Lamp L-FIN move.
#
# BOTH ARE CORRECT; the default is a MEASUREMENT, and the measurement says ROOT.  See the
# note at `_combine_fin_spread` for the numbers and for why spreading a REPLICATED (as
# opposed to divisible) term cannot shorten the chain.
COMBINE_FIN_SPREAD = False

# The stat multicast's PRE-HANDSHAKE.  `mcast_pipe` defaults to `handshake=True`: every
# receiver remote-atomic-incs the sender's consumer-ready semaphore and the sender waits
# for `num_active` of them BEFORE it puts any data on the wire.  That protects a landing
# buffer that a previous round might still be draining -- which cannot exist when the
# combine runs exactly ONE round, because then every receiver's landing CB is still at its
# boot state and its write pointer is still the ring base.
#
# The safety argument is an ORDERING one and it is exact, not statistical: a receiver
# constructs its ReceiverPipe (whose ctor is what writes the data-ready flag cell to
# INVALID) BEFORE it ships its own partial into the gather, and the root cannot send until
# every partial has arrived.  So the ctor's INVALID store provably precedes the root's
# VALID broadcast, which is the one race `PRE_HANDSHAKE=false` would otherwise expose.
# With more than one round that argument evaporates (round n+1's send races round n's
# drain), so the gate is `num_blocks == 1` and nothing else.
COMBINE_MCAST_FIRE_AND_FORGET = True

# Faces per stat tile the IDENTITY-path (BLOCK_ROWS == 1) multicast carries.  0 == the
# whole tile, which is also what the COMPACT path is forced to (its un-permute matmul sums
# 32 products, so every column must be finite).  3 is the smallest CONTIGUOUS prefix that
# covers both column-carrying faces (0 and 2) in ONE transaction -- and one transaction is
# the point, because this stage pays per call at a GROUP_SIZE fan-out.  The measured
# licence is D26's, re-used in the other direction; the writer kernel carries the full
# argument at MCAST_FACES.
COMBINE_MCAST_FACES = 3


def _combine_fin_spread(combine: bool) -> bool:
    """Where the finalize runs.  ONE definition, read by the CB table, the L1 solve and
    the compute kernel's CT arg.

    MEASURED and PARKED at ROOT (blackhole p150b 1350 MHz) -- see the changelog entry for
    Refinement 2 for the table.  The reason it does not pay is structural and worth
    keeping next to the knob: the finalize is a REPLICATED term, not a divisible one.
    Spreading work helps when N cores can each do 1/N of it; here every core needs the
    SAME finalized value, so moving the rsqrt off the root does not delete it anywhere --
    it just relocates it from before the multicast to after it, on the identical serial
    chain (fold -> [rsqrt] -> send -> recv -> [rsqrt] -> pass B).  And D22 fused the
    root's rsqrt INTO the fold's DEST window, so at ROOT it costs no pack at all, while a
    spread finalize needs its own copy_tile + pack + unpack on every core.  D27 already
    took the O(BLOCK_ROWS) out of it: the compact tile makes the finalize ONE tile-op per
    round whatever BLOCK_ROWS is, which is the half of Lamp L-FIN that was actually
    expensive.  Kept as a live knob (byte-identical at its default) rather than deleted,
    because it is the only lever that moves `compute_root_fused` at all.
    """
    return bool(combine) and COMBINE_FIN_SPREAD


def _combine_gather_faces_ct(combine: bool, compact: bool) -> int:
    """The writer's ONE packed face-count word: gather faces in the low byte, multicast
    faces in the high byte (0 == whole tile).

    Packed rather than appended because the writer's CT-arg LIST SHAPE is a checked seed
    property; a build at the defaults emits the seed's literal `GATHER_FACES`.
    """
    mcast_faces = 0 if (compact or not combine) else (COMBINE_MCAST_FACES & 0xFF)
    return (GATHER_FACES & 0xFF) | (mcast_faces << 8)


def _combine_mcast_pre_handshake(combine: bool, single_round: bool) -> bool:
    """Whether the stat multicast keeps its receiver-readiness pre-handshake."""
    if not combine:
        return True
    return not (COMBINE_MCAST_FIRE_AND_FORGET and single_round)


# ---------------------------------------------------------------------------------------
# WHICH NoC CARRIES THE COMBINE (Refinement 1, lever 2) -- ONE pair of constants.
# ---------------------------------------------------------------------------------------
# The combine (gather + level-1 forward + stat multicast) lives entirely in the WRITER
# kernel, and a writer is NOC_1 by Metal's own default (`preferred_noc_for_dram_write`),
# so the reader's NOC_0 keeps streaming x through pass A while the combine runs.  That
# trade is only worth paying for WHEN THE READER ACTUALLY HAS AN ACTIVATION STREAM.  It
# does not on any `native_in` plan: x is a resident L1 shard aliased straight into
# cb_input_tiles, so the reader reads NOTHING but the (tiny, face-trimmed) per-channel
# vectors and its NoC sits idle while the combine serialises on the other one.  And
# `master.md`'s `tensix_all_reduce_ring_transport` measures NOC_1 at 6.07-6.14x slower
# than NOC_0 for forwarding across a rectangular group that spans grid ROWS -- which is
# the shape of every WIDTH/BLOCK shard group.
#
# So the NoC is chosen PER PLAN, on exactly that predicate, and it is a SWAP of both
# kernels rather than a one-sided move -- see `_combine_noc_swapped`.
#
# MEASURED (blackhole p150b 1350 MHz, in-process profiler, median of 5, two reps;
# whole-op DEVICE KERNEL DURATION, NOC_1 -> NOC_0):
#
#   native_in (x resident)                                   NOC_1     NOC_0
#     (1,1,32,7168)  WIDTH [32,256] (7,4)   28c   gamma       5768      5698   1.012x
#     (1,1,32,5120)  WIDTH [32,160] (8,4)   32c   gbr/f32     6615      6553   1.009x
#     (1,1,32,5120)  WIDTH [32,160] (8,4)   32c   gamma       5042      4968   1.015x
#     (1,1,32,4800)  WIDTH [32,160] (10,3)  30c   gamma       5072      5049   1.005x
#     (1,1,32,5120)  WIDTH [32,128] (10,4)  40c   gamma       5123      5039   1.017x
#     (1,1,32,8192)  WIDTH [32,128] (8,8)   64c   gamma       5870      5500   1.067x
#     (1,1,8192,1024) BLOCK [1024,128] (8,8) 64c  gamma      24360     23611   1.032x
#     (1,1,32,1024)  WIDTH [32,128] (8,1)    8c   gamma       3727      3694   1.009x
#     (1,1,32,2304)  WIDTH [32,256] (9,1)    9c   gamma       4387      4427   0.991x
#   streamed (x from DRAM through the accessor)
#     (1,1,32,7168)  INTERLEAVED width split 16c  gamma       9044     13550   0.667x  <--
#
# The interleaved row is the whole reason this is a predicate and not a constant: there
# the reader carries every activation byte, and NOC_1 is the wrong engine for a DRAM READ
# (`preferred_noc_for_dram_read` is NOC_0 on every arch).  A 1.5x regression on the op's
# hardest interleaved target is not a trade, it is the gate telling you where it lives.
COMBINE_NOC_RESIDENT = ttnn.NOC.NOC_0  # x is a resident shard: the reader's NoC is free
COMBINE_NOC_STREAMED = ttnn.NOC.NOC_1  # the reader streams x from DRAM: leave NOC_0 to it


def _combine_noc(native_in: bool):
    """The NoC the combine runs on -- ONE definition, read by the host wire AND the kernels."""
    return COMBINE_NOC_RESIDENT if native_in else COMBINE_NOC_STREAMED


def _mcast_cfg(native_in: bool, base_sem_id: int = 0):
    """The combine's McastConfig.

    The `noc` here is not cosmetic: NOC_0 and NOC_1 traverse a rectangle from opposite
    corners, so the host orders the multicast bounding box differently for each.  It must
    agree with the writer kernel's own NoC or the broadcast covers the wrong box, which is
    why both come off `_combine_noc` and nothing else.
    """
    return ttnn.McastConfig(noc=_combine_noc(native_in), handshake=True, base_sem_id=base_sem_id)


def _combine_noc_swapped(plan) -> bool:
    """True when this build moves the combine off NOC_1, which SWAPS BOTH kernels' NoCs.

    The two data-movement processors must sit on DIFFERENT NoCs: `DM_DEDICATED_NOC` gives
    each RISC its own engine, and putting the writer on NOC_0 while the reader is still
    there is not "sharing", it is two kernels driving one engine's command buffers.
    MEASURED: writer -> NOC_0 with the reader left on NOC_0 HANGS -- (1,1,32,4800) WIDTH
    30c timed out with two physical cores never finishing.  So the knob is a SWAP, never a
    one-sided move: writer RISCV_0 + NOC_0, reader RISCV_1 + NOC_1.  That pairing is
    Metal's own `RISCV_n_default`, just the mirror of the reader/writer defaults.
    """
    # Compared by VALUE: the nanobind `NOC` enum exposes NOC_0 / NOC_1 as aliases of
    # RISCV_0_default / RISCV_1_default that do not compare equal to their own alias, so
    # `is` / `==` on the members is not a reliable identity test.
    return bool(plan.combine) and _combine_noc(plan.native_in).value != ttnn.NOC.NOC_1.value


def _writer_dm_config(plan):
    """The writer kernel's data-movement config.

    Off the swap this is exactly `ttnn.WriterConfigDescriptor()` -- Metal's RISCV_0 +
    NOC_1 writer default -- so every build the swap does not select stays byte-identical
    to the seed's.
    """
    if not _combine_noc_swapped(plan):
        return ttnn.WriterConfigDescriptor()
    return ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0,
        noc=_combine_noc(plan.native_in),
    )


def _reader_noc(plan):
    """`per_channel_mcast_v2`: the NoC the READER kernel runs on.

    The per-channel multicast rides the reader's own NoC, and `McastRect` orders the
    broadcast bounding box from that NoC's routing corner, so the host wire must agree
    with the kernel's NoC or the box would be walked from the wrong corner.
    """
    return ttnn.NOC.NOC_1 if _combine_noc_swapped(plan) else ttnn.NOC.NOC_0


def _reader_dm_config(plan):
    """The reader kernel's data-movement config -- the other half of the NoC swap.

    `ttnn.ReaderConfigDescriptor()` (RISCV_1 + NOC_0) unless the writer took NOC_0, in
    which case the reader must vacate it; see `_combine_noc_swapped`.
    """
    if not _combine_noc_swapped(plan):
        return ttnn.ReaderConfigDescriptor()
    return ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.NOC_1,
    )


# Smallest number of tile-rows a core must own before the ROW_RESIDENT regime
# (Lamp L5, D14) is taken at a SHALLOWER CB depth than STREAM would have used.
#
# L5 trades DRAM BYTES (x read once instead of twice, and gamma once per core
# instead of once per pass-B chunk of every row-block) for L1: holding a whole
# tile-row of x can leave no room for the depth the two cross-processor CBs were
# spending on movement<->compute overlap.  At the SAME depth L5 is pure profit and
# this knob does not apply; only the depth-sacrificing case is gated, and there the
# sacrifice only bites when the core has ONE row-block -- nothing to overlap with.
#
# MEASURED (blackhole p150b, 110-core grid, bf16 / HiFi2 / fp32_dest_acc_en=False
# except the RM row, which is the perf probe's HiFi4 config; one fresh-cache
# profiled run per variant; see D14):
#
#   shape                        depth   rows/core   STREAM       ROW_RESIDENT
#   (1,1,8192,5120)  TILE         2->1       3        753345 ns    466594 ns  1.61x
#   (1,1,8192,7168)  TILE         2->1       3       1043918 ns    645578 ns  1.62x
#   (1,1,32,7168)    TILE GRID_W=1 2->1      1         41779 ns     50598 ns  0.83x
#   (1,1,32,4096)    ROW_MAJOR    1->1       1         52197 ns     47226 ns  1.11x
#
# The last two are the whole reason the gate is on the depth SACRIFICE rather than
# on L5: the ROW_MAJOR path is already depth 1 in both regimes, so it gives up
# nothing and wins on bytes even at one row-block.
# 0 would take L5 whenever it fits; a very large value disables it on the TILE path.
ROW_RESIDENT_MIN_ROWS_PER_CORE = 2

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


def _width_chunk(wt_core: int, cap: int, ragged_ok: bool) -> tuple[int, int]:
    """(WT_CHUNK, NUM_W_CHUNKS) for `wt_core` width tiles under an L1 cap of `cap`
    tiles per chunk.  ONE source of truth for the chunk-count decision (D33).

    D1 constrained WT_CHUNK to a DIVISOR of wt_core, which is a GRANULARITY CLIFF at
    a prime width: Wt = 127 has no divisor between 1 and 127, so a cap of 32 collapses
    the chunk to ONE tile and repays every per-phase init / reconfig / pipeline
    fill-and-drain 127 times per block.  The ragged split instead takes the coarsest
    BALANCED chunk `ceil(wt_core / ceil(wt_core / cap))` and PADS the last chunk out to
    it, so every chunk -- and therefore every CB ring, every helper block width and
    every batched NoC group -- stays uniform.  127 at cap 32 becomes 4 chunks of 32
    with ONE pad tile, i.e. 0.8% of wasted width against 32x fewer chunks.

    The balanced form is what keeps the pad negligible: pad = n*wtc - wt_core < n, so
    the wasted fraction is under 1/cap.  Returns the divisor whenever the divisor is
    at least as coarse, so every shape that already had one is byte-identical.
    """
    cap = max(1, min(cap, wt_core))
    div = _largest_divisor_at_most(wt_core, cap)
    if not RAGGED_WIDTH_CHUNK or not ragged_ok or div >= cap:
        return div, wt_core // div
    n = _div_up(wt_core, cap)  # chunks the cap forces
    wtc = _div_up(wt_core, n)  # balanced -> the smallest pad for that chunk count
    if wtc <= div:
        return div, wt_core // div
    return wtc, n


def _norm_cb_depth(has_gamma: bool, has_bias: bool, block_rows: int = 0) -> int:
    """Ring depth of cb_normalized, in whole blocks.  ONE source of truth (D29).

    0  no post-normalize stage at all -- normalize packs cb_output_tiles direct.
    1  exactly ONE post-stage (scale OR bias): it reads cb_normalized and packs
       cb_output_tiles, so the ring is filled once and drained once per block.
    2  BOTH stages: the scale transforms cb_normalized IN PLACE (design B10) and
       the bias then reads it.  transform-in-place ROTATES the ring -- the chain
       pops PASS_B_BLK then reserves PASS_B_BLK -- so a block advances the front
       by `rows * WT_CHUNK`.  With a ring of exactly BLOCK_ROWS * WT_CHUNK that
       is a whole revolution for a FULL block but not for the PARTIAL final one,
       and the bias stage's bulk `cb_wait_front(rows * WT_CHUNK)` + linear tile
       indexing would then read past the ring end.  This is D6's hazard on a
       different CB, and it takes D6's fix: doubling the ring keeps the window
       contiguous for ANY rows <= BLOCK_ROWS.
    """
    if not (has_gamma or has_bias):
        return 0
    if not (has_gamma and has_bias):
        return 1
    # `block_rows == 1` is not a guard, it is the ABSENCE of the hazard: with a
    # one-row block every block has `rows == 1 == BLOCK_ROWS`, so there is no
    # partial final block and the rotation is always a whole revolution.  Depth 1
    # is then exactly correct, and it is worth having as its own case rather than
    # rounding up: every ROW_RESIDENT / STREAM / BAND / width-shard build solves to
    # BLOCK_ROWS == 1, i.e. the L1-tightest regimes are precisely the ones that
    # would otherwise pay a whole extra block of cb_normalized for a hazard they
    # cannot have.  `block_rows == 0` means "not decided yet" and takes the
    # conservative 2 (the RESIDENT search prices it before it knows the block).
    return 1 if block_rows == 1 else 2


def _cb_block_mult(
    depth_x: int,
    depth_out: int,
    has_gamma: bool,
    has_bias: bool = False,
    has_residual: bool = False,
    depth_r: int = 0,
    block_rows: int = 0,
) -> int:
    """Tiles-per-block-tile summed over the BLOCK-SCOPED CBs (op_design.md 1.4).

    ONE source of truth for "which CBs scale with BLOCK_ROWS * WT_CHUNK, and at
    what depth":

        cb_input_tiles (depth_x) + cb_x_squared (1)
        + cb_normalized (_norm_cb_depth) + cb_output_tiles (depth_out)
        + cb_residual_tiles (depth_r) + cb_x_sum (1)     [residual only]

    Both the L1 fit predicate and the STREAM chunk-size solve call this, so a new
    block-scoped CB (or a depth change) is a one-line edit that cannot drift
    between the two solves.

    A1/A2: the residual terms are multiplied by `has_residual` and the bias term
    rides `_norm_cb_depth`, so an operand-free build returns EXACTLY the seed's
    `depth_x + 1 + has_gamma + depth_out` -- nothing is sized for an absent
    operand and no worst-case bucket is taken over the possible operand sets.
    """
    mult = depth_x + 1 + _norm_cb_depth(has_gamma, has_bias, block_rows) + depth_out
    if has_residual:
        mult += depth_r + 1  # cb_residual_tiles + cb_x_sum
    return mult


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
            f"rms_norm_ttnn: {tensor.dtype} is a block-float format and cannot be ROW_MAJOR "
            f"(got layout {tensor.layout})"
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
# Perf 3 / D27 re-cuts the middle of this chain: cb_sum_handoff and cb_row_final are
# now COMPUTE-PRIVATE (the permute consumes one and produces the other), and the two
# CBs that cross to the writer carry ONE COMPACT tile per round instead of BLOCK_ROWS
# column-shaped ones.
CB_SUM_HANDOFF = 10  # fp32: pass A's raw per-row partials, reduce pack -> compact pack
CB_PARTIALS_GATHERED = 11  # fp32: root's per-sender landing slots (ONE page per sender)
CB_STAT_HANDOFF = 12  # fp32: root's finalized COMPACT stat, compute -> writer (mcast src)
CB_ROW_FINAL = 13  # fp32: the un-permuted per-row 1/rms, compute -> compute (pass B)
CB_BANK = 14  # bf16: the one-hot permutation bank E_r, reader -> compute (never popped)
CB_COMPACT_HANDOFF = 15  # fp32: this core's COMPACT partial, compute -> writer (gather src)
CB_MCAST_IN = 16  # fp32: multicast landing of the COMPACT stat, writer -> compute
# --- Perf 3 / D28: the two CBs the SLOT TREE adds (allocated only when it is taken) --
# CB_PARTIALS_GATHERED becomes the LEVEL-0 ring (f0 rounded up to even pages) on the tree
# path -- same index, same producer/consumer roles, just a shorter ring on more cores.
CB_GATHER_L1 = 17  # fp32: the ROOT's level-1 landing ring (f1 rounded up to even pages)
CB_NODE_OUT = 18  # fp32: an interior gatherer's RAW folded sum, compute -> writer

# --- A1 (residual) + A2 (bias): NEW indices, allocated ONLY under their HAS_* flag ----
# Deliberately at NEW slots 19..23 rather than re-numbering 0..18, because "the
# operand-free program is byte-identical to the seed's" is a checked property and a
# renumbering would break it for every configuration, not just the new ones.
CB_RESIDUAL_STICKS = 19  # ROW_MAJOR only: padded row-major staging of the residual
CB_RESIDUAL_TILES = 20  # residual tiles (reader for TILE / zero-copy shard, tilize for RM)
CB_X_SUM = 21  # t = x + r.  Takes over cb_input_tiles' HELD role when a residual is present
CB_BIAS_STICKS = 22  # ROW_MAJOR bias only
CB_BIAS_TILES = 23  # bias tiles (row 0 valid)
# --- D39: the reader-private COMPACT per-channel caches -----------------------
# Producer AND consumer are the reader (it fills them at boot and copies out of
# them per chunk), so they are scratch, never pushed and never popped.
CB_GAMMA_COMPACT = 24  # gamma, 2 face-rows per width tile
CB_BIAS_COMPACT = 25  # bias, ditto


def _combine_tree_candidates(group_size: int):
    """The level-0 fan-ins to try, best-measured first.  See COMBINE_TREE_F0_MAX.

    EXACT DIVISORS OF `group_size` FIRST, LARGEST DOWN, then the plain cap as a ragged
    fallback for a group no divisor in the band can cover (34 = 2 x 17, 31 prime, ...).
    Duplicates are dropped so the fallback never re-tries a divisor.
    """
    band = range(COMBINE_TREE_F0_MAX, COMBINE_TREE_F0_MIN - 1, -1)
    ordered = [f for f in band if group_size % f == 0] + [COMBINE_TREE_F0_MAX]
    seen, out = set(), []
    for f in ordered:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _combine_tree_arity(group_size: int, rows_per_round: int):
    """(f0, f1) for the combine's two-level slot tree, or None to keep the FLAT root.

    THE ONE PLACE THE TREE IS DECIDED (Perf 3 / D28, re-measured in Refinement 1).  Gated
    purely on the derived quantities the mechanism is about -- the level-1 fan-in `f1` and
    the root fold-tiles the tree deletes per round -- so it is blind to shape, dtype,
    layout and placement.  Every constant carries its measured bracket above.

    `f0` is DERIVED rather than fixed: the candidates are walked best-first and the first
    one both gates admit is taken.  A group no candidate satisfies keeps the flat root.
    """
    for f0 in _combine_tree_candidates(group_size):
        f1 = _div_up(group_size, f0)
        # EXPRESSIBILITY: a level that gathers one member is not a fold, it is a hop.
        if f1 < 2:
            continue
        # COST: the fold-tiles taken off the root's critical path, against the extra hop.
        if rows_per_round * (group_size - f0 - f1) < COMBINE_TREE_MIN_DELETED_FOLD_TILES:
            continue
        return f0, f1
    return None


def _combine_fixed_pages(plan, compact: bool, tree) -> int:
    """fp32 pages of the combine's BLOCK_ROWS-INDEPENDENT CBs, per core.

    ONE definition, read by the L1 blocking solve AND by the CB table below, because the
    solve must agree page-for-page with the allocation or BLOCK_ROWS is solved against a
    budget that does not exist.
    """
    if not plan.combine:
        return 0
    if tree is None:
        # cb_partials_gathered: one page per sender, GROUP_SIZE rounded UP TO EVEN (D22).
        pages = plan.group_size + plan.group_size % 2
    else:
        f0, f1 = tree
        # The level-0 ring shrinks to f0 slots and the level-1 ring holds f1; both are
        # rounded UP TO EVEN so every fold is a pairwise DEST walk.  Plus cb_node_out.
        pages = (f0 + f0 % 2) + (f1 + f1 % 2) + CB_COMBINE_FLAT_DEPTH
    pages += CB_COMBINE_FLAT_DEPTH  # cb_stat_handoff
    if compact:
        pages += 2 * CB_COMBINE_FLAT_DEPTH  # cb_compact_handoff + cb_mcast_in
    if _combine_fin_spread(plan.combine):
        # Lamp L-FIN: the spread finalize's landing CB (cb_row_stat, re-used -- it is dead
        # on the combine path otherwise).  BLOCK_ROWS-independent: one raw compact tile in,
        # one finalized tile out, per round.
        pages += CB_COMBINE_FLAT_DEPTH
    return pages


# ---------------------------------------------------------------------------
# A10 — the per-channel operand's two physical FORMS
# ---------------------------------------------------------------------------


def per_channel_form(operand, width: int):
    """`(is_blocked, channel_extent)` for a `weight` / `bias` tensor (A10).

    A per-channel operand carries the same values in two physical forms and only
    the ROW_MAJOR reader path can tell them apart, so a form is NOT a layout and
    the `gamma_layout` axis cannot express it:

      flat     (1, 1, 1, Wg), Wg >= W -- legal at either layout; channel extent
               is the logical last dim.
      blocked  (Wt, 32) ROW_MAJOR, Wt = ceil(W / 32) -- one tile COLUMN per row,
               the trailing lanes of the last row zero-padded; channel extent is
               Wt * 32.

    The detection is deliberately four conjuncts wide (rank 2, trailing 32,
    leading == Wt, ROW_MAJOR) because reading the blocked form's trailing `32` as
    its channel count would refuse every blocked operand with W > 32, and
    treating a flat rank-2 vector as blocked would fetch it column-wise.  Where
    the two are genuinely indistinguishable -- W <= 32, so Wt == 1 and the shape
    is (1, 32) either way -- both readings produce the SAME staged bytes, so the
    ambiguity costs nothing.

    ONE definition, imported by validate() for the support floor and read here
    for the reader's fetch granularity.
    """
    shape = list(operand.shape)
    wt = _div_up(max(1, int(width)), TILE_DIM)
    if operand.layout == ttnn.ROW_MAJOR_LAYOUT and len(shape) == 2 and shape[-1] == TILE_DIM and shape[0] == wt:
        return True, shape[0] * TILE_DIM
    return False, (shape[-1] if shape else 1)


# ---------------------------------------------------------------------------
# DEST budget — the host-side mirror of kernel_lib's DEST_AUTO_LIMIT
# ---------------------------------------------------------------------------


def dest_tile_limit(compute_kernel_config) -> int:
    """DEST tiles available, from the caller's precision config.

    `dest_helpers.hpp:get_dest_limit()` is the kernel-side authority and the
    kernels always read it there; this is the HOST mirror, needed for exactly one
    thing -- refusing a caller's `program_config.subblock_w` that exceeds the DEST
    capacity their own `fp32_dest_acc_en` bought (A5 / X-06).  Kept next to the
    knobs so the two cannot drift silently: full-sync 16/8, half-sync 8/4.
    """
    fp32 = bool(getattr(compute_kernel_config, "fp32_dest_acc_en", False))
    full_sync = bool(getattr(compute_kernel_config, "dst_full_sync_en", False))
    if full_sync:
        return 8 if fp32 else 16
    return 4 if fp32 else 8


# ---------------------------------------------------------------------------
# A5 — `program_config`: the caller's own blocking, consumed
# ---------------------------------------------------------------------------


class ResolvedProgramConfig(NamedTuple):
    """What the op actually takes from a caller-supplied `program_config`.

    Two fields, because only two of the object's fields are degrees of freedom:

      subblock_w  the caller's pass-B DEST-lane block size, or 0 for "the op's
                  own choice" (`pass_b_blk(WT_CHUNK, DEST_AUTO_LIMIT)` in the
                  compute kernel).  HONOURED, never clamped, never absorbed.
      inplace     the output IS the input tensor object.

    Everything else on the object is checked and then NOT acted upon:
    `block_h` / `block_w` restate the input's shard geometry (X-01),
    `compute_with_storage_grid_size` is range-checked but is not a placement
    contract (X-02), and `legacy_reduction` / `legacy_rsqrt` are algorithm
    switches this op is numerically correct at either setting of (so it picks its
    own and answers the same).
    """

    subblock_w: int
    inplace: bool


_PC_NONE = ResolvedProgramConfig(subblock_w=0, inplace=False)


def _pc_is_sharded_variant(program_config) -> bool:
    """Tell the two variants apart by which FIELDS are present, never by type.

    The suite hands the op an object that mirrors the real C++ class field-for-
    field; a `isinstance` gate would refuse it.  `compute_with_storage_grid_size`
    -- and the blocking that travels with it -- marks the sharded variant.
    """
    return hasattr(program_config, "compute_with_storage_grid_size")


def _same_placement(memory_config, tensor) -> bool:
    """Does `memory_config` name the placement `tensor` already has?"""
    tc = tensor.memory_config()
    if memory_config.memory_layout != tc.memory_layout:
        return False
    a, b = memory_config.shard_spec, tc.shard_spec
    if (a is None) != (b is None):
        return False
    if a is None:
        return memory_config.buffer_type == tc.buffer_type
    return tuple(a.shape) == tuple(b.shape) and a.grid == b.grid and a.orientation == b.orientation


def resolve_program_config(input_tensor, *, program_config, memory_config, dest_limit: int) -> ResolvedProgramConfig:
    """Validate a caller's `program_config` and reduce it to what it decides.

    A PURE function of (input placement, the object, the requested output
    placement, the DEST limit), called twice -- once from validate() so the
    refusals land before any device work, once from the entry point so the
    resolved record is not smuggled through global state.

    `program_config=None` does NOT mean "no config": for a sharded input the
    contract is that the op synthesises one from the input's OWN shard spec with
    `inplace` off.  That synthesis is exactly `_PC_NONE` here -- the shard
    geometry is already where every extent comes from (X-01), `inplace` is off,
    and `subblock_w` stays the op's own choice.  DEVIATION worth stating: the
    suite's `sharded_config_for()` derivation writes `subblock_w=1`, and binding
    that would force PASS_B_BLK to 1 on every sharded call that passes no config
    -- a measured 1.12x regression against the seed on the 64-core BLOCK shard
    (feature_spec's own note) for a knob the caller never touched.  The design
    binds `subblock_w` "when supplied" (op_design.md's Axes table), and an
    absent argument supplies nothing.
    """
    is_sharded = input_tensor.memory_config().memory_layout in _SHARDED
    if program_config is None:
        return _PC_NONE

    # `use_welford` sits on BOTH variants and is REFUSED on both: there is no
    # Welford single-pass path here, and a caller who sets it must get an error
    # rather than the two-pass reduce under a name it did not ask for.
    if bool(getattr(program_config, "use_welford", False)):
        raise ValueError(
            "rms_norm_ttnn: program_config.use_welford=True is refused -- this op has no Welford "
            "single-pass statistics path; unset the field to get the two-pass reduce"
        )

    variant_sharded = _pc_is_sharded_variant(program_config)
    if variant_sharded != is_sharded:
        raise ValueError(
            f"rms_norm_ttnn: program_config variant does not match the input's placement -- a "
            f"{'SHARDED' if variant_sharded else 'DEFAULT'} config was supplied for an "
            f"{'sharded' if is_sharded else 'interleaved'} input "
            f"(memory_layout={input_tensor.memory_config().memory_layout!r})"
        )
    if not variant_sharded:
        # The DEFAULT variant carries only the three flags, two of which are
        # accepted-and-unread and the third of which was refused above.
        return _PC_NONE

    # ---- the SHARDED variant -------------------------------------------------
    shard = _shard_shape(input_tensor)
    assert shard is not None, "rms_norm_ttnn: a sharded memory_layout with no shard spec"
    block_h = shard[0] // TILE_DIM
    block_w = shard[1] // TILE_DIM

    # X-02: range-checked, and NOT a placement contract.  Placement follows the
    # input's shard spec, so every in-range value builds the same program.
    grid_req = getattr(program_config, "compute_with_storage_grid_size", None)
    if grid_req is not None:
        device_grid = input_tensor.device().compute_with_storage_grid_size()
        if not (1 <= int(grid_req.x) <= int(device_grid.x) and 1 <= int(grid_req.y) <= int(device_grid.y)):
            raise ValueError(
                f"rms_norm_ttnn: program_config.compute_with_storage_grid_size "
                f"({grid_req.x}, {grid_req.y}) is outside the device's compute grid "
                f"({device_grid.x}, {device_grid.y})"
            )

    # X-01: block_h / block_w RESTATE the shard geometry.  Check the restatement,
    # then take the geometry from the shard.
    for name, supplied, derived in (
        ("block_h", getattr(program_config, "block_h", block_h), block_h),
        ("block_w", getattr(program_config, "block_w", block_w), block_w),
    ):
        if int(supplied) != int(derived):
            raise ValueError(
                f"rms_norm_ttnn: program_config.{name}={int(supplied)} disagrees with the input's shard "
                f"geometry, which fixes {name}={derived} (shard shape {shard}, tile {TILE_DIM}). "
                f"{name} restates the shard spec and cannot express a different blocking"
            )

    # X-06: subblock_w is the ONE free blocking knob.  Honoured, never clamped --
    # each refusal names both operands of the constraint it broke.
    subblock_w = int(getattr(program_config, "subblock_w", 0) or 0)
    if subblock_w < 1:
        raise ValueError(
            f"rms_norm_ttnn: program_config.subblock_w must be at least 1 (it is used as a divisor of "
            f"the pass-B width block); got {subblock_w}"
        )
    if block_w % subblock_w != 0:
        raise ValueError(
            f"rms_norm_ttnn: program_config.subblock_w={subblock_w} does not divide block_w={block_w} "
            f"(the shard's width in tiles), so an outer iteration would be partly empty"
        )
    if subblock_w > dest_limit:
        raise ValueError(
            f"rms_norm_ttnn: program_config.subblock_w={subblock_w} exceeds the DEST capacity "
            f"{dest_limit} tiles available at this compute config "
            f"(fp32_dest_acc_en halves it); lower subblock_w or turn fp32_dest_acc_en off"
        )

    inplace = bool(getattr(program_config, "inplace", False))
    if inplace and memory_config is not None and not _same_placement(memory_config, input_tensor):
        # X-03: under `inplace` the output IS the input, so its placement is the
        # input's.  Refuse a placement the op would have to discard.
        raise ValueError(
            f"rms_norm_ttnn: program_config.inplace=True makes the output the INPUT tensor, whose "
            f"placement is {input_tensor.memory_config()}; the supplied memory_config "
            f"{memory_config} disagrees with it and would be discarded"
        )
    return ResolvedProgramConfig(subblock_w=subblock_w, inplace=inplace)


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
    assert sh is not None, "rms_norm_ttnn: _shard_tile_extent on an interleaved tensor"
    assert sh[0] % TILE_DIM == 0 and sh[1] % TILE_DIM == 0, f"rms_norm_ttnn: TILE shard {sh} is not tile-aligned"
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
    mc_cfg = _mcast_cfg(native_in=False)  # x streams from DRAM here
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
        assert gh == 1, "rms_norm_ttnn: a width group wider than the grid must be the only group"
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
    assert row_cursor == Rt, f"rms_norm_ttnn: work split covers {row_cursor} of {Rt} tile-rows"
    assert len(assignment) == num_cores, f"rms_norm_ttnn: {len(assignment)} cores assigned, expected {num_cores}"
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
            f"rms_norm_ttnn: a ROW_MAJOR {in_ml} input needs an output that is either the SAME "
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
                _mcast_cfg(native_in=False),  # BAND stages x, it does not alias it
                group_size - 1,
            )
            if group_size > 1
            else None
        )
    else:  # _BLOCK_SHARDED -- a true rectangle; each grid ROW is a width group
        nx = bbox.end.x - bbox.start.x + 1
        ny = bbox.end.y - bbox.start.y + 1
        assert (
            shard_grid.num_cores() == nx * ny
        ), f"rms_norm_ttnn: BLOCK shard grid {shard_grid} is not a full rectangle"
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
                _mcast_cfg(native_in=False),  # BAND stages x, it does not alias it
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
                _mcast_cfg(native_in=True),  # x is the resident shard
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
        assert (
            shard_grid.num_cores() == nx * ny
        ), f"rms_norm_ttnn: BLOCK shard grid {shard_grid} is not a full rectangle"
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
                _mcast_cfg(native_in=True),  # x is the resident shard
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


# ---------------------------------------------------------------------------
# A4 — the zero-volume program
# ---------------------------------------------------------------------------


def _zero_volume_descriptor(all_cores, compute_kernel_config):
    """ONE degenerate device program for an input with a 0 dimension.

    The requirement is one native device-program dispatch per invocation, not one
    per element, so a zero-volume call still dispatches -- it does not return
    early on the host and it does not sequence a copy op.

    Every core is `row_count == 0`, which is the INACTIVE-core early return all
    three kernels already take (the seed's width-combine needed it for the cores
    inside a multicast box but outside the shard grid).  That return sits BEFORE
    any accessor construction, any CB touch and `compute_kernel_hw_startup`, so
    the kernels never look at a buffer that has no pages: the accessor CT args
    are the NULL form and every address is 0.  The scalar CT args are the
    smallest values that satisfy the kernels' own static_asserts (one width tile,
    one chunk, one row-block, resident, no combine, no operand).
    """
    bt = ttnn.tile_size(ttnn.bfloat16)
    cbs = [
        _cb(CB_INPUT_TILES, bt, 1, ttnn.bfloat16, all_cores),
        _cb(CB_X_SQUARED, bt, 1, ttnn.bfloat16, all_cores),
        _cb(CB_SCALER, bt, 1, ttnn.bfloat16, all_cores),
        _cb(CB_ROW_STAT, ttnn.tile_size(ttnn.float32), 1, ttnn.float32, all_cores),
        _cb(CB_OUTPUT_TILES, bt, 1, ttnn.bfloat16, all_cores),
    ]
    null_acc = list(ttnn.TensorAccessorArgs().get_compile_time_args())

    reader_ct = [1, 1, 1, 1, 1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    reader_ct += [0, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 0, 0]
    assert len(reader_ct) == READER_CT_SCALARS
    reader_ct += null_acc * 4

    writer_ct = [1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 0, 1, 0, 0, 0, GATHER_FACES, 0, 0]
    assert len(writer_ct) == 18
    writer_ct += [0] * 6 + null_acc

    compute_ct = [1, 1, 1, 1, 0, 0, 0, _f32_bits(1.0), _f32_bits(0.0), REDUCE_BULK, 0, 1, 0, 1, 1, 1, 0, 0, 0]
    compute_ct += [0, 0, 0, 0, 0, 0, 0] + [0]
    assert len(compute_ct) == COMPUTE_CT_SCALARS

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for core in _cores_in(all_cores):
        reader_rt[core.x][core.y] = [0] * 12
        writer_rt[core.x][core.y] = [0] * 12
        compute_rt[core.x][core.y] = [0, 0, 0, 0]

    return ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "rms_norm_ttnn_reader.cpp"),
                core_ranges=all_cores,
                defines=_kernel_defines(),
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "rms_norm_ttnn_writer.cpp"),
                core_ranges=all_cores,
                defines=_kernel_defines(),
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "rms_norm_ttnn_compute.cpp"),
                core_ranges=all_cores,
                defines=_kernel_defines(),
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_kernel_config,
            ),
        ],
        semaphores=[],
        cbs=cbs,
    )


# ---------------------------------------------------------------------------
# CT-arg contract — the ONE place each kernel's scalar-arg count is spelled
# ---------------------------------------------------------------------------
#
# The kernels read their accessor args at TensorAccessorArgs<N>(), so N must
# equal the scalar count exactly.  Named here (and asserted at both emission
# sites) so appending an arg fails in Python instead of mis-parsing on device.
READER_CT_SCALARS = 33
COMPUTE_CT_SCALARS = 27


def create_program_descriptor(
    input_tensor: "ttnn.Tensor",
    output_tensor: "ttnn.Tensor",
    *,
    weight: "ttnn.Tensor" = None,
    bias: "ttnn.Tensor" = None,
    residual: "ttnn.Tensor" = None,
    epsilon: float = 1e-12,
    compute_kernel_config: "ttnn.ComputeConfigDescriptor" = None,
    program_config: "ResolvedProgramConfig" = None,
) -> "ttnn.ProgramDescriptor":
    device = input_tensor.device()
    shape = list(input_tensor.shape)
    resolved_pc = program_config if program_config is not None else _PC_NONE

    is_tile = input_tensor.layout == ttnn.TILE_LAYOUT
    # A1 / A2 -- operand presence is COMPILE-TIME SPECIALIZATION.  Every term
    # below that an operand adds is multiplied by its HAS_* flag, so a call that
    # omits an operand builds the seed's program: no buffer, no CT arg payload,
    # no blocking term, and no worst-case bucket over the possible operand sets.
    has_gamma = weight is not None
    has_bias = bias is not None
    has_residual = residual is not None
    # weight and bias must SHARE a layout when both are present (validated), so
    # one flag covers the per-channel reader path.  Read it off whichever operand
    # is present -- a bias-only call is a per-channel call too.
    per_channel = weight if has_gamma else bias
    per_channel_is_rm = per_channel is not None and per_channel.layout == ttnn.ROW_MAJOR_LAYOUT
    gamma_is_rm = has_gamma and weight.layout == ttnn.ROW_MAJOR_LAYOUT

    # ---- geometry (alignment-aware: ceil everywhere, per image) ------------
    # A3: no rank floor and no rank ceiling.  A tensor with no last extent is a
    # SCALAR -- its reduction is over one element, so W = 1 -- and one with no
    # second-to-last extent has a single row.  Rank 5 needs no case at all: it is
    # rank 4 with one more leading factor in the same product.
    W = shape[-1] if len(shape) >= 1 else 1
    Wt = _div_up(max(1, W), TILE_DIM)
    partial_w = W % TILE_DIM  # 0 => tile-aligned width

    # A4: any zero dimension -> the degenerate program.  Checked here, after the
    # layout/operand flags and before anything divides by W or Rt.
    if _prod(shape) == 0:
        one_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        return _zero_volume_descriptor(one_core, compute_kernel_config)

    if is_tile:
        # Every (..., H, W) image is tile-padded independently.
        Rt = _prod(shape[:-2]) * _div_up(shape[-2], TILE_DIM) if len(shape) >= 2 else 1
        R_rm = 0  # unused
    else:
        # ROW_MAJOR has no implicit H padding: all images' rows are contiguous.
        R_rm = _prod(shape[:-1]) if len(shape) >= 1 else 1
        Rt = _div_up(R_rm, TILE_DIM)

    elem_bytes = _stick_elem_bytes(input_tensor)
    gamma_elem_bytes = _stick_elem_bytes(weight) if has_gamma else 0
    bias_elem_bytes = _stick_elem_bytes(bias) if has_bias else 0

    bt = ttnn.tile_size(input_tensor.dtype)  # x / r / t / x^2 / normalized / out
    gt = ttnn.tile_size(weight.dtype) if has_gamma else 0
    bit = ttnn.tile_size(bias.dtype) if has_bias else 0
    st = ttnn.tile_size(ttnn.bfloat16)  # scaler CB (R4: value exactly 1.0)
    ft = ttnn.tile_size(ttnn.float32)  # cb_row_stat & the combine CBs

    # ---- A10: which physical FORM did each per-channel operand arrive in? ---
    # Only the ROW_MAJOR reader can tell the two apart, and it needs to: the flat
    # form is ONE wide read at a byte offset, the blocked form is WT_CHUNK
    # per-page reads landing at tile-COLUMN offsets of the same staged stick.
    # Both produce byte-identical staging, which is why nothing downstream of the
    # reader knows the difference.
    gamma_blocked = per_channel_form(weight, W)[0] if has_gamma else False
    bias_blocked = per_channel_form(bias, W)[0] if has_bias else False

    # ---- TILE per-channel read granularity (Perf 2, D23) -------------------
    # A TILE-layout per-channel operand is a (1,1,1,W) vector, so the tensor is
    # tile-padded to 32 rows and 31 of every 32 rows are PADDING.  Its only
    # consumer is pass B's mul/add<BroadcastDim::Row>, which reads TILE ROW 0 --
    # so the reader has been moving a whole tile where a couple of face-rows are
    # meaningful.  See the seed's D23 note for the measurement (1.10-1.18x) and
    # for why row 0 really is the only row read (proved by seeding rows 1..31
    # with 1e5x garbage and getting a BIT-IDENTICAL result).
    #
    #   2  FACE-ROWS: two reads of TILE_DIM * elem bytes at page offsets 0 and
    #      tile/4 -- the top row-group of faces 0 and 1, where row 0 lives.
    #      Legal only when a FACE OFFSET is 64-byte DRAM aligned.
    #   1  HALF PAGE: one read of tile/2 from offset 0 == faces 0 and 1 in EVERY
    #      tiled format, block-float included; needs no face-stride alignment.
    #   0  WHOLE TILE / not applicable (a ROW_MAJOR operand has no tile padding).
    #
    # bfloat8_b is the measured exception and it is a FORMAT FACT: its 1088-byte
    # tile has a 272-byte face (16 shared-exponent bytes + 256 mantissa) which is
    # NOT 64-byte aligned, so a face-offset read would be silently truncated down
    # to the alignment.  It demotes to the half page.
    #
    # A2 / Lamp L-OPERAND-TRIM: the bias takes the SAME policy, derived from its
    # OWN tile size rather than copying gamma's number -- the two operands may be
    # at different dtypes, so a shared constant would truncate one of them.  The
    # lamp records that two trimmed reads per chunk change the transaction count,
    # which is a measurement to take, not an argument to settle here.
    # Refinement 3 / Lamp L-OPERAND-TRIM.  The policy above is D23's DERIVED one and
    # stays the single source of the *legality* question (which granularities the
    # dtype's tile layout admits).  `PER_CHANNEL_TRIM_GAMMA` / `_BIAS` are the two
    # measured OVERRIDES on top of it -- `TRIM_DERIVED` means "take the derived
    # answer", and an explicit 0/1/2 forces a granularity.  A forced value is still
    # filtered through the legality rule (a block-float operand can never take the
    # face-row form), so a knob can never produce a truncated read.
    def _trim_for(tile_bytes: int, present: bool, override: int) -> int:
        if not present or per_channel_is_rm:
            return 0
        legal_2 = tile_bytes % 4 == 0 and (tile_bytes // 4) % 64 == 0
        derived = 2 if legal_2 else 1
        if override == TRIM_DERIVED:
            return derived
        if override == 2 and not legal_2:
            return 1  # the dtype refuses the face-row form; fall back, never truncate
        return override

    gamma_trim = _trim_for(gt, has_gamma, PER_CHANNEL_TRIM_GAMMA)
    bias_trim = _trim_for(bit, has_bias, PER_CHANNEL_TRIM_BIAS)

    # ---- D39: is the COMPACT per-channel hold expressible here? --------------
    # It is exactly D23's TRIM == 2 question -- the face-row form has to be legal
    # for EVERY per-channel operand present, because the cache stores what the
    # trim fetches.  A ROW_MAJOR operand already arrives compact (it IS a stick),
    # so it is out of scope; a block-float one demotes to TRIM 1 and is refused.
    pc_compact_ok = (
        bool(PC_COMPACT_HOLD)
        and (has_gamma or has_bias)
        and not per_channel_is_rm
        and (gamma_trim == 2 if has_gamma else True)
        and (bias_trim == 2 if has_bias else True)
    )
    # Compact bytes per WIDTH TILE, summed over the operands present.
    pc_compact_bytes = ((2 * TILE_DIM * gamma_elem_bytes) if has_gamma else 0) + (
        (2 * TILE_DIM * bias_elem_bytes) if has_bias else 0
    )
    pc_compact_rr = pc_compact_ok and bool(ROW_RESIDENT_COMPACT_PC)

    # ---- placement -> scheme, cores, per-core (row, width) extents ---------
    # The scheme decides which axis the cores cut, whether the x / residual / out
    # CBs alias a resident shard, and whether a cross-core combine runs;
    # everything below reads it from the plan rather than re-deriving it.
    def _plan(force_rows: bool = False):
        p = _plan_placement(
            device,
            input_tensor,
            output_tensor,
            is_tile=is_tile,
            Rt=Rt,
            Wt=Wt,
            W=W,
            R_rm=R_rm,
            partial_w=partial_w,
            force_rows=force_rows,
        )
        # The resident shards share L1 with the CB arena, so EVERY one of them is
        # budget the CBs may not spend.  Two corrections to the plan's own
        # (input, output) sum, and both are load-bearing:
        #   * A1 -- a residual on a native scheme is a SECOND resident shard
        #     (feature_spec's own perf note: block-sharded fits <= 120 tiles/core
        #     with one, and fp32_dest_acc_en=True does not fit at all);
        #   * A5 -- under `inplace` the output IS the input tensor, so counting
        #     `_shard_l1_bytes` for both would double-charge one buffer and shrink
        #     the block for no reason.
        seen = []
        for t in (input_tensor, output_tensor, residual):
            if t is None or any(t is s for s in seen):
                continue
            seen.append(t)
        p.l1_reserved = sum(_shard_l1_bytes(t) for t in seen)
        return p

    plan = _plan()

    # ---- the reduce's partial-W mechanism ----------------------------------
    # PARTIAL_W as the KERNELS see it.  The BAND scheme stages each core's band at
    # its REAL element width into a ring whose trailing bytes are zero, so its pad
    # lanes contribute an exact 0 to sum(t^2) with no scaler / mask tile at all --
    # and a per-core band boundary is not expressible as one program-wide
    # PARTIAL_W anyway.  Every other scheme is unchanged.
    kernel_partial_w = 0 if plan.band else partial_w

    # Scaler CB page count: one source of truth for the budget term, the CB
    # allocation and (via PARTIAL_W) the compute kernel's final pop.
    scaler_pages = 2 if kernel_partial_w else 1
    scaler_bytes = st * scaler_pages

    # ROW_MAJOR staging rings that scale with the chunk width: x, the output, and
    # (A1) the residual.  ONE count, read by both L1 solves and the CB table.
    rm_stage_rings = 2 + (1 if has_residual else 0)

    def _solve_blocking(plan):
        """(block_rows, wt_chunk, num_w_chunks, cb_x_depth, cb_out_depth, x_resident,
        rm_stage_depth, narrow_pc_stage) or None.

        None => this plan's per-core block does not fit L1 at all, and the caller
        must fall back to SCHEME_ROWS (which can chunk `width`).
        """
        wt_core = plan.wt_per_core
        # D33's gate, in ONE place.  A ragged (padded) chunk needs the pad tiles to
        # contribute exactly 0 to sum(t^2), which the reader arranges by zeroing them
        # -- and that is only sound when the row's LAST REAL tile is fully valid.  With
        # PARTIAL_W != 0 the reduce's partial scaler / 0-1 mask is aimed at the last
        # tile of the block, which padding would make a pad tile rather than the real
        # one, so a non-tile-aligned width keeps D1's divisor clamp.
        ragged_ok = kernel_partial_w == 0
        avail = ttnn.get_max_worker_l1_unreserved_size()
        if plan.l1_reserved:
            avail -= plan.l1_reserved + L1_CB_ARENA_BASE_RESERVE
        budget = int(max(0, avail) * L1_SAFETY_FRACTION)
        max_rows = max((a.row_count for a in plan.assignment), default=1) or 1
        # HARD CAP on the combine path (D27): the compact partial packs one
        # tile-row's sum into one COLUMN of a single tile, and a tile has TILE_DIM
        # columns.  A combine row-block can therefore never exceed TILE_DIM
        # tile-rows however much L1 is free.  The kernels assert the same bound.
        if plan.combine:
            max_rows = min(max_rows, TILE_DIM)
        # A CB backed on a shard costs ZERO arena bytes (it aliases the tensor's
        # own L1), so its depth term drops out of the block multiplier -- and with
        # no NoC read to overlap, depth buys nothing there either.  A1: the
        # residual's CB is zero-copy on exactly the schemes x's is, because the
        # contract pins its shard spec to the input's.
        dx0 = 0 if plan.native_in else None
        do0 = 0 if plan.native_out else None
        dr0 = 0 if (has_residual and plan.native_in) else None
        depth_candidates = CB_DEPTH_CANDIDATES if is_tile else (1,)
        # D42 retired D41's second, deeper ladder: under D42's block rule a deeper
        # ring can never raise the block count (see the proof at the RESIDENT search),
        # so RESIDENT and everything else share the one ladder again.
        resident_depths = depth_candidates

        combine_tree = _combine_tree_arity(plan.group_size, 1) if plan.combine else None
        # Pages the NARROW per-channel staging ring gets (D30): a knob-derived depth
        # for reader <-> tilize overlap, not a width.  One name so the solve and the
        # CB table cannot disagree.
        rm_stage_depth_for_pc = CB_RM_STAGE_DEPTH

        def _f32_terms(compact):
            per_row = (CB_ROW_STAT_DEPTH + CB_ROW_STAT_DEPTH) if plan.combine else CB_ROW_STAT_DEPTH
            fixed_pages = _combine_fixed_pages(plan, compact, combine_tree)
            bank = 0
            if plan.combine and compact:
                bank = st
            return per_row * ft + bank, fixed_pages * ft

        # Per-channel bytes that scale with the HELD width (one tile CB per
        # operand, plus its stick staging ring on the ROW_MAJOR path).
        #
        # `narrow` is the D30 staging fallback: a per-channel operand is ONE stick,
        # but a `tilize<WT_CHUNK>` staging ring reserves 32 rows' worth of pages to
        # carry it, so at a wide chunk it is the single biggest term in the solve
        # (25 fp32 tiles = 100 kB per operand on the widest band).  Staged one tile
        # COLUMN at a time -- `tilize<1>(WT_CHUNK)` -- the ring is `rm_depth` pages
        # instead of `WT_CHUNK`, for the same tiles bit-for-bit.
        def _per_channel_bytes(width_tiles, staged_tiles, narrow=False):
            total = 0
            staged = 0 if not per_channel_is_rm else (rm_stage_depth_for_pc if narrow else staged_tiles)
            if has_gamma:
                total += width_tiles * gt + staged * gt
            if has_bias:
                total += width_tiles * bit + staged * bit
            return total

        def _resident_fit(depth, compact, rm_depth=CB_RM_STAGE_DEPTH, block_rows=0, narrow_pc=False):
            mult = _cb_block_mult(
                depth if dx0 is None else dx0,
                depth if do0 is None else do0,
                has_gamma,
                has_bias,
                has_residual,
                _residual_depth(depth if dr0 is None else dr0),
                block_rows,
            )
            per_row_bytes, combine_fixed = _f32_terms(compact)
            fixed = (
                _per_channel_bytes(wt_core, wt_core, narrow_pc)
                + (rm_stage_rings * rm_depth * wt_core * bt if not is_tile else 0)
                + scaler_bytes
                + combine_fixed
            )
            # R3: cb_x_squared's REAL width.  The RESIDENT chunk IS `wt_core`, so the
            # D12 fold's predicate is already decidable here; `mult` prices it at the
            # full width, so subtract the difference back off when the fold is on.
            # `CB_SQ_EXACT = 0` keeps the seed's conservative price exactly.
            sq_wt = _x_squared_wt(wt_core, kernel_partial_w) if CB_SQ_EXACT else wt_core
            per_tilerow = wt_core * bt * mult - (wt_core - sq_wt) * bt + per_row_bytes
            return max(0, (budget - fixed) // max(1, per_tilerow)), mult

        # D41 -- RESIDENT PICKS ON ROW-BLOCKS PER CORE, NOT ON DEPTH ORDER.
        # The seed took the coarsest BLOCK_ROWS that fit, which on a shape whose
        # whole assignment fits in one block leaves the core with ONE row-block: it
        # reads all of its work, then computes all of it, then writes all of it,
        # with zero cross-stage overlap.  A deeper ring buys a SMALLER block and
        # therefore MORE blocks, which is the only thing there is to pipeline over.
        # MEASURED: (1,1,8192,1024) interleaved is BLOCK_ROWS == rows_max == 3 -- one
        # block -- at depth 2 and BLOCK_ROWS 2 at depth 3, 84,906 -> 82,961 ns, 1.023x.
        # The TIE-BREAK IS THE SHALLOWEST, and it is what keeps this from costing
        # anything: where the deeper ring cannot raise the block count it buys no
        # overlap at all and is pure L1 -- one tile-row per core (every WIDTH shard,
        # the interleaved decode), or a block count the shallow ring already reaches
        # (the BLOCK shard).  Every one of those keeps its pre-D41 program, byte for
        # byte, because the shallow candidate is evaluated first and only a STRICTLY
        # larger block count displaces it.
        # D42 -- THE BLOCK IS PICKED, NOT INHERITED FROM WHAT FITS.
        #
        # D41 (above) got the OBJECTIVE right -- row-blocks per core are the only
        # thing there is to pipeline read / compute / write over -- but reached for it
        # with the wrong lever: it raised the block count only by DEEPENING the ring,
        # and at each depth it still took `min(max_rows, brmax)`, the LARGEST block
        # that fits.  A smaller block always fits, so the search never even offered
        # itself the finest split.  On the round-3 focus shape that left
        # `BLOCK_ROWS == 2` with 74 of 110 cores holding exactly TWO tile-rows, i.e.
        # ONE row-block: read all of it, compute all of it, write all of it, nothing
        # overlapped.  Perf 3 measured that this is the whole non-roofline half of
        # that shape's wall -- with every payload stubbed the op still costs
        # 14,225 ns (16.8%), it is TRISC-bound (NCRISC span 2,493 ns against a TRISC
        # span of 14,377), and the peel shows it ADDS to the payload rather than
        # hiding behind it (70,923 + 14,225 = 85,148 = the wall).
        #
        # THE RULE IS REGIME-SPLIT, because the two regimes measured OPPOSITE SIGNS:
        #
        #   * x READ OVER THE NoC (`not native_in`): take the FINEST block, one tile
        #     row.  The blocks ARE the pipeline.  MEASURED (1,1,8192,1024) INT
        #     83,997 -> 82,889 ns (1.013x, ~6.7 sigma over 17-20 reads), the masked
        #     (1,1,8192,1000) 84,766 -> 83,512 (1.015x), and FLAT -- never worse --
        #     where a core already holds many row-blocks ((1,1,28160,1024) and up).
        #
        #   * x RESIDENT IN L1 (`native_in`, a zero-copy CB): there is NO read to
        #     overlap, so every extra block is pure per-block overhead.  MEASURED
        #     br 16 -> 1 on the (1,1,8192,1024) BLOCK shard is 20,372 -> 40,618 ns
        #     (0.50x) and on (1,1,7168,1024) BLOCK gbr 28,880 -> 69,566 (0.42x).
        #     That is the carve-out, and it is written as the narrow exception.  What
        #     a native shard CAN still take is the BALANCED block at the SAME block
        #     count -- 32 rows in 2 blocks is 20+12 today and 16+16 balanced, and
        #     that is 20,372 -> 19,493 ns (1.045x).  Only when it divides EXACTLY:
        #     an inexact rebalance (11 -> 10 on case 17) measured 0.993x.
        #
        # DEPTH IS NOW PROVABLY INERT HERE, which is why D41's second ladder is gone
        # (see CB_DEPTH_CANDIDATES).  `br = 1` is the CHEAPEST resident configuration
        # in L1, so if it fits at any depth it fits at the shallowest; and it already
        # yields `max_rows` blocks, the maximum, so no deeper candidate can strictly
        # exceed it.  On a native shard the depth term drops out of the block
        # multiplier altogether (a shard-backed CB costs zero arena bytes), so deeper
        # is at best equal there too.  MEASURED across the focus shape at depths
        # 2/3/4/5/6/8 under this rule: 82,924 / 82,961 / 83,153 / 82,570 / 82,713 /
        # 82,673 ns -- a 0.7% band, i.e. nothing.  Taking depth 2 hands back the L1
        # D41 was spending on depth 3.
        best = None
        for depth in reversed(resident_depths):
            brmax, _ = _resident_fit(depth, compact=True)
            if brmax < 2:
                brmax = min(1, _resident_fit(depth, compact=False)[0])
            if brmax >= 1:
                top = min(max_rows, brmax)
                if plan.native_in:
                    # The exception: a resident shard keeps its coarsest block, and
                    # only levels it out at the same block count.
                    blocks0 = -(-max_rows // top)
                    br0 = -(-max_rows // blocks0)
                    br = br0 if br0 * blocks0 == max_rows else top
                else:
                    br = 1
                blocks = -(-max_rows // br)
                if best is None or blocks > best[0]:
                    best = (blocks, depth, br)
        if best is not None:
            _, depth, br = best
            return br, wt_core, 1, depth, depth, True, CB_RM_STAGE_DEPTH, False, False

        if plan.band:
            # The BAND scheme's WIDTH is shard-derived, so it cannot be chunked, and
            # SCHEME_ROWS cannot address an RM width shard at all (the page is a row
            # SEGMENT).  Its block is therefore fixed at ONE tile-row -- but the
            # STAGING DEPTH is still a live knob, and with three activation rings
            # (x, the residual and the output) instead of two it is the biggest term
            # the band has left to give back.
            #
            # A1: this search is what the residual made necessary.  At the seed's
            # two operands the fixed depth fit every band geometry; a third
            # activation ring plus cb_x_sum plus a second per-channel operand pushed
            # the widest fp32 band (a 8192-element row block-sharded over 88 cores,
            # 25 tile columns per band) to 1.93 MB of CBs against a 1.57 MB L1.
            # Stepping the ring depth to 1 costs the reader<->tilize overlap on a
            # path whose reads are LOCAL L1 rather than DRAM -- the cheapest overlap
            # in the op to give up -- and it is taken only when the budget says so,
            # so every band build that already fit is byte-identical.
            #
            # The last resort is unchanged from the seed: take the shallowest depth
            # and let metal's own CB-region check be the arbiter rather than
            # pre-refusing on a proportional safety margin.
            band_depths = tuple(dict.fromkeys((CB_RM_STAGE_DEPTH, 1)))
            # Ordered COARSEST-FIRST on the thing each step costs: the activation
            # rings' depth first (it costs reader<->tilize overlap on reads that are
            # LOCAL L1 here, the cheapest overlap in the op to give up), then the
            # per-channel staging width (D30 -- it costs WT_CHUNK LLK block calls
            # instead of one, at boot).  Every band build that already fit takes the
            # first candidate and is byte-identical.
            for narrow_pc in (False, True):
                for rm_depth in band_depths:
                    fit = _resident_fit(
                        depth_candidates[0], compact=False, rm_depth=rm_depth, block_rows=1, narrow_pc=narrow_pc
                    )[0]
                    if fit >= 1:
                        return (
                            1,
                            wt_core,
                            1,
                            depth_candidates[0],
                            depth_candidates[0],
                            True,
                            rm_depth,
                            narrow_pc,
                            False,
                        )
            # Last resort, unchanged from the seed: take the smallest footprint the
            # knobs can express and let metal's own CB-region check be the arbiter
            # rather than pre-refusing on a proportional safety margin.
            return 1, wt_core, 1, depth_candidates[0], depth_candidates[0], True, band_depths[-1], True, False
        if plan.scheme != SCHEME_ROWS:
            return None  # a shard-derived width cannot be chunked -> caller falls back

        # ---- the width has to be chunked.  Which of the two chunked regimes? ----
        # ROW_RESIDENT (Lamp L5, D14) first: hold ONE tile-row of the X-role CB and
        # the whole row of each per-channel operand resident, and chunk only the
        # DERIVED CBs, so pass B re-reads NOTHING.
        def _row_resident_chunk(depth_x, depth_out, compact=False):
            """(WT_CHUNK, NUM_W_CHUNKS) for the L5 regime, or None if it cannot fit."""

            # A1: with a residual the HELD role moves from cb_input_tiles to
            # cb_x_sum -- which is depth 1 (compute-private) rather than depth_x --
            # while cb_input_tiles and cb_residual_tiles become chunked streams.
            # That is why the residual does not simply double the held term.
            #
            # D33: the HELD CBs span the PADDED row (`NUM_W_CHUNKS * WT_CHUNK`), not
            # the real one, so the fixed term is a function of the hold width and the
            # ragged candidate is re-priced against it below.
            def _fixed(hold_wt):
                held = (1 if has_residual else depth_x) * hold_wt * bt
                # D39: the COMPACT hold prices the per-channel operands at
                # the 2 face-rows that carry row 0 instead of their whole tiles -- 1/16
                # -- and pays for it with a per-chunk LOCAL expand.  It is a FALLBACK,
                # tried only after the tiled hold has been refused (see the search).
                if compact:
                    held += hold_wt * pc_compact_bytes
                else:
                    held += _per_channel_bytes(hold_wt, 0)
                per_row_bytes, combine_fixed = _f32_terms(compact=False)
                return held + scaler_bytes + per_row_bytes + combine_fixed

            # Per width tile of a CHUNK: cb_x_squared + cb_normalized(+in-place
            # depth) + cb_output_tiles, the two streamed activations when a
            # residual is present, the per-channel stick rings, and the ROW_MAJOR
            # staging rings (D2).
            per_chunk_tile = (
                bt * (1 + _norm_cb_depth(has_gamma, has_bias, 1) + depth_out)
                + (bt * (depth_x + _residual_depth(depth_x)) if has_residual else 0)
                + _per_channel_bytes(0, 1)
                + (rm_stage_rings * CB_RM_STAGE_DEPTH * bt if not is_tile else 0)
                # D39: with a compact hold the per-channel TILE ring becomes
                # CHUNKED (one WT_CHUNK window, popped per chunk) instead of held.
                + (PC_RING_CHUNKS * ((gt if has_gamma else 0) + (bit if has_bias else 0)) if compact else 0)
            )
            room = (budget - _fixed(wt_core)) // per_chunk_tile
            if room < 1:
                return None
            cap = min(room, wt_core - 1) if wt_core > 1 else 1
            if ROW_RESIDENT_CAP_OVERRIDE:
                cap = min(cap, ROW_RESIDENT_CAP_OVERRIDE)
            # The pad tiles are HELD too, so the cap and the chunk it admits are
            # mutually dependent: price each candidate's OWN pad and re-cap until it
            # fits.  Every step strictly shrinks `cap`, so this terminates -- and in
            # practice it takes one step, because pad < NUM_W_CHUNKS.  Re-capping
            # rather than falling straight back to the divisor is what keeps the fix
            # a fix: at Wt = 127 the first candidate (19 x 7, 6 pad tiles) misses by
            # one tile of L1, and the divisor below it is 1 -- the very cliff.
            while True:
                wtc, n = _width_chunk(wt_core, cap, ragged_ok)
                if wtc < 1 or wtc >= wt_core or n <= 1:
                    return None
                pad = n * wtc - wt_core
                if pad == 0:
                    break
                room_pad = (budget - _fixed(wt_core + pad)) // per_chunk_tile
                if wtc <= room_pad:
                    break
                cap = min(cap - 1, room_pad)
                if cap < 1:
                    return None
            if wtc < (ROW_RESIDENT_COMPACT_MIN_CHUNK_WT if compact else ROW_RESIDENT_MIN_CHUNK_WT):
                # L5's hold has priced the chunk below the granularity floor: decline
                # the regime rather than the chunk, so STREAM -- which holds nothing --
                # can take a coarse one.
                return None
            return wtc, n

        stream_depth = depth_candidates[0]
        # D39 -- ORDERED COARSEST-FIRST ON WHAT EACH STEP COSTS, the same
        # discipline the BAND search above uses.  The TILED per-channel hold is tried
        # at every depth FIRST: it stages the operands once per core and pass B reads
        # them with no producer handshake at all.  Only when no depth admits it does
        # the COMPACT hold get a turn -- it buys the regime (and with it the deletion
        # of pass B's re-read of x AND the residual) at the price of a per-chunk local
        # expand sitting on pass B's Upfront wait.  Ordering it second is what keeps
        # every shape that ALREADY fit ROW_RESIDENT byte-identical; measured, taking
        # the compact hold where the tiled one fits COSTS 0.84-0.94x
        # ((1,1,8192,5120) gamma_bias_residual 658,766 -> 782,650 ns,
        #  (1,1,8192,5120) gamma 419,090 -> 454,511, (1,1,8192,7168) gamma
        #  576,476 -> 614,070).
        _hold_forms = (True, False) if COMPACT_FIRST else (False, True)
        _grid = device.compute_with_storage_grid_size()
        _active = sum(1 for a in plan.assignment if a.row_count)
        _compact_ok = pc_compact_rr and _active >= ROW_RESIDENT_COMPACT_MIN_GRID_FRACTION * (_grid.x * _grid.y)
        for compact in _hold_forms if _compact_ok else (False,):
            for depth in tuple(dict.fromkeys(depth_candidates + (1,))):
                if depth < stream_depth and max_rows < ROW_RESIDENT_MIN_ROWS_PER_CORE:
                    continue
                fit = _row_resident_chunk(depth, depth, compact)
                if fit:
                    return 1, fit[0], fit[1], depth, depth, True, CB_RM_STAGE_DEPTH, False, compact

        # STREAM: not even ONE tile-row of the X-role CB fits -> chunk it and
        # re-read x (and the residual) in pass B.  An L1 fallback, not a
        # parallelization.
        depth = depth_candidates[0]
        mult = _cb_block_mult(depth, depth, has_gamma, has_bias, has_residual, _residual_depth(depth), 1)
        per_chunk_tile_bytes = (
            bt * mult + _per_channel_bytes(1, 1) + (rm_stage_rings * CB_RM_STAGE_DEPTH * bt if not is_tile else 0)  # D2
        )
        stream_per_row, stream_combine_fixed = _f32_terms(compact=False)
        fixed_stream = scaler_bytes + stream_per_row + stream_combine_fixed
        wt_chunk_l1_max = max(1, (budget - fixed_stream) // per_chunk_tile_bytes)
        # D1's divisor clamp, or D33's balanced ragged chunk where it is coarser.
        # STREAM holds nothing, so the pad costs no L1 at all here -- only the pad
        # tiles' (zero-valued) compute.
        wtc, n = _width_chunk(wt_core, wt_chunk_l1_max, ragged_ok)
        return 1, wtc, n, depth, depth, False, CB_RM_STAGE_DEPTH, False, False

    solved = _solve_blocking(plan)
    if solved is None:
        # L1 bail-out: the tiled shard's per-core block does not fit even at
        # BLOCK_ROWS == 1, and a shard-derived width is not chunkable.  Re-plan on
        # SCHEME_ROWS, which reads x through the accessor and CAN chunk `width`.
        plan = _plan(force_rows=True)
        solved = _solve_blocking(plan)
        assert solved is not None, "rms_norm_ttnn: no admissible blocking even on SCHEME_ROWS"
    (
        block_rows,
        wt_chunk,
        num_w_chunks,
        cb_x_depth,
        cb_out_depth,
        x_resident,
        rm_stage_depth,
        narrow_pc_stage,
        pc_compact,
    ) = solved
    all_cores = plan.all_cores
    assignment = plan.assignment
    wt_per_core = plan.wt_per_core
    combine = plan.combine
    # A1: the residual carries the input's IDENTICAL shard spec (validated), so it
    # is already in this core's L1 wherever x is -- reading it through a
    # TensorAccessor would re-fetch bytes that are already resident AND add a
    # redundant arena copy.  Two flags rather than one so a future divergence
    # fails loudly instead of silently producing norm(x) for norm(x + r).
    native_residual = has_residual and plan.native_in and _same_shard_spec(input_tensor, residual)
    assert not has_residual or not plan.native_in or native_residual, (
        "rms_norm_ttnn: a native-in scheme with a residual whose shard spec differs from the "
        "input's would silently skip the residual's read"
    )

    assert x_resident or num_w_chunks > 1, "rms_norm_ttnn: a one-chunk width is resident by definition"
    assert not (combine and num_w_chunks > 1), "rms_norm_ttnn: a width-split core takes its slice in one chunk"
    row_resident = x_resident and num_w_chunks > 1
    assert not row_resident or block_rows == 1, "rms_norm_ttnn: ROW_RESIDENT holds ONE tile-row of x"
    compact_combine = combine and block_rows > 1
    assert block_rows <= TILE_DIM or not combine, "rms_norm_ttnn: a compact combine block is at most 32 tile-rows"
    # ---- Refinement 2 (Lamp L-FIN) -- the two derived combine facts ------------------
    # `fin_spread`: WHO runs the rsqrt (see `_combine_fin_spread`).
    # `single_round`: whether EVERY core's combine loop runs exactly one round, which is
    # the multicast pre-handshake's only safety precondition.  `num_blocks` is a per-core
    # runtime quantity (`ceil(num_rows / BLOCK_ROWS)` in both dataflow kernels), and the
    # CT emission is program-wide, so the gate takes the MAX over the assignment.
    fin_spread = _combine_fin_spread(combine)
    single_round = combine and max((a.row_count for a in assignment), default=0) <= block_rows
    mcast_pre_handshake = _combine_mcast_pre_handshake(combine, single_round)
    combine_tree = _combine_tree_arity(plan.group_size, 1) if combine else None
    tree_f0, tree_f1 = combine_tree if combine_tree else (0, 0)
    # D33: PAD tiles the chunking added past the core's real width.  ONE derivation,
    # read by the CB sizes, by both dataflow kernels and by the asserts below; 0 on
    # every divisor build, which is every INPUTS and every perf-group shape.
    wt_pad = wt_chunk * num_w_chunks - wt_per_core
    assert wt_pad >= 0, "rms_norm_ttnn: the width chunking cannot be NARROWER than the core's width"
    assert wt_pad == 0 or (kernel_partial_w == 0 and block_rows == 1 and not combine), (
        "rms_norm_ttnn: a ragged width chunk requires a tile-aligned width, one tile-row "
        "per block and no cross-core width combine"
    )
    # Width tiles the HELD CBs span -- the PADDED row when the chunking is ragged,
    # because chunk c indexes them at c * WT_CHUNK and the last chunk runs to the pad.
    x_hold_wt = wt_chunk * num_w_chunks if x_resident else wt_chunk
    # ---- D39: the compact hold's three derived facts -------------------------
    # `pc_hold_wt` is the CACHE's width -- always the core's whole (padded) row,
    # in every regime, because the cache's whole job is to be read once.
    # `pc_compact` comes back FROM the solve -- it is true exactly when the regime
    # search had to fall back to the compact hold to reach ROW_RESIDENT at all.
    # Every RESIDENT, every already-fitting ROW_RESIDENT and every STREAM build
    # therefore has it False and is BYTE-IDENTICAL to the op's.
    pc_hold_wt = wt_chunk * num_w_chunks
    assert not pc_compact or (x_resident and num_w_chunks > 1), "D39: the compact hold is ROW_RESIDENT's"
    # The per-channel TILE ring is a CHUNKED window (WT_CHUNK pages, popped per
    # chunk) rather than a held row.  Already true in STREAM; the compact cache is
    # what makes it possible under ROW_RESIDENT.
    pc_chunked = (not x_resident) or pc_compact
    pc_tile_pages = (PC_RING_CHUNKS * wt_chunk) if pc_chunked else x_hold_wt

    # A one-line, env-gated dump of the blocking solve.  Not a knob and not read by
    # anything -- it exists so a perf round can see WHICH regime and WHICH chunk a
    # shape resolved to without re-deriving the solve by hand.
    if os.environ.get("RMS_TRACE_BLOCKING"):
        print(
            f"RMS_BLOCKING scheme={plan.scheme} cores={len(assignment)} "
            f"wt_per_core={wt_per_core} BLOCK_ROWS={block_rows} WT_CHUNK={wt_chunk} "
            f"NUM_W_CHUNKS={num_w_chunks} X_RESIDENT={int(x_resident)} "
            f"depth=({cb_x_depth},{cb_out_depth}) partial_w={kernel_partial_w} "
            f"rows_max={max((a.row_count for a in assignment), default=0)} "
            f"PC_COMPACT={int(pc_compact)} PC_CHUNKED={int(pc_chunked)} PC_RING={pc_tile_pages}",
            flush=True,
        )

    # ---- pass A's square: fold into DEST, or pack to L1?  (Lamp L6d, D12) ---
    # With the fold on, `square` runs DestAccumulation::PerRow: the chunk's width
    # tiles are multiplied and ACCUMULATED in DEST, so cb_x_squared receives ONE
    # tile per tile-row instead of WT_CHUNK, and the reduce's per-call width drops
    # to 1.
    #
    # Gated on PARTIAL_W == 0: the pad lanes of the row's last width tile are
    # folded in BEFORE the reduce runs, so the reduce's partial scaler / 0-1 mask
    # can no longer reach them.  A1 does not change that gate -- with a residual
    # the folded operand is `cb_x_sum`, whose pad lanes are `x_pad + r_pad`, and
    # the same gate excludes exactly the same builds.
    # D43: `_x_squared_wt` is the ONE definition; the fold is on whenever it narrows.
    x_squared_wt = _x_squared_wt(wt_chunk, kernel_partial_w)
    square_dest_acc_per_row = x_squared_wt != wt_chunk

    # ---- reduce datapath (D7, refined by D8, carve-out added by D20) -------
    # THREE floors, measuring THREE different quantities -- see the seed's D7/D8/D20
    # notes for the measured brackets behind each.  Unchanged by A1/A2: the reduce
    # sees `cb_x_squared` whatever produced it.
    reduce_acc_via_add = (
        REDUCE_BULK == 1
        and wt_per_core >= REDUCE_ACC_VIA_ADD_MIN_WT
        and wt_chunk >= REDUCE_ACC_VIA_ADD_MIN_CHUNK_WT
        and not (num_w_chunks == 1 and x_squared_wt < REDUCE_ACC_VIA_ADD_MIN_CALL_WT)
    )
    scaler_tiles = 1 if reduce_acc_via_add else scaler_pages
    assert scaler_tiles <= scaler_pages, "rms_norm_ttnn: cb_scaler is sized below the tiles the reader pushes"

    # ---- A9: epsilon = 0.0 on the COMPACT combine path ---------------------
    # The compact finalize's UNUSED columns (rows..31) hold rsqrt(0 * 1/W + eps).
    # At eps > 0 that is finite and the un-permute matmul -- which sums 32
    # products -- is safe.  At eps == 0 exactly they would be +inf, and inf * 0
    # is NaN in EVERY column of the un-permuted result.  The design's stated fix
    # is to clamp the finalize's ADDITIVE TERM rather than widen the lane scope,
    # so the smallest positive normal float stands in for a caller's 0.0 on that
    # one path.  It shifts the real columns' denominator by 1e-38, which is below
    # any representable difference in the result, and it leaves every eps > 0
    # build byte-identical.
    eps_bits = _f32_bits(epsilon)
    if compact_combine and float(epsilon) == 0.0:
        eps_bits = _f32_bits(1.1754944e-38)

    # ---- A5: PASS_B_BLK, the caller's `subblock_w` when supplied -----------
    # 0 means "the op's own choice" and the compute kernel then computes
    # pass_b_blk(WT_CHUNK, DEST_AUTO_LIMIT) exactly as the seed does -- so an
    # absent config is byte-identical.  A supplied value is HONOURED, not
    # clamped; every way it could be illegal was refused in
    # resolve_program_config, which is the only place that decision lives.
    # R3 lever 3: the NoC transaction unit, a divisor of BLOCK_ROWS (one source).
    dm_txn_rows = _dm_txn_rows(block_rows)

    pass_b_blk_ct = int(resolved_pc.subblock_w)
    assert pass_b_blk_ct == 0 or wt_chunk % pass_b_blk_ct == 0, (
        f"rms_norm_ttnn: program_config.subblock_w={pass_b_blk_ct} does not divide the resolved " f"WT_CHUNK={wt_chunk}"
    )

    # ---- circular buffers -------------------------------------------------
    # Every page count below is a function of the block/depth knobs only.
    # cb_input_tiles / cb_residual_tiles / cb_output_tiles are ZERO-COPY when the
    # plan says native: cb_descriptor_from_sharded_tensor aliases the tensor's own
    # resident L1, so there is no NoC read and no arena allocation at all.
    cbs = []
    if not is_tile:
        cbs.append(_cb(CB_INPUT_STICKS, bt, rm_stage_depth * wt_chunk, input_tensor.dtype, all_cores))
        cbs.append(_cb(CB_OUTPUT_STICKS, bt, rm_stage_depth * wt_chunk, output_tensor.dtype, all_cores))
        if has_residual:
            cbs.append(_cb(CB_RESIDUAL_STICKS, bt, rm_stage_depth * wt_chunk, input_tensor.dtype, all_cores))
    in_shard_pages = 0
    out_shard_pages = 0
    if plan.native_in:
        sh_t, sw_t = _shard_tile_extent(input_tensor)
        in_shard_pages = sh_t * sw_t
        assert sw_t == wt_chunk, f"rms_norm_ttnn: native x CB row stride {sw_t} != WT_CHUNK {wt_chunk}"
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_TILES, input_tensor))
    else:
        # A1: with a residual, cb_input_tiles drops from HELD to STREAMING -- the
        # held role (and the `x_hold_wt` span that goes with it) moves to
        # cb_x_sum, which is what pass B indexes back into.  Without one this is
        # the seed's `cb_x_depth * block_rows * x_hold_wt` exactly.
        x_span = wt_chunk if has_residual else x_hold_wt
        cbs.append(_cb(CB_INPUT_TILES, bt, cb_x_depth * block_rows * x_span, input_tensor.dtype, all_cores))
    if has_residual:
        if native_residual:
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_RESIDUAL_TILES, residual))
        else:
            r_depth = _residual_depth(cb_x_depth)
            cbs.append(_cb(CB_RESIDUAL_TILES, bt, r_depth * block_rows * wt_chunk, input_tensor.dtype, all_cores))
        # cb_x_sum spans `row` x `width` at HOLD scope: it is the tensor pass A
        # squares and pass B normalizes, so it takes over cb_input_tiles' held
        # role.  Depth 1 is a whole ring revolution per FULL block, and only the
        # LAST block of a core can be partial, so D6's straddle cannot arise here.
        cbs.append(_cb(CB_X_SUM, bt, block_rows * x_hold_wt, input_tensor.dtype, all_cores))
    # x_squared_wt == 1 under the DEST fold (D12), else wt_chunk -- one source.
    cbs.append(_cb(CB_X_SQUARED, bt, block_rows * x_squared_wt, input_tensor.dtype, all_cores))
    cbs.append(_cb(CB_SCALER, st, scaler_pages, ttnn.bfloat16, all_cores))
    # D6: CB_ROW_STAT_DEPTH * block_rows, so transform_in_place's rotation leaves
    # a PARTIAL final block's stat tiles contiguous for pass B's indexed read.
    # NOT ALLOCATED on the combine path (D27) -- it is strictly dead there.
    if not combine:
        cbs.append(_cb(CB_ROW_STAT, ft, CB_ROW_STAT_DEPTH * block_rows, ttnn.float32, all_cores))
    elif fin_spread:
        # Lamp L-FIN: cb_row_stat comes BACK on the combine path as the spread finalize's
        # output -- the one place a per-core finalized stat can land without giving a CB two
        # producers (cb_mcast_in / cb_row_final are the WRITER's).  Flat in BLOCK_ROWS for
        # the same reason the rest of the combine is since D27: the round's unit is ONE
        # compact tile.
        cbs.append(_cb(CB_ROW_STAT, ft, CB_COMBINE_FLAT_DEPTH, ttnn.float32, all_cores))
    # D30: pages of a per-channel STAGING ring.  `wt_chunk` feeds one
    # `tilize<WT_CHUNK>(1)` call; the narrow fallback feeds `tilize<1>(WT_CHUNK)`
    # from a `rm_stage_depth`-page ring, which is the same tiles bit-for-bit at
    # 1/WT_CHUNK of the L1.  ONE expression, read by both operands.
    pc_stage_pages = rm_stage_depth if narrow_pc_stage else wt_chunk
    if has_gamma:
        if per_channel_is_rm:
            # The stick staging stays CHUNKED even under ROW_RESIDENT -- the tilize
            # consumes it a chunk at a time into the whole-row cb_gamma_tiles.
            cbs.append(_cb(CB_GAMMA_STICKS, gt, pc_stage_pages, weight.dtype, all_cores))
        cbs.append(_cb(CB_GAMMA_TILES, gt, pc_tile_pages, weight.dtype, all_cores))
        if pc_compact:
            cbs.append(_cb(CB_GAMMA_COMPACT, 2 * TILE_DIM * gamma_elem_bytes, pc_hold_wt, weight.dtype, all_cores))
    if has_bias:
        # A2: mirrors gamma exactly, at the bias's OWN dtype -- the two per-channel
        # operands share a layout but not a format, so each CB declares its own.
        if per_channel_is_rm:
            cbs.append(_cb(CB_BIAS_STICKS, bit, pc_stage_pages, bias.dtype, all_cores))
        cbs.append(_cb(CB_BIAS_TILES, bit, pc_tile_pages, bias.dtype, all_cores))
        if pc_compact:
            cbs.append(_cb(CB_BIAS_COMPACT, 2 * TILE_DIM * bias_elem_bytes, pc_hold_wt, bias.dtype, all_cores))
    norm_depth = _norm_cb_depth(has_gamma, has_bias, block_rows)
    if norm_depth:
        cbs.append(_cb(CB_NORMALIZED, bt, norm_depth * block_rows * wt_chunk, input_tensor.dtype, all_cores))
    if plan.native_out:
        osh_t, osw_t = _shard_tile_extent(output_tensor)
        out_shard_pages = osh_t * osw_t
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT_TILES, output_tensor))
    else:
        cbs.append(_cb(CB_OUTPUT_TILES, bt, cb_out_depth * block_rows * wt_chunk, output_tensor.dtype, all_cores))
    if combine:
        # op_design.md's combine section.  See the seed's D18/D22/D25/D27/D28 notes
        # for why each page count is what it is; A1/A2 change none of them (the
        # combine carries a per-row STATISTIC, which no operand touches).
        cbs.append(_cb(CB_SUM_HANDOFF, ft, CB_ROW_STAT_DEPTH * block_rows, ttnn.float32, all_cores))
        if combine_tree is None:
            cbs.append(_cb(CB_PARTIALS_GATHERED, ft, plan.group_size + plan.group_size % 2, ttnn.float32, all_cores))
        else:
            cbs.append(_cb(CB_PARTIALS_GATHERED, ft, tree_f0 + tree_f0 % 2, ttnn.float32, all_cores))
            cbs.append(_cb(CB_GATHER_L1, ft, tree_f1 + tree_f1 % 2, ttnn.float32, all_cores))
            cbs.append(_cb(CB_NODE_OUT, ft, CB_COMBINE_FLAT_DEPTH, ttnn.float32, all_cores))
        cbs.append(_cb(CB_STAT_HANDOFF, ft, CB_COMBINE_FLAT_DEPTH, ttnn.float32, all_cores))
        cbs.append(_cb(CB_ROW_FINAL, ft, CB_ROW_STAT_DEPTH * block_rows, ttnn.float32, all_cores))
        if compact_combine:
            cbs.append(_cb(CB_COMPACT_HANDOFF, ft, CB_COMBINE_FLAT_DEPTH, ttnn.float32, all_cores))
            cbs.append(_cb(CB_MCAST_IN, ft, CB_COMBINE_FLAT_DEPTH, ttnn.float32, all_cores))
            cbs.append(_cb(CB_BANK, st, block_rows, ttnn.bfloat16, all_cores))

    # ---- ROW_MAJOR staging-ring zero (R3, generalized by Refinement 2b) ----
    # The pad bytes of a staged stick are never written by a read, so whatever L1
    # garbage was there would survive into the reduce (and inf*0 / nan*0 = NaN
    # would poison the whole row).  Zeroing the ring ONCE at boot establishes
    # "every pad byte is either zero or real tensor data".  A1: the reader zeroes
    # the RESIDUAL ring too, because `x_pad + r_pad` reaches cb_x_sum and the mask
    # can only rescue it if it is finite.
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

    # ======================================================================
    # D40 (Perf 2) -- the GAMMA_MCAST regime row of the Blocking Model, closed.
    # ======================================================================
    # PRECONDITIONS (all host-checkable; any failure falls back to the pre-D40
    # per-core read and emits the pre-D40 program byte for byte):
    #   * NOT a D39 COMPACT hold.  The two do not compose: with a compact hold the
    #     per-channel staging is a LOCAL L1 expand out of the reader's own cache, so
    #     there is no DRAM read left to broadcast and `stage_per_channel` issues no
    #     NoC transaction for the mcast to replace.  The compact plan's own boot
    #     FILL is broadcastable in principle and is left on the table (a round-3
    #     follow-up), not claimed here.  The reader pins this with a static_assert.
    #   * TILE per-channel operands -- the RM staging ring is a different shape
    #   * X_RESIDENT -- the operands are staged ONCE per core at boot, so every
    #     member of a group runs the same number of multicast rounds.  In STREAM the
    #     round count is ceil(row_count / BLOCK_ROWS) * NUM_W_CHUNKS, which is
    #     PER CORE, so a group would desync; see the STREAM note in the README.
    #   * the core set is a FULL rectangle and every core in it is ACTIVE -- the
    #     reader returns early on `num_rows == 0`, and an inactive receiver would
    #     leave the landing box short of a core that never consumes its bytes
    #   * (w_start, w_real) is CONSTANT inside each group, or the members do not
    #     want the same bytes.  This is what makes the WIDTH / BLOCK shards inert
    #     under "one" / "split": their cores own DISJOINT width slices, so there is
    #     no reuse to remove and the build falls straight back.
    pc_mcast = None
    pc_role = {}
    pc_n = 1
    PC_OPT_OUT = 0xFFFFFFFF
    pc_lines_on = 0
    pc_lines = 0
    if PC_MCAST_MODE is not None and (has_gamma or has_bias) and not per_channel_is_rm and not pc_compact:
        cores = _cores_in(all_cores)
        xs = sorted({c.x for c in cores})
        ys = sorted({c.y for c in cores})
        rect = (
            len(cores) == len(xs) * len(ys)
            and xs == list(range(xs[0], xs[-1] + 1))
            and ys == list(range(ys[0], ys[-1] + 1))
        )
        act = {(a.core.x, a.core.y): a for a in assignment if a.row_count}
        # `one` / `split` broadcast over the WHOLE rectangle, so they need one global
        # group; `col` / `row` cut it into independent LINES and can engage them one at
        # a time.  STREAM's round count is per core, which only a line can reconcile.
        ok = rect and len(act) == len(cores) and (x_resident or PC_MCAST_MODE in ("col", "row"))
        if ok:
            if PC_MCAST_MODE == "col":
                groups = [[(cx, cy) for cy in ys] for cx in xs]
            elif PC_MCAST_MODE == "row":
                groups = [[(cx, cy) for cx in xs] for cy in ys]
            else:
                groups = [[(cx, cy) for cy in ys for cx in xs]]

            def _key(k):
                a = act[k]
                # A group's members must want the SAME BYTES and run the SAME NUMBER OF
                # ROUNDS.  In the resident regimes the round count is NUM_W_CHUNKS on
                # every core, so only the width slice matters; in STREAM it is
                # ceil(row_count / BLOCK_ROWS) * NUM_W_CHUNKS, so the block count joins
                # the key.  A group that fails either test takes the shipped per-core
                # read (role = OPT_OUT) and costs the others nothing.
                blocks = 0 if x_resident else -(-a.row_count // block_rows)
                return (a.w_start, a.w_real, blocks)

            # D40's ONE MEASURED CARVE-OUT.  What the broadcast SAVES is one DRAM read
            # per RECEIVER -- it scales with `len(g) - 1` -- while what it COSTS is a
            # handshake plus a multicast round, which is essentially fixed.  A line of
            # TWO has exactly ONE receiver, so it removes one core's read and pays the
            # whole fixed cost for it.  MEASURED: `(1,1,64,128)` ROW_MAJOR with a TILE
            # weight (Rt=2 -> a 2-core line) is 5,164 -> 5,540 ns, 0.932x, reproducible
            # on min AND median and far outside that cell's noise.
            # `PC_MCAST_MIN_GROUP` is the narrow carve-out around exactly that, and no
            # wider: a line of THREE is UNTESTED and therefore INCLUDED, because the
            # saving doubles while the cost does not.  Lines of 4, 7, 8 and 11 all
            # measured at or above parity (`(1,1,128,4096)` 4 cores 1.030-1.049x, the
            # 64-core HEIGHT shard's 8-core lines 1.592x, the focus shape's 11-core
            # lines 1.073x), so the constant is at the edge of the bracket it was
            # measured in and shrinks if a 3-core line is ever measured to win.
            live = [g for g in groups if len(g) >= PC_MCAST_MIN_GROUP and len({_key(k) for k in g}) == 1]
            pc_lines, pc_lines_on = len(groups), len(live)
            ok = pc_lines_on > 0
        if ok:
            base_sem = 0
            if combine:
                base_sem = plan.gather_sem_id + (1 if combine_tree is None else 2)
            rotating = PC_MCAST_MODE == "split"
            # THE HANDSHAKE IS NOT OPTIONAL IN STREAM.  The resident regimes may elide it
            # because their per-channel CBs are boot-filled once and never popped, so the
            # landing slot is provably idle.  In STREAM the CB is a ring the compute kernel
            # pops, so a broadcast that arrives before the receiver has reserved would land
            # on tiles still being read.
            handshake = PC_MCAST_HANDSHAKE or not x_resident
            cfg = ttnn.McastConfig(
                noc=_reader_noc(plan),
                handshake=handshake,
                data_ready=(ttnn.McastDataReady.Flag if handshake else ttnn.McastDataReady.Counter),
                base_sem_id=base_sem,
                rotating_sender=rotating,
            )
            if PC_MCAST_MODE in ("col", "row"):
                shape = ttnn.Mcast1DShape.PerColumn if PC_MCAST_MODE == "col" else ttnn.Mcast1DShape.PerRow
                if PC_MCAST_DIAGONAL:
                    pc_mcast = ttnn.Mcast1D(
                        device,
                        all_cores,
                        shape,
                        PC_MCAST_SENDER_INDEX,
                        ttnn.Mcast1DSenderPlacement.Diagonal,
                        cfg,
                    )
                else:
                    pc_mcast = ttnn.Mcast1D(device, all_cores, shape, PC_MCAST_SENDER_INDEX, cfg)
                on = {k for g in live for k in g}
                pc_role = {
                    (c.x, c.y): ((1 if pc_mcast.is_sender(c) else 0) if (c.x, c.y) in on else PC_OPT_OUT) for c in cores
                }
            elif PC_MCAST_MODE == "one":
                pc_mcast = ttnn.Mcast2D(device, all_cores, cores[0], cfg)
                pc_role = {(c.x, c.y): (1 if pc_mcast.is_sender(c) else 0) for c in cores}
            else:
                # SPLIT.  The injectors are spread EVENLY through the rectangle in
                # (y, x) order -- the same order `Mcast2D::senders_from_grid_` sorts
                # them into -- so slice i lands on rotating round i.  Spread and not
                # "the first N": N adjacent injectors would hammer one NoC row and
                # would all be far from the far corner of the box.
                pc_n = min(int(PC_MCAST_SPLIT), len(cores), wt_chunk)
                by_yx = sorted(cores, key=lambda c: (c.y, c.x))
                pick = [by_yx[(i * len(by_yx)) // pc_n] for i in range(pc_n)]
                sender_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in pick])
                pc_mcast = ttnn.Mcast2D(device, all_cores, pick[0], cfg, 0, sender_grid)
                order = {(c.x, c.y): i for i, c in enumerate(sorted(pick, key=lambda c: (c.y, c.x)))}
                pc_role = {(c.x, c.y): (order.get((c.x, c.y), -1) + 1) for c in cores}
    if os.environ.get("RMS_PC_TRACE"):
        print(
            f"PC_MCAST mode={PC_MCAST_MODE} engaged={pc_mcast is not None} "
            f"cores={len(_cores_in(all_cores))} "
            f"injectors={sum(1 for v in pc_role.values() if v not in (0, 0xFFFFFFFF))} "
            f"optout={sum(1 for v in pc_role.values() if v == 0xFFFFFFFF)} "
            f"lines={pc_lines_on}/{pc_lines} n={pc_n} "
            f"wt_chunk={wt_chunk} num_w_chunks={num_w_chunks} x_resident={x_resident} "
            f"pc_rm={per_channel_is_rm} combine={combine} handshake={PC_MCAST_HANDSHAKE}",
            flush=True,
        )

    # ---- reader -----------------------------------------------------------
    reader_ct_args = [
        1 if is_tile else 0,  # 0  IS_TILE
        Wt,  # 1  WT (whole-row width tiles: per-channel / x tile ids)
        wt_chunk,  # 2  WT_CHUNK
        num_w_chunks,  # 3  NUM_W_CHUNKS
        _pack_txn_rows(block_rows, dm_txn_rows),  # 4  BLOCK_ROWS | (TXN_ROWS-1)<<16 (R3 lever 3)
        kernel_partial_w,  # 5  PARTIAL_W (0 => aligned, or the BAND scheme)
        1 if has_gamma else 0,  # 6  HAS_GAMMA
        1 if per_channel_is_rm else 0,  # 7  PER_CHANNEL_IS_RM (weight and bias share a layout)
        elem_bytes,  # 8  input element bytes
        gamma_elem_bytes,  # 9  weight element bytes
        R_rm,  # 10 total ROW_MAJOR sticks (0 for TILE)
        W,  # 11 logical width (elements)
        1 if reduce_acc_via_add else 0,  # 12 REDUCE_ACC_VIA_ADD (picks mask vs scaler pair)
        1 if plan.native_in else 0,  # 13 NATIVE_IN: cb_input_tiles aliases the shard
        in_shard_pages,  # 14 pages of the resident x shard to publish (native only)
        1 if plan.band else 0,  # 15 BAND: stage x from THIS core's own RM shard
        plan.shard_row_bytes,  # 16 L1 stick stride inside that shard (0 if !BAND)
        1 if stage_zero else 0,  # 17 zero the RM staging rings at boot
        1 if x_resident else 0,  # 18 X_RESIDENT: held across both passes (D14)
        gamma_trim,  # 19 TILE-weight read granularity (D23)
        (block_rows if compact_combine else 0),  # 20 BANK_PAGES (D27)
        # ---- A1 + A2 + A10: appended, so every operand-free build is unchanged ----
        1 if has_bias else 0,  # 21 HAS_BIAS
        bias_elem_bytes,  # 22 bias element bytes
        bias_trim,  # 23 TILE-bias read granularity (its OWN tile size, not gamma's)
        1 if has_residual else 0,  # 24 HAS_RESIDUAL
        1 if native_residual else 0,  # 25 NATIVE_RESIDUAL: cb_residual_tiles aliases its shard
        1 if gamma_blocked else 0,  # 26 GAMMA_BLOCKED: the (Wt, 32) ROW_MAJOR form
        1 if bias_blocked else 0,  # 27 BIAS_BLOCKED
        # 28 D30: stage a ROW_MAJOR per-channel operand one tile COLUMN per page
        # (`tilize<1>(WT_CHUNK)`) instead of one wide block (`tilize<WT_CHUNK>(1)`).
        # Same staged tiles, 1/WT_CHUNK of the ring; taken only when the L1 solve
        # asks for it, so every build that already fit is byte-identical.
        1 if narrow_pc_stage else 0,
        # 29 D33: PAD width tiles in the LAST chunk (0 on every divisor build).  The
        # reader owns the pad's invariant -- those tiles are never read from the
        # tensor and are zeroed in L1, so they contribute exactly 0 to sum(t^2).
        wt_pad,
        # ---- D39: the COMPACT per-channel hold, appended ------------------------
        (0 if not pc_compact else (2 if PC_COMPACT_LAZY else 1)),  # 30 PC_COMPACT 0/1 eager/2 lazy
        pc_hold_wt,  # 31 width tiles the cache spans (the core's padded row)
        1 if pc_chunked else 0,  # 32 PC_CHUNKED: the per-channel TILE ring is one chunk
    ]
    assert (
        len(reader_ct_args) == READER_CT_SCALARS
    ), f"rms_norm_ttnn_reader.cpp expects TensorAccessorArgs<{READER_CT_SCALARS}>()"
    assert reader_ct_args[READER_CT_BAND] == (1 if plan.band else 0), "READER_CT_BAND index drifted"
    # FOUR accessor blocks, ALWAYS emitted in this order, present or not: the
    # kernel chains its offsets at compile time, so an absent operand must still
    # contribute its (null) block or every later block mis-parses.
    for tensor in (input_tensor, weight, bias, residual):
        args = ttnn.TensorAccessorArgs(tensor) if tensor is not None else ttnn.TensorAccessorArgs()
        reader_ct_args.extend(args.get_compile_time_args())
    # `per_channel_mcast_v2`: appended ONLY when the multicast engages.  Round 1
    # emitted these seven words unconditionally "so the offset chain is
    # build-independent" and paid 4-5% for it on every plan that never engages.
    if pc_mcast is not None:
        inj_g = gamma_trim if PC_MCAST_INJ_TRIM is None else PC_MCAST_INJ_TRIM
        inj_b = bias_trim if PC_MCAST_INJ_TRIM is None else PC_MCAST_INJ_TRIM
        pc_late = 1 if (PC_MCAST_LATE and num_w_chunks == 1) else 0
        reader_ct_args.append(pc_n | (int(inj_g) << 8) | (int(inj_b) << 16) | (pc_late << 24))
        reader_ct_args.extend(pc_mcast.compile_time_args())

    # ---- writer -----------------------------------------------------------
    # The writer owns the whole cross-core combine (gather -> root -> mcast back):
    # it runs on NoC1, which is idle through pass A, so the reader's NoC0 keeps
    # streaming x / the residual / the per-channel operands while the handshake runs.
    # A1/A2 add nothing here -- neither operand crosses the writer's half.
    writer_ct_args = [
        1 if is_tile else 0,  # 0  IS_TILE
        Wt,  # 1  WT
        wt_chunk | (wt_pad << 16),  # 2  WT_CHUNK | WT_PAD<<16 (D33)
        num_w_chunks,  # 3  NUM_W_CHUNKS
        _pack_txn_rows(block_rows, dm_txn_rows),  # 4  BLOCK_ROWS | (TXN_ROWS-1)<<16 (R3 lever 3)
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
        # 15 PACKED face counts: low byte = the IDENTITY-path gather's faces (D13/D26),
        # high byte = the IDENTITY-path stat MULTICAST's faces (Refinement 2; 0 == whole
        # tile, which is also what the compact path is forced to).
        _combine_gather_faces_ct(combine, compact_combine),
        tree_f0,  # 16/17 the SLOT TREE's arity (D28); 0 == keep the flat root
        tree_f1,
    ]
    assert len(writer_ct_args) == 18, "rms_norm_ttnn_writer.cpp expects McastArgs<18, 12>()"
    assert (
        writer_ct_args[WRITER_CT_OUT_SHARD_ROW_BYTES] == plan.out_shard_row_bytes
    ), "WRITER_CT_OUT_SHARD_ROW_BYTES index drifted"
    # Refinement 2: the mcast helper's own per-kernel `pre_handshake` override is the whole
    # of the fire-and-forget lever -- one flags bit, no new CT arg, no kernel change (both
    # SenderPipe and ReceiverPipe read the same bit).  `_combine_mcast_pre_handshake` is the
    # single place the gate lives.
    writer_ct_args.extend(plan.mcast.compile_time_args(pre_handshake=mcast_pre_handshake) if combine else [0] * 6)
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # ---- compute ----------------------------------------------------------
    compute_ct_args = [
        1 if is_tile else 0,  # 0  IS_TILE
        wt_chunk,  # 1  WT_CHUNK
        num_w_chunks,  # 2  NUM_W_CHUNKS
        block_rows,  # 3  BLOCK_ROWS
        kernel_partial_w,  # 4  PARTIAL_W (0 on the BAND scheme)
        1 if has_gamma else 0,  # 5  HAS_GAMMA
        1 if per_channel_is_rm else 0,  # 6  PER_CHANNEL_IS_RM
        _f32_bits(1.0 / float(W)),  # 7  INV_W (raw fp32 bits) -- the LOGICAL width
        eps_bits,  # 8  EPS (raw fp32 bits; A9 clamps a 0.0 on the compact path)
        REDUCE_BULK,  # 9  reduce input policy knob
        1 if reduce_acc_via_add else 0,  # 10 REDUCE_ACC_VIA_ADD (reduce datapath, D7)
        scaler_tiles,  # 11 tiles the reader pushed into cb_scaler
        1 if combine else 0,  # 12 COMBINE: partial sum -> gather -> mcast-back
        plan.group_size,  # 13 cores per width group (GRID_W)
        x_squared_wt,  # 14 reduce's per-call width == cb_x_squared's row stride (D12)
        1 if x_resident else 0,  # 15 X_RESIDENT: held across both passes (D14)
        1 if plan.native_in else 0,  # 16 NATIVE_IN: cb_input_tiles aliases the shard (D25)
        tree_f0,  # 17 SLOT TREE level-0 fan-in, 0 == flat root (D28)
        tree_f1,  # 18 SLOT TREE level-1 fan-in (== ceil(group_size / f0))
        # ---- A1 + A2 + A5: appended, so every operand-free build is unchanged ----
        1 if has_bias else 0,  # 19 HAS_BIAS
        1 if has_residual else 0,  # 20 HAS_RESIDUAL
        pass_b_blk_ct,  # 21 PASS_B_BLK override (0 == the op's own pass_b_blk)
        1 if narrow_pc_stage else 0,  # 22 NARROW_PC_STAGE (D30) -- tilize<1>(WT_CHUNK)
        # 23 Refinement 2 / Lamp L-FIN: FIN_SPREAD.  0 == the root finalizes inside its
        # fold's DEST window and the multicast carries the finalized stat; 1 == the root
        # forwards the RAW group sum and every core finalizes its own copy.  Appended, so
        # every build at the default is byte-identical to the seed's prefix.
        1 if fin_spread else 0,
        # ---- Refinement 3: the two pass-A knobs, appended (default 0 == the seed) ----
        PASS_A_SQ_BLOCK,  # 24 pass A's square takes pass B's DEST-lane block size
        RES_FUSE,  # 25 Lamp L-RES-FUSE: t = x + r and its square as ONE chain
        # 26 D39: the per-channel TILE CB is a CHUNKED window (WT_CHUNK
        # pages, popped after every chunk) rather than a held whole row.  Was
        # implicitly `!X_RESIDENT`; the compact hold decouples the two.
        1 if pc_chunked else 0,
    ]
    assert len(compute_ct_args) == COMPUTE_CT_SCALARS, "compute CT-arg count drifted"
    assert x_squared_wt >= 1 and wt_chunk % x_squared_wt == 0, (
        "rms_norm_ttnn: x_squared_wt must divide WT_CHUNK (1 == the flat DEST fold, D43)"
    )

    # ---- the SLOT TREE's ONE extra runtime fact: my level-0 gatherer's coords -------
    tree_parent = {}
    if combine_tree is not None:
        per_row_groups = isinstance(plan.mcast, ttnn.Mcast1D)
        slot_core = {}
        for w in assignment:
            if w.row_count:
                slot_core[(w.core.y if per_row_groups else 0, w.slot)] = w.core
        for w in assignment:
            if not w.row_count:
                continue
            key = w.core.y if per_row_groups else 0
            parent = slot_core[(key, (w.slot // tree_f0) * tree_f0)]
            v = device.worker_core_from_logical_core(ttnn.CoreCoord(parent.x, parent.y))
            tree_parent[(w.core.x, w.core.y)] = [v.x, v.y]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    x_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    g_addr = weight.buffer_address() if has_gamma else 0
    b_addr = bias.buffer_address() if has_bias else 0
    r_addr = residual.buffer_address() if has_residual else 0
    for w in assignment:
        core = w.core
        # owns_last_w: only the core holding the row's LAST width tile applies the
        # partial-W scaler / mask.
        owns_last_w = 1 if (w.w_start + w.w_real >= Wt) else 0
        band_rt = [w.stick_base, w.stick_count, w.w_off_elems, w.w_real_elems]
        # 10/11 appended (A2/A1), so indices 0..9 are the seed's exactly.
        reader_rt[core.x][core.y] = (
            [x_addr, g_addr, w.row_start, w.row_count, w.w_start, w.w_real]
            + band_rt
            + [
                b_addr,
                r_addr,
            ]
            # `per_channel_mcast_v2`: 12 = this core's ROLE (0 = pure receiver,
            # 1 + i = injector of slice i), 13.. = the multicast wire.  Appended
            # ONLY on an engaged plan -- an off build ships the shipped 12-word block.
            + ([pc_role.get((core.x, core.y), 0)] + list(pc_mcast.runtime_args(core)) if pc_mcast is not None else [])
        )
        writer_rt[core.x][core.y] = (
            [out_addr, w.row_start, w.row_count, w.w_start, 1 if w.is_root else 0, w.slot]
            + band_rt
            + tree_parent.get((core.x, core.y), [0, 0])
            + (list(plan.mcast.runtime_args(core)) if combine else [])
        )
        compute_rt[core.x][core.y] = [w.row_count, owns_last_w, 1 if w.is_root else 0, w.slot]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_ttnn_reader.cpp"),
        core_ranges=all_cores,
        # `per_channel_mcast_v2`: the ONE define that switches the multicast on.  The
        # writer and compute kernels never see it, so their build keys are untouched.
        defines=_kernel_defines() + ([("RMS_PC_MCAST", "1")] if pc_mcast is not None else []),
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=_reader_dm_config(plan),  # NoC0, or NoC1 when the combine swaps
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_ttnn_writer.cpp"),
        core_ranges=all_cores,
        defines=_kernel_defines(),
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=_writer_dm_config(plan),  # NoC1, or NoC0 on a resident-x combine
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_ttnn_compute.cpp"),
        core_ranges=all_cores,
        defines=_kernel_defines(),
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=compute_kernel_config,  # passed through unmodified
    )

    semaphores = []
    if combine:
        semaphores = list(plan.mcast.owned_semaphores())
        for lvl in range(1 if combine_tree is None else 2):
            semaphores.append(
                ttnn.SemaphoreDescriptor(id=plan.gather_sem_id + lvl, core_ranges=all_cores, initial_value=0)
            )

    if pc_mcast is not None:
        semaphores = semaphores + list(pc_mcast.owned_semaphores())

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=cbs,
    )
