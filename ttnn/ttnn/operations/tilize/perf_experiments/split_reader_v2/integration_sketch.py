# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PATCH SKETCH for graduating `split_reader_v2` into the real tilize op.
# NOT executed and NOT imported by anything — this file is the recipe, written as
# code so the coordinator can lift it straight into
# `tilize_program_descriptor.py` / `kernels/`. Measured numbers: results.tsv.
#
# ===========================================================================
# 0. WHAT THE MEASUREMENT SAYS THE PREDICATE IS
# ===========================================================================
# Written in the exception polarity the op already uses (`if cannot: legacy`).
# Every clause is a MEASURED regression or an inexpressibility, never a
# suspicion — the reasons are in results.tsv.
#
#   split_reader_ok = (
#       out_placement == P_LOCAL_SHARD          # BRISC has NO write duty to give up.
#       and in_placement == P_ACCESSOR          # there IS a read to split.
#       and not out_fill                        # the writer must still stamp pads.
#       and blocks_per_core >= 2                # nothing to split otherwise (flat).
#       and regime != R_RETILE                  # untested; reader is L1-permute bound.
#   )
#
# and the FLAVOR is picked by the SOURCE BUFFER TYPE, not by the read regime
# (`gather_h` proves this: an L1 source in the R_ALIGNED regime behaves like the
# paged L1 gather, not like DRAM):
#
#   if source is DRAM:
#       # NOC_1 is the slow NoC for a DRAM read (Metal's own
#       # preferred_noc_for_dram_read() == NOC_0), so BOTH readers go on NOC_0
#       # under DM_DYNAMIC_NOC and separate their barriers with per-RISC trids.
#       flavor = SHARED_NOC0_TRID          # 1.50x / 1.55x / 1.65x measured
#       # ... but ONLY while the read is issue-bound. Measured ladder:
#       #   512 B/stick -> 1.50-1.65x | 1024 B -> 1.07x flat | 2048 B -> 0.86x
#       #   4096 B -> the second CB does not fit L1 at all.
#       if read_bytes_per_stick(wt_chunk, in_tile_bytes, tile_h) > 1024:
#           split_reader_ok = False        # measured regression, see results.tsv
#   else:                                  # a NON-LOCAL L1 shard (any regime)
#       flavor = DEDICATED_DUAL_NOC        # 1.72x / 1.77x / 1.60x / 1.24x measured
#       # no transfer-size clause: this flavor still wins 1.60x at a 2 KB span.
#
# Consequence worth stating explicitly: on the DRAM flavor the transfer-size gate
# BOUNDS the L1 cost by itself. read_bytes_per_stick <= 1024 means one block is
# <= 32 KB, so two depth-2 input CBs are <= 128 KB against the current 64 KB —
# the extra L1 can never exceed 64 KB per core. On the L1-source flavor the extra
# is one more `cb_pages(cb_depth, wt_chunk) * in_tile_bytes`, i.e. exactly one
# more streaming input CB, which is what item 3 below feeds back into `wt_cap()`.

# ===========================================================================
# 1. HOST — tilize_program_descriptor.py
# ===========================================================================
# 1a. new CB index (CB_RETILE_STAGE=1 is already taken; the retile path and the
#     split path are mutually exclusive by the predicate above, so slot 1 could be
#     shared — but a distinct index is one less thing to reason about).
CB_INPUT_STICKS_B = 3  # reader#2 -> compute (the split's second input CB)

# 1b. new lever, with its OFF arm (this is how the bench arms become the ledger's
#     counterfactual):
#     LEVERS["split_reader"] = 1   # 0 -> single reader, byte-identical to today

# 1c. the derivation, placed right after `out_fill` / `read_trid` are known:
"""
    SPLIT_NONE, SPLIT_DUAL_NOC, SPLIT_SHARED_NOC0 = 0, 1, 2

    src_is_dram = input_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
    split_reader = SPLIT_NONE
    if (
        LEVERS["split_reader"]
        and out_placement == P_LOCAL_SHARD
        and in_placement == P_ACCESSOR
        and not out_fill
        and regime != R_RETILE
        and min(blocks_per_shard) >= 2
    ):
        if src_is_dram:
            # Measured: flat at a 1 KB stick, 0.86x at 2 KB, L1-infeasible at 4 KB.
            if read_bytes_per_stick(wt_chunk, in_tile_bytes, tile_h) <= 1024:
                split_reader = SPLIT_SHARED_NOC0
        else:
            split_reader = SPLIT_DUAL_NOC
"""

# 1d. the L1 ceiling must learn about the second CB, or a wide shard OOMs
#     (measured: crossover_2048 [1,1,2048,2048] -> H x8 throws at program
#     creation, "static circular buffers clash with L1 buffers"). ONE knob:
"""
    in_cbs = 2 if split_reader else 1
    # every existing call site: wt_cap(...) / derive_shard_blocking(...) /
    # the `cb_bytes(...) > CB_L1_BUDGET` fallbacks take `in_cbs * stream_in`
    # instead of `stream_in`.
"""
#     Deriving `split_reader` BEFORE the blocking makes it self-consistent; the
#     transfer-size gate then keeps the DRAM flavor's blocking unchanged in
#     practice (a <=1 KB stick never approaches the cap).

# 1e. the CB descriptors: the second input CB is the SAME descriptor as the first
#     (same page size, same tile, same dtype), at `cb_pages(cb_depth, wt_chunk)`
#     pages. Depth 2 each is what every winning arm measured (the periodic
#     weighted arms wanted depth 4+2; they are NOT the recommendation).
#     TWO SEPARATE CBs, not one CB with two issuers — see §3.

# 1f. kernels: the WRITER kernel keeps its descriptor but takes a second role.
#     On the split path the two data-movement kernels become symmetric readers:
#       reader kernel   (NCRISC)  ct: ..., split=1, phase=0, cb_in=CB_INPUT_STICKS
#       writer kernel   (BRISC)   ct: ..., split=1, phase=1, cb_in=CB_INPUT_STICKS_B
#     with `out_placement` forced to a NEW value P_NONE on the writer (it has no
#     drain duty any more — compute took it), and the configs:
"""
    if split_reader == SPLIT_SHARED_NOC0:
        types = ttnn._ttnn.types
        reader_config = ttnn.DataMovementConfigDescriptor(
            types.DataMovementProcessor.RISCV_1, types.NOC.RISCV_0_default, ttnn.NOC_MODE.DM_DYNAMIC_NOC)
        writer_config = ttnn.DataMovementConfigDescriptor(
            types.DataMovementProcessor.RISCV_0, types.NOC.RISCV_0_default, ttnn.NOC_MODE.DM_DYNAMIC_NOC)
        # BOTH must be DYNAMIC: the dynamic read barrier sums the two RISCs'
        # issue counters, so a dedicated-mode partner would not publish into the
        # shared counter. Each reader then tags its own reads with a trid
        # (1 and 2) and barriers on that id alone, which is per-transaction-id
        # hardware state and therefore RISC-agnostic.
    else:
        reader_config = ttnn.ReaderConfigDescriptor()   # NCRISC / NOC_0
        writer_config = ttnn.WriterConfigDescriptor()   # BRISC  / NOC_1
"""

# 1g. runtime args: NOTHING new. Both DM kernels get the SAME per-core args they
#     get today (region origin + block count); the block subset is a COMPILE-TIME
#     phase (0/1) plus the stride, so there is no host-side block ledger to keep
#     in sync. That is the whole reason the stride-2 interleave is the shipped
#     split and the contiguous half is not.

# ===========================================================================
# 2. KERNELS
# ===========================================================================
# 2a. kernels/tilize_reader.cpp — two new compile-time args, `split` and `phase`,
#     and ONE change in the W_REGION / R_ALIGNED / R_PAD block loops:
#
#         for (uint32_t i = 0; i < num_blocks; ++i) {
#     ->  for (uint32_t i = phase; i < num_blocks; i += (split ? 2 : 1)) {
#
#     The `n_chunks == 1` FAST PATH (one `read_sticks_for_tilize` call over
#     num_blocks*tile_h contiguous sticks) has to become one call per OWNED block
#     on the split path — which costs nothing, because the helper already
#     barriers and pushes PER BLOCK internally
#     (tilize_helpers_dataflow.inl:110-127): splitting one N-block call into N
#     one-block calls changes no transfer, no barrier and no push. Measured
#     `raw_baseline` vs `op_baseline` (the whole per-call prologue, raw vs helper)
#     is 0.99-1.01x on both focus plans, i.e. inside noise.
#
#     On the SHARED_NOC0 flavor the reads must be issued RAW (see §4) and tagged:
#         noc_async_read_set_trid(TRID);       // 1 on NCRISC, 2 on BRISC
#         ... noc_async_read(...) per stick ...
#         noc_async_read_barrier_with_trid(TRID);
#         ... and noc_async_read_set_trid(0) at the end, for cmd-buffer hygiene.
#
# 2b. kernels/tilize_writer.cpp — on the split path this kernel IS the second
#     reader. The cleanest shape is to give the reader source file a `phase`
#     compile-time arg and point BOTH kernel descriptors at
#     `tilize_reader.cpp`, leaving `tilize_writer.cpp` untouched and unused on
#     that path. (The bench does exactly this: one `sr2_dm.cpp` for both RISCs,
#     so the baseline and candidate cannot diverge in anything but CT args.)
#
# 2c. kernels/tilize_compute.cpp — the alternation. The library helper ALREADY
#     supports it through its documented back-to-back lifecycle, so NO raw LLK:
#
"""
        using namespace compute_kernel_lib::tilize_config;
        compute_kernel_lib::tilize<wt_chunk, CB_A, CB_OUT, InitUninitMode::InitOnly, ...>(0);
        for (uint32_t i = 0; i < num_blocks; ++i) {
            if (i & 1) compute_kernel_lib::tilize<wt_chunk, CB_B, CB_OUT, InitUninitMode::Neither, ...>(1);
            else       compute_kernel_lib::tilize<wt_chunk, CB_A, CB_OUT, InitUninitMode::Neither, ...>(1);
            cb_wait_front(CB_OUT, wt_chunk);   // compute takes over the drain
            cb_pop_front(CB_OUT, wt_chunk);    // so the aliased CB keeps ONE consumer
        }
        compute_kernel_lib::tilize<wt_chunk, CB_A, CB_OUT, InitUninitMode::UninitOnly, ...>(0);
"""
#     THE ALTERNATION IS FREE. Control `alt_tax` (one reader, one CB, but this
#     per-block form) measured 14,768 / 14,727 ns against `op_baseline`'s
#     14,780 / 14,908 on the crossover and 18,281 / 18,142 against 18,406 /
#     18,613 on the reshard — flat within 1% on both. That is the single result
#     that closes Perf-1's "nobody has measured whether the alternation eats the
#     win".
#
#     The compute-side drain likewise costs nothing on the DRAM flavor
#     (split_trid 9,857 vs split_trid_cdrain 9,919) and ~6% on the L1 flavor
#     (split_il 10,134 vs split_il_cdrain 10,814) — worth paying to keep the
#     "exactly one consumer" contract the design doc states.

# ===========================================================================
# 3. CB TOPOLOGY — measured/derived decision
# ===========================================================================
# TWO SEPARATE CBs, depth 2 blocks each.
#   * ONE CB filled by two issuers is INEXPRESSIBLE safely: `cb_push_back` moves a
#     single shared write pointer, so two RISCs pushing into one CB race, and
#     ordering them back into block order needs a semaphore handshake per block —
#     which re-serializes exactly the issue the split is trying to parallelize.
#     Not benched; the contract rules it out.
#   * Contiguous halves (Perf-1's `split_w75`) need the second CB to be as deep as
#     the second RISC's WHOLE half — compute does not touch that half until it has
#     drained the first, so a capped depth just serializes the two halves. That is
#     L1 that scales with BLOCKS-PER-CORE, i.e. with the tensor, which the op's CB
#     contract forbids. Measured anyway (1.19x / 1.28x), and beaten by the
#     bounded-depth arms.
#   * The PERIODIC weighted interleave (new here, `split_p60/p67/p75`) reproduces
#     the same 3:1 ratio at bounded depth and is the best PREDICATE-FREE option
#     (1.29x crossover / 1.31x reshard / 1.34x big / 1.38x tall / 1.26x gather_h)
#     — but it still regresses on the >=2 KB-stick DRAM plans (0.72-0.88x), so it
#     removes the flavor predicate, not the transfer-size one. Recommended only if
#     the coordinator wants one path instead of two.
#
# L1 cost, per core, on the recommended arms (from results.tsv):
#   crossover      32,768 -> 65,536 B   (+32 KB)
#   crossover_512  65,536 -> 131,072 B  (+64 KB)   <- the gate's worst case
#   reshard        32,768 -> 65,536 B   (+32 KB)
#   reshard_wide  131,072 -> 262,144 B  (+128 KB)  <- L1-source flavor, ungated
#   crossover_2048 262,144 -> 524,288 B => DOES NOT FIT (excluded by the gate)

# ===========================================================================
# 4. HELPER BYPASSES TO DECLARE TO THE VERIFIER
# ===========================================================================
# dataflow_kernel_lib::read_sticks_for_tilize — bypassed ONLY on the
#   SHARED_NOC0 (DRAM) flavor. Gap class: `capability`. What is missing: the
#   helper owns its `noc_async_read_barrier()` internally and its contract
#   exposes no transaction id, so two readers sharing NOC_0 cannot barrier on
#   their own reads through it. Same transfers, same one-barrier-per-block
#   policy — only the barrier's SCOPE changes. The op ALREADY documents this
#   exact substitution for master.md B8 (`read_trid`), so the bypass is not new
#   surface. On the DEDICATED_DUAL_NOC (L1-source) flavor the helper is used
#   verbatim, once per owned block — no bypass at all.
# compute_kernel_lib::tilize — NOT bypassed anywhere. The alternating form is the
#   helper's own documented InitOnly / Neither / UninitOnly lifecycle. No
#   `capability` gap on the compute side.
