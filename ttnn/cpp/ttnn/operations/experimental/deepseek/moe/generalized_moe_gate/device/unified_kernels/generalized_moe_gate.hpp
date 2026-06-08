// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include <cstdint>
#ifndef REDUCE_OP
#define REDUCE_OP PoolType::SUM
#endif
#ifndef REDUCE_DIM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#endif
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/generalized_moe_gate.h"
#endif

namespace deepseek_v3_ops {

// ============================================================================
// GeneralizedMoeGate micro-op
//
// Computes top-8 expert selection with normalized scores.
// Input: router logits [16, 16] + bias [16, 16] + indices [16, 16]
// Output: top8 scores [1, 16] + top8 indices [1, 16] (only first 8 valid)
//
// CB States:
//   NCRISC: No-op (sharded buffer setup done in kernel file via setup_sharded_buffer)
//   BRISC: Waits for output CBs (scores, indices)
//   TRISC: Computes sigmoid (optional), bias add, sorting, normalization
// ============================================================================
struct GeneralizedMoeGate {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC) - empty, sharded buffer setup done in kernel file
    struct ReaderCTArgs {};

    // Writer CTArgs (BRISC)
    template <uint32_t output_cb_, uint32_t output_indices_cb_>
    struct WriterCTArgs {
        static constexpr uint32_t output_cb = output_cb_;
        static constexpr uint32_t output_indices_cb = output_indices_cb_;
    };

    // Compute CTArgs (TRISC)
    // enable_sigmoid must be compile-time (template parameter for generalized_moe_gate<>)
    template <
        uint32_t input_cb_,
        uint32_t bias_cb_,
        uint32_t input_indices_cb_,
        uint32_t output_cb_,
        uint32_t output_indices_cb_,
        uint32_t eps_,
        uint32_t scaling_factor_,
        uint32_t enable_sigmoid_,
        uint32_t num_blocks_ = 1,
        uint32_t run_scores_cb_ = 5,
        uint32_t run_idx_cb_ = 6,
        uint32_t run_bias_cb_ = 7,
        uint32_t cb_tilize_ = 8,
        uint32_t cb_tilize_idx_ = 9>
    struct ComputeCTArgs {
        static constexpr uint32_t input_cb = input_cb_;
        static constexpr uint32_t bias_cb = bias_cb_;
        static constexpr uint32_t input_indices_cb = input_indices_cb_;
        static constexpr uint32_t output_cb = output_cb_;
        static constexpr uint32_t output_indices_cb = output_indices_cb_;
        static constexpr uint32_t eps = eps_;
        static constexpr uint32_t scaling_factor = scaling_factor_;
        static constexpr bool enable_sigmoid = enable_sigmoid_ == 1;
        static constexpr uint32_t num_blocks = num_blocks_;
        static constexpr uint32_t run_scores_cb = run_scores_cb_;
        static constexpr uint32_t run_idx_cb = run_idx_cb_;
        static constexpr uint32_t run_bias_cb = run_bias_cb_;
        static constexpr uint32_t cb_tilize = cb_tilize_;
        static constexpr uint32_t cb_tilize_idx = cb_tilize_idx_;
    };

    // ========================================================================
    // Op - the actual operation, templated on CTArgs and IsActiveCore
    // ========================================================================
    template <typename CTArgs, bool IsActiveCore>
    class Op {
    public:
        void operator()() {
            if constexpr (IsActiveCore) {
                impl();
            }
        }

    private:
#if defined(COMPILE_FOR_TRISC)
        // Multi-block combine helpers (TRISC). ------------------------------------------------
        // Run the full per-256 pipeline on block b, ending at merge16_to_run (a re-mergeable top-8
        // RUN at rows {0,2}, idx made global via +b*256), and pack its 3 fields (score/idx/bias)
        // to the L1 stash CBs (page b).
        template <uint32_t b>
        void process_block_to_run() {
            cb_wait_front(CTArgs::bias_cb, 1);
            cb_wait_front(CTArgs::input_cb, 1);
            reconfig_data_format_srca(CTArgs::input_indices_cb);
            copy_tile_to_dst_init_short(CTArgs::input_indices_cb);
            tile_regs_acquire();
            // Per-block GLOBAL indices: block b uses indices tile b (uploaded as arange + b*256). The
            // in-kernel idx_offset add was a no-op (sfpi/TTI both failed), so global ids come from the
            // per-block indices tile, with idx_offset=0.
            copy_tile(CTArgs::input_indices_cb, b, 1);
            reconfig_data_format_srca(CTArgs::input_cb);
            generalized_moe_gate_init<CTArgs::enable_sigmoid>(CTArgs::input_cb, CTArgs::bias_cb);
            generalized_moe_gate<CTArgs::enable_sigmoid, false, /*produce_run=*/true, 0, 2, 0>(
                CTArgs::input_cb, CTArgs::bias_cb, CTArgs::eps, CTArgs::scaling_factor);  // math run at {0,2}
            // Relocate {0,2}->{0,4} BEFORE step2: step2 is built for the finalize layout (store8_even_cols
            // at offsets {0,4}); applied to a {0,2} run it mis-strides and 2-period-duplicates col 2. This
            // matches the proven produce_run->relocate<0,2,0,4>->normalize_step2 path. The run is now at {0,4}.
            generalized_moe_gate_relocate_run<0, 2, 0, 4>();
            // step2-only (transpose math->standard, NO normalize): PACK can only read the STANDARD layout,
            // so the math-layout run must be transposed to standard BEFORE pack_untilize, else pack reads
            // empty standard cells -> all-zero. No normalize (the combine needs raw scores for global top-8;
            // normalize runs once after the combine). transpose_wh restores standard->math on the way back.
            generalized_moe_gate_step2_only<false>();
            tile_regs_commit();
            cb_reserve_back(CTArgs::run_scores_cb, 1);
            cb_reserve_back(CTArgs::run_idx_cb, 1);
            cb_reserve_back(CTArgs::run_bias_cb, 1);
            tile_regs_wait();
            // pack_untilize each region (DEST tile 0/1/2) to its run CB as ROW-MAJOR (linearized). Unlike
            // pack_tile, this survives the round-trip in a form tilize+transpose_wh can restore to math.
            // NOTE: the DEST tile to pack FROM is the RUNTIME tile_dst_rt_offset (last arg), NOT the 3rd
            // positional arg (that is block_c_index, only used when full_ct_dim > block_ct_dim). Passing
            // the tile as block_c_index left all three packs reading DEST tile 0 (scores) -> run_idx/run_bias
            // got the scores. tile_dst_rt_offset = 0/1/2 selects scores/idx/bias.
            pack_reconfig_data_format(CTArgs::run_scores_cb);
            pack_untilize_dest_init<1, 1>(CTArgs::run_scores_cb);
            pack_untilize_dest<1, 1>(CTArgs::run_scores_cb, 1, 0, 16, 4, 0);  // DEST tile 0 (scores)
            pack_untilize_uninit(CTArgs::run_scores_cb);
            cb_push_back(CTArgs::run_scores_cb, 1);
            pack_reconfig_data_format(CTArgs::run_idx_cb);
            pack_untilize_dest_init<1, 1>(CTArgs::run_idx_cb);
            pack_untilize_dest<1, 1>(CTArgs::run_idx_cb, 1, 0, 16, 4, 1);  // DEST tile 1 (idx)
            pack_untilize_uninit(CTArgs::run_idx_cb);
            cb_push_back(CTArgs::run_idx_cb, 1);
            pack_reconfig_data_format(CTArgs::run_bias_cb);
            pack_untilize_dest_init<1, 1>(CTArgs::run_bias_cb);
            pack_untilize_dest<1, 1>(CTArgs::run_bias_cb, 1, 0, 16, 4, 2);  // DEST tile 2 (bias)
            pack_untilize_uninit(CTArgs::run_bias_cb);
            cb_push_back(CTArgs::run_bias_cb, 1);
            tile_regs_release();
            cb_pop_front(CTArgs::input_cb, 1);
            cb_pop_front(CTArgs::bias_cb, 1);
        }

        // Place the run stashed at `page` into the merge slot {dst_lo,dst_hi}: unpack each field-tile
        // into the interm region (DEST tile 3) and SFPU row-copy it into its home region. Must be
        // called inside a tile_regs_acquire/commit (writes DEST). Row-selective, so two runs placed
        // at {0,2} and {4,6} coexist for the finalize merge.
        template <uint32_t dst_lo, uint32_t dst_hi>
        void place_run_at(uint32_t page) {
            // Unpack each field of block0's stashed run with TRANSPOSE (transpose_wh: L1 standard layout
            // -> DEST math layout, inverting pack_tile's implicit math->standard). A plain copy_tile
            // leaves it in standard layout, so the SFPU's math-offset reads hit the wrong cells. After
            // the transpose the run sits in interm at math cols {0,2}; SFPU-place each field at {dst}.
            UNPACK((llk_unpack_set_srcb_dummy_valid()));
            reconfig_data_format_srca(CTArgs::run_scores_cb);
            transpose_wh_init_short(CTArgs::run_scores_cb);
            transpose_wh_tile(CTArgs::run_scores_cb, page, 3);
            generalized_moe_gate_place_field_from_interm<2, dst_lo, dst_hi>();  // field 2 = score
            reconfig_data_format_srca(CTArgs::run_idx_cb);
            transpose_wh_init_short(CTArgs::run_idx_cb);
            transpose_wh_tile(CTArgs::run_idx_cb, page, 3);
            generalized_moe_gate_place_field_from_interm<1, dst_lo, dst_hi>();  // field 1 = idx
            reconfig_data_format_srca(CTArgs::run_bias_cb);
            transpose_wh_init_short(CTArgs::run_bias_cb);
            transpose_wh_tile(CTArgs::run_bias_cb, page, 3);
            generalized_moe_gate_place_field_from_interm<0, dst_lo, dst_hi>();  // field 0 = bias
        }

        // Unpack a stashed run's 3 fields directly into the scores/idx/bias regions at cols {0,2}
        // (whole-tile unpack — overwrites the regions). Used by the combine DIAG to validate
        // pack/unpack + the proven copy_topk_run relocate, independent of place_run_at.
        void unpack_run_to_regions(uint32_t page) {
            reconfig_data_format_srca(CTArgs::run_scores_cb);
            copy_tile_to_dst_init_short(CTArgs::run_scores_cb);
            copy_tile(CTArgs::run_scores_cb, page, 0);  // scores region (DEST tile 0)
            reconfig_data_format_srca(CTArgs::run_idx_cb);
            copy_tile_to_dst_init_short(CTArgs::run_idx_cb);
            copy_tile(CTArgs::run_idx_cb, page, 1);  // indices region (DEST tile 1)
            reconfig_data_format_srca(CTArgs::run_bias_cb);
            copy_tile_to_dst_init_short(CTArgs::run_bias_cb);
            copy_tile(CTArgs::run_bias_cb, page, 2);  // bias region (DEST tile 2)
        }

        // Unpack a stashed STANDARD-layout run (packed after step2) back into the scores/idx/bias regions
        // at math cols {0,2} via transpose_wh (standard->math, the inverse of the step2 done before pack).
        // Used by the linchpin test to verify the L1 round-trip + layout convert.
        void unpack_run_to_regions_transpose(uint32_t page) {
            // FULL transpose_wh_init per field: the run CBs are new CBs that never got the full
            // transpose_wh hw-config (unpack_A_init with transpose_of_faces + 16x16_transpose +
            // datacopy_init + transpose_dest_init). init_short alone (assumes a prior full init) leaves
            // transpose_wh producing nothing (all-0). ocb = output_cb just for the pack-config slot.
            transpose_wh_init(CTArgs::run_scores_cb, CTArgs::output_cb);
            transpose_wh_tile(CTArgs::run_scores_cb, page, 0);  // -> scores region (math {0,2})
            transpose_wh_init(CTArgs::run_idx_cb, CTArgs::output_cb);
            transpose_wh_tile(CTArgs::run_idx_cb, page, 1);  // -> indices region
            transpose_wh_init(CTArgs::run_bias_cb, CTArgs::output_cb);
            transpose_wh_tile(CTArgs::run_bias_cb, page, 2);  // -> bias region
        }
#endif

        void impl() {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC: No-op - sharded buffer setup done in kernel file
            // ================================================================

#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC: Wait for compute to finish
            // ================================================================
            cb_wait_front(CTArgs::output_indices_cb, 1);
            cb_wait_front(CTArgs::output_cb, 1);

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC: Compute gate logic
            // ================================================================

            // Input indices CB should have the same tile shape as the input CB
            reconfig_data_format<false, true>(CTArgs::input_indices_cb, CTArgs::bias_cb);
            // Output indices CB should have the same tile shape as the output CB
            pack_reconfig_data_format<true>(CTArgs::output_cb);

            // input_indices: one persistent sharded tile PER block, holding that block's GLOBAL expert
            // ids (block b = arange + b*256). Block b copies tile b, so the pipeline tracks global ids
            // directly (no in-kernel offset add).
            cb_wait_front(CTArgs::input_indices_cb, CTArgs::num_blocks);

            if constexpr (CTArgs::num_blocks == 1) {
                // ============ single 256-block path ============
#ifdef GMG_TEST_STASH
                // STASH round-trip test: produce_run -> step2(math->standard) -> pack_untilize (row-major L1,
                // in process_block_to_run) -> tilize (row-major -> tiled scratch) -> transpose_wh (tiled
                // standard -> DEST math {0,2}) -> dump the IDX region. Now restores BOTH scores (bf16, via
                // cb_tilize) AND idx (uint16, via cb_tilize_idx — bit-preserved so the SFPU reads the raw
                // expert id). Golden top-8 ids at math {0,2} in the idx dump => the round-trip is faithful.
                process_block_to_run<0>();
                // compute_kernel_hw_startup is REQUIRED before the first tilize_block — without it the
                // MATH<->PACK DST semaphore is uninitialized and tilize_block deadlocks immediately.
                compute_kernel_hw_startup(CTArgs::run_scores_cb, CTArgs::cb_tilize);
                // ---- tilize run_scores_cb (row-major bf16) -> cb_tilize (tiled). self-manages DEST. ----
                cb_wait_front(CTArgs::run_scores_cb, 1);
                reconfig_data_format_srca(CTArgs::run_scores_cb);
                tilize_init(CTArgs::run_scores_cb, 1, CTArgs::cb_tilize);
                cb_reserve_back(CTArgs::cb_tilize, 1);
                tilize_block(CTArgs::run_scores_cb, 1, CTArgs::cb_tilize);
                cb_push_back(CTArgs::cb_tilize, 1);
                tilize_uninit(CTArgs::run_scores_cb, CTArgs::cb_tilize);
                cb_pop_front(CTArgs::run_scores_cb, 1);
                // ---- tilize run_idx_cb (row-major uint16) -> cb_tilize_idx (tiled uint16). ----
                // Fresh hw_startup with the UInt16 CBs: the bf16 scores tilize (+ its hw_startup) left the
                // unpack/pack runtime format at bf16, and tilize_uninit doesn't fully restore it on WH, so
                // tilize_block would decode the raw uint16 idx as bf16. This configures a clean uint16 path.
                compute_kernel_hw_startup(CTArgs::run_idx_cb, CTArgs::cb_tilize_idx);
                cb_wait_front(CTArgs::run_idx_cb, 1);
                reconfig_data_format_srca(CTArgs::run_idx_cb);
                tilize_init(CTArgs::run_idx_cb, 1, CTArgs::cb_tilize_idx);
                cb_reserve_back(CTArgs::cb_tilize_idx, 1);
                tilize_block(CTArgs::run_idx_cb, 1, CTArgs::cb_tilize_idx);
                cb_push_back(CTArgs::cb_tilize_idx, 1);
                tilize_uninit(CTArgs::run_idx_cb, CTArgs::cb_tilize_idx);
                cb_pop_front(CTArgs::run_idx_cb, 1);
                cb_pop_front(CTArgs::run_bias_cb, 1);  // bias not restored in this test
                // ---- DIAGNOSTIC (option A): transpose_wh scores -> interm (tile 3), then DUMP interm RAW two
                //      ways (bf16 via output_cb, uint16 via output_indices) — NO place/normalize. Shows where
                //      transpose_wh puts the score in the interm cell: which OFFSET (expect {0,4}) and which
                //      16-bit half (bf16 view shows the score 0.6-0.7 values; uint16 view shows the raw bits).
                //      This tells us the load/store modes place_field_from_interm needs for the standalone field. ----
                // PLACE isolation (srcb FIXED): transpose_wh each field -> interm (tile 3) [NO srcb_dummy_valid
                // before — the transpose's TRNSPSRCB must NOT read a dummy SrcB], SFPU place src{0,4} -> home
                // {0,4}; THEN combine_init + srcb_dummy_valid (for step2) + normalize_step2. Validates place
                // against the 256 golden. (idst=2/3 are reachable; earlier all-0 was the srcb-before-transpose bug.)
                cb_wait_front(CTArgs::cb_tilize, 1);
                cb_wait_front(CTArgs::cb_tilize_idx, 1);
                tile_regs_acquire();
                reconfig_data_format_srca(CTArgs::cb_tilize);
                transpose_wh_init_short(CTArgs::cb_tilize);
                transpose_wh_tile(CTArgs::cb_tilize, 0, 3);                     // scores -> interm (tile 3)
                generalized_moe_gate_place_field_from_interm<2, 0, 4, 0, 4>();  // score: interm{0,4} -> scores {0,4}
                reconfig_data_format_srca(CTArgs::cb_tilize_idx);
                transpose_wh_init_short(CTArgs::cb_tilize_idx);
                transpose_wh_tile(CTArgs::cb_tilize_idx, 0, 3);  // idx -> interm (tile 3, overwrites score interm)
                generalized_moe_gate_place_field_from_interm<1, 0, 4, 0, 4>();  // idx: interm{0,4} -> indices {0,4}
                generalized_moe_gate_combine_init<false>();
                UNPACK((llk_unpack_set_srcb_dummy_valid()));  // AFTER transposes, before step2
                generalized_moe_gate_normalize_step2<false>(CTArgs::eps, CTArgs::scaling_factor);
                tile_regs_commit();
                cb_pop_front(CTArgs::cb_tilize, 1);
                cb_pop_front(CTArgs::cb_tilize_idx, 1);
                cb_reserve_back(CTArgs::output_cb, 1);
                cb_reserve_back(CTArgs::output_indices_cb, 1);
                tile_regs_wait();
                pack_reconfig_data_format(CTArgs::output_cb);
                pack_tile(0, CTArgs::output_cb);
                cb_push_back(CTArgs::output_cb, 1);
                pack_reconfig_data_format(CTArgs::output_indices_cb);
                pack_tile(1, CTArgs::output_indices_cb);
                cb_push_back(CTArgs::output_indices_cb, 1);
                tile_regs_release();
#else
                cb_wait_front(CTArgs::bias_cb, 1);
                copy_tile_to_dst_init_short(CTArgs::input_indices_cb);
                tile_regs_acquire();
                copy_tile(CTArgs::input_indices_cb, 0, 1);
                reconfig_data_format_srca(CTArgs::input_cb);
                generalized_moe_gate_init<CTArgs::enable_sigmoid>(CTArgs::input_cb, CTArgs::bias_cb);
                cb_wait_front(CTArgs::input_cb, 1);
#if defined(GMG_DUMP_OCCUPANCY)
                // OCCUPANCY PROBE: the indices region was pre-filled with the arange (each datum = its
                // expert id) by the copy_tile above. Run ONE full pipeline (produce_run: sum_top2 +
                // transpose + merge) and pack the scores/idx regions RAW (no relocate/normalize/step2).
                // Diffing the dumped idx region vs the original arange shows which offsets the pipeline
                // WROTE (occupied) vs left as arange (free) -> tells us where a run can be safely parked.
                generalized_moe_gate<CTArgs::enable_sigmoid, false, true, 0, 2, 0>(
                    CTArgs::input_cb, CTArgs::bias_cb, CTArgs::eps, CTArgs::scaling_factor);
#elif defined(GMG_TEST_PARK2)
                // ISOLATION: produce run -> park {32,34} -> run the pipeline AGAIN (same input) ->
                // restore the parked run to {4,6} -> merge. If a full second pipeline pass between park
                // and restore leaves {32,34} intact, dev_idx == block0's top-8 (two identical runs);
                // if the pipeline clobbers offset 32-63, it's garbage -> the park location isn't free.
                generalized_moe_gate<CTArgs::enable_sigmoid, false, true, 0, 2, 0>(
                    CTArgs::input_cb, CTArgs::bias_cb, CTArgs::eps, CTArgs::scaling_factor);
                generalized_moe_gate_relocate_run<0, 2, 16, 18>();  // park at the REAL free cols 16,18
                generalized_moe_gate_init<CTArgs::enable_sigmoid>(CTArgs::input_cb, CTArgs::bias_cb);
                generalized_moe_gate<CTArgs::enable_sigmoid, false, true, 0, 2, 0>(
                    CTArgs::input_cb, CTArgs::bias_cb, CTArgs::eps, CTArgs::scaling_factor);  // run again @ {0,2}
                generalized_moe_gate_relocate_run<16, 18, 4, 6>();  // restore parked run -> {4,6}
                generalized_moe_gate_combine_finalize<false>(CTArgs::eps, CTArgs::scaling_factor);
#elif defined(GMG_TEST_PARK)
                // ISOLATION: produce_run -> PARK to the free upper-half offset {32,34} -> RESTORE to the
                // normalize slot {0,4} -> normalize. Tests whether copy_topk_run can address offset
                // 32-63 (the in-DEST park location). Matches 256 golden => offset-32 park works.
                generalized_moe_gate<CTArgs::enable_sigmoid, false, true, 0, 2, 0>(
                    CTArgs::input_cb, CTArgs::bias_cb, CTArgs::eps, CTArgs::scaling_factor);
                generalized_moe_gate_relocate_run<0, 2, 32, 34>();  // park
                generalized_moe_gate_relocate_run<32, 34, 0, 4>();  // restore to normalize slot
                generalized_moe_gate_normalize_step2<false>(CTArgs::eps, CTArgs::scaling_factor);
#elif defined(GMG_TEST_PRODUCE_RUN)
                // ISOLATION: same 256 top-8 but via the A2 building blocks WITHOUT the L1 stash —
                // produce_run (merge16_to_run -> run at cols {0,2}) + proven relocate {0,2}->{0,4} +
                // normalize+step2. If this matches the 256 golden, merge16_to_run/relocate/normalize
                // are good and the combine bug is the pack/unpack stash.
                generalized_moe_gate<CTArgs::enable_sigmoid, false, true, 0, 2, 0>(
                    CTArgs::input_cb, CTArgs::bias_cb, CTArgs::eps, CTArgs::scaling_factor);
                generalized_moe_gate_relocate_run<0, 2, 0, 4>();
                generalized_moe_gate_normalize_step2<false>(CTArgs::eps, CTArgs::scaling_factor);
#else
                generalized_moe_gate<CTArgs::enable_sigmoid>(
                    CTArgs::input_cb, CTArgs::bias_cb, CTArgs::eps, CTArgs::scaling_factor);
#endif
                cb_pop_front(CTArgs::input_cb, 1);
                tile_regs_commit();

                cb_reserve_back(CTArgs::output_cb, 1);
                cb_reserve_back(CTArgs::output_indices_cb, 1);
                tile_regs_wait();
#if defined(GMG_DUMP_AFTER_SUM_TOP2) || defined(GMG_DUMP_AFTER_STEP1) || defined(GMG_DUMP_AFTER_STEP0)
                pack_tile(2, CTArgs::output_cb);
                cb_push_back(CTArgs::output_cb, 1);
                pack_reconfig_data_format(CTArgs::output_indices_cb);
                pack_tile(1, CTArgs::output_indices_cb);
                cb_push_back(CTArgs::output_indices_cb, 1);
#else
                pack_tile(0, CTArgs::output_cb);
                cb_push_back(CTArgs::output_cb, 1);
                pack_reconfig_data_format(CTArgs::output_indices_cb);
                pack_tile(1, CTArgs::output_indices_cb);
                cb_push_back(CTArgs::output_indices_cb, 1);
#endif
                tile_regs_release();
#endif  // GMG_TEST_STASH
            } else {
                // ============ multi-block path ============
#ifndef GMG_DIAG_BLOCK
                // A2 combine (L1 stash, v3 — MERGE-ONLY acquire). Both blocks are stashed to L1 via the
                // proven round-trip (process_block_to_run). Then ALL fields are tilize'd into scratch, and a
                // SINGLE merge-only acquire (NO produce_run inside it — produce_run's SFPU/srcb state poisons a
                // same-acquire transpose_wh) restores both runs via transpose_wh -> interm -> SFPU place
                // (block1->{0,2}, block0->{4,6}) and merges them. This mirrors the clean state of the proven
                // 256 place-isolation. Per-block indices carry global ids.
                process_block_to_run<0>();  // -> L1 run CBs page 0
                process_block_to_run<1>();  // -> L1 run CBs page 1
                // ---- tilize all fields BEFORE the merge acquire (tilize self-manages DEST). cb_tilize (bf16):
                //      p0=b0 score, p1=b1 score, p2=b0 bias, p3=b1 bias. cb_tilize_idx (uint16): p0=b0 idx, p1=b1 idx.
                compute_kernel_hw_startup(CTArgs::run_scores_cb, CTArgs::cb_tilize);
                cb_wait_front(CTArgs::run_scores_cb, CTArgs::num_blocks);
                reconfig_data_format_srca(CTArgs::run_scores_cb);
                tilize_init(CTArgs::run_scores_cb, 1, CTArgs::cb_tilize);
                for (uint32_t b = 0; b < CTArgs::num_blocks; ++b) {
                    cb_reserve_back(CTArgs::cb_tilize, 1);
                    tilize_block(CTArgs::run_scores_cb, 1, CTArgs::cb_tilize);
                    cb_push_back(CTArgs::cb_tilize, 1);
                    cb_pop_front(CTArgs::run_scores_cb, 1);
                }
                tilize_uninit(CTArgs::run_scores_cb, CTArgs::cb_tilize);
                cb_wait_front(CTArgs::run_bias_cb, CTArgs::num_blocks);
                reconfig_data_format_srca(CTArgs::run_bias_cb);
                tilize_init(CTArgs::run_bias_cb, 1, CTArgs::cb_tilize);
                for (uint32_t b = 0; b < CTArgs::num_blocks; ++b) {
                    cb_reserve_back(CTArgs::cb_tilize, 1);
                    tilize_block(CTArgs::run_bias_cb, 1, CTArgs::cb_tilize);
                    cb_push_back(CTArgs::cb_tilize, 1);
                    cb_pop_front(CTArgs::run_bias_cb, 1);
                }
                tilize_uninit(CTArgs::run_bias_cb, CTArgs::cb_tilize);
                compute_kernel_hw_startup(CTArgs::run_idx_cb, CTArgs::cb_tilize_idx);
                cb_wait_front(CTArgs::run_idx_cb, CTArgs::num_blocks);
                reconfig_data_format_srca(CTArgs::run_idx_cb);
                tilize_init(CTArgs::run_idx_cb, 1, CTArgs::cb_tilize_idx);
                for (uint32_t b = 0; b < CTArgs::num_blocks; ++b) {
                    cb_reserve_back(CTArgs::cb_tilize_idx, 1);
                    tilize_block(CTArgs::run_idx_cb, 1, CTArgs::cb_tilize_idx);
                    cb_push_back(CTArgs::cb_tilize_idx, 1);
                    cb_pop_front(CTArgs::run_idx_cb, 1);
                }
                tilize_uninit(CTArgs::run_idx_cb, CTArgs::cb_tilize_idx);
                // ---- merge-only acquire: restore block1 -> {0,2}, block0 -> {4,6}, then merge ----
                cb_wait_front(CTArgs::cb_tilize, 2 * CTArgs::num_blocks);
                cb_wait_front(CTArgs::cb_tilize_idx, CTArgs::num_blocks);
                tile_regs_acquire();
                // block1 -> {0,2}: scores(cb_tilize p1), idx(cb_tilize_idx p1), bias(cb_tilize p3)
                reconfig_data_format_srca(CTArgs::cb_tilize);
                transpose_wh_init_short(CTArgs::cb_tilize);
                transpose_wh_tile(CTArgs::cb_tilize, 1, 3);
                generalized_moe_gate_place_field_from_interm<2, 0, 2, 0, 4>();
                reconfig_data_format_srca(CTArgs::cb_tilize_idx);
                transpose_wh_init_short(CTArgs::cb_tilize_idx);
                transpose_wh_tile(CTArgs::cb_tilize_idx, 1, 3);
                generalized_moe_gate_place_field_from_interm<1, 0, 2, 0, 4>();
                reconfig_data_format_srca(CTArgs::cb_tilize);
                transpose_wh_init_short(CTArgs::cb_tilize);
                transpose_wh_tile(CTArgs::cb_tilize, 3, 3);
                generalized_moe_gate_place_field_from_interm<0, 0, 2, 0, 4>();
                // block0 -> {4,6}: scores(cb_tilize p0), idx(cb_tilize_idx p0), bias(cb_tilize p2)
                transpose_wh_tile(CTArgs::cb_tilize, 0, 3);
                generalized_moe_gate_place_field_from_interm<2, 4, 6, 0, 4>();
                reconfig_data_format_srca(CTArgs::cb_tilize_idx);
                transpose_wh_init_short(CTArgs::cb_tilize_idx);
                transpose_wh_tile(CTArgs::cb_tilize_idx, 0, 3);
                generalized_moe_gate_place_field_from_interm<1, 4, 6, 0, 4>();
                reconfig_data_format_srca(CTArgs::cb_tilize);
                transpose_wh_init_short(CTArgs::cb_tilize);
                transpose_wh_tile(CTArgs::cb_tilize, 2, 3);
                generalized_moe_gate_place_field_from_interm<0, 4, 6, 0, 4>();
                // merge {0,2}+{4,6} -> global top-8 + normalize + step2. srcb dummy-valid AFTER the transposes.
                generalized_moe_gate_combine_init<false>();
                UNPACK((llk_unpack_set_srcb_dummy_valid()));
                generalized_moe_gate_combine_finalize<false>(CTArgs::eps, CTArgs::scaling_factor);
                tile_regs_commit();
                cb_pop_front(CTArgs::cb_tilize, 2 * CTArgs::num_blocks);
                cb_pop_front(CTArgs::cb_tilize_idx, CTArgs::num_blocks);
                cb_reserve_back(CTArgs::output_cb, 1);
                cb_reserve_back(CTArgs::output_indices_cb, 1);
                tile_regs_wait();
                pack_reconfig_data_format(CTArgs::output_cb);
                pack_tile(0, CTArgs::output_cb);
                cb_push_back(CTArgs::output_cb, 1);
                pack_reconfig_data_format(CTArgs::output_indices_cb);
                pack_tile(1, CTArgs::output_indices_cb);
                cb_push_back(CTArgs::output_indices_cb, 1);
                tile_regs_release();
#else
                // A1 diagnostic: run the full per-256 pipeline on each block and OUTPUT only block
                // GMG_DIAG_BLOCK, to validate each block's top-8 in isolation.
                for (uint32_t b = 0; b < CTArgs::num_blocks; ++b) {
                    cb_wait_front(CTArgs::bias_cb, 1);   // block b's bias tile (front)
                    cb_wait_front(CTArgs::input_cb, 1);  // block b's logit tile (front)

                    // Reset SrcA to the input_indices (uint16) format before copying indices: the
                    // previous block's pipeline left SrcA configured for input_cb (bf16), which would
                    // otherwise mis-read the uint16 indices as garbage (e.g. bf16 score bits).
                    reconfig_data_format_srca(CTArgs::input_indices_cb);
                    copy_tile_to_dst_init_short(CTArgs::input_indices_cb);
                    tile_regs_acquire();
                    copy_tile(CTArgs::input_indices_cb, 0, 1);
                    reconfig_data_format_srca(CTArgs::input_cb);
                    generalized_moe_gate_init<CTArgs::enable_sigmoid>(CTArgs::input_cb, CTArgs::bias_cb);
                    generalized_moe_gate<CTArgs::enable_sigmoid>(
                        CTArgs::input_cb, CTArgs::bias_cb, CTArgs::eps, CTArgs::scaling_factor);
                    tile_regs_commit();

                    // Complete the full math/pack handshake EVERY iteration (commit -> wait -> release),
                    // even for non-output blocks; skipping wait/release misaligns the semaphore and
                    // corrupts the next iteration. Pack only the GMG_DIAG_BLOCK block's result.
#ifdef GMG_DIAG_BLOCK
                    const bool do_output = (b == (uint32_t)GMG_DIAG_BLOCK);
#else
                    const bool do_output = false;
#endif
                    if (do_output) {
                        cb_reserve_back(CTArgs::output_cb, 1);
                        cb_reserve_back(CTArgs::output_indices_cb, 1);
                    }
                    tile_regs_wait();
                    if (do_output) {
                        pack_tile(0, CTArgs::output_cb);
                        cb_push_back(CTArgs::output_cb, 1);
                        pack_reconfig_data_format(CTArgs::output_indices_cb);
                        pack_tile(1, CTArgs::output_indices_cb);
                        cb_push_back(CTArgs::output_indices_cb, 1);
                    }
                    tile_regs_release();
                    cb_pop_front(CTArgs::input_cb, 1);
                    cb_pop_front(CTArgs::bias_cb, 1);
                }
#endif  // GMG_DIAG_BLOCK (combine vs A1 diagnostic)
            }
#endif
        }
    };  // class Op

};  // struct GeneralizedMoeGate

}  // namespace deepseek_v3_ops
