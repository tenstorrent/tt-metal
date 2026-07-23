// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
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
#include "api/compute/experimental/generalized_moe_gate.h"
#include "api/dataflow/circular_buffer.h"
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
        uint32_t cb_tilize_idx_ = 9,
        uint32_t topk_ = 8,
        uint32_t output_softmax_ = 0>
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
        static constexpr uint32_t topk = topk_;
        static constexpr bool output_softmax = output_softmax_ == 1;
        // topk is restricted to {4, 6, 8}: the finalize rank-mask is correct only for these (topk 1-3 leave
        // ranks 0-3 unmasked, 5/7 are untested). The device op rejects other values on the host (TT_FATAL)
        // before this kernel is ever compiled; this static_assert is the compile-time mirror, so a kernel
        // built with an unsupported topk fails to compile instead of silently mis-normalizing.
        static_assert(topk == 4 || topk == 6 || topk == 8, "generalized_moe_gate: topk must be one of {4, 6, 8}");
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
            CircularBuffer bias_cb(CTArgs::bias_cb);
            CircularBuffer input_cb(CTArgs::input_cb);
            CircularBuffer run_scores_cb(CTArgs::run_scores_cb);
            CircularBuffer run_idx_cb(CTArgs::run_idx_cb);
            CircularBuffer run_bias_cb(CTArgs::run_bias_cb);
            bias_cb.wait_front(1);
            input_cb.wait_front(1);
            reconfig_data_format_srca(CTArgs::input_indices_cb);
            copy_init(CTArgs::input_indices_cb);
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
            // at offsets {0,4}); applied to a {0,2} run it mis-strides and 2-period-duplicates col 2 (and
            // bias) — which corrupts the merge's sort key. Aligning to {0,4} fixes it. The run is now at {0,4}.
            generalized_moe_gate_relocate_run<0, 2, 0, 4>();
            // step2-only (transpose math->standard, NO normalize): PACK can only read the STANDARD layout,
            // so the math-layout run must be transposed to standard BEFORE pack_untilize, else pack reads
            // empty standard cells -> all-zero. No normalize (the combine needs raw scores for global top-8;
            // normalize runs once after the combine). transpose_wh restores standard->math on the way back.
            generalized_moe_gate_step2_only<false>();
            tile_regs_commit();
            run_scores_cb.reserve_back(1);
            run_idx_cb.reserve_back(1);
            run_bias_cb.reserve_back(1);
            tile_regs_wait();
            // pack_untilize each region (DEST tile 0/1/2) to its run CB as ROW-MAJOR (linearized). Unlike
            // pack_tile, this survives the round-trip in a form tilize+transpose_wh can restore to math.
            // NOTE: the DEST tile to pack FROM is the RUNTIME tile_dst_rt_offset (last arg), NOT the 3rd
            // positional arg (that is block_c_index, only used when full_ct_dim > block_ct_dim). Passing
            // the tile as block_c_index left all three packs reading DEST tile 0 (scores) -> run_idx/run_bias
            // got the scores. tile_dst_rt_offset = 0/1/2 selects scores/idx/bias.
            pack_reconfig_data_format(CTArgs::run_scores_cb);
            pack_untilize_dest_init<1, 1>(CTArgs::run_scores_cb);
            pack_untilize_dest<1, 1>(CTArgs::run_scores_cb, 1, 0, 0);  // DEST tile 0 (scores)
            pack_untilize_uninit(CTArgs::run_scores_cb);
            run_scores_cb.push_back(1);
            pack_reconfig_data_format(CTArgs::run_idx_cb);
            pack_untilize_dest_init<1, 1>(CTArgs::run_idx_cb);
            pack_untilize_dest<1, 1>(CTArgs::run_idx_cb, 1, 0, 1);  // DEST tile 1 (idx)
            pack_untilize_uninit(CTArgs::run_idx_cb);
            run_idx_cb.push_back(1);
            pack_reconfig_data_format(CTArgs::run_bias_cb);
            pack_untilize_dest_init<1, 1>(CTArgs::run_bias_cb);
            pack_untilize_dest<1, 1>(CTArgs::run_bias_cb, 1, 0, 2);  // DEST tile 2 (bias)
            pack_untilize_uninit(CTArgs::run_bias_cb);
            run_bias_cb.push_back(1);
            tile_regs_release();
            input_cb.pop_front(1);
            bias_cb.pop_front(1);
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
            CircularBuffer output_cb(CTArgs::output_cb);
            CircularBuffer output_indices_cb(CTArgs::output_indices_cb);
            output_indices_cb.wait_front(1);
            output_cb.wait_front(1);

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
            CircularBuffer input_indices_cb(CTArgs::input_indices_cb);
            input_indices_cb.wait_front(CTArgs::num_blocks);

            if constexpr (CTArgs::num_blocks == 1) {
                CircularBuffer bias_cb(CTArgs::bias_cb);
                CircularBuffer input_cb(CTArgs::input_cb);
                CircularBuffer output_cb(CTArgs::output_cb);
                CircularBuffer output_indices_cb(CTArgs::output_indices_cb);
                // ============ single ≤256-block path: the proven fused single-op gate (128/256 top-8) ============
                bias_cb.wait_front(1);
                copy_init(CTArgs::input_indices_cb);
                tile_regs_acquire();
                copy_tile(CTArgs::input_indices_cb, 0, 1);  // indices (arange 0-255) -> idx region
                reconfig_data_format_srca(CTArgs::input_cb);
                generalized_moe_gate_init<CTArgs::enable_sigmoid>(CTArgs::input_cb, CTArgs::bias_cb);
                input_cb.wait_front(1);
                generalized_moe_gate<
                    CTArgs::enable_sigmoid,
                    false,
                    false,
                    0,
                    2,
                    0,
                    CTArgs::topk,
                    CTArgs::output_softmax>(CTArgs::input_cb, CTArgs::bias_cb, CTArgs::eps, CTArgs::scaling_factor);
                input_cb.pop_front(1);
                tile_regs_commit();

                output_cb.reserve_back(1);
                output_indices_cb.reserve_back(1);
                tile_regs_wait();
                pack_tile(0, CTArgs::output_cb);
                output_cb.push_back(1);
                pack_reconfig_data_format(CTArgs::output_indices_cb);
                pack_tile(1, CTArgs::output_indices_cb);
                output_indices_cb.push_back(1);
                tile_regs_release();
            } else {
                CircularBuffer run_scores_cb(CTArgs::run_scores_cb);
                CircularBuffer run_idx_cb(CTArgs::run_idx_cb);
                CircularBuffer run_bias_cb(CTArgs::run_bias_cb);
                CircularBuffer cb_tilize(CTArgs::cb_tilize);
                CircularBuffer cb_tilize_idx(CTArgs::cb_tilize_idx);
                CircularBuffer output_cb(CTArgs::output_cb);
                CircularBuffer output_indices_cb(CTArgs::output_indices_cb);
                // ============ multi-block path ============
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
                run_scores_cb.wait_front(CTArgs::num_blocks);
                reconfig_data_format_srca(CTArgs::run_scores_cb);
                tilize_init(CTArgs::run_scores_cb, 1, CTArgs::cb_tilize);
                for (uint32_t b = 0; b < CTArgs::num_blocks; ++b) {
                    cb_tilize.reserve_back(1);
                    tilize_block(CTArgs::run_scores_cb, 1, CTArgs::cb_tilize);
                    cb_tilize.push_back(1);
                    run_scores_cb.pop_front(1);
                }
                tilize_uninit(CTArgs::run_scores_cb, CTArgs::cb_tilize);
                run_bias_cb.wait_front(CTArgs::num_blocks);
                reconfig_data_format_srca(CTArgs::run_bias_cb);
                tilize_init(CTArgs::run_bias_cb, 1, CTArgs::cb_tilize);
                for (uint32_t b = 0; b < CTArgs::num_blocks; ++b) {
                    cb_tilize.reserve_back(1);
                    tilize_block(CTArgs::run_bias_cb, 1, CTArgs::cb_tilize);
                    cb_tilize.push_back(1);
                    run_bias_cb.pop_front(1);
                }
                tilize_uninit(CTArgs::run_bias_cb, CTArgs::cb_tilize);
                compute_kernel_hw_startup(CTArgs::run_idx_cb, CTArgs::cb_tilize_idx);
                run_idx_cb.wait_front(CTArgs::num_blocks);
                reconfig_data_format_srca(CTArgs::run_idx_cb);
                tilize_init(CTArgs::run_idx_cb, 1, CTArgs::cb_tilize_idx);
                for (uint32_t b = 0; b < CTArgs::num_blocks; ++b) {
                    cb_tilize_idx.reserve_back(1);
                    tilize_block(CTArgs::run_idx_cb, 1, CTArgs::cb_tilize_idx);
                    cb_tilize_idx.push_back(1);
                    run_idx_cb.pop_front(1);
                }
                tilize_uninit(CTArgs::run_idx_cb, CTArgs::cb_tilize_idx);
                // ---- merge-only acquire: restore block1 -> {0,2}, block0 -> {4,6}, then merge ----
                cb_tilize.wait_front(2 * CTArgs::num_blocks);
                cb_tilize_idx.wait_front(CTArgs::num_blocks);
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
                generalized_moe_gate_combine_finalize<false, CTArgs::topk, CTArgs::output_softmax>(
                    CTArgs::eps, CTArgs::scaling_factor);
                tile_regs_commit();
                cb_tilize.pop_front(2 * CTArgs::num_blocks);
                cb_tilize_idx.pop_front(CTArgs::num_blocks);
                output_cb.reserve_back(1);
                output_indices_cb.reserve_back(1);
                tile_regs_wait();
                pack_reconfig_data_format(CTArgs::output_cb);
                pack_tile(0, CTArgs::output_cb);
                output_cb.push_back(1);
                pack_reconfig_data_format(CTArgs::output_indices_cb);
                pack_tile(1, CTArgs::output_indices_cb);
                output_indices_cb.push_back(1);
                tile_regs_release();
            }
#endif
        }
    };  // class Op

};  // struct GeneralizedMoeGate

}  // namespace deepseek_v3_ops
