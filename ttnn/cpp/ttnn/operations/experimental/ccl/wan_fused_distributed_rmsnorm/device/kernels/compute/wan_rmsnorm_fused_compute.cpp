// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Fused Wan2.2 distributed RMSNorm compute kernel — CHUNKED.
 *
 * Processes the core's tile-row slice in fixed-size windows ("chunks") of
 * chunk_size_rows. For each chunk we:
 *   1) Wait for the chunk's input tiles in input_cb (reader streams them).
 *   2) PRE: compute sum(x**2) per row, push one stat tile per row to
 *      stats_local_cb (forwarder will ring-gather across TP).
 *   3) Wait for stats_gathered_cb to fill (chunk_size_rows × stats_tiles_cols
 *      tiles per chunk, where stats_tiles_cols == ring_size).
 *   4) POST: reduce gathered stats across TP, eps+rsqrt, multiply input by
 *      1/rms (input still L1-resident in input_cb), optional gamma row-bcast,
 *      optional RoPE; push outputs to output_cb.
 *   5) Pop the chunk's input + gathered stats from their CBs.
 *
 * The chunk window is the L1-residency window: input lives in L1 from step 1
 * through step 4, then is released in step 5. Sizing chunk_size_rows lets the
 * program factory trade off L1 footprint vs AG amortization.
 *
 * For TP=1 (ring_size==1), the forwarder is a no-op that just promotes
 * stats_local_cb tiles to stats_gathered_cb; the per-row reduce in post
 * degenerates to a pass-through (stats_tiles_cols==1).
 */

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/layernorm.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose_wh.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    // === Compile-time args ===
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t stats_local_cb = get_compile_time_arg_val(1);
    constexpr uint32_t stats_gathered_cb = get_compile_time_arg_val(2);
    constexpr uint32_t weight_cb = get_compile_time_arg_val(3);
    constexpr uint32_t reduce_scalar_sum_cb = get_compile_time_arg_val(4);
    constexpr uint32_t reduce_scalar_avg_cb = get_compile_time_arg_val(5);
    constexpr uint32_t epsilon_cb = get_compile_time_arg_val(6);
    constexpr uint32_t reduce_result_cb = get_compile_time_arg_val(7);
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(8);
    constexpr uint32_t pre_intermediate_cb = get_compile_time_arg_val(9);
    constexpr uint32_t output_cb = get_compile_time_arg_val(10);
    constexpr uint32_t transformation_mat_cb = get_compile_time_arg_val(11);
    constexpr uint32_t rope_cos_cb = get_compile_time_arg_val(12);
    constexpr uint32_t rope_sin_cb = get_compile_time_arg_val(13);
    constexpr uint32_t rotated_input_cb = get_compile_time_arg_val(14);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(15);
    constexpr uint32_t block_size = get_compile_time_arg_val(16);
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(17);
    constexpr uint32_t chunk_size_rows = get_compile_time_arg_val(18);
    constexpr bool use_legacy_rsqrt = get_compile_time_arg_val(19);
    constexpr uint32_t has_weight = get_compile_time_arg_val(20);
    constexpr uint32_t fuse_rope = get_compile_time_arg_val(21);
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(22);
    // When is_tp_1 is true, the compute kernel skips stats_local_cb entirely
    // and pushes per-row stats directly into stats_gathered_cb. This makes
    // TP=1 (single-device) operation self-contained — no forwarder needed.
    constexpr uint32_t is_tp_1 = get_compile_time_arg_val(23);
    // Phase 9 packed-page AG (only the MUX writer uses it). When enabled the
    // pre phase transposes the per-row stat tile (real data in col 0 → row 0)
    // so the writer can extract two contiguous 64-byte spans per tile and
    // pack `window_size` of them into a single fabric packet. The post phase
    // then transposes the row-0 gathered tiles back to col 0 before the
    // existing reduce<AVG,REDUCE_ROW> chain runs.
    constexpr uint32_t stats_transposed_local_cb = get_compile_time_arg_val(24);
    constexpr uint32_t stats_transposed_gathered_cb = get_compile_time_arg_val(25);
    constexpr uint32_t packed_ag_enabled = get_compile_time_arg_val(26);
    // Per-head RoPE: reader pushes num_tile_cols (num_heads * head_dim_tiles)
    // cos/sin tiles per row (one per col). The post phase reads them by
    // absolute index (col_tile + i) — no wrap-cycling. Broadcast RoPE
    // (per_head_rope=0) pushes only head_dim_tiles and wraps.
    constexpr uint32_t per_head_rope = get_compile_time_arg_val(27);
    // Optional row-broadcast bias: tilized [1, H] tensor added after weight
    // multiply (sub-phase 2.5). Mirrors weight handling.
    constexpr uint32_t bias_cb = get_compile_time_arg_val(28);
    constexpr uint32_t has_bias = get_compile_time_arg_val(29);
    // Per-head normalization (FLUX.2): reduce over head_dim per head instead
    // of the full row. Forces is_tp_1 path (skip AG entirely); pre phase
    // produces num_heads_per_device stat tiles per row, post phase computes
    // an rsqrt per head and applies it to that head's columns.
    constexpr uint32_t per_head_norm = get_compile_time_arg_val(30);
    constexpr uint32_t num_heads_per_device = get_compile_time_arg_val(31);
    // Per-token weight / bias: [N, H] (vs broadcast [1, H]). Reader pushes
    // per-row tiles. Compute uses mul_tiles / add_tiles (no _bcast_rows) and
    // pops the row's weight/bias tiles at end of each row in the chunk.
    constexpr uint32_t per_token_weight = get_compile_time_arg_val(32);
    constexpr uint32_t per_token_bias = get_compile_time_arg_val(33);
    // Bit-packed fp32 epsilon for the fused +eps SFPU scalar-add in the reduce
    // post-op (replaces the prior bf16 epsilon_cb add_tiles path; fp32 is at
    // least as precise).
    constexpr uint32_t eps_bits = get_compile_time_arg_val(34);
    // Streaming low-L1: input_cb is block-sized, so PRE reads input block by
    // block (popping each) and POST sub-phase 1 (x*1/rms) re-reads a second
    // streamed pass from the reader (also popping each block). Only the whole-
    // row reduce path is supported (per_head_norm stays resident). The math /
    // accumulation order is identical to the resident path, so the result is
    // bit-exact; only the input_cb residency window changes.
    constexpr uint32_t streaming_low_l1 = get_compile_time_arg_val(35);

    constexpr uint32_t stats_dest_cb = (is_tp_1 != 0) ? stats_gathered_cb : stats_local_cb;
    // Per-row post reduce reads ring_size tiles. With packed AG enabled the
    // ring_size tiles live in stats_transposed_gathered_cb (post-transpose);
    // otherwise the legacy path uses stats_gathered_cb directly.
    constexpr uint32_t stats_reduce_src_cb =
        (packed_ag_enabled != 0) ? stats_transposed_gathered_cb : stats_gathered_cb;

    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    mm_init(intermediate_cb, transformation_mat_cb, rotated_input_cb);
    binary_op_init_common(input_cb, input_cb, input_cb);

    // One-time waits for reader-produced singletons.
    cb_wait_front(reduce_scalar_sum_cb, 1);
    cb_wait_front(reduce_scalar_avg_cb, 1);
    cb_wait_front(epsilon_cb, 1);
    if constexpr (fuse_rope) {
        cb_wait_front(transformation_mat_cb, 1);
    }

    constexpr uint32_t mul_rms_result_cb = (fuse_rope || has_weight) ? intermediate_cb : output_cb;
    // has_bias implies has_weight (enforced in validate). Weight output stays
    // in intermediate when bias or rope follows.
    constexpr uint32_t mul_weight_result_cb = (fuse_rope || has_bias) ? intermediate_cb : output_cb;
    constexpr uint32_t add_bias_result_cb = fuse_rope ? intermediate_cb : output_cb;

    // Process the core's tile rows in chunks of chunk_size_rows.
    uint32_t row_processed = 0;
    while (row_processed < num_tile_rows) {
        const uint32_t rows_in_chunk =
            (row_processed + chunk_size_rows <= num_tile_rows) ? chunk_size_rows : (num_tile_rows - row_processed);
        const uint32_t chunk_input_tiles = rows_in_chunk * num_tile_cols;
        // per_head_norm produces num_heads_per_device stats per row (one per
        // head) instead of stats_tiles_cols (== ring_size) for the AG path.
        constexpr uint32_t per_row_stats_count = (per_head_norm != 0) ? num_heads_per_device : stats_tiles_cols;
        const uint32_t chunk_stats_tiles = rows_in_chunk * per_row_stats_count;

        // -------- PHASE 1: PRE — sum(x**2) per row --------
        // Cumulative input wait (Phase 4): instead of one cb_wait_front for the
        // whole chunk, wait per col-block. Lets the reader push block N+1 while
        // compute is processing block N. Counter resets per chunk because we
        // cb_pop_front(input_cb, chunk_input_tiles) at the end.
        // Per-head norm: inner reduce spans head_dim_tiles columns per head;
        // we run num_heads_per_device reduces per row. Whole-row norm: single
        // reduce over num_tile_cols per row.
        constexpr uint32_t pre_groups_per_row = (per_head_norm != 0) ? num_heads_per_device : 1u;
        constexpr uint32_t pre_group_width = (per_head_norm != 0) ? head_dim_tiles : num_tile_cols;
        {
            DeviceZoneScopedN("RMS_PRE");
            if constexpr (streaming_low_l1) {
                // Streamed whole-row PRE (rows_in_chunk == 1, one group). Input
                // arrives block by block from the reader's first pass; each
                // block is popped after its x**2 is accumulated into
                // pre_intermediate_cb[0]. The accumulation order (and l1_acc
                // sequencing) is identical to the resident path, so the stat
                // tile is bit-exact — only the input residency window differs.
                for (uint32_t r = 0; r < rows_in_chunk; r++) {
                    reconfig_data_format(input_cb, input_cb);
                    pack_reconfig_data_format(pre_intermediate_cb);
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    mul_tiles_init(input_cb, input_cb);

                    cb_reserve_back(pre_intermediate_cb, 1);
                    for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                        const uint32_t tiles_in_block =
                            ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                        cb_wait_front(input_cb, tiles_in_block);
                        tile_regs_acquire();
                        for (uint32_t i = 0; i < tiles_in_block; i++) {
                            mul_tiles(input_cb, input_cb, i, i, i);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < tiles_in_block; i++) {
                            pack_tile<true>(i, pre_intermediate_cb, 0);
                            if (col_tile == 0 && i == 0) {
                                PACK((llk_pack_reconfig_l1_acc(1)));
                            }
                        }
                        tile_regs_release();
                        cb_pop_front(input_cb, tiles_in_block);
                    }
                    cb_push_back(pre_intermediate_cb, 1);
                    PACK((llk_pack_reconfig_l1_acc(0)));

                    compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
                        pre_intermediate_cb,
                        reduce_scalar_sum_cb,
                        stats_dest_cb,
                        compute_kernel_lib::ReduceInputBlockShape::single());
                }
            } else {
                uint32_t input_tiles_waited = 0;
                for (uint32_t r = 0; r < rows_in_chunk; r++) {
                    const uint32_t row_base = r * num_tile_cols;

                    for (uint32_t g = 0; g < pre_groups_per_row; g++) {
                        const uint32_t group_base = row_base + g * pre_group_width;

                        reconfig_data_format(input_cb, input_cb);
                        pack_reconfig_data_format(pre_intermediate_cb);
                        PACK((llk_pack_reconfig_l1_acc(0)));
                        mul_tiles_init(input_cb, input_cb);

                        cb_reserve_back(pre_intermediate_cb, 1);

                        for (uint32_t col_tile = 0; col_tile < pre_group_width; col_tile += block_size) {
                            const uint32_t tiles_in_block = ((pre_group_width - col_tile) >= block_size)
                                                                ? block_size
                                                                : (pre_group_width - col_tile);
                            // Cumulative wait covers the absolute tile range we need:
                            // r*num_tile_cols + g*pre_group_width + col_tile + tiles_in_block.
                            // Reader pushes block_size at a time across the whole row, so
                            // a wait for fewer-than-block_size tiles is satisfied by the
                            // next reader push regardless.
                            const uint32_t need = group_base + col_tile + tiles_in_block;
                            if (need > input_tiles_waited) {
                                cb_wait_front(input_cb, need);
                                input_tiles_waited = need;
                            }

                            tile_regs_acquire();
                            for (uint32_t i = 0; i < block_size && col_tile + i < pre_group_width; i++) {
                                const uint32_t abs_idx = group_base + col_tile + i;
                                mul_tiles(input_cb, input_cb, abs_idx, abs_idx, i);
                            }
                            tile_regs_commit();

                            tile_regs_wait();
                            for (uint32_t i = 0; i < block_size && col_tile + i < pre_group_width; i++) {
                                pack_tile<true>(i, pre_intermediate_cb, 0);
                                if (col_tile == 0 && i == 0) {
                                    PACK((llk_pack_reconfig_l1_acc(1)));
                                }
                            }
                            tile_regs_release();
                        }
                        cb_push_back(pre_intermediate_cb, 1);
                        PACK((llk_pack_reconfig_l1_acc(0)));

                        // Row/head reduce → 1 stat tile. SUM (col 0 = sum). Post phase
                        // divides by H_full or head_dim via the AVG scalar.
                        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
                            pre_intermediate_cb,
                            reduce_scalar_sum_cb,
                            stats_dest_cb,
                            compute_kernel_lib::ReduceInputBlockShape::single());
                    }
                }
            }
        }  // RMS_PRE

        // Phase 9 pre: NO compute-side transpose needed. The writer extracts
        // col 0 of each stats_local_cb tile directly via strided L1 loads
        // (col 0 lives in face_00 col 0 + face_10 col 0 at byte offsets
        // {0,64,...,960} and {2048,...,3008}). Skipping the transpose saves
        // a TRISC-state reconfig + pack pass per chunk (~hundreds of cycles).

        // -------- WAIT FOR FORWARDER TO COMPLETE AG FOR THIS CHUNK --------
        {
            DeviceZoneScopedN("RMS_AGWAIT");
            cb_wait_front(stats_gathered_cb, chunk_stats_tiles);
        }

        // -------- PHASE 3: POST — finalize normalization --------
        {
            DeviceZoneScopedN("RMS_POST");
            for (uint32_t r = 0; r < rows_in_chunk; r++) {
                const uint32_t row_base = r * num_tile_cols;
                uint32_t rope_cos_tile_in_head = 0;
                uint32_t rope_sin_tile_in_head = 0;

                // Per-head norm: do reduce + eps+rsqrt + sub-phase 1 per head, so
                // each head's rsqrt only stays in reduce_result_cb long enough to
                // be consumed by that head's mul. Whole-row norm: one iteration.
                constexpr uint32_t post_groups_per_row = (per_head_norm != 0) ? num_heads_per_device : 1u;
                constexpr uint32_t post_group_width = (per_head_norm != 0) ? head_dim_tiles : num_tile_cols;
                {
                    DeviceZoneScopedN("P_NORM");
                    for (uint32_t g = 0; g < post_groups_per_row; g++) {
                        const uint32_t group_col_base = g * post_group_width;
                        const uint32_t group_abs_base = row_base + group_col_base;

                        // Phase 9 post (TP>1 path): writer scattered the gathered
                        // packed pages directly into COL 0 of stats_gathered_cb tiles
                        // (strided stores), so reduce<AVG,REDUCE_ROW> over the row of
                        // ring_size tiles runs on stats_gathered_cb unchanged. For the
                        // per-head path each head's stat is a single tile.
                        // Fused reduce + eps + rsqrt: the post_reduce_op runs on DST
                        // (after reduce math, before pack) so we drop a separate
                        // tile_regs cycle, a reduce_result_cb round-trip, and the two
                        // eps reconfigs per row. fp32 scalar eps >= the prior bf16
                        // epsilon_cb add in precision, and rsqrt sees the un-truncated
                        // fp32 mean still in DST.
                        auto eps_rsqrt = [](uint32_t dst_idx) {
                            binop_with_scalar_tile_init();
                            add_unary_tile(dst_idx, eps_bits);
                            rsqrt_tile_init<use_legacy_rsqrt>();
                            rsqrt_tile<use_legacy_rsqrt>(dst_idx);
                        };
                        {
                            DeviceZoneScopedN("P_NRED");
                            if constexpr (per_head_norm != 0) {
                                compute_kernel_lib::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW>(
                                    stats_gathered_cb,
                                    reduce_scalar_avg_cb,
                                    reduce_result_cb,
                                    compute_kernel_lib::ReduceInputBlockShape::single(),
                                    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                                    compute_kernel_lib::NoAccumulation{},
                                    eps_rsqrt);
                            } else {
                                compute_kernel_lib::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW>(
                                    stats_gathered_cb,
                                    reduce_scalar_avg_cb,
                                    reduce_result_cb,
                                    compute_kernel_lib::ReduceInputBlockShape::row(stats_tiles_cols),
                                    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                                    compute_kernel_lib::NoAccumulation{},
                                    eps_rsqrt);
                            }

                            cb_wait_front(reduce_result_cb, 1);
                        }  // P_NRED

                        DeviceZoneScopedN("P_NMUL");
                        // ----- Sub-phase 1: x * (1/rms) → mul_rms_result_cb -----
                        reconfig_data_format(input_cb, reduce_result_cb);
                        pack_reconfig_data_format(mul_rms_result_cb);
                        mul_bcast_cols_init_short(input_cb, reduce_result_cb);
                        if constexpr (streaming_low_l1) {
                            // Streamed POST re-read: reader's 2nd pass pushes the
                            // row's input block by block. Each block is waited at a
                            // block-relative index, multiplied by 1/rms, packed, then
                            // popped — matching the reader's 2nd-pass pushes.
                            // streaming_low_l1 implies per_head_norm==0, so there is a
                            // single group with post_group_width == num_tile_cols, and
                            // num_tile_cols % block_size == 0 (host TT_FATAL invariant).
                            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                                cb_wait_front(input_cb, block_size);
                                cb_reserve_back(mul_rms_result_cb, block_size);
                                tile_regs_acquire();
                                for (uint32_t i = 0; i < block_size; i++) {
                                    mul_tiles_bcast_cols(input_cb, reduce_result_cb, i, 0, i);
                                }
                                tile_regs_commit();
                                tile_regs_wait();
                                for (uint32_t i = 0; i < block_size; i++) {
                                    pack_tile(i, mul_rms_result_cb);
                                }
                                tile_regs_release();
                                cb_push_back(mul_rms_result_cb, block_size);
                                cb_pop_front(input_cb, block_size);
                            }
                        } else {
                            for (uint32_t col_tile = 0; col_tile < post_group_width; col_tile += block_size) {
                                // Per_head_norm pushes head_dim_tiles per head (no padding)
                                // so multiple heads don't blow past intermediate_cb. The
                                // legacy whole-row path keeps the block_size-padded push so
                                // downstream sub-phases (still block_size-driven) consume
                                // matching counts even when num_tile_cols < block_size.
                                const uint32_t tiles_in_block = (per_head_norm != 0)
                                                                    ? (((post_group_width - col_tile) >= block_size)
                                                                           ? block_size
                                                                           : (post_group_width - col_tile))
                                                                    : block_size;
                                cb_reserve_back(mul_rms_result_cb, tiles_in_block);
                                tile_regs_acquire();
                                for (uint32_t i = 0; i < block_size && col_tile + i < post_group_width; i++) {
                                    const uint32_t abs_idx = group_abs_base + col_tile + i;
                                    mul_tiles_bcast_cols(input_cb, reduce_result_cb, abs_idx, 0, i);
                                }
                                tile_regs_commit();
                                tile_regs_wait();
                                for (uint32_t i = 0; i < block_size && col_tile + i < post_group_width; i++) {
                                    pack_tile(i, mul_rms_result_cb);
                                }
                                tile_regs_release();
                                cb_push_back(mul_rms_result_cb, tiles_in_block);
                            }
                        }

                        cb_pop_front(reduce_result_cb, 1);
                    }
                }  // P_NORM

                if constexpr (has_weight) {
                    DeviceZoneScopedN("P_WEIGHT");
                    // ----- Sub-phase 2: (x * 1/rms) * weight → mul_weight_result_cb -----
                    // Broadcast weight (default): weight_cb holds num_tile_cols
                    // row-broadcast tiles pushed once per worker; we use
                    // mul_tiles_bcast_rows so the same weight column applies to
                    // every token row.
                    // Per-token weight: weight_cb holds chunk_size_rows * num_tile_cols
                    // tiles, with this row's slice at the front. Use mul_tiles
                    // directly (no broadcast) and pop the row's slice at end of
                    // row.
                    reconfig_data_format(mul_rms_result_cb, weight_cb);
                    pack_reconfig_data_format(mul_weight_result_cb);
                    if constexpr (per_token_weight != 0) {
                        mul_tiles_init(mul_rms_result_cb, weight_cb);
                    } else {
                        mul_bcast_rows_init_short(mul_rms_result_cb, weight_cb);
                    }

                    for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                        const uint32_t tiles_in_block =
                            (col_tile + block_size <= num_tile_cols) ? block_size : (num_tile_cols - col_tile);
                        cb_wait_front(weight_cb, col_tile + tiles_in_block);
                        cb_wait_front(mul_rms_result_cb, block_size);
                        tile_regs_acquire();
                        for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                            if constexpr (per_token_weight != 0) {
                                mul_tiles(mul_rms_result_cb, weight_cb, i, col_tile + i, i);
                            } else {
                                mul_tiles_bcast_rows(mul_rms_result_cb, weight_cb, i, col_tile + i, i);
                            }
                        }
                        tile_regs_commit();
                        cb_pop_front(mul_rms_result_cb, block_size);
                        cb_reserve_back(mul_weight_result_cb, block_size);
                        tile_regs_wait();
                        for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                            pack_tile(i, mul_weight_result_cb);
                        }
                        tile_regs_release();
                        cb_push_back(mul_weight_result_cb, block_size);
                    }
                }

                if constexpr (has_bias) {
                    // ----- Sub-phase 2.5: + bias → add_bias_result_cb -----
                    // Broadcast bias uses add_tiles_bcast_rows; per-token bias
                    // uses add_tiles. Same per-row pop pattern as weight when
                    // per_token_bias.
                    reconfig_data_format(mul_weight_result_cb, bias_cb);
                    pack_reconfig_data_format(add_bias_result_cb);
                    if constexpr (per_token_bias != 0) {
                        add_tiles_init(mul_weight_result_cb, bias_cb);
                    } else {
                        add_bcast_rows_init_short(mul_weight_result_cb, bias_cb);
                    }
                    for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                        const uint32_t tiles_in_block =
                            (col_tile + block_size <= num_tile_cols) ? block_size : (num_tile_cols - col_tile);
                        cb_wait_front(bias_cb, col_tile + tiles_in_block);
                        cb_wait_front(mul_weight_result_cb, block_size);
                        tile_regs_acquire();
                        for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                            if constexpr (per_token_bias != 0) {
                                add_tiles(mul_weight_result_cb, bias_cb, i, col_tile + i, i);
                            } else {
                                add_tiles_bcast_rows(mul_weight_result_cb, bias_cb, i, col_tile + i, i);
                            }
                        }
                        tile_regs_commit();
                        cb_pop_front(mul_weight_result_cb, block_size);
                        cb_reserve_back(add_bias_result_cb, block_size);
                        tile_regs_wait();
                        for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                            pack_tile(i, add_bias_result_cb);
                        }
                        tile_regs_release();
                        cb_push_back(add_bias_result_cb, block_size);
                    }
                }

                if constexpr (fuse_rope) {
                    {
                        DeviceZoneScopedN("P_MM");
                        // ----- Sub-phase 3a: matmul(intermediate, trans_mat) → rotated -----
                        reconfig_data_format(transformation_mat_cb, intermediate_cb);
                        pack_reconfig_data_format(rotated_input_cb);
                        mm_init_short(intermediate_cb, transformation_mat_cb);
                        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                            // Don't pop intermediate — Phase 3b needs to read it again.
                            cb_wait_front(intermediate_cb, col_tile + block_size);
                            cb_reserve_back(rotated_input_cb, block_size);
                            // Standard acquire→math→commit→wait→pack→release.
                            tile_regs_acquire();
                            for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                                matmul_tiles(intermediate_cb, transformation_mat_cb, col_tile + i, 0, i);
                            }
                            tile_regs_commit();
                            tile_regs_wait();
                            for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                                pack_tile(i, rotated_input_cb);
                            }
                            tile_regs_release();
                            cb_push_back(rotated_input_cb, block_size);
                        }
                    }  // P_MM

                    {
                        DeviceZoneScopedN("P_COS");
                        // ----- Sub-phase 3b: intermediate * cos → intermediate (in-place) -----
                        // Per-head RoPE (per_head_rope=1): rope_cos_cb holds
                        // num_tile_cols tiles (one per col), indexed directly by
                        // col_tile+i. Broadcast RoPE (per_head_rope=0): rope_cos_cb
                        // holds head_dim_tiles tiles, cycled with rope_cos_tile_in_head.
                        // Cumulative wait lets compute start as soon as first cos
                        // tiles arrive.
                        reconfig_data_format(intermediate_cb, rope_cos_cb);
                        pack_reconfig_data_format(intermediate_cb);
                        mul_tiles_init(intermediate_cb, rope_cos_cb);
                        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                            const uint32_t cos_tiles_needed =
                                (per_head_rope != 0)
                                    ? ((col_tile + block_size <= num_tile_cols) ? (col_tile + block_size)
                                                                                : num_tile_cols)
                                    : (((col_tile + block_size) < head_dim_tiles) ? (col_tile + block_size)
                                                                                  : head_dim_tiles);
                            cb_wait_front(rope_cos_cb, cos_tiles_needed);
                            cb_wait_front(intermediate_cb, block_size);
                            tile_regs_acquire();
                            for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                                if constexpr (per_head_rope != 0) {
                                    mul_tiles(intermediate_cb, rope_cos_cb, i, col_tile + i, i);
                                } else {
                                    mul_tiles(intermediate_cb, rope_cos_cb, i, rope_cos_tile_in_head, i);
                                    rope_cos_tile_in_head++;
                                    if (rope_cos_tile_in_head == head_dim_tiles) {
                                        rope_cos_tile_in_head = 0;
                                    }
                                }
                            }
                            tile_regs_commit();
                            cb_pop_front(intermediate_cb, block_size);
                            cb_reserve_back(intermediate_cb, block_size);
                            tile_regs_wait();
                            for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                                pack_tile(i, intermediate_cb);
                            }
                            tile_regs_release();
                            cb_push_back(intermediate_cb, block_size);
                        }
                    }  // P_COS

                    {
                        DeviceZoneScopedN("P_SIN");
                        // ----- Sub-phase 3c: rotated * sin → rotated (in-place) -----
                        // Same pattern as cos: per_head_rope=1 uses col_tile+i index
                        // directly; per_head_rope=0 cycles within head_dim_tiles.
                        reconfig_data_format(rotated_input_cb, rope_sin_cb);
                        pack_reconfig_data_format(rotated_input_cb);
                        mul_tiles_init(rotated_input_cb, rope_sin_cb);
                        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                            const uint32_t sin_tiles_needed =
                                (per_head_rope != 0)
                                    ? ((col_tile + block_size <= num_tile_cols) ? (col_tile + block_size)
                                                                                : num_tile_cols)
                                    : (((col_tile + block_size) < head_dim_tiles) ? (col_tile + block_size)
                                                                                  : head_dim_tiles);
                            cb_wait_front(rope_sin_cb, sin_tiles_needed);
                            cb_wait_front(rotated_input_cb, block_size);
                            tile_regs_acquire();
                            for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                                if constexpr (per_head_rope != 0) {
                                    mul_tiles(rotated_input_cb, rope_sin_cb, i, col_tile + i, i);
                                } else {
                                    mul_tiles(rotated_input_cb, rope_sin_cb, i, rope_sin_tile_in_head, i);
                                    rope_sin_tile_in_head++;
                                    if (rope_sin_tile_in_head == head_dim_tiles) {
                                        rope_sin_tile_in_head = 0;
                                    }
                                }
                            }
                            tile_regs_commit();
                            cb_pop_front(rotated_input_cb, block_size);
                            cb_reserve_back(rotated_input_cb, block_size);
                            tile_regs_wait();
                            for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                                pack_tile(i, rotated_input_cb);
                            }
                            tile_regs_release();
                            cb_push_back(rotated_input_cb, block_size);
                        }
                    }  // P_SIN

                    {
                        DeviceZoneScopedN("P_ADD");
                        // ----- Sub-phase 3d: intermediate + rotated → output -----
                        reconfig_data_format(intermediate_cb, rotated_input_cb);
                        pack_reconfig_data_format(output_cb);
                        add_tiles_init(intermediate_cb, rotated_input_cb);
                        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                            cb_wait_front(intermediate_cb, block_size);
                            cb_wait_front(rotated_input_cb, block_size);
                            cb_reserve_back(output_cb, block_size);
                            // Standard acquire→math→commit→wait→pack→release.
                            tile_regs_acquire();
                            for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                                add_tiles(intermediate_cb, rotated_input_cb, i, i, i);
                            }
                            tile_regs_commit();
                            tile_regs_wait();
                            for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                                pack_tile(i, output_cb);
                            }
                            tile_regs_release();
                            cb_push_back(output_cb, block_size);
                            cb_pop_front(intermediate_cb, block_size);
                            cb_pop_front(rotated_input_cb, block_size);
                        }
                    }  // P_ADD
                }

                if constexpr (fuse_rope) {
                    // Pop matches per-row push count: num_tile_cols for per-head
                    // RoPE, head_dim_tiles for broadcast RoPE.
                    constexpr uint32_t rope_pop_per_row = (per_head_rope != 0) ? num_tile_cols : head_dim_tiles;
                    cb_pop_front(rope_cos_cb, rope_pop_per_row);
                    cb_pop_front(rope_sin_cb, rope_pop_per_row);
                }

                // Per-token weight/bias: pop the row's slice now so the next
                // row's slice is at the front. (Broadcast path keeps the same
                // tiles across all rows and pops at end of kernel.)
                if constexpr (per_token_weight != 0) {
                    cb_pop_front(weight_cb, num_tile_cols);
                }
                if constexpr (per_token_bias != 0) {
                    cb_pop_front(bias_cb, num_tile_cols);
                }
            }
        }  // RMS_POST

        // -------- RELEASE THIS CHUNK --------
        // Streaming popped input block-by-block in PRE (num_tile_cols) +
        // P_NMUL (num_tile_cols) = 2*num_tile_cols, matching the reader's two
        // passes. Only the resident path holds the chunk and drains it here.
        if constexpr (!streaming_low_l1) {
            cb_pop_front(input_cb, chunk_input_tiles);
        }
        // NOTE: stats_gathered_cb is NOT popped here — the reduce<AVG> with
        // default WaitAndPopPerTile policy already drains chunk_stats_tiles
        // (4 tiles × 3 rows for the AG path, or num_heads × rows for per_head).
        // A manual pop here would be a DOUBLE-POP, advancing rd_ptr past
        // wr_ptr and causing subsequent chunks to read stale L1 contents.

        row_processed += rows_in_chunk;
    }

    cb_pop_front(reduce_scalar_sum_cb, 1);
    cb_pop_front(reduce_scalar_avg_cb, 1);
    cb_pop_front(epsilon_cb, 1);
    // Broadcast weight/bias: pop the worker's single row slice at end of
    // kernel. Per-token already popped per-row inside the chunk loop.
    if constexpr (has_weight && per_token_weight == 0) {
        cb_pop_front(weight_cb, num_tile_cols);
    }
    if constexpr (has_bias && per_token_bias == 0) {
        cb_pop_front(bias_cb, num_tile_cols);
    }
    if constexpr (fuse_rope) {
        cb_pop_front(transformation_mat_cb, 1);
    }
}
