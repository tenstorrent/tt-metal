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
#include "api/compute/layernorm.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

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

    constexpr uint32_t stats_dest_cb = (is_tp_1 != 0) ? stats_gathered_cb : stats_local_cb;

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
    constexpr uint32_t mul_weight_result_cb = fuse_rope ? intermediate_cb : output_cb;

    // Process the core's tile rows in chunks of chunk_size_rows.
    uint32_t row_processed = 0;
    while (row_processed < num_tile_rows) {
        const uint32_t rows_in_chunk =
            (row_processed + chunk_size_rows <= num_tile_rows) ? chunk_size_rows : (num_tile_rows - row_processed);
        const uint32_t chunk_input_tiles = rows_in_chunk * num_tile_cols;
        const uint32_t chunk_stats_tiles = rows_in_chunk * stats_tiles_cols;

        // -------- PHASE 1: PRE — sum(x**2) per row --------
        // Cumulative input wait (Phase 4): instead of one cb_wait_front for the
        // whole chunk, wait per col-block. Lets the reader push block N+1 while
        // compute is processing block N. Counter resets per chunk because we
        // cb_pop_front(input_cb, chunk_input_tiles) at the end.
        uint32_t input_tiles_waited = 0;
        for (uint32_t r = 0; r < rows_in_chunk; r++) {
            const uint32_t row_base = r * num_tile_cols;

            reconfig_data_format(input_cb, input_cb);
            pack_reconfig_data_format(pre_intermediate_cb);
            PACK((llk_pack_reconfig_l1_acc(0)));
            mul_tiles_init(input_cb, input_cb);

            cb_reserve_back(pre_intermediate_cb, 1);

            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                input_tiles_waited += tiles_in_block;
                cb_wait_front(input_cb, input_tiles_waited);

                tile_regs_acquire();
                for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                    const uint32_t abs_idx = row_base + col_tile + i;
                    mul_tiles(input_cb, input_cb, abs_idx, abs_idx, i);
                }
                tile_regs_commit();

                tile_regs_wait();
                for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                    pack_tile<true>(i, pre_intermediate_cb, 0);
                    if (col_tile == 0 && i == 0) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
                }
                tile_regs_release();
            }
            cb_push_back(pre_intermediate_cb, 1);
            PACK((llk_pack_reconfig_l1_acc(0)));

            // Row-reduce → 1 stat tile per row. SUM (col 0 = sum of row).
            // Post phase will divide by H_full via AVG scalar.
            compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
                pre_intermediate_cb,
                reduce_scalar_sum_cb,
                stats_dest_cb,
                compute_kernel_lib::ReduceInputBlockShape::single());
        }

        // -------- WAIT FOR FORWARDER TO COMPLETE AG FOR THIS CHUNK --------
        cb_wait_front(stats_gathered_cb, chunk_stats_tiles);

        // -------- PHASE 3: POST — finalize normalization --------
        for (uint32_t r = 0; r < rows_in_chunk; r++) {
            const uint32_t row_base = r * num_tile_cols;
            uint32_t rope_cos_tile_in_head = 0;
            uint32_t rope_sin_tile_in_head = 0;

            // Reduce gathered TP partials. For TP=1 (stats_tiles_cols==1), this
            // is effectively a pass-through (single tile averaged with itself).
            compute_kernel_lib::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW>(
                stats_gathered_cb,
                reduce_scalar_avg_cb,
                reduce_result_cb,
                compute_kernel_lib::ReduceInputBlockShape::row(stats_tiles_cols));

            // mean + eps; rsqrt → reduce_result_cb (in-place)
            cb_wait_front(reduce_result_cb, 1);
            reconfig_data_format(reduce_result_cb, epsilon_cb);
            pack_reconfig_data_format(reduce_result_cb);
            add_tiles_init(reduce_result_cb, epsilon_cb);
            tile_regs_acquire();
            add_tiles(reduce_result_cb, epsilon_cb, 0, 0, 0);
            rsqrt_tile_init<use_legacy_rsqrt>();
            rsqrt_tile<use_legacy_rsqrt>(0);
            tile_regs_commit();
            cb_pop_front(reduce_result_cb, 1);
            cb_reserve_back(reduce_result_cb, 1);
            tile_regs_wait();
            pack_tile(0, reduce_result_cb);
            tile_regs_release();
            cb_push_back(reduce_result_cb, 1);

            // Phase 7: hoist reconfig_data_format + *_init_short out of the
            // col-tile loop. Each sub-phase (mul_rms / weight / matmul / cos /
            // sin / add) does ONE reconfig at its start, then a tight col-tile
            // loop. intermediate_cb and rotated_input_cb are sized num_tile_cols
            // (not block_size) to hold a full row between phases.

            cb_wait_front(reduce_result_cb, 1);

            // ----- Sub-phase 1: x * (1/rms) → mul_rms_result_cb -----
            reconfig_data_format(input_cb, reduce_result_cb);
            pack_reconfig_data_format(mul_rms_result_cb);
            mul_bcast_cols_init_short(input_cb, reduce_result_cb);
            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                cb_reserve_back(mul_rms_result_cb, block_size);
                tile_regs_acquire();
                tile_regs_wait();
                for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                    const uint32_t abs_idx = row_base + col_tile + i;
                    mul_tiles_bcast_cols(input_cb, reduce_result_cb, abs_idx, 0, i);
                    pack_tile(i, mul_rms_result_cb);
                }
                tile_regs_commit();
                tile_regs_release();
                cb_push_back(mul_rms_result_cb, block_size);
            }

            if constexpr (has_weight) {
                // ----- Sub-phase 2: (x * 1/rms) * weight → mul_weight_result_cb -----
                // weight is row-broadcast (same per col), pushed to weight_cb by
                // the reader once on the worker's first row. cb_wait_front it
                // here at row granularity; pop happens at end of kernel.
                reconfig_data_format(mul_rms_result_cb, weight_cb);
                pack_reconfig_data_format(mul_weight_result_cb);
                mul_bcast_rows_init_short(mul_rms_result_cb, weight_cb);
                cb_wait_front(weight_cb, num_tile_cols);

                for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                    cb_wait_front(mul_rms_result_cb, block_size);
                    tile_regs_acquire();
                    for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                        mul_tiles_bcast_rows(mul_rms_result_cb, weight_cb, i, col_tile + i, i);
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

            if constexpr (fuse_rope) {
                // ----- Sub-phase 3a: matmul(intermediate, trans_mat) → rotated -----
                reconfig_data_format(transformation_mat_cb, intermediate_cb);
                pack_reconfig_data_format(rotated_input_cb);
                mm_init_short(intermediate_cb, transformation_mat_cb);
                for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                    // Don't pop intermediate — Phase 3b needs to read it again.
                    cb_wait_front(intermediate_cb, col_tile + block_size);
                    cb_reserve_back(rotated_input_cb, block_size);
                    tile_regs_acquire();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                        matmul_tiles(intermediate_cb, transformation_mat_cb, col_tile + i, 0, i);
                        pack_tile(i, rotated_input_cb);
                    }
                    tile_regs_commit();
                    tile_regs_release();
                    cb_push_back(rotated_input_cb, block_size);
                }

                // ----- Sub-phase 3b: intermediate * cos → intermediate (in-place) -----
                reconfig_data_format(intermediate_cb, rope_cos_cb);
                pack_reconfig_data_format(intermediate_cb);
                mul_tiles_init(intermediate_cb, rope_cos_cb);
                cb_wait_front(rope_cos_cb, head_dim_tiles);
                for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                    cb_wait_front(intermediate_cb, block_size);
                    tile_regs_acquire();
                    for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                        mul_tiles(intermediate_cb, rope_cos_cb, i, rope_cos_tile_in_head, i);
                        rope_cos_tile_in_head++;
                        if (rope_cos_tile_in_head == head_dim_tiles) {
                            rope_cos_tile_in_head = 0;
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

                // ----- Sub-phase 3c: rotated * sin → rotated (in-place) -----
                reconfig_data_format(rotated_input_cb, rope_sin_cb);
                pack_reconfig_data_format(rotated_input_cb);
                mul_tiles_init(rotated_input_cb, rope_sin_cb);
                cb_wait_front(rope_sin_cb, head_dim_tiles);
                for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                    cb_wait_front(rotated_input_cb, block_size);
                    tile_regs_acquire();
                    for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                        mul_tiles(rotated_input_cb, rope_sin_cb, i, rope_sin_tile_in_head, i);
                        rope_sin_tile_in_head++;
                        if (rope_sin_tile_in_head == head_dim_tiles) {
                            rope_sin_tile_in_head = 0;
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

                // ----- Sub-phase 3d: intermediate + rotated → output -----
                reconfig_data_format(intermediate_cb, rotated_input_cb);
                pack_reconfig_data_format(output_cb);
                add_tiles_init(intermediate_cb, rotated_input_cb);
                for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                    cb_wait_front(intermediate_cb, block_size);
                    cb_wait_front(rotated_input_cb, block_size);
                    cb_reserve_back(output_cb, block_size);
                    tile_regs_acquire();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                        add_tiles(intermediate_cb, rotated_input_cb, i, i, i);
                        pack_tile(i, output_cb);
                    }
                    tile_regs_commit();
                    tile_regs_release();
                    cb_push_back(output_cb, block_size);
                    cb_pop_front(intermediate_cb, block_size);
                    cb_pop_front(rotated_input_cb, block_size);
                }
            }
            cb_pop_front(reduce_result_cb, 1);

            if constexpr (fuse_rope) {
                cb_pop_front(rope_cos_cb, head_dim_tiles);
                cb_pop_front(rope_sin_cb, head_dim_tiles);
            }
        }

        // -------- RELEASE THIS CHUNK --------
        cb_pop_front(input_cb, chunk_input_tiles);
        cb_pop_front(stats_gathered_cb, chunk_stats_tiles);

        row_processed += rows_in_chunk;
    }

    cb_pop_front(reduce_scalar_sum_cb, 1);
    cb_pop_front(reduce_scalar_avg_cb, 1);
    cb_pop_front(epsilon_cb, 1);
    if constexpr (has_weight) {
        cb_pop_front(weight_cb, num_tile_cols);
    }
    if constexpr (fuse_rope) {
        cb_pop_front(transformation_mat_cb, 1);
    }
}
