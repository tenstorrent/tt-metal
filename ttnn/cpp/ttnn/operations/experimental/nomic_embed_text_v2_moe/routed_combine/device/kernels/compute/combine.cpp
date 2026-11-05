// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Every core: for each unit, tilizes each slot's 32 gathered row segments, then builds each output
// tile in DST as the bias broadcast down the rows plus each slot's tile times its weights broadcast
// along the columns, accumulated in fp32, and packs it to bf16 once.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
    constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t chunk_tiles = get_named_compile_time_arg_val("chunk_tiles");
    constexpr uint32_t dest_tiles = get_named_compile_time_arg_val("dest_tiles");
    constexpr uint32_t rows_cb_id = get_named_compile_time_arg_val("rows_cb");
    constexpr uint32_t tiles_cb_id = get_named_compile_time_arg_val("tiles_cb");
    constexpr uint32_t weights_cb_id = get_named_compile_time_arg_val("weights_cb");
    constexpr uint32_t bias_cb_id = get_named_compile_time_arg_val("bias_cb");
    constexpr uint32_t output_cb_id = get_named_compile_time_arg_val("output_cb");

    const uint32_t first_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);

    constexpr uint32_t chunks = hidden_tiles / chunk_tiles;
    DataflowBuffer rows_cb(rows_cb_id);
    DataflowBuffer tiles_cb(tiles_cb_id);
    DataflowBuffer weights_cb(weights_cb_id);
    DataflowBuffer bias_cb(bias_cb_id);
    DataflowBuffer output_cb(output_cb_id);

    compute_kernel_hw_startup(tiles_cb_id, weights_cb_id, output_cb_id);
    // The reader fills the bias tiles of the chunks this core covers; the rest stay unread.
    bias_cb.wait_front(hidden_tiles);

    for (uint32_t unit = first_unit; unit < first_unit + num_units; ++unit) {
        const uint32_t first_bias_tile = (unit % chunks) * chunk_tiles;
        // Each slot's 32 rows of chunk_tiles * 32 values become chunk_tiles tiles. The _with_dt init
        // also resets SrcB, which the multiply below leaves in the weights' fp32 format.
        pack_reconfig_data_format(tiles_cb_id);
        fast_tilize_init_with_dt(rows_cb_id, chunk_tiles, tiles_cb_id);
        for (uint32_t k = 0; k < top_k; ++k) {
            rows_cb.wait_front(chunk_tiles);
            tiles_cb.reserve_back(chunk_tiles);
            fast_tilize_block(rows_cb_id, chunk_tiles, tiles_cb_id);
            tiles_cb.push_back(chunk_tiles);
            rows_cb.pop_front(chunk_tiles);
        }
        fast_tilize_uninit(rows_cb_id, tiles_cb_id, chunk_tiles);

        tiles_cb.wait_front(top_k * chunk_tiles);
        weights_cb.wait_front(top_k);
        pack_reconfig_data_format(output_cb_id);
        for (uint32_t first = 0; first < chunk_tiles; first += dest_tiles) {
            const uint32_t count = first + dest_tiles <= chunk_tiles ? dest_tiles : chunk_tiles - first;
            tile_regs_acquire();
            // DST starts as the bias, its one row broadcast down the 32 rows.
            reconfig_data_format_srcb(bias_cb_id);
            unary_bcast_init<BroadcastType::ROW>(bias_cb_id);
            for (uint32_t tile = 0; tile < count; ++tile) {
                unary_bcast<BroadcastType::ROW>(bias_cb_id, first_bias_tile + first + tile, tile);
            }
            unary_bcast_uninit<BroadcastType::ROW>(bias_cb_id);
            // Then each slot's tile times its weights, column 0 broadcast along the columns, accumulates
            // into it: weights at TF32 in SrcB, products exact in the fp32 DST.
            reconfig_data_format(tiles_cb_id, weights_cb_id);
            bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(tiles_cb_id, weights_cb_id);
            MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(
                tiles_cb_id, weights_cb_id, 1 /*acc_to_dest*/)));
            for (uint32_t k = 0; k < top_k; ++k) {
                for (uint32_t tile = 0; tile < count; ++tile) {
                    mul_tiles_bcast_cols(tiles_cb_id, weights_cb_id, k * chunk_tiles + first + tile, k, tile);
                }
            }
            tile_regs_commit();
            output_cb.reserve_back(count);
            tile_regs_wait();
            for (uint32_t tile = 0; tile < count; ++tile) {
                pack_tile(tile, output_cb_id);
            }
            tile_regs_release();
            output_cb.push_back(count);
        }
        tiles_cb.pop_front(top_k * chunk_tiles);
        weights_cb.pop_front(top_k);
    }
    bias_cb.pop_front(hidden_tiles);
}
