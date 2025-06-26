// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"

constexpr uint32_t NUM_TILES_IN_TILIZED_CHUNK = 32;

constexpr uint32_t cb_a_in = get_compile_time_arg_val(0);
constexpr uint32_t cb_bx_in = get_compile_time_arg_val(1);
constexpr uint32_t cb_h_in = get_compile_time_arg_val(2);

constexpr uint32_t cb_a_tilize_in = get_compile_time_arg_val(3);
constexpr uint32_t cb_bx_tilize_in = get_compile_time_arg_val(4);

constexpr uint32_t cb_h_prev = get_compile_time_arg_val(5);
constexpr uint32_t cb_ah = get_compile_time_arg_val(6);
constexpr uint32_t cb_h = get_compile_time_arg_val(7);

constexpr uint32_t cb_tilize_out = get_compile_time_arg_val(8);
constexpr uint32_t cb_out = get_compile_time_arg_val(9);

constexpr uint32_t cb_h_acc = get_compile_time_arg_val(10);

// This function relies on untilizing NUM_TILES_IN_TILIZED_CHUNK tiles so we pad up to that amount
FORCE_INLINE void pack_block_rows_into_tiles(uint32_t cb_in, uint32_t cb_out, uint32_t num_tiles) {
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);

    untilize_init_short(cb_in);

    cb_wait_front(cb_in, num_tiles);
    cb_reserve_back(cb_out, NUM_TILES_IN_TILIZED_CHUNK);

    untilize_block(cb_in, NUM_TILES_IN_TILIZED_CHUNK, cb_out);

    cb_push_back(cb_out, NUM_TILES_IN_TILIZED_CHUNK);
    cb_pop_front(cb_in, num_tiles);

    untilize_uninit(cb_in);
}

// This function relies on tilizing NUM_TILES_IN_TILIZED_CHUNK tiles so we pad up to that amount
FORCE_INLINE void pack_block_tiles_into_rows(uint32_t cb_in, uint32_t cb_out, uint32_t num_tiles) {
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);

    tilize_init(cb_in, NUM_TILES_IN_TILIZED_CHUNK, cb_out);

    cb_wait_front(cb_in, NUM_TILES_IN_TILIZED_CHUNK);
    cb_reserve_back(cb_out, num_tiles);

    tilize_block(cb_in, NUM_TILES_IN_TILIZED_CHUNK, cb_out);

    cb_push_back(cb_out, num_tiles);
    cb_pop_front(cb_in, NUM_TILES_IN_TILIZED_CHUNK);

    tilize_uninit(cb_in, cb_out);
}

FORCE_INLINE void mul(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_out);

    mul_tiles_init(cb_a, cb_b);

    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    mul_tiles(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, 1);
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
}

FORCE_INLINE void sum(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_out);

    add_tiles_init(cb_a, cb_b);

    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    add_tiles(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, 1);
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
}

FORCE_INLINE void copy(uint32_t cb_in, uint32_t cb_out, uint32_t num_input_units = 1) {
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);

    copy_tile_to_dst_init_short(cb_in);

    cb_wait_front(cb_in, num_input_units);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    // Don't pop the copied tile - caller can do it
    cb_push_back(cb_out, 1);
}

FORCE_INLINE void compute_ht(uint32_t cb_a, uint32_t cb_bx, uint32_t cb_out, uint32_t num_tiles) {
    for (uint32_t idx = 0; idx < num_tiles; idx++) {
        mul(cb_a, cb_h_prev, cb_ah);
        sum(cb_ah, cb_bx, cb_h);
        copy(cb_h, cb_h_prev);
        copy(cb_h, cb_out);  // TODO: Get rid of this extraneous copy
        cb_pop_front(cb_h, 1);
    }
    copy(cb_h_prev, cb_h_acc);  // Store the last row of this tile for the next iteration

    // Make sure to remove the last hidden state
    cb_wait_front(cb_h_prev, 1);
    cb_pop_front(cb_h_prev, 1);
}

namespace NAMESPACE {
void MAIN {
    const uint32_t total_tiles = get_arg_val<uint32_t>(0);
    const uint32_t total_tiles_per_row = get_arg_val<uint32_t>(1);
    const uint32_t total_tiles_per_col = get_arg_val<uint32_t>(2);
    const uint32_t num_chunks_per_row = get_arg_val<uint32_t>(3);

    binary_op_init_common(cb_a_in, cb_bx_in, cb_out);
    const uint32_t num_tiles_last_chunk = total_tiles_per_row % NUM_TILES_IN_TILIZED_CHUNK == 0
                                              ? NUM_TILES_IN_TILIZED_CHUNK
                                              : total_tiles_per_row % NUM_TILES_IN_TILIZED_CHUNK;

    // Fill initial hidden states
    for (uint32_t tilized_chunk_idx = 0; tilized_chunk_idx < num_chunks_per_row; tilized_chunk_idx++) {
        const uint32_t remaining_tiles_in_chunk =
            tilized_chunk_idx == num_chunks_per_row - 1 ? num_tiles_last_chunk : NUM_TILES_IN_TILIZED_CHUNK;
        copy(cb_h_in, cb_h_acc, remaining_tiles_in_chunk);
        cb_pop_front(cb_h_in, remaining_tiles_in_chunk);
    }

    // For each row of tiles we want to tilize chunks of 32 tiles to pack the rows into tiles
    for (uint32_t row_idx = 0; row_idx < total_tiles_per_col; row_idx++) {
        for (uint32_t tilized_chunk_idx = 0; tilized_chunk_idx < num_chunks_per_row; tilized_chunk_idx++) {
            // Load the last row from the hidden state above this row
            copy(cb_h_acc, cb_h_prev);
            cb_pop_front(cb_h_acc, 1);

            // If we don't have a full chunk (NUM_TILES_IN_TILIZED_CHUNK tiles) we should figure out how many tiles we
            // have left. This only runs 2-3 tiles per shard so no need to unroll.
            const uint32_t remaining_tiles_in_chunk =
                tilized_chunk_idx == num_chunks_per_row - 1 ? num_tiles_last_chunk : NUM_TILES_IN_TILIZED_CHUNK;

            pack_block_rows_into_tiles(cb_a_in, cb_a_tilize_in, remaining_tiles_in_chunk);
            pack_block_rows_into_tiles(cb_bx_in, cb_bx_tilize_in, remaining_tiles_in_chunk);

            compute_ht(cb_a_tilize_in, cb_bx_tilize_in, cb_tilize_out, NUM_TILES_IN_TILIZED_CHUNK);

            pack_block_tiles_into_rows(cb_tilize_out, cb_out, remaining_tiles_in_chunk);
        }
    }
}
}  // namespace NAMESPACE
