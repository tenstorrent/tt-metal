// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t NUM_TILES_IN_TILIZED_CHUNK = 32;

// pack_untilize pulls block_ct_dim tiles into DEST before packing, so block_ct_dim is bounded by
// DEST capacity (16-bit half-sync = 8 tiles; this op runs with fp32_dest_acc_en = false). The
// 32-tile-wide chunk is therefore untilized in DEST-sized sub-blocks, with block_c_index placing
// each sub-block in the correct output columns (full_ct_dim = 32) so tile ordering matches the
// legacy unpack-untilize reorg.
constexpr uint32_t UNTILIZE_SUBBLOCK_CT = 8;
constexpr uint32_t UNTILIZE_FULL_CT = NUM_TILES_IN_TILIZED_CHUNK;
constexpr uint32_t UNTILIZE_NUM_SUBBLOCKS = UNTILIZE_FULL_CT / UNTILIZE_SUBBLOCK_CT;
static_assert(UNTILIZE_FULL_CT % UNTILIZE_SUBBLOCK_CT == 0);

// Staging CB always has NUM_TILES_IN_TILIZED_CHUNK tiles; pop the full chunk to keep it clean.
FORCE_INLINE void pack_block_rows_into_tiles(uint32_t cb_in, uint32_t cb_out) {
    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_out_obj(cb_out);

    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);

    pack_untilize_init<UNTILIZE_SUBBLOCK_CT, UNTILIZE_FULL_CT>(cb_in, cb_out);

    cb_in_obj.wait_front(NUM_TILES_IN_TILIZED_CHUNK);
    cb_out_obj.reserve_back(NUM_TILES_IN_TILIZED_CHUNK);

    // pack_untilize_block does not offset its input read by block_c_index, so pop the input
    // incrementally; each sub-block reads the next UNTILIZE_SUBBLOCK_CT tiles (32 popped total).
    for (uint32_t b = 0; b < UNTILIZE_NUM_SUBBLOCKS; ++b) {
        pack_untilize_block<UNTILIZE_SUBBLOCK_CT, UNTILIZE_FULL_CT>(
            cb_in, /*block_rt_dim=*/1, cb_out, /*block_c_index=*/b);
        cb_in_obj.pop_front(UNTILIZE_SUBBLOCK_CT);
    }

    cb_out_obj.push_back(NUM_TILES_IN_TILIZED_CHUNK);

    pack_untilize_uninit(cb_out);
}

// Tilize the full 32-tile block from cb_in to cb_out, but only push num_valid_tiles.
// The tilize must always process the full NUM_TILES_IN_TILIZED_CHUNK block to correctly
// reconstruct the tile layout from row-major format (inverse of pack_block_rows_into_tiles, i.e. the 32-tile untilize).
FORCE_INLINE void pack_block_tiles_into_rows(uint32_t cb_in, uint32_t cb_out, uint32_t num_valid_tiles) {
    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_out_obj(cb_out);

    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);

    tilize_init(cb_in, NUM_TILES_IN_TILIZED_CHUNK, cb_out);

    cb_in_obj.wait_front(NUM_TILES_IN_TILIZED_CHUNK);
    cb_out_obj.reserve_back(num_valid_tiles);

    tilize_block(cb_in, NUM_TILES_IN_TILIZED_CHUNK, cb_out);

    cb_out_obj.push_back(num_valid_tiles);
    cb_in_obj.pop_front(NUM_TILES_IN_TILIZED_CHUNK);

    tilize_uninit(cb_in, cb_out);
}

FORCE_INLINE void mul(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    CircularBuffer cb_a_obj(cb_a);
    CircularBuffer cb_b_obj(cb_b);
    CircularBuffer cb_out_obj(cb_out);

    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_out);

    mul_init(cb_a, cb_b);

    cb_a_obj.wait_front(1);
    cb_b_obj.wait_front(1);
    cb_out_obj.reserve_back(1);

    tile_regs_acquire();
    mul_tiles(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_out_obj.push_back(1);
    cb_a_obj.pop_front(1);
    cb_b_obj.pop_front(1);
}

FORCE_INLINE void sum(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    CircularBuffer cb_a_obj(cb_a);
    CircularBuffer cb_b_obj(cb_b);
    CircularBuffer cb_out_obj(cb_out);

    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_out);

    add_init(cb_a, cb_b);

    cb_a_obj.wait_front(1);
    cb_b_obj.wait_front(1);
    cb_out_obj.reserve_back(1);

    tile_regs_acquire();
    add_tiles(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_out_obj.push_back(1);
    cb_a_obj.pop_front(1);
    cb_b_obj.pop_front(1);
}

FORCE_INLINE void copy(uint32_t cb_in, uint32_t cb_out, uint32_t num_input_units = 1) {
    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_out_obj(cb_out);

    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);

    copy_tile_to_dst_init_short(cb_in);

    cb_in_obj.wait_front(num_input_units);
    cb_out_obj.reserve_back(1);

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    // Don't pop the copied tile - caller can do it
    cb_out_obj.push_back(1);
}

FORCE_INLINE void compute_ht(
    uint32_t cb_a,
    uint32_t cb_bx,
    uint32_t cb_out,
    uint32_t cb_h_prev,
    uint32_t cb_ah,
    uint32_t cb_h,
    uint32_t cb_h_acc,
    uint32_t num_tiles) {
    CircularBuffer cb_h_obj(cb_h);
    CircularBuffer cb_h_prev_obj(cb_h_prev);
    for (uint32_t idx = 0; idx < num_tiles; idx++) {
        mul(cb_a, cb_h_prev, cb_ah);
        sum(cb_ah, cb_bx, cb_h);
        copy(cb_h, cb_h_prev);
        copy(cb_h, cb_out);  // TODO: Get rid of this extraneous copy
        cb_h_obj.pop_front(1);
    }
    copy(cb_h_prev, cb_h_acc);  // Store the last row of this tile for the next iteration

    // Make sure to remove the last hidden state
    cb_h_prev_obj.wait_front(1);
    cb_h_prev_obj.pop_front(1);
}

void kernel_main() {
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

    const uint32_t total_tiles = get_arg_val<uint32_t>(0);
    const uint32_t total_tiles_per_row = get_arg_val<uint32_t>(1);
    const uint32_t total_tiles_per_col = get_arg_val<uint32_t>(2);
    const uint32_t num_chunks_per_row = get_arg_val<uint32_t>(3);

    CircularBuffer cb_h_in_obj(cb_h_in);
    CircularBuffer cb_h_acc_obj(cb_h_acc);

    compute_kernel_hw_startup(cb_a_in, cb_bx_in, cb_out);
    const uint32_t num_tiles_last_chunk = total_tiles_per_row % NUM_TILES_IN_TILIZED_CHUNK == 0
                                              ? NUM_TILES_IN_TILIZED_CHUNK
                                              : total_tiles_per_row % NUM_TILES_IN_TILIZED_CHUNK;

    // Fill initial hidden states
    for (uint32_t tilized_chunk_idx = 0; tilized_chunk_idx < num_chunks_per_row; tilized_chunk_idx++) {
        const uint32_t remaining_tiles_in_chunk =
            tilized_chunk_idx == num_chunks_per_row - 1 ? num_tiles_last_chunk : NUM_TILES_IN_TILIZED_CHUNK;
        copy(cb_h_in, cb_h_acc, remaining_tiles_in_chunk);
        cb_h_in_obj.pop_front(remaining_tiles_in_chunk);
    }

    // For each row of tiles we want to tilize chunks of 32 tiles to pack the rows into tiles
    for (uint32_t row_idx = 0; row_idx < total_tiles_per_col; row_idx++) {
        for (uint32_t tilized_chunk_idx = 0; tilized_chunk_idx < num_chunks_per_row; tilized_chunk_idx++) {
            // Load the last row from the hidden state above this row
            copy(cb_h_acc, cb_h_prev);
            cb_h_acc_obj.pop_front(1);

            // If we don't have a full chunk (NUM_TILES_IN_TILIZED_CHUNK tiles) we should figure out how many tiles we
            // have left. This only runs 2-3 tiles per shard so no need to unroll.
            const uint32_t remaining_tiles_in_chunk =
                tilized_chunk_idx == num_chunks_per_row - 1 ? num_tiles_last_chunk : NUM_TILES_IN_TILIZED_CHUNK;

            pack_block_rows_into_tiles(cb_a_in, cb_a_tilize_in);
            pack_block_rows_into_tiles(cb_bx_in, cb_bx_tilize_in);

            compute_ht(
                cb_a_tilize_in,
                cb_bx_tilize_in,
                cb_tilize_out,
                cb_h_prev,
                cb_ah,
                cb_h,
                cb_h_acc,
                NUM_TILES_IN_TILIZED_CHUNK);

            pack_block_tiles_into_rows(cb_tilize_out, cb_out, remaining_tiles_in_chunk);
        }
    }
}
