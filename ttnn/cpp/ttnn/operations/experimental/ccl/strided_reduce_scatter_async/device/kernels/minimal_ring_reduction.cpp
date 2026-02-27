// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/debug/dprint.h"
#include "strided_ring_reduce_scatter_common.hpp"

void kernel_main() {
    // Compile-time arguments (must match reader/writer on same device)
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb = get_compile_time_arg_val(2);
    constexpr uint32_t tile_granularity = get_compile_time_arg_val(3);
    constexpr uint32_t ring_size = get_compile_time_arg_val(4);
    constexpr uint32_t input_tensor_B = get_compile_time_arg_val(5);
    constexpr uint32_t mm_M_unit_blocks_per_core = get_compile_time_arg_val(6);
    constexpr uint32_t mm_N_full_blocks_per_slice = get_compile_time_arg_val(7);
    constexpr uint32_t mm_block_ht = get_compile_time_arg_val(8);
    constexpr uint32_t mm_cores_y = get_compile_time_arg_val(9);
    constexpr uint32_t chunk_width_in_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t chunks_per_mm_N_full_block = get_compile_time_arg_val(11);
    constexpr uint32_t mm_N_full_block_wt = get_compile_time_arg_val(13);
    constexpr uint32_t slice_Ht_per_core = get_compile_time_arg_val(14);
    constexpr uint32_t slice_Ht = get_compile_time_arg_val(15);

    uint32_t arg_idx = 0;
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_workers = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t batch_size = input_tensor_B;
    const uint32_t last_mm_core_idx = mm_cores_y - 1;
    const uint32_t effective_worker_id = worker_id + (direction ? num_workers : 0);
    const uint32_t effective_advance_by_tiles = 2 * num_workers;

    binary_op_init_common(input_cb, intermediate_cb, output_cb);
    add_tiles_init(input_cb, intermediate_cb, false);

    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t m_block_iter = 0; m_block_iter < mm_M_unit_blocks_per_core; m_block_iter++) {
            const uint32_t current_mm_block_ht =
                get_current_mm_block_ht(m_block_iter, mm_M_unit_blocks_per_core, mm_block_ht, slice_Ht_per_core);
            for (uint32_t chunk_idx = 0; chunk_idx < chunks_per_mm_N_full_block; chunk_idx++) {
                const uint32_t effective_chunk_width_in_tiles =
                    get_effective_chunk_width_in_tiles(chunk_idx, chunk_width_in_tiles, mm_N_full_block_wt);
                const uint32_t effective_subchunk_size = current_mm_block_ht * effective_chunk_width_in_tiles;

                for (uint32_t i = 1; i < ring_size; i++) {
                    for (uint32_t chunk_piece_idx = 0; chunk_piece_idx < mm_N_full_blocks_per_slice;
                         chunk_piece_idx++) {
                        uint32_t tile_row_in_mm_M_unit_block = 0;
                        uint32_t chunk_col_in_tiles = 0;
                        uint32_t mm_core_idx = 0;

                        get_next_tile_coordinates(
                            tile_row_in_mm_M_unit_block,
                            chunk_col_in_tiles,
                            mm_core_idx,
                            effective_worker_id,
                            effective_subchunk_size,
                            effective_chunk_width_in_tiles,
                            current_mm_block_ht);
                        uint32_t tiles_to_read = how_many_tiles_to_read_formula(
                            tile_row_in_mm_M_unit_block,
                            chunk_col_in_tiles,
                            mm_core_idx,
                            effective_advance_by_tiles,
                            last_mm_core_idx,
                            effective_subchunk_size,
                            effective_chunk_width_in_tiles);

                        while (tiles_to_read > 0) {
                            const uint32_t tiles_to_read_in_this_step = std::min(tiles_to_read, tile_granularity);
                            tiles_to_read -= tiles_to_read_in_this_step;

                            cb_wait_front(input_cb, tile_granularity);
                            cb_wait_front(intermediate_cb, tile_granularity);
                            cb_reserve_back(output_cb, tile_granularity);
                            acquire_dst();
                            for (uint32_t tile_id = 0; tile_id < tiles_to_read_in_this_step; tile_id++) {
                                add_tiles(input_cb, intermediate_cb, tile_id, tile_id, tile_id);
                                pack_tile(tile_id, output_cb);
                            }
                            release_dst();
                            cb_pop_front(input_cb, tile_granularity);
                            cb_pop_front(intermediate_cb, tile_granularity);
                            cb_push_back(output_cb, tile_granularity);
                        }
                    }
                }
            }
        }
    }
}
