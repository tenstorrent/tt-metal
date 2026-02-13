// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/debug/dprint.h"
#include "strided_ring_reduce_scatter_common.hpp"

void kernel_main() {
    // Compile-time arguments (must match reader/writer on same device)
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb = get_compile_time_arg_val(2);
    constexpr uint32_t tile_granularity = get_compile_time_arg_val(3);
    constexpr uint32_t ring_size = get_compile_time_arg_val(4);
    constexpr uint32_t input_tensor_B = get_compile_time_arg_val(5);
    constexpr uint32_t mm_M_blocks_per_core = get_compile_time_arg_val(6);
    constexpr uint32_t mm_N_blocks_per_slice = get_compile_time_arg_val(7);
    constexpr uint32_t mm_block_ht = get_compile_time_arg_val(8);
    constexpr uint32_t mm_cores_y = get_compile_time_arg_val(9);
    constexpr uint32_t chunk_width_in_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t chunks_per_mm_N_block = get_compile_time_arg_val(11);
    constexpr uint32_t slice_Wt = get_compile_time_arg_val(12);

    uint32_t arg_idx = 0;
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_workers = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t batch_size = input_tensor_B;
    const uint32_t last_mm_core_idx = mm_cores_y - 1;
    const uint32_t effective_worker_id = worker_id + (direction ? num_workers : 0);
    const uint32_t effective_advance_by_tiles = 2 * num_workers;

    DPRINT << "compile time args:" << ENDL();
    DPRINT << "input_cb_id: " << input_cb_id << ENDL();
    DPRINT << "intermediate_cb: " << intermediate_cb << ENDL();
    DPRINT << "output_cb: " << output_cb << ENDL();
    DPRINT << "tile_granularity: " << tile_granularity << ENDL();
    DPRINT << "ring_size: " << ring_size << ENDL();
    DPRINT << "batch_size: " << input_tensor_B << ENDL();
    DPRINT << "effective_worker_id: " << effective_worker_id << ENDL();
    DPRINT << "effective_advance_by_tiles: " << effective_advance_by_tiles << ENDL();
    DPRINT << "last_mm_core_idx: " << last_mm_core_idx << ENDL();
    DPRINT << "slice_Wt: " << slice_Wt << ENDL();
    DPRINT << "tile_granularity: " << tile_granularity << ENDL();
    DPRINT << "direction: " << (uint32_t)direction << ENDL();
    DPRINT << "worker_id: " << worker_id << ENDL();
    DPRINT << "num_workers: " << num_workers << ENDL();
    DPRINT << "batch_size: " << batch_size << ENDL();
    DPRINT << "mm_M_blocks_per_core: " << mm_M_blocks_per_core << ENDL();
    DPRINT << "chunks_per_mm_N_block: " << chunks_per_mm_N_block << ENDL();
    DPRINT << "ring_size: " << ring_size << ENDL();
    DPRINT << "mm_N_blocks_per_slice: " << mm_N_blocks_per_slice << ENDL();
    DPRINT << "mm_block_ht: " << mm_block_ht << ENDL();
    DPRINT << "mm_cores_y: " << mm_cores_y << ENDL();
    DPRINT << "chunk_width_in_tiles: " << chunk_width_in_tiles << ENDL();

    DPRINT << "The reduction kernel running its loop." << ENDL();

    binary_op_init_common(input_cb_id, intermediate_cb, output_cb);
    add_tiles_init(input_cb_id, intermediate_cb, false);

    for (uint32_t b = 0; b < batch_size; b++) {
        DPRINT << "================================================" << ENDL();
        DPRINT << "batch: " << b << " started" << ENDL();

        for (uint32_t m_block_iter = 0; m_block_iter < mm_M_blocks_per_core; m_block_iter++) {
            DPRINT << "--------------------------------" << ENDL();
            DPRINT << "m_block_iter: " << m_block_iter << " started" << ENDL();

            for (uint32_t chunk_idx = 0; chunk_idx < chunks_per_mm_N_block; chunk_idx++) {
                DPRINT << "chunk_idx: " << chunk_idx << " started" << ENDL();

                for (uint32_t i = 1; i < ring_size; i++) {
                    DPRINT << "************************************************" << ENDL();
                    DPRINT << "ring iteration: " << i << " started" << ENDL();
                    DPRINT << "direction: " << (uint32_t)direction << ENDL();

                    for (uint32_t chunk_piece_idx = 0; chunk_piece_idx < mm_N_blocks_per_slice; chunk_piece_idx++) {
                        DPRINT << "chunk_piece_idx: " << chunk_piece_idx << " started" << ENDL();

                        uint32_t first_tile_row_in_mm_M_block = 0;
                        uint32_t first_chunk_col_in_tiles = 0;
                        uint32_t first_mm_core_idx = 0;
                        uint32_t effective_chunk_width_in_tiles =
                            get_effective_chunk_width_in_tiles(chunk_idx, chunk_width_in_tiles, slice_Wt);
                        uint32_t effective_chunk_piece_size = mm_block_ht * effective_chunk_width_in_tiles;
                        get_next_tile_coordinates(
                            first_tile_row_in_mm_M_block,
                            first_chunk_col_in_tiles,
                            first_mm_core_idx,
                            effective_worker_id,
                            effective_chunk_piece_size,
                            effective_chunk_width_in_tiles,
                            mm_block_ht);
                        uint32_t tiles_to_read = how_many_tiles_to_read_formula(
                            first_tile_row_in_mm_M_block,
                            first_chunk_col_in_tiles,
                            first_mm_core_idx,
                            effective_advance_by_tiles,
                            last_mm_core_idx,
                            effective_chunk_piece_size,
                            effective_chunk_width_in_tiles);

                        DPRINT << "tiles_to_read: " << tiles_to_read << ENDL();

                        while (tiles_to_read > 0) {
                            uint32_t tiles_to_read_in_this_step = std::min(tiles_to_read, tile_granularity);
                            tiles_to_read -= tiles_to_read_in_this_step;

                            cb_wait_front(input_cb_id, tile_granularity);
                            cb_wait_front(intermediate_cb, tile_granularity);
                            cb_reserve_back(output_cb, tile_granularity);
                            acquire_dst();
                            for (uint32_t tile_id = 0; tile_id < tiles_to_read_in_this_step; tile_id++) {
                                add_tiles(input_cb_id, intermediate_cb, tile_id, tile_id, tile_id);
                                pack_tile(tile_id, output_cb);
                            }
                            release_dst();
                            cb_pop_front(input_cb_id, tile_granularity);
                            cb_pop_front(intermediate_cb, tile_granularity);
                            cb_push_back(output_cb, tile_granularity);
                        }

                        DPRINT << "chunk_piece_idx: " << chunk_piece_idx << " done" << ENDL();
                    }

                    DPRINT << "ring iteration: " << i << " done" << ENDL();
                }

                DPRINT << "chunk_idx: " << chunk_idx << " done" << ENDL();
            }

            DPRINT << "m_block_iter: " << m_block_iter << " done" << ENDL();
        }

        DPRINT << "batch: " << b << " done" << ENDL();
    }
}
