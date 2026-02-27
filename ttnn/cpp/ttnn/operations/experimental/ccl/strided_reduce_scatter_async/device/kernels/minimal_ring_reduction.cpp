// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
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
<<<<<<< HEAD
    constexpr uint32_t mm_block_ht = get_compile_time_arg_val(7);
    constexpr uint32_t mm_cores_y = get_compile_time_arg_val(8);
    constexpr uint32_t chunk_width_in_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t chunks_per_mm_N_full_block = get_compile_time_arg_val(10);
    constexpr uint32_t slice_Wt = get_compile_time_arg_val(11);
    constexpr uint32_t mm_N_full_block_wt = get_compile_time_arg_val(12);
    constexpr uint32_t slice_Ht_per_core = get_compile_time_arg_val(13);
    constexpr uint32_t slice_Ht = get_compile_time_arg_val(14);
    constexpr uint32_t my_chip_id = get_compile_time_arg_val(15);

#ifdef FUSE_RS_ADDCMUL
    constexpr uint32_t addcmul_temp_cb = get_compile_time_arg_val(16);
    constexpr uint32_t addcmul_a_cb = get_compile_time_arg_val(17);
    constexpr uint32_t addcmul_b_cb = get_compile_time_arg_val(18);
#endif
=======
    constexpr uint32_t mm_N_full_blocks_per_slice = get_compile_time_arg_val(7);
    constexpr uint32_t mm_block_ht = get_compile_time_arg_val(8);
    constexpr uint32_t mm_cores_y = get_compile_time_arg_val(9);
    constexpr uint32_t chunk_width_in_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t chunks_per_mm_N_full_block = get_compile_time_arg_val(11);
    constexpr uint32_t mm_N_full_block_wt = get_compile_time_arg_val(13);
    constexpr uint32_t slice_Ht_per_core = get_compile_time_arg_val(14);
    constexpr uint32_t slice_Ht = get_compile_time_arg_val(15);
>>>>>>> b56ae1d9f9 (handle any M continue -- requires cleanup)

    uint32_t arg_idx = 0;
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_workers = get_arg_val<uint32_t>(arg_idx++);
#ifdef FUSE_RS_ADDCMUL
    const uint32_t fused_ternary_scalar_uint = get_arg_val<uint32_t>(arg_idx++);
#endif

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

                // Same slice_idx pattern as the reader, but starting at i=1 (skipping
                // the read-only i=0 pass that the compute kernel does not participate in).
                int32_t slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;
                slice_idx += direction ? -1 : 1;
                for (uint32_t i = 1; i < ring_size; i++) {
                    const uint32_t actual_slice_idx = wrap_slice_idx(slice_idx, direction, ring_size);
                    const uint32_t mm_N_full_blocks_per_slice =
                        get_slice_N_block_info(actual_slice_idx, slice_Wt, mm_N_full_block_wt).first;
#ifdef FUSE_RS_ADDCMUL
                    const bool is_final_ring_step = (i == ring_size - 1);
#endif

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

#ifdef FUSE_RS_ADDCMUL
                            if (is_final_ring_step) {
                                // -------------------------------------------------------
                                // Fused addcmul at the final ring step:
                                //   output = a + scalar * (input + intermediate) * b
                                //
                                // Step 1: acc = input + intermediate -> addcmul_temp_cb
                                // -------------------------------------------------------
                                cb_wait_front(input_cb, tile_granularity);
                                cb_wait_front(intermediate_cb, tile_granularity);
                                cb_reserve_back(addcmul_temp_cb, tile_granularity);

                                add_tiles_init(input_cb, intermediate_cb, false);
                                reconfig_data_format(input_cb, intermediate_cb);
                                pack_reconfig_data_format(addcmul_temp_cb);

                                acquire_dst();
                                for (uint32_t tile_id = 0; tile_id < tiles_to_read_in_this_step; tile_id++) {
                                    add_tiles(input_cb, intermediate_cb, tile_id, tile_id, tile_id);
                                    pack_tile(tile_id, addcmul_temp_cb);
                                }
                                release_dst();
                                cb_pop_front(input_cb, tile_granularity);
                                cb_pop_front(intermediate_cb, tile_granularity);
                                cb_push_back(addcmul_temp_cb, tile_granularity);

                                // -------------------------------------------------------
                                // Step 2: scalar * acc * b -> pack back to addcmul_temp_cb
                                // b has 1 row per tile, broadcast across acc's rows.
                                // -------------------------------------------------------
                                cb_wait_front(addcmul_temp_cb, tile_granularity);
                                cb_wait_front(addcmul_b_cb, tile_granularity);

                                mul_bcast_rows_init_short(addcmul_temp_cb, addcmul_b_cb);
                                reconfig_data_format(addcmul_temp_cb, addcmul_b_cb);
                                pack_reconfig_data_format(addcmul_temp_cb);
                                binop_with_scalar_tile_init();

                                for (uint32_t tile_id = 0; tile_id < tiles_to_read_in_this_step; tile_id++) {
                                    tile_regs_acquire();
                                    mul_tiles_bcast<BroadcastType::ROW>(
                                        addcmul_temp_cb, addcmul_b_cb, tile_id, tile_id, 0);
                                    mul_unary_tile(0, fused_ternary_scalar_uint);
                                    tile_regs_commit();
                                    tile_regs_wait();
                                    pack_tile(0, addcmul_temp_cb);
                                    tile_regs_release();
                                }
                                cb_pop_front(addcmul_b_cb, tile_granularity);
                                cb_pop_front(addcmul_temp_cb, tile_granularity);
                                cb_reserve_back(addcmul_temp_cb, tile_granularity);
                                cb_push_back(addcmul_temp_cb, tile_granularity);

                                // -------------------------------------------------------
                                // Step 3: a + scalar*acc*b -> output_cb
                                // -------------------------------------------------------
                                cb_wait_front(addcmul_temp_cb, tile_granularity);
                                cb_wait_front(addcmul_a_cb, tile_granularity);
                                cb_reserve_back(output_cb, tile_granularity);

                                add_tiles_init(addcmul_temp_cb, addcmul_a_cb, false);
                                reconfig_data_format(addcmul_temp_cb, addcmul_a_cb);
                                pack_reconfig_data_format(output_cb);

                                acquire_dst();
                                for (uint32_t tile_id = 0; tile_id < tiles_to_read_in_this_step; tile_id++) {
                                    add_tiles(addcmul_temp_cb, addcmul_a_cb, tile_id, tile_id, tile_id);
                                    pack_tile(tile_id, output_cb);
                                }
                                release_dst();
                                cb_pop_front(addcmul_temp_cb, tile_granularity);
                                cb_pop_front(addcmul_a_cb, tile_granularity);
                                cb_push_back(output_cb, tile_granularity);
                            } else {
#endif
                                // Normal ring accumulation step: acc = input + intermediate
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
#ifdef FUSE_RS_ADDCMUL
                            }  // end else (non-final ring step)
#endif
                        }
                    }
                    slice_idx += direction ? -1 : 1;
                }
            }
        }
    }
}
