// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"

#include "api/debug/dprint.h"
#include "api/debug/dprint_tile.h"
#include "tt-metalium/constants.hpp"

void kernel_main() {
    // Define all compile-time arguments at the beginning
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_interm_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_interm2_id = 2;  // TODO hardcoded
    constexpr uint32_t cb_compute_output_id = get_compile_time_arg_val(2);
    constexpr uint32_t tile_granularity = get_compile_time_arg_val(3);
    constexpr uint32_t ring_size = get_compile_time_arg_val(4);
    constexpr uint32_t input_tensor_B = get_compile_time_arg_val(5);
    constexpr uint32_t slice_C = get_compile_time_arg_val(6);

    uint32_t arg_idx = 0;
    uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);

    compute_kernel_hw_startup(cb_input_id, cb_interm_id, cb_compute_output_id);

    for (uint32_t b = 0; b < input_tensor_B; ++b) {
        uint32_t num_iters = (ring_size / 2) + 1;
        for (uint32_t i = 0; i < num_iters; ++i) {
            // State machine for control variables
            bool even_chunks, odd_chunks, reduce_even_chunks, reduce_odd_chunks, reduce_output;
            switch (i) {
                case 0: {
                    even_chunks = direction;     // process the even chunks (half the tensor slice)
                    odd_chunks = !direction;     // process the odd chunks (other half of tensor slice)
                    reduce_even_chunks = false;  // (input_tensor + interm_tensor) or (input_tensor)
                    reduce_odd_chunks = false;   // (input_tensor + interm_tensor) or (input_tensor)
                    reduce_output =
                        false;  // (input_tensor + interm_tensor + output_tensor) or (input_tensor + interm_tensor)
                    break;
                }
                case (ring_size / 2): {
                    even_chunks = direction;
                    odd_chunks = !direction;
                    reduce_even_chunks = even_chunks;
                    reduce_odd_chunks = odd_chunks;
                    reduce_output = true;
                    break;
                }
                case 1: {
                    even_chunks = true;
                    odd_chunks = true;
                    reduce_even_chunks = direction;
                    reduce_odd_chunks = !direction;
                    reduce_output = false;
                    break;
                }
                default: {
                    even_chunks = true;
                    odd_chunks = true;
                    reduce_even_chunks = even_chunks;
                    reduce_odd_chunks = odd_chunks;
                    reduce_output = false;
                    break;
                }
            }

            for (uint32_t c = 0; c < slice_C; ++c) {
                uint32_t tiles_read = start_tiles_read;
                uint32_t total_tiles_to_read = start_tiles_to_read;

                bool is_even_chunk = true;
                while (tiles_read < total_tiles_to_read) {
                    uint32_t tiles_to_read = 0;
                    uint32_t tiles_remaining = total_tiles_to_read - tiles_read;
                    if (is_even_chunk) {
                        tiles_to_read = std::min(tiles_remaining / 2, tile_granularity);
                    } else {
                        tiles_to_read = std::min(tiles_remaining, tile_granularity);
                    }

                    if ((is_even_chunk && !even_chunks) || (!is_even_chunk && !odd_chunks)) {
                        // Skip this chunk
                        tiles_read += tiles_to_read;
                    } else {
                        const bool reduce_interm =
                            (is_even_chunk && reduce_even_chunks) || (!is_even_chunk && reduce_odd_chunks);

                        if (reduce_interm) {
                            // If reduce_output, add 3 tensors. Else add 2 tensors.
                            if (reduce_output) {
                                cb_wait_front(cb_interm2_id, tile_granularity);
                            }
                            cb_wait_front(cb_input_id, tile_granularity);
                            cb_wait_front(cb_interm_id, tile_granularity);

                            tile_regs_acquire();  // acquire DST registers for MATH thread, resets DST to 0
                            if (reduce_output) {
                                copy_tile_init(cb_interm2_id);
                                // copy_tile_to_dst_init_short(cb_interm2_id);
                                for (uint32_t tile_id = 0; tile_id < tiles_to_read; ++tile_id) {
                                    copy_tile(cb_interm2_id, tile_id, tile_id);  // load DST
                                }
                                add_tiles_init(cb_input_id, cb_interm_id, true);  // DST = srcA + srcB + DST
                            } else {
                                add_tiles_init(cb_input_id, cb_interm_id, false);  // DST = srcA + srcB
                            }
                            for (uint32_t tile_id = 0; tile_id < tiles_to_read; ++tile_id) {
                                add_tiles(cb_input_id, cb_interm_id, tile_id, tile_id, tile_id);
                            }
                            tile_regs_commit();  // release lock on DST by MATH thread, signal the PACK thread

                            if (reduce_output) {
                                cb_pop_front(cb_interm2_id, tile_granularity);
                            }
                            cb_pop_front(cb_input_id, tile_granularity);
                            cb_pop_front(cb_interm_id, tile_granularity);

                            cb_reserve_back(cb_compute_output_id, tile_granularity);
                            tile_regs_wait();  // acquire lock on DST for PACK thread
                            for (uint32_t tile_id = 0; tile_id < tiles_to_read; ++tile_id) {
                                pack_tile(tile_id, cb_compute_output_id);  // pack results from DST registers to output
                                                                           // circular buffers
                            }
                            tile_regs_release();  // release lock on DST by PACK thread
                            cb_push_back(cb_compute_output_id, tile_granularity);

                            /*// based on ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/compute.cpp
                            for (uint32_t tile_id = 0; tile_id < tiles_to_read; ++tile_id) {
                                tile_regs_acquire(); // acquire DST registers for MATH thread, resets DST to 0
                                if (reduce_output) {
                                    copy_tile_init(cb_interm2_id);
                                    //copy_tile_to_dst_init_short(cb_interm2_id);
                                    copy_tile(cb_interm2_id, 0, 0);  // load DST
                                    add_tiles_init(cb_input_id, cb_interm_id, true);  // DST = srcA + srcB + DST
                                } else {
                                    add_tiles_init(cb_input_id, cb_interm_id, false);  // DST = srcA + srcB
                                }
                                add_tiles(cb_input_id, cb_interm_id, 0, 0, 0);
                                tile_regs_commit();  // release lock on DST by MATH thread, signal the PACK thread

                                if (reduce_output) {
                                    cb_pop_front(cb_interm2_id, 1);
                                }
                                cb_pop_front(cb_input_id, 1);
                                cb_pop_front(cb_interm_id, 1);

                                cb_reserve_back(cb_compute_output_id, 1);
                                tile_regs_wait();  // acquire lock on DST for PACK thread
                                pack_tile(0, cb_compute_output_id);  // pack results from DST registers to output
                            circular buffers tile_regs_release();  // release lock on DST by PACK thread
                                cb_push_back(cb_compute_output_id, 1);
                            }*/
                        }
                        tiles_read += tiles_to_read;

                    }  // if skip or process

                    is_even_chunk = !is_even_chunk;
                }  // while tiles_read
            }  // for slice_C
        }  // for ring_size
    }  // for input_tensor_B
}
