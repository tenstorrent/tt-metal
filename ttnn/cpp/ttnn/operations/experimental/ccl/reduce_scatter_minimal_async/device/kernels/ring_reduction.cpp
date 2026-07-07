// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "cpp/ttnn/operations/experimental/ccl/reduce_scatter_common/kernels/common.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    // Define all compile-time arguments at the beginning
    constexpr uint32_t cb_input_id = get_named_compile_time_arg_val("cb_input_id");
    constexpr uint32_t cb_interm_id = get_named_compile_time_arg_val("cb_interm_id");
    constexpr uint32_t cb_interm2_id = get_named_compile_time_arg_val("cb_interm2_id");
    constexpr uint32_t cb_compute_output_id = get_named_compile_time_arg_val("cb_compute_output_id");
    constexpr uint32_t tile_granularity = get_named_compile_time_arg_val("tile_granularity");
    constexpr uint32_t ring_size = get_named_compile_time_arg_val("ring_size");
    constexpr uint32_t input_tensor_B = get_named_compile_time_arg_val("input_tensor_B");
    constexpr uint32_t slice_C = get_named_compile_time_arg_val("slice_C");

    uint32_t arg_idx = 0;
    uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);

    CircularBuffer cb_input(cb_input_id);
    CircularBuffer cb_interm(cb_interm_id);
    CircularBuffer cb_interm2(cb_interm2_id);
    CircularBuffer cb_compute_output(cb_compute_output_id);

    compute_kernel_hw_startup(cb_interm_id, cb_input_id, cb_compute_output_id);

    for (uint32_t b = 0; b < input_tensor_B; ++b) {
        constexpr uint32_t ring_size_by_2 = ring_size / 2;
        uint32_t num_iters = ring_size_by_2 + 1;
        for (uint32_t i = 0; i < num_iters; ++i) {
            // State machine for control variables
            bool even_chunks, odd_chunks, reduce_even_chunks, reduce_odd_chunks, reduce_output;
            if (i == 0) {
                even_chunks = direction;     // process the even chunks (half the tensor slice)
                odd_chunks = !direction;     // process the odd chunks (other half of tensor slice)
                reduce_even_chunks = false;  // (input_tensor + interm_tensor) or (input_tensor)
                reduce_odd_chunks = false;   // (input_tensor + interm_tensor) or (input_tensor)
                reduce_output =
                    false;  // (input_tensor + interm_tensor + output_tensor) or (input_tensor + interm_tensor)
            } else if (i == ring_size_by_2) {
                even_chunks = direction;
                odd_chunks = !direction;
                reduce_even_chunks = even_chunks;
                reduce_odd_chunks = odd_chunks;
                reduce_output = true;
            } else if (i == 1) {
                even_chunks = true;
                odd_chunks = true;
                reduce_even_chunks = direction;
                reduce_odd_chunks = !direction;
                reduce_output = false;
            } else {
                even_chunks = true;
                odd_chunks = true;
                reduce_even_chunks = even_chunks;
                reduce_odd_chunks = odd_chunks;
                reduce_output = false;
            }

            for (uint32_t c = 0; c < slice_C; ++c) {
                uint32_t tiles_read = start_tiles_read;
                uint32_t total_tiles_to_read = start_tiles_to_read;

                while (tiles_read < total_tiles_to_read) {
                    const auto [is_even_chunk, tiles_to_read] =
                        reduce_scatter_common::chunk_ring_parity<tile_granularity>(tiles_read, total_tiles_to_read);

                    if ((is_even_chunk && !even_chunks) || (!is_even_chunk && !odd_chunks) || tiles_to_read == 0) {
                        // Skip this chunk
                        tiles_read += tiles_to_read;
                    } else {
                        const bool reduce_interm =
                            (is_even_chunk && reduce_even_chunks) || (!is_even_chunk && reduce_odd_chunks);

                        if (reduce_interm) {
                            // DETERMINISM: reserve the output slot BEFORE the reduce and pop the input
                            // slots AFTER the pack (the original popped inputs before packing and
                            // reserved the output after). This pins the compute<->reader/writer CB
                            // handshake so the cross-device partials are consumed in a fixed order,
                            // making the ring reduce run-to-run deterministic. Without it the near-tie
                            // Llama-70B greedy logits flip (test_topk/test_seeding nondeterminism); the
                            // reduce is scheduling/accumulation-order sensitive, not a precision issue.
                            // If reduce_output, add 3 tensors. Else add 2 tensors.
                            if (reduce_output) {
                                cb_interm2.wait_front(tile_granularity);
                            }
                            cb_input.wait_front(tile_granularity);
                            cb_interm.wait_front(tile_granularity);
                            cb_compute_output.reserve_back(tile_granularity);

                            tile_regs_acquire();  // acquire DST registers for MATH thread, resets DST to 0
                            if (reduce_output) {
                                copy_tile_init(cb_interm2_id);
                                for (uint32_t tile_id = 0; tile_id < tiles_to_read; ++tile_id) {
                                    copy_tile(cb_interm2_id, tile_id, tile_id);  // load DST
                                }
                                add_tiles_init(cb_interm_id, cb_input_id, true);  // DST = srcA + srcB + DST
                            } else {
                                add_tiles_init(cb_interm_id, cb_input_id, false);  // DST = srcA + srcB
                            }
                            for (uint32_t tile_id = 0; tile_id < tiles_to_read; ++tile_id) {
                                add_tiles(cb_interm_id, cb_input_id, tile_id, tile_id, tile_id);
                            }
                            tile_regs_commit();  // release lock on DST by MATH thread, signal the PACK thread

                            tile_regs_wait();  // acquire lock on DST for PACK thread
                            for (uint32_t tile_id = 0; tile_id < tiles_to_read; ++tile_id) {
                                pack_tile(tile_id, cb_compute_output_id, tile_id);  // pack results from DST registers
                                                                                    // to output circular buffers
                            }
                            tile_regs_release();  // release lock on DST by PACK thread

                            if (reduce_output) {
                                cb_interm2.pop_front(tile_granularity);
                            }
                            cb_input.pop_front(tile_granularity);
                            cb_interm.pop_front(tile_granularity);
                            cb_compute_output.push_back(tile_granularity);
                        }
                        tiles_read += tiles_to_read;

                    }  // if skip or process
                }  // while total_tiles_to_read
            }  // for slice_C
        }  // for num_iters
    }  // for input_tensor_B
}
