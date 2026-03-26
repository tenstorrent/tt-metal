// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include <cstdint>
#include <utility>

#include "api/debug/dprint.h"

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t ring_size = get_compile_time_arg_val(1);
constexpr uint32_t cb_input_id = get_compile_time_arg_val(2);
constexpr uint32_t cb_intermediate_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(4);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(5);
constexpr uint32_t page_size = get_compile_time_arg_val(6);
constexpr uint32_t input_batch_num_pages = get_compile_time_arg_val(7);
constexpr uint32_t input_channel_num_pages = get_compile_time_arg_val(8);
constexpr uint32_t input_tensor_B = get_compile_time_arg_val(9);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(10);
constexpr uint32_t slice_C = get_compile_time_arg_val(11);
constexpr uint32_t slice_Ht = get_compile_time_arg_val(12);
constexpr uint32_t slice_Wt = get_compile_time_arg_val(13);
constexpr uint32_t fuse_op = get_compile_time_arg_val(14);
constexpr uint32_t dim = get_compile_time_arg_val(15);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    // Load the input tensor spec
    address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t intermediate_tensor_address = get_arg_val<address_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);
    const int32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t ct_idx = 16;
    constexpr auto input_tensor_args = TensorAccessorArgs<ct_idx>();
    constexpr uint32_t ct_offset = input_tensor_args.num_compile_time_args();
    auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_address);

    constexpr auto intermediate_tensor_args = TensorAccessorArgs<ct_idx + ct_offset>();
    auto intermediate_tensor_accessor = TensorAccessor(intermediate_tensor_args, intermediate_tensor_address);

    ReduceScatterOpReceiver matmul_receiver;
    if constexpr (fuse_op) {
        matmul_receiver = ReduceScatterOpReceiver(arg_idx);
    }

    uint32_t chunk_count = 0;
    uint32_t sem_target = 0;

    for (uint32_t b = 0; b < input_tensor_B; b++) {
        if constexpr (fuse_op) {
            matmul_receiver.wait_for_matmul_batch(b);
        }
        int slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;
        uint32_t batch_offset = input_batch_num_pages * b;

        // Loop over the slices, starting from the furthest, and working backwards until we get to ourselves
        // Read our local slice at this slice idx into cb_input_id or cb_output_id
        // If we are not the first slice, then read intermediate into the cb_intermediate_id
        // Then reduce those two CB's, and push that to cb_output_id
        // If slices_forwarded in writer is 7, we don't forward anymore and write it to output_buffer
        // Otherwise, the writer will write cb_output_id to the next chip in the forward direction
        for (uint32_t i = 0; i < ring_size; ++i) {
            const bool full_slice = false;       // TODO ...
            const bool even_chunks = direction;  // TODO ...
            const bool do_reduce = i != 0;
            uint32_t cb_in0 = do_reduce ? cb_input_id : cb_reader_output_id;

            // slice_idx = slice_idx % ring_size
            if (direction) {
                slice_idx = slice_idx < 0 ? slice_idx + ring_size : slice_idx;
            } else {
                slice_idx = slice_idx >= (int)ring_size ? (uint32_t)slice_idx - ring_size : (uint32_t)slice_idx;
            }

            uint32_t input_tile_id_start;
            uint32_t intermediate_tile_id_start;
            if constexpr (dim == 3) {
                input_tile_id_start = slice_idx * slice_Wt + batch_offset;
                intermediate_tile_id_start = slice_idx * slice_Wt;
            } else if constexpr (dim == 2) {
                input_tile_id_start = slice_idx * slice_Ht * slice_Wt + batch_offset;
                intermediate_tile_id_start = slice_idx * slice_Ht * slice_Wt;
            } else if constexpr (dim == 1) {
                input_tile_id_start = slice_idx * slice_C * slice_Ht * slice_Wt + batch_offset;
                intermediate_tile_id_start = slice_idx * slice_C * slice_Ht * slice_Wt;
            } else {
                ASSERT(false);
            }
            chunk_count = 0;
            for (uint32_t c = 0; c < slice_C; ++c) {
                uint32_t input_pages_read_in_row = start_pages_read_in_row;
                uint32_t input_row_offset = start_row_offset;

                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;

                if (!full_slice && !even_chunks) {
                    uint32_t first_even_chunk = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                    for (uint32_t k = 0; k < first_even_chunk; ++k) {
                        input_pages_read_in_row++;
                        if (input_pages_read_in_row == slice_Wt) {
                            input_row_offset += input_tensor_Wt;
                            input_pages_read_in_row -= slice_Wt;
                        }
                    }
                    tiles_read += first_even_chunk;
                }

                /**
                 * Interleave forward and backward ring reads
                 * forward handles even chunks, backward handles odd chunks (1 chunk = tile_granularity tiles)
                 * after ring_size-1 steps, we've transferred all tiles
                 */
                while (tiles_read < tiles_to_read) {
                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;

                    if (do_reduce && (chunk_count % chunks_per_sync == 0)) {
                        noc_semaphore_wait_min(
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), sem_target + 1);
                        sem_target++;
                    }
                    chunk_count++;

                    uint32_t tiles_to_read_in_current_direction = 0;
                    if (full_slice || !even_chunks) {
                        tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read, tile_granularity);
                    } else {
                        tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                    }

                    cb_reserve_back(cb_in0, tile_granularity);
                    uint32_t l1_write_addr = get_write_ptr(cb_in0);
                    uint32_t intermediate_l1_write_addr;
                    if (do_reduce) {
                        cb_reserve_back(cb_intermediate_id, tile_granularity);
                        intermediate_l1_write_addr = get_write_ptr(cb_intermediate_id);
                    }
                    for (uint32_t j = 0; j < tiles_to_read_in_current_direction; ++j) {
                        uint32_t input_tile_id = input_tile_id_start + input_row_offset + input_pages_read_in_row;
                        // DPRINT << "R: dir=" << direction << " i=" << i << " tile=" << input_tile_id << ENDL();
                        uint64_t noc_read_addr = input_tensor_accessor.get_noc_addr(input_tile_id);
                        noc_async_read(noc_read_addr, l1_write_addr, page_size);
                        l1_write_addr += page_size;

                        if (do_reduce) {
                            uint32_t intermediate_tile_id =
                                intermediate_tile_id_start + input_row_offset + input_pages_read_in_row;
                            uint64_t intermediate_noc_read_addr =
                                intermediate_tensor_accessor.get_noc_addr(intermediate_tile_id);
                            noc_async_read(intermediate_noc_read_addr, intermediate_l1_write_addr, page_size);
                            intermediate_l1_write_addr += page_size;
                        }

                        input_pages_read_in_row++;
                        if (input_pages_read_in_row == slice_Wt) {
                            input_row_offset += input_tensor_Wt;
                            input_pages_read_in_row -= slice_Wt;
                        }
                    }
                    tiles_read += tiles_to_read_in_current_direction;
                    noc_async_read_barrier();
                    cb_push_back(cb_in0, tile_granularity);
                    if (do_reduce) {
                        cb_push_back(cb_intermediate_id, tile_granularity);
                    }

                    // Skip the tiles going the other direction
                    tiles_remaining_to_read = tiles_to_read - tiles_read;
                    if (!full_slice && tiles_remaining_to_read > 0) {
                        uint32_t tiles_to_read_in_other_direction = 0;
                        if (!even_chunks) {
                            tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                        } else {
                            tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read, tile_granularity);
                        }

                        for (uint32_t k = 0; k < tiles_to_read_in_other_direction; ++k) {
                            input_pages_read_in_row++;
                            if (input_pages_read_in_row == slice_Wt) {
                                input_row_offset += input_tensor_Wt;
                                input_pages_read_in_row -= slice_Wt;
                            }
                        }
                        tiles_read += tiles_to_read_in_other_direction;
                    }
                }
                input_tile_id_start += input_channel_num_pages;
                intermediate_tile_id_start += input_channel_num_pages;
            }

            // Next slice idx
            slice_idx = direction ? (slice_idx - 1) : (slice_idx + 1);
        }

        // Reset the semaphore before the next batch
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
        sem_target = 0;
    }
}
