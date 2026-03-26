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
constexpr uint32_t cb_interm_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_interm2_id = 2;  // TODO hardcoded
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(4);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(5);
constexpr uint32_t page_size = get_compile_time_arg_val(6);
constexpr uint32_t input_batch_num_pages = get_compile_time_arg_val(7);
constexpr uint32_t output_batch_num_pages = get_compile_time_arg_val(8);
constexpr uint32_t input_channel_num_pages = get_compile_time_arg_val(9);
constexpr uint32_t input_tensor_B = get_compile_time_arg_val(10);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(11);
constexpr uint32_t slice_C = get_compile_time_arg_val(12);
constexpr uint32_t slice_Ht = get_compile_time_arg_val(13);
constexpr uint32_t slice_Wt = get_compile_time_arg_val(14);
constexpr uint32_t fuse_op = get_compile_time_arg_val(15);
constexpr uint32_t dim = get_compile_time_arg_val(16);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    // Load the input tensor spec
    address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t interm_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t out2_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);
    const int32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t ct_idx = 17;
    constexpr auto input_tensor_args = TensorAccessorArgs<ct_idx>();
    auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_address);
    constexpr uint32_t ct_idx2 = ct_idx + input_tensor_args.num_compile_time_args();

    constexpr auto interm_tensor_args = TensorAccessorArgs<ct_idx2>();
    auto interm_tensor_accessor = TensorAccessor(interm_tensor_args, interm_tensor_address);
    constexpr uint32_t ct_idx3 = ct_idx2 + interm_tensor_args.num_compile_time_args();

    constexpr auto output_tensor_args = TensorAccessorArgs<ct_idx3>();
    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);
    constexpr uint32_t ct_idx4 = ct_idx3 + output_tensor_args.num_compile_time_args();

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
        // If we are not the first slice, then read interm_tensor into the cb_interm_id
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

            // address incrementer for input_tensor and interm_tensor
            uint32_t input_tile_id_start, interm_tile_id_start;
            if constexpr (dim == 3) {
                input_tile_id_start = slice_idx * slice_Wt + batch_offset;
                interm_tile_id_start = slice_idx * slice_Wt;
            } else if constexpr (dim == 2) {
                input_tile_id_start = slice_idx * slice_Ht * slice_Wt + batch_offset;
                interm_tile_id_start = slice_idx * slice_Ht * slice_Wt;
            } else if constexpr (dim == 1) {
                input_tile_id_start = slice_idx * slice_C * slice_Ht * slice_Wt + batch_offset;
                interm_tile_id_start = slice_idx * slice_C * slice_Ht * slice_Wt;
            } else {
                ASSERT(false);
            }
            uint32_t input_pages_read_in_row = start_pages_read_in_row,
                     interm_pages_read_in_row = start_pages_read_in_row;
            uint32_t input_row_offset = start_row_offset, interm_row_offset = start_row_offset;
            auto get_next_input_tile_id = [&]() -> uint32_t {
                uint32_t tile_id = input_tile_id_start + input_row_offset + input_pages_read_in_row;
                input_pages_read_in_row++;
                if (input_pages_read_in_row == slice_Wt) {
                    input_row_offset += input_tensor_Wt;
                    input_pages_read_in_row -= slice_Wt;
                }
                return tile_id;
            };
            auto get_next_interm_tile_id = [&]() -> uint32_t {
                uint32_t tile_id = interm_tile_id_start + interm_row_offset + interm_pages_read_in_row;
                interm_pages_read_in_row++;
                if (interm_pages_read_in_row == slice_Wt) {
                    interm_row_offset += input_tensor_Wt;
                    interm_pages_read_in_row -= slice_Wt;
                }
                return tile_id;
            };

            // address incrementer for output_tensor
            uint32_t output_tile_id_start = b * output_batch_num_pages;
            uint32_t output_tiles_read = start_tiles_read;
            auto get_next_output_tile_id = [&]() -> uint32_t { return output_tile_id_start + (output_tiles_read++); };

            chunk_count = 0;
            for (uint32_t c = 0; c < slice_C; ++c) {
                // reset addr counters
                input_pages_read_in_row = interm_pages_read_in_row = start_pages_read_in_row;
                input_row_offset = interm_row_offset = start_row_offset;
                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;

                if (!full_slice && !even_chunks) {
                    uint32_t first_even_chunk = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                    tiles_read += first_even_chunk;
                    for (uint32_t k = 0; k < first_even_chunk; ++k) {
                        get_next_input_tile_id();
                        get_next_interm_tile_id();
                    }
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
                    uint32_t interm_l1_write_addr;
                    if (do_reduce) {
                        cb_reserve_back(cb_interm_id, tile_granularity);
                        interm_l1_write_addr = get_write_ptr(cb_interm_id);
                    }
                    for (uint32_t j = 0; j < tiles_to_read_in_current_direction; ++j) {
                        // DPRINT << "R: dir=" << direction << " i=" << i << " tile=" << input_tile_id << ENDL();
                        uint64_t noc_read_addr = input_tensor_accessor.get_noc_addr(get_next_input_tile_id());
                        noc_async_read(noc_read_addr, l1_write_addr, page_size);
                        l1_write_addr += page_size;

                        if (do_reduce) {
                            uint64_t interm_noc_read_addr =
                                interm_tensor_accessor.get_noc_addr(get_next_interm_tile_id());
                            noc_async_read(interm_noc_read_addr, interm_l1_write_addr, page_size);
                            interm_l1_write_addr += page_size;
                        }
                    }
                    tiles_read += tiles_to_read_in_current_direction;
                    noc_async_read_barrier();
                    cb_push_back(cb_in0, tile_granularity);
                    if (do_reduce) {
                        cb_push_back(cb_interm_id, tile_granularity);
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

                        tiles_read += tiles_to_read_in_other_direction;
                        for (uint32_t k = 0; k < tiles_to_read_in_other_direction; ++k) {
                            get_next_input_tile_id();
                            get_next_interm_tile_id();
                        }
                    }
                }

                input_tile_id_start += input_channel_num_pages;
                interm_tile_id_start += input_channel_num_pages;
            }

            // Next slice idx
            slice_idx = direction ? (slice_idx - 1) : (slice_idx + 1);
        }

        // Reset the semaphore before the next batch
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
        sem_target = 0;
    }
}
