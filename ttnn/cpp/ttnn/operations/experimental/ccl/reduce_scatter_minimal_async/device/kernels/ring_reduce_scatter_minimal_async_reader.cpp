// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr BufferType input_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(1));
constexpr BufferType intermediate_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(2));
constexpr uint32_t cb_input_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_intermediate_id = get_compile_time_arg_val(4);
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(5);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(6);
constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(7);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(8);
constexpr uint32_t batch_slice_num_pages = get_compile_time_arg_val(9);
constexpr uint32_t ring_size = get_compile_time_arg_val(10);
constexpr uint32_t num_batches = get_compile_time_arg_val(11);
constexpr uint32_t fuse_op = get_compile_time_arg_val(12);
constexpr bool direction = get_compile_time_arg_val(13);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    // Load the input tensor spec
    address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t intermediate_tensor_address = get_arg_val<address_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t batch_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    uint32_t link = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_links = get_arg_val<uint32_t>(arg_idx++);

    uint32_t slice_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);
    int32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);

    ReduceScatterOpReceiver matmul_receiver;
    if constexpr (fuse_op) {
        matmul_receiver = ReduceScatterOpReceiver(arg_idx);
    }

    constexpr uint32_t batch_num_pages = batch_slice_num_pages * ring_size;

    constexpr bool input_tensor_is_dram = input_buffer_type == tt::tt_metal::BufferType::DRAM;
    auto input_tensor_addrgen = InterleavedAddrGenFast<input_tensor_is_dram>{
        .bank_base_address = input_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_input_id)};
    constexpr bool intermediate_tensor_is_dram = intermediate_buffer_type == tt::tt_metal::BufferType::DRAM;
    auto intermediate_tensor_addrgen = InterleavedAddrGenFast<intermediate_tensor_is_dram>{
        .bank_base_address = intermediate_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_input_id)};

    for (uint32_t b = 0; b < num_batches; b++) {
        if constexpr (fuse_op) {
            matmul_receiver.wait_for_matmul_batch(b);
        }
        int slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;
        uint32_t batch_offset = batch_num_pages * b;

        // Loop over the slices, starting from the furthest, and working backwards until we get to ourselves
        // Read our local slice at this slice idx into cb_input_id or cb_output_id
        // If we are not the first slice, then read intermediate into the cb_intermediate_id
        // Then reduce those two CB's, and push that to cb_output_id
        // If slices_forwarded in writer is 7, we don't forward anymore and write it to output_buffer
        // Otherwise, the writer will write cb_output_id to the next chip in the forward direction
        for (uint32_t i = 0; i < ring_size; ++i) {
            const bool do_reduce = i != 0;
            uint32_t cb_in0 = do_reduce ? cb_input_id : cb_reader_output_id;

            uint32_t actual_slice_idx;
            if constexpr (direction) {
                actual_slice_idx = slice_idx < 0 ? slice_idx + ring_size : slice_idx;
            } else {
                actual_slice_idx = slice_idx >= (int)ring_size ? (uint32_t)slice_idx - ring_size : (uint32_t)slice_idx;
            }

            uint32_t input_tile_id_start = actual_slice_idx * slice_Wt + batch_offset;
            uint32_t intermediate_tile_id_start = actual_slice_idx * slice_Wt;
            uint32_t stride_Wt = input_tensor_Wt;
            uint32_t pages_read_in_row = start_pages_read_in_row;
            uint32_t row_offset = start_row_offset;
            uint32_t intermediate_pages_read_in_row = pages_read_in_row;
            uint32_t intermediate_row_offset = row_offset;
            uint32_t tiles_read = start_tiles_read;
            uint32_t tiles_to_read = start_tiles_to_read;

            if constexpr (!direction) {
                uint32_t backwards_offset = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                for (uint32_t k = 0; k < backwards_offset; ++k) {
                    pages_read_in_row++;
                    if (pages_read_in_row == slice_Wt) {
                        row_offset += stride_Wt;
                        pages_read_in_row = pages_read_in_row - slice_Wt;
                    }
                }

                tiles_read += backwards_offset;
                intermediate_pages_read_in_row = pages_read_in_row;
                intermediate_row_offset = row_offset;
            }

            /**
             * Interleave forward and backward ring reads
             * forward handles even tiles, backward handles odd tiles
             * after ring_size-1 steps, we've transferred all tiles
             */
            if (do_reduce) {
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), i);
                if (i == (ring_size - 1)) {
                    // Reset the semaphore before the next batch
                    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
                }
            }
            while (tiles_read < tiles_to_read) {
                uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;

                uint32_t tiles_to_read_in_current_direction = 0;
                if constexpr (direction) {
                    tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                } else {
                    tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read, tile_granularity);
                }

                cb_reserve_back(cb_in0, tile_granularity);
                uint32_t l1_write_addr = get_write_ptr(cb_in0);
                for (uint32_t j = 0; j < tiles_to_read_in_current_direction; ++j) {
                    uint32_t tile_id = input_tile_id_start + row_offset + pages_read_in_row;
                    noc_async_read_tile(tile_id, input_tensor_addrgen, l1_write_addr);
                    l1_write_addr += input_tensor_page_size;
                    tiles_read++;

                    pages_read_in_row++;
                    if (pages_read_in_row == slice_Wt) {
                        row_offset += stride_Wt;
                        pages_read_in_row = 0;
                    }
                }

                if (do_reduce) {
                    // read the next intermediate slice out of the intermediate buffer, and put it in intermediate CB
                    cb_reserve_back(cb_intermediate_id, tile_granularity);
                    uint32_t intermediate_l1_write_addr = get_write_ptr(cb_intermediate_id);
                    for (uint32_t j = 0; j < tiles_to_read_in_current_direction; ++j) {
                        uint32_t intermediate_tile_id =
                            intermediate_tile_id_start + intermediate_row_offset + intermediate_pages_read_in_row;
                        noc_async_read_tile(
                            intermediate_tile_id, intermediate_tensor_addrgen, intermediate_l1_write_addr);
                        intermediate_l1_write_addr += input_tensor_page_size;

                        intermediate_pages_read_in_row++;
                        if (intermediate_pages_read_in_row == slice_Wt) {
                            intermediate_row_offset += stride_Wt;
                            intermediate_pages_read_in_row = 0;
                        }
                    }

                    noc_async_read_barrier();
                    cb_push_back(cb_intermediate_id, tile_granularity);
                }

                noc_async_read_barrier();
                cb_push_back(cb_in0, tile_granularity);

                // Skip the tiles going the other direction
                tiles_remaining_to_read = tiles_to_read - tiles_read;
                if (tiles_remaining_to_read > 0) {
                    uint32_t tiles_to_read_in_other_direction = 0;
                    if constexpr (!direction) {
                        tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                    } else {
                        tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read, tile_granularity);
                    }

                    for (uint32_t k = 0; k < tiles_to_read_in_other_direction; ++k) {
                        pages_read_in_row++;
                        if (pages_read_in_row == slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = pages_read_in_row - slice_Wt;
                        }
                    }

                    tiles_read += tiles_to_read_in_other_direction;
                    intermediate_pages_read_in_row = pages_read_in_row;
                    intermediate_row_offset = row_offset;
                }
            }

            // Next slice idx
            if constexpr (direction) {
                slice_idx--;
            } else {
                slice_idx++;
            }
        }
    }
}
