// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include <cstdint>
#include <utility>

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t ring_size = get_compile_time_arg_val(1);
constexpr uint32_t cb_input_id = get_compile_time_arg_val(2);
constexpr uint32_t cb_intermediate_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(4);
constexpr uint32_t page_size = get_compile_time_arg_val(5);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(6);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(7);
constexpr uint32_t slice_Wt = get_compile_time_arg_val(8);

constexpr uint32_t initial_ct_idx = 9;

void kernel_main() {
    uint32_t arg_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t intermediate_tensor_address = get_arg_val<uint32_t>(arg_idx++);
    size_t op_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    const int32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);

    constexpr auto input_tensor_args = TensorAccessorArgs<initial_ct_idx>();
    constexpr uint32_t input_ct_offset = input_tensor_args.num_compile_time_args();
    auto input_tensor_addrgen = TensorAccessor(input_tensor_args, input_tensor_address, page_size);

    constexpr auto intermediate_tensor_args = TensorAccessorArgs<initial_ct_idx + input_ct_offset>();
    auto intermediate_tensor_addrgen = TensorAccessor(intermediate_tensor_args, intermediate_tensor_address, page_size);

    uint32_t semaphore_target_val = 0;
    int slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;
    for (uint32_t i = 0; i < ring_size; ++i) {
        uint32_t actual_slice_idx;
        if (direction) {
            actual_slice_idx = slice_idx < 0 ? slice_idx + ring_size : slice_idx;
        } else {
            actual_slice_idx = slice_idx >= (int)ring_size ? (uint32_t)slice_idx - ring_size : (uint32_t)slice_idx;
        }

        bool do_reduce = i != 0;
        uint32_t cb_in0 = do_reduce ? cb_input_id : cb_reader_output_id;

        uint32_t input_tile_id_start = actual_slice_idx * slice_Wt;
        uint32_t intermediate_tile_id_start = actual_slice_idx * slice_Wt;

        uint32_t input_pages_read_in_row = start_pages_read_in_row;
        uint32_t input_row_offset = start_row_offset;

        uint32_t intermediate_pages_read_in_row = input_pages_read_in_row;
        uint32_t intermediate_row_offset = input_row_offset;

        uint32_t tiles_read = start_tiles_read;
        uint32_t tiles_to_read = start_tiles_to_read;

        if (!direction) {
            for (uint32_t k = 0; k < tile_granularity; ++k) {
                input_pages_read_in_row++;
                if (input_pages_read_in_row == slice_Wt) {
                    input_row_offset += input_tensor_Wt;
                    input_pages_read_in_row -= slice_Wt;
                }
            }
            tiles_read += tile_granularity;

            intermediate_pages_read_in_row = input_pages_read_in_row;
            intermediate_row_offset = input_row_offset;
        }

        while (tiles_read < tiles_to_read) {
            if (do_reduce) {
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(op_semaphore), semaphore_target_val + 1);
                semaphore_target_val++;
            }

            cb_reserve_back(cb_in0, tile_granularity);
            uint32_t input_l1_write_addr = get_write_ptr(cb_in0);
            for (uint32_t j = 0; j < tile_granularity; ++j) {
                uint32_t input_tile_id = input_tile_id_start + input_row_offset + input_pages_read_in_row;
                uint64_t input_noc_read_addr = get_noc_addr(input_tile_id, input_tensor_addrgen);
                noc_async_read(input_noc_read_addr, input_l1_write_addr, page_size);
                input_l1_write_addr += page_size;

                input_pages_read_in_row++;
                if (input_pages_read_in_row == slice_Wt) {
                    input_row_offset += input_tensor_Wt;
                    input_pages_read_in_row -= slice_Wt;
                }
            }

            if (do_reduce) {
                cb_reserve_back(cb_intermediate_id, tile_granularity);
                uint32_t intermediate_l1_write_addr = get_write_ptr(cb_intermediate_id);
                for (uint32_t j = 0; j < tile_granularity; ++j) {
                    uint32_t intermediate_tile_id =
                        intermediate_tile_id_start + intermediate_row_offset + intermediate_pages_read_in_row;
                    uint64_t intermediate_noc_read_addr =
                        get_noc_addr(intermediate_tile_id, intermediate_tensor_addrgen);
                    noc_async_read(intermediate_noc_read_addr, intermediate_l1_write_addr, page_size);
                    intermediate_l1_write_addr += page_size;

                    intermediate_pages_read_in_row++;
                    if (intermediate_pages_read_in_row == slice_Wt) {
                        intermediate_row_offset += input_tensor_Wt;
                        intermediate_pages_read_in_row -= slice_Wt;
                    }
                }

                noc_async_read_barrier();
                cb_push_back(cb_intermediate_id, tile_granularity);
            }

            tiles_read += tile_granularity;
            noc_async_read_barrier();
            cb_push_back(cb_in0, tile_granularity);

            uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
            if (tiles_remaining_to_read > 0) {
                for (uint32_t k = 0; k < tile_granularity; ++k) {
                    input_pages_read_in_row++;
                    if (input_pages_read_in_row == slice_Wt) {
                        input_row_offset += input_tensor_Wt;
                        input_pages_read_in_row -= slice_Wt;
                    }
                }
                tiles_read += tile_granularity;

                intermediate_pages_read_in_row = input_pages_read_in_row;
                intermediate_row_offset = input_row_offset;
            }
        }

        // Next slice idx
        if (direction) {
            slice_idx--;
        } else {
            slice_idx++;
        }
    }

    // reset the semaphore
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(op_semaphore), 0);
}
