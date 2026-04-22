// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "experimental/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t num_pages_in_row = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t size_of_valid_data_in_last_page_in_row = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t padded_stick_size = get_arg_val<uint32_t>(1);
    const uint32_t unpadded_stick_size = get_arg_val<uint32_t>(2);
    const uint32_t stick_size_offset = get_arg_val<uint32_t>(3);
    const uint32_t num_dims = get_arg_val<uint32_t>(4);
    const uint32_t misalignment = get_arg_val<uint32_t>(5);
    const uint32_t start_id = get_arg_val<uint32_t>(6);
    const uint32_t num_sticks_per_core = get_arg_val<uint32_t>(7);
    const uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(8);
    const uint32_t num_read_per_barrier = get_arg_val<uint32_t>(9);

    tt_l1_ptr uint32_t* num_unpadded_sticks = (tt_l1_ptr uint32_t*)(get_arg_addr(10));
    volatile tt_l1_ptr uint32_t* num_padded_sticks = num_unpadded_sticks + num_dims;
    volatile tt_l1_ptr uint32_t* id_per_dim = num_padded_sticks + num_dims;

    uint32_t read_size = unpadded_stick_size + misalignment;

    const auto s0 = TensorAccessor(src_args, src_addr);

    constexpr uint32_t cb_id_in0 = 0;

    // Create experimental CircularBuffer for Device 2.0 API
    experimental::CircularBuffer cb_in0(cb_id_in0);

    uint32_t src_stick_id = start_id;
    uint32_t sticks_read = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
        cb_in0.reserve_back(num_read_per_barrier);
        uint32_t src_buffer_l1_addr = cb_in0.get_write_ptr();

        for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            if (num_pages_in_row == 1) {
                uint64_t src_noc_addr = s0.get_noc_addr(src_stick_id);
                noc_async_read(src_noc_addr, src_buffer_l1_addr, read_size);
                if (misalignment != 0) {
                    noc_async_read_barrier();
                    tt::data_movement::common::tt_memmove<false, false, false, 0>(
                        src_buffer_l1_addr, src_buffer_l1_addr + misalignment, unpadded_stick_size);
                }
                src_buffer_l1_addr += stick_size_offset;
                src_stick_id += 1;
            } else {
                uint32_t row_l1 = src_buffer_l1_addr;
                for (uint32_t p = 0; p < num_pages_in_row - 1; p++) {
                    uint64_t src_noc_addr = s0.get_noc_addr(src_stick_id);
                    noc_async_read(src_noc_addr, row_l1, page_size);
                    noc_async_read_barrier();
                    row_l1 += page_size;
                    src_stick_id += 1;
                }
                uint64_t src_noc_addr = s0.get_noc_addr(src_stick_id);
                noc_async_read(src_noc_addr, row_l1, size_of_valid_data_in_last_page_in_row);
                noc_async_read_barrier();
                src_stick_id += 1;
                if (misalignment != 0) {
                    tt::data_movement::common::tt_memmove<false, false, false, 0>(
                        src_buffer_l1_addr, src_buffer_l1_addr + misalignment, unpadded_stick_size);
                }
                src_buffer_l1_addr += stick_size_offset;
            }
            for (uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == num_unpadded_sticks[j]) {
                    id_per_dim[j] = 0;
                    src_stick_id += num_padded_sticks[j] * num_pages_in_row;
                } else {
                    break;
                }
            }
        }
        noc_async_read_barrier();
        cb_in0.push_back(num_read_per_barrier);
    }
}
