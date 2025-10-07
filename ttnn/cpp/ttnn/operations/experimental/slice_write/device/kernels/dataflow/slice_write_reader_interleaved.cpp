// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t stick_size_offset = get_arg_val<uint32_t>(2);
    uint32_t num_sticks_per_core = get_arg_val<uint32_t>(3);
    uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(4);
    uint32_t num_read_per_barrier = get_arg_val<uint32_t>(5);
    uint32_t start_id = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t page_offset = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();

    const auto s0 = TensorAccessor(src_args, src_addr, stick_size);

    uint32_t i_stick = start_id;
    uint32_t sticks_read = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
        cb_reserve_back(cb_id_in0, num_read_per_barrier);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

        for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            uint64_t src_noc_addr = get_noc_addr(i_stick, s0);
            noc_async_read(src_noc_addr, l1_write_addr, stick_size);
#ifdef LAST_DIM
            // align data if slicing on last dim
            noc_async_read_barrier();
            tt::data_movement::common::tt_memmove<false, false, false, 0>(
                l1_write_addr + page_offset,  // destination (shifted right)
                l1_write_addr,                // source (original location)
                stick_size                    // total bytes to move
            );
#endif
            l1_write_addr += stick_size_offset;
            i_stick += 1;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, num_read_per_barrier);
    }
}
