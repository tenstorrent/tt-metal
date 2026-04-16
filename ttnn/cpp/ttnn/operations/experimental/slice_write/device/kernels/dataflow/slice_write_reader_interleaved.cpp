// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

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

    // Third argument page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program cache hits.
    const auto s0 = TensorAccessor(src_args, src_addr, stick_size);

    experimental::Noc noc;
    experimental::CB cb_in0(cb_id_in0);

    uint32_t i_stick = start_id;
    uint32_t sticks_read = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
        cb_in0.reserve_back(num_read_per_barrier);
        uint32_t l1_write_addr = cb_in0.get_write_ptr();

        for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            noc.async_read(
                s0, experimental::CoreLocalMem<uint32_t>(l1_write_addr), stick_size, {.page_id = i_stick}, {});
#ifdef LAST_DIM
            // align data if slicing on last dim
            noc.async_read_barrier();
            tt::data_movement::common::tt_memmove<false, false, false, 0>(
                l1_write_addr + page_offset,  // destination (shifted right)
                l1_write_addr,                // source (original location)
                stick_size                    // total bytes to move
            );
#endif
            l1_write_addr += stick_size_offset;
            i_stick += 1;
        }
        noc.async_read_barrier();
        cb_in0.push_back(num_read_per_barrier);
    }
}
