// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel
//
// Waits for cb_rm_out (Wt tiles = 32 RM sticks per block).
// Extracts each stick and writes to DRAM via TensorAccessor.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_rm_out = 17;

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    // TensorAccessorArgs for output tensor follow at index 2+
    constexpr auto output_accessor_args = TensorAccessorArgs<2>();

    // Runtime args
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_blocks = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    // Output TensorAccessor
    const auto output_accessor = TensorAccessor(output_accessor_args, dst_addr, stick_size);

    uint32_t stick_id = start_stick_id;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Wait for Wt pages of untilized RM data (32 sticks)
        cb_wait_front(cb_rm_out, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_rm_out);

        // Write 32 RM sticks to DRAM
        for (uint32_t stick = 0; stick < 32; ++stick) {
            uint64_t noc_addr = output_accessor.get_noc_addr(stick_id);
            noc_async_write(l1_read_addr, noc_addr, stick_size);
            l1_read_addr += stick_size;
            stick_id++;
        }
        noc_async_write_barrier();

        // Pop Wt pages
        cb_pop_front(cb_rm_out, Wt);
    }
}
