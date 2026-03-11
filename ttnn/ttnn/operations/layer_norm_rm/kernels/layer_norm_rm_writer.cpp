// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel
// Reads untilized RM sticks from c_16 and writes to DRAM via TensorAccessor.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr auto output_accessor_args = TensorAccessorArgs<1>();

    // ========== CB indices ==========
    constexpr uint32_t cb_rm_output = 16;  // Untilized RM output

    // ========== Runtime args ==========
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);
    const uint32_t Wt = get_arg_val<uint32_t>(3);

    // ========== Setup TensorAccessor for output ==========
    const auto output_accessor = TensorAccessor(output_accessor_args, dst_addr, stick_size);

    // ========== Main loop: write 32 RM sticks per block ==========
    uint32_t stick_id = start_stick_id;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Wait for Wt pages (one tile-row of untilized data)
        cb_wait_front(cb_rm_output, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_rm_output);

        // Write 32 sticks to DRAM
        for (uint32_t stick = 0; stick < 32; ++stick) {
            uint64_t noc_addr = output_accessor.get_noc_addr(stick_id);
            noc_async_write(l1_read_addr, noc_addr, stick_size);
            l1_read_addr += stick_size;
            stick_id++;
        }
        noc_async_write_barrier();

        // Release the pages
        cb_pop_front(cb_rm_output, Wt);
    }
}
