// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Writer Kernel
// Drains untilized RM sticks from CB c_17 and writes them to DRAM via
// TensorAccessor.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);          // CB_OUTPUT_RM = 17
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(1);  // W * 2 bytes
    constexpr auto output_accessor_args = TensorAccessorArgs<2>();

    // Runtime args
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(1);
    const uint32_t num_sticks = get_arg_val<uint32_t>(2);

    if (num_sticks == 0) {
        return;
    }

    // TensorAccessor for output
    const auto output_accessor = TensorAccessor(output_accessor_args, output_addr, output_stick_size);

    // Derive Wt from output_stick_size
    constexpr uint32_t Wt = output_stick_size / (32 * 2);

    // Number of tile-row blocks
    const uint32_t nblocks = num_sticks / 32;

    for (uint32_t block = 0; block < nblocks; block++) {
        // Wait for Wt pages of untilized RM data
        cb_wait_front(cb_id_out, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);

        // Write 32 sticks to DRAM
        for (uint32_t stick = 0; stick < 32; stick++) {
            uint32_t stick_id = start_stick_id + block * 32 + stick;
            uint64_t noc_addr = output_accessor.get_noc_addr(stick_id);
            noc_async_write(l1_read_addr, noc_addr, output_stick_size);
            l1_read_addr += output_stick_size;
        }
        noc_async_write_barrier();

        cb_pop_front(cb_id_out, Wt);
    }
}
