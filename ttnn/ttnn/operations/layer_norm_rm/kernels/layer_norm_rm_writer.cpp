// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Writer Kernel
// Waits for Wt tiles in c_16 per block (untilized RM data).
// Extracts 32 RM sticks and writes them to interleaved RM output via TensorAccessor.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t c_16 = 16;  // Untilized output CB

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);  // W * 2 bytes
    constexpr uint32_t Wt = get_compile_time_arg_val(1);          // tiles per row
    constexpr auto output_tensor_args = TensorAccessorArgs<2>();  // Output TensorAccessor starts at index 2

    // Runtime args
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    // Create TensorAccessor for output
    const auto output_accessor = TensorAccessor(output_tensor_args, output_addr, stick_size);

    // Per-block loop
    for (uint32_t block = 0; block < num_rows_per_core; block++) {
        // Wait for Wt tiles of untilized data in c_16
        cb_wait_front(c_16, Wt);
        uint32_t l1_read_addr = get_read_ptr(c_16);

        // Extract 32 RM sticks from the untilized CB
        for (uint32_t stick = 0; stick < 32; stick++) {
            uint64_t noc_addr = output_accessor.get_noc_addr(start_stick_id);
            noc_async_write(l1_read_addr, noc_addr, stick_size);
            l1_read_addr += stick_size;
            start_stick_id++;
        }
        noc_async_write_barrier();

        // Release the block from c_16
        cb_pop_front(c_16, Wt);
    }
}
