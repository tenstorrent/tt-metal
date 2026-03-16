// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Writer Kernel
// Writes RM sticks from untilized output CB to DRAM via TensorAccessor.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);

    // TensorAccessor args for output (starts at CT arg index 2)
    constexpr auto output_args = TensorAccessorArgs<2>();

    // ========== Runtime args ==========
    const uint32_t output_addr = get_arg_val<uint32_t>(0);

    // ========== Constants ==========
    constexpr uint32_t cb_out = 16;

    // Stick size = W * sizeof(bf16) = Wt * 32 * 2 = Wt * 64
    constexpr uint32_t stick_size = Wt * 32 * 2;

    // Create output TensorAccessor with stick_size as page_size (RM output)
    const auto output_accessor = TensorAccessor(output_args, output_addr, stick_size);

    // ========== Main loop: write output RM sticks ==========
    uint32_t stick_id = 0;
    for (uint32_t ht = 0; ht < Ht; ht++) {
        // Wait for Wt tile-sized pages of untilized RM data
        cb_wait_front(cb_out, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_out);

        // Each tile-sized page holds 32 sticks of 32 elements each.
        // But after untilize, the data is laid out as 32 full-width sticks
        // across Wt tile-sized pages. Each stick is stick_size bytes.
        // The total data is 32 * stick_size = 32 * Wt * 64 bytes.
        for (uint32_t row = 0; row < 32; row++) {
            noc_async_write_page(stick_id, output_accessor, l1_read_addr);
            l1_read_addr += stick_size;
            stick_id++;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, Wt);
    }
}
