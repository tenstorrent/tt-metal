// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel
// Waits for Wt pages in c_16 (untilized RM sticks), extracts 32 sticks per block,
// writes each stick to DRAM via TensorAccessor.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    // ========== COMPILE-TIME ARGS ==========
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);  // W * 2 bytes
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr auto output_accessor_args = TensorAccessorArgs<2>();

    // ========== RUNTIME ARGS ==========
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    // Early exit for idle cores
    if (num_blocks == 0) {
        return;
    }

    // ========== TENSOR ACCESSOR ==========
    const auto output_accessor = TensorAccessor(output_accessor_args, dst_addr, stick_size);

    // ========== CB CONSTANTS ==========
    constexpr uint32_t cb_output_rm = 16;  // c_16: output RM sticks

    // ========== MAIN LOOP: WRITE OUTPUT STICKS ==========
    uint32_t stick_id = start_stick_id;

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Wait for Wt pages (containing 32 RM sticks from untilize)
        cb_wait_front(cb_output_rm, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_output_rm);

        // Write 32 sticks to DRAM
        for (uint32_t s = 0; s < 32; s++) {
            uint64_t noc_addr = output_accessor.get_noc_addr(stick_id);
            noc_async_write(l1_read_addr, noc_addr, stick_size);
            l1_read_addr += stick_size;
            stick_id++;
        }
        noc_async_write_barrier();

        // Pop Wt pages
        cb_pop_front(cb_output_rm, Wt);
    }
}
