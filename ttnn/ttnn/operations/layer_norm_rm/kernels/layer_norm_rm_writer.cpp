// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel
// Runs on RISCV_1 (NCRISC), writes untilized RM sticks to DRAM via NOC1.
//
// Per block: cb_wait_front(c_17, Wt) -> extract 32 sticks -> write to DRAM -> cb_pop_front(c_17, Wt)

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr auto output_tensor_args = TensorAccessorArgs<2>();

    // Runtime args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(1);
    const uint32_t num_sticks = get_arg_val<uint32_t>(2);

    // CB index
    constexpr uint32_t cb_output_rm = 17;  // c_17

    // Set up TensorAccessor for output
    const auto output_accessor = TensorAccessor(output_tensor_args, dst_addr, stick_size);

    // Main loop: write blocks of 32 sticks
    uint32_t stick_id = start_stick_id;
    for (uint32_t s = 0; s < num_sticks; s += 32) {
        // Wait for Wt tile-pages of untilized data
        cb_wait_front(cb_output_rm, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_output_rm);

        // Write 32 sticks to DRAM
        for (uint32_t i = 0; i < 32; i++) {
            uint64_t noc_addr = output_accessor.get_noc_addr(stick_id);
            noc_async_write(l1_read_addr, noc_addr, stick_size);
            l1_read_addr += stick_size;
            stick_id++;
        }
        noc_async_write_barrier();

        // Release the Wt pages
        cb_pop_front(cb_output_rm, Wt);
    }
}
