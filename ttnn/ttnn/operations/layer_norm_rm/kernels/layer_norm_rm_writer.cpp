// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel
//
// Reads untilized RM sticks from cb_out_rm and writes to DRAM output buffer.
// Each tile-row produces 32 sticks of stick_size bytes.
//
// Compile-time args:
//   [0]  stick_size  - W * element_size bytes
//   [1]  Wt          - Tiles per row
//   [2+] TensorAccessorArgs(output)
//
// Runtime args:
//   [0] dst_addr       - Output buffer base address
//   [1] num_blocks     - Number of tile-rows for this core
//   [2] start_stick_id - First output stick index

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out_rm = 16;

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr auto output_tensor_args = TensorAccessorArgs<2>();

    // Runtime args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    // Create TensorAccessor for output (RM tensor: page = 1 stick = stick_size bytes)
    const auto output_accessor = TensorAccessor(output_tensor_args, dst_addr, stick_size);

    // Early exit for idle cores
    if (num_blocks == 0) {
        return;
    }

    uint32_t stick_id = start_stick_id;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Wait for Wt pages of untilized data
        cb_wait_front(cb_out_rm, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_out_rm);

        // Extract 32 RM sticks and write to DRAM
        for (uint32_t stick = 0; stick < 32; ++stick) {
            uint64_t noc_addr = output_accessor.get_noc_addr(stick_id);
            noc_async_write(l1_read_addr, noc_addr, stick_size);
            l1_read_addr += stick_size;
            stick_id++;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out_rm, Wt);
    }
}
