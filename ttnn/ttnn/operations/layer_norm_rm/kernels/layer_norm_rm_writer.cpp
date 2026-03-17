// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Writer Kernel
// Waits for untilized RM sticks in output CB, writes them to DRAM using TensorAccessor.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(0);   // c_16
    constexpr uint32_t stick_size = get_compile_time_arg_val(1);  // W * 2 bytes
    constexpr uint32_t Wt = get_compile_time_arg_val(2);          // tiles per row

    // TensorAccessor args for output tensor start at index 3
    constexpr auto output_args = TensorAccessorArgs<3>();

    // ========== Runtime args ==========
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(1);
    const uint32_t start_page_id = get_arg_val<uint32_t>(2);

    // ========== Construct TensorAccessor ==========
    auto output_accessor = TensorAccessor(output_args, output_addr, stick_size);

    // ========== Main loop: write RM sticks per tile-row ==========
    uint32_t page_id = start_page_id;

    for (uint32_t tr = 0; tr < num_tile_rows; tr++) {
        // Wait for Wt pages of untilized data from compute
        cb_wait_front(cb_out_rm, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_out_rm);

        // Write 32 RM sticks (one tile-row) to DRAM
        for (uint32_t s = 0; s < 32; s++) {
            uint64_t noc_addr = output_accessor.get_noc_addr(page_id);
            noc_async_write(l1_read_addr, noc_addr, stick_size);
            l1_read_addr += stick_size;
            page_id++;
        }
        noc_async_write_barrier();

        cb_pop_front(cb_out_rm, Wt);
    }
}
