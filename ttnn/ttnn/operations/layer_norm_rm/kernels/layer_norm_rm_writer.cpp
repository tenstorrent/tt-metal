// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm — Writer Kernel
//
// Extracts RM sticks from tile-sized CB pages (cb_out_rm) and writes them
// to DRAM. Per tile-row block:
//   1. cb_wait_front(cb_out_rm, Wt)
//   2. For each of 32 rows: extract stick and write to DRAM
//   3. noc_async_write_barrier()
//   4. cb_pop_front(cb_out_rm, Wt)
//
// Compile-time args:
//   [0]  output_stick_size  — W * sizeof(bfloat16) bytes
//   [1]  Wt                 — tiles per row
//   [2+] TensorAccessorArgs(output) — interleaved output accessor
//
// Runtime args:
//   [0] dst_addr       — output buffer address
//   [1] start_stick_id — first output RM stick for this core
//   [2] num_blocks     — number of tile-row blocks to write

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr auto output_tensor_args = TensorAccessorArgs<2>();

    // ========== Runtime args ==========
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);

    // ========== Constants ==========
    constexpr uint32_t cb_out_rm = 17;

    // ========== Setup TensorAccessor ==========
    // Output is ROW_MAJOR, each page = 1 stick = output_stick_size bytes
    const auto output_accessor = TensorAccessor(output_tensor_args, dst_addr, output_stick_size);

    // ========== Write RM sticks ==========
    uint32_t stick_id = start_stick_id;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Wait for full tile-row block from compute
        cb_wait_front(cb_out_rm, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_out_rm);

        // Extract and write 32 RM sticks
        // After untilize, the data in cb_out_rm is arranged as:
        // Wt tile-sized pages, each containing 32 rows of 32 elements
        // But the overall layout is 32 sticks of W elements each,
        // contiguous in L1: stick[j] at offset j * output_stick_size
        for (uint32_t j = 0; j < 32; ++j) {
            noc_async_write_page(stick_id, output_accessor, l1_read_addr);
            l1_read_addr += output_stick_size;
            stick_id++;
        }
        noc_async_write_barrier();

        // Release the block
        cb_pop_front(cb_out_rm, Wt);
    }
}
