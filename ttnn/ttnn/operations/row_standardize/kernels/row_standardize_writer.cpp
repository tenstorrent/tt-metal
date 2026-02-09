// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row Standardize - Writer Kernel
// Runs on RISCV_1 (NCRISC), writes RM sticks to DRAM via NOC1
//
// Responsibilities:
// For each block: write 32 RM sticks from cb_rm_out to DRAM
//
// Compile-time args:
//   0: stick_size_bytes - Size of one output RM stick (W * datum_size)
//   1: Wt - Number of tiles per row (for CB wait count)
//   2+: TensorAccessorArgs (dst)
//
// Runtime args:
//   0: dst_addr - Destination buffer base address in DRAM
//   1: num_blocks - Number of tile-row blocks to write
//   2: start_stick_id - First output stick ID for this core (0 for single-core)

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr auto dst_tensor_args = TensorAccessorArgs<2>();

    // ========== Runtime args ==========
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    // ========== CB indices ==========
    constexpr uint32_t cb_rm_out = tt::CBIndex::c_16;

    // ========== Constants ==========
    constexpr uint32_t tile_height = 32;

    // ========== TensorAccessor for output ==========
    const auto s = TensorAccessor(dst_tensor_args, dst_addr, stick_size_bytes);

    // ========== Per-block loop: Write 32 RM sticks per block ==========
    uint32_t stick_id = start_stick_id;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Wait for Wt pages in cb_rm_out (32 sticks worth of data)
        cb_wait_front(cb_rm_out, Wt);

        // Get L1 base read address
        uint32_t l1_read_addr = get_read_ptr(cb_rm_out);

        // Write 32 sticks row-by-row to DRAM
        for (uint32_t j = 0; j < tile_height; ++j) {
            uint64_t dst_noc_addr = get_noc_addr(stick_id, s);
            noc_async_write(l1_read_addr, dst_noc_addr, stick_size_bytes);
            l1_read_addr += stick_size_bytes;
            stick_id++;
        }

        noc_async_write_barrier();
        cb_pop_front(cb_rm_out, Wt);
    }
}
