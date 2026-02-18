// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row Centralize - Writer Kernel
//
// Runs on RISCV_1 (NCRISC), writes RM sticks to DRAM via NOC1.
//
// Per tile-row (num_tile_rows iterations):
//   1. Wait for Wt tiles in CB c_16 (one tile-row of RM output)
//   2. For each of 32 rows within the tile-row:
//      - Compute L1 read address within CB c_16
//      - Compute output page ID (stick ID) = tile_row_idx * TILE_H + row_in_tile_row
//      - Write output_stick_size bytes to DRAM via noc_async_write with TensorAccessor
//   3. Barrier after all 32 rows written
//   4. Pop Wt tiles from CB c_16
//
// Compile-time args:
//   [0] cb_rm_out         - CB c_16 ID
//   [1] output_stick_size - W * 2 (bytes per output RM stick)
//   [2] tile_height       - 32 (TILE_H)
//   [3] Wt               - tiles per tile-row
//   [4+] TensorAccessorArgs(dst)
//
// Runtime args:
//   [0] dst_addr          - Output buffer DRAM base address
//   [1] num_tile_rows     - Total tile-rows to write (Ht_total)
//   [2] start_tile_row    - First tile-row index (0 for single-core)

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t cb_rm_out = get_compile_time_arg_val(0);
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t tile_height = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr auto dst_tensor_args = TensorAccessorArgs<4>();

    // ========== Runtime args ==========
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(2);

    // ========== TensorAccessor for output ==========
    const auto dst_accessor = TensorAccessor(dst_tensor_args, dst_addr, output_stick_size);

    // ========== Per tile-row: write 32 RM sticks from cb_rm_out to DRAM ==========
    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        // Wait for Wt tiles (one tile-row of untilized output)
        cb_wait_front(cb_rm_out, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_rm_out);

        // Compute the starting stick ID for this tile-row
        uint32_t stick_id = (start_tile_row + tr) * tile_height;

        // Write 32 sticks to DRAM
        for (uint32_t s = 0; s < tile_height; ++s) {
            uint64_t noc_addr = dst_accessor.get_noc_addr(stick_id);
            noc_async_write(l1_read_addr, noc_addr, output_stick_size);
            l1_read_addr += output_stick_size;
            stick_id++;
        }

        noc_async_write_barrier();
        cb_pop_front(cb_rm_out, Wt);
    }
}
