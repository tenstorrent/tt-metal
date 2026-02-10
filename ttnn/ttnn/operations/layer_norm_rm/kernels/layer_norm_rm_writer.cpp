// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Writer Kernel
// Runs on RISCV_1 (NCRISC), writes data to DRAM via NOC1
//
// This kernel:
// For each tile-row: writes 32 output sticks from c_16 to DRAM

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ========== RUNTIME ARGUMENTS ==========
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);

    // ========== COMPILE-TIME ARGUMENTS ==========
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t tile_height = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks_across_height = get_compile_time_arg_val(3);
    constexpr uint32_t num_blocks_per_row = get_compile_time_arg_val(4);
    constexpr uint32_t num_tiles_per_block = get_compile_time_arg_val(5);
    constexpr uint32_t block_width_bytes = get_compile_time_arg_val(6);
    constexpr auto dst_args = TensorAccessorArgs<7>();

    // ========== TENSOR ACCESSOR ==========
    const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);

    // ========== PER TILE-ROW LOOP ==========
    uint64_t base_dst_noc_addr[tile_height];

    for (uint32_t i = 0; i < num_blocks_across_height; ++i) {
        // Resolve 32 NoC addresses for output sticks in this tile-row
        for (uint32_t k = 0; k < tile_height; ++k) {
            uint32_t stick_id = i * tile_height + k;
            base_dst_noc_addr[k] = s.get_noc_addr(stick_id);
        }

        // WSmall pattern: only 1 block per row
        for (uint32_t j = 0; j < num_blocks_per_row; ++j) {
            cb_wait_front(cb_id_out, num_tiles_per_block);

            uint32_t l1_read_addr = get_read_ptr(cb_id_out);
            for (uint32_t l = 0; l < tile_height; ++l) {
                noc_async_write(l1_read_addr, base_dst_noc_addr[l], block_width_bytes);
                l1_read_addr += block_width_bytes;
                base_dst_noc_addr[l] += block_width_bytes;
            }

            noc_async_write_barrier();
            cb_pop_front(cb_id_out, num_tiles_per_block);
        }
    }
}
