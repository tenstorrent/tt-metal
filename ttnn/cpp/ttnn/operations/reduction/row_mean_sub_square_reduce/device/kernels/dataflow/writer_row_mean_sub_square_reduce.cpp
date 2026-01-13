// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

/**
 * Writer kernel for row_mean_sub_square_reduce operation
 *
 * Per the design document:
 * - Wait for 1 tile (untilized variance) from CB c_16
 * - Write 32 output sticks (one per tile row) to DRAM
 * - Output stick size = TILE_WIDTH * element_size = 32 * 2 = 64 bytes (bfloat16)
 *
 * CB Flow:
 * - CB c_16 (rm_out): Wait for 1 tile, write 32 sticks, pop 1 tile per tile-row
 */
void kernel_main() {
    // Compile-time args
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(0);  // 32 * element_size = 64 bytes
    constexpr auto dst_tensor_args = TensorAccessorArgs<1>();            // TensorAccessor args start at index 1

    // Runtime args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(1);   // Tile-rows to write
    const uint32_t start_tile_row = get_arg_val<uint32_t>(2);  // First tile-row for this core

    constexpr uint32_t cb_rm_out = tt::CBIndex::c_16;
    constexpr uint32_t TILE_HEIGHT = 32;

    // Setup TensorAccessor (page_size = output_stick_size for row-major output)
    const auto d = TensorAccessor(dst_tensor_args, dst_addr, output_stick_size);

    // Write each output tile-row
    for (uint32_t tile_row = 0; tile_row < num_tile_rows; ++tile_row) {
        // Wait for 1 tile (contains 32 untilized sticks)
        cb_wait_front(cb_rm_out, 1);

        // Get read pointer to CB
        uint32_t l1_read_addr = get_read_ptr(cb_rm_out);

        // Calculate starting output stick ID for this tile-row
        uint32_t base_stick_id = (start_tile_row + tile_row) * TILE_HEIGHT;

        // Write 32 sticks to DRAM
        for (uint32_t row = 0; row < TILE_HEIGHT; ++row) {
            uint32_t stick_id = base_stick_id + row;
            uint64_t noc_addr = d.get_noc_addr(stick_id);
            noc_async_write(l1_read_addr, noc_addr, output_stick_size);
            l1_read_addr += output_stick_size;
        }

        // Barrier after all 32 sticks are written
        noc_async_write_barrier();

        // Pop the tile
        cb_pop_front(cb_rm_out, 1);
    }
}
