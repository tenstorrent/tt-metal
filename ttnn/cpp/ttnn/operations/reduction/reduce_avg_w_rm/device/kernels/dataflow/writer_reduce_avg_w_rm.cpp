// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for reduce_avg_w_rm operation
// Writes row-major output sticks from CB_rm_out to DRAM.
//
// Per Kernel Design Document:
// - For each tile row: wait for 1 tile, write 32 output sticks, barrier, pop 1 tile
// - Output stick width is 32 elements (one tile width after width reduction)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time args (from spec)
    constexpr uint32_t output_stick_size =
        get_compile_time_arg_val(0);                           // Size of one output row in bytes (32 * element_size)
    constexpr auto dst_tensor_args = TensorAccessorArgs<1>();  // TensorAccessor args start at index 1

    // Runtime args (from spec)
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(2);

    // CB index
    constexpr uint32_t cb_rm_out = tt::CBIndex::c_16;

    // Initialize TensorAccessor for stick-based writes
    // Each page is one output stick (output_stick_size bytes = 32 * element_size)
    const auto dst_accessor = TensorAccessor(dst_tensor_args, dst_addr, output_stick_size);

    // =========================================================================
    // Write sticks for each tile row
    // For each tile row, untilize produces 1 tile = 32 sticks of 32 elements each
    // =========================================================================
    constexpr uint32_t TILE_HEIGHT = 32;

    for (uint32_t tile_row = 0; tile_row < num_tile_rows; ++tile_row) {
        // Calculate the base output stick index for this tile row
        uint32_t stick_base = (start_tile_row + tile_row) * TILE_HEIGHT;

        // Wait for 1 tile of untilized data
        cb_wait_front(cb_rm_out, 1);

        // Get the L1 read address
        uint32_t l1_read_addr = get_read_ptr(cb_rm_out);

        // Write 32 output sticks (one tile height)
        for (uint32_t stick = 0; stick < TILE_HEIGHT; ++stick) {
            uint32_t global_stick_id = stick_base + stick;
            uint64_t noc_addr = dst_accessor.get_noc_addr(global_stick_id);
            noc_async_write(l1_read_addr, noc_addr, output_stick_size);
            l1_read_addr += output_stick_size;
        }

        // Wait for all writes to complete
        noc_async_write_barrier();

        // Pop the tile from CB
        cb_pop_front(cb_rm_out, 1);
    }
}
