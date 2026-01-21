// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr auto tensor_args = TensorAccessorArgs<2>();

    // Runtime args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);

    // CB IDs
    constexpr uint32_t cb_out_rm = tt::CBIndex::c_16;  // Output RM sticks (width=32)

    // Create tensor accessor for output
    // Output stick size is 32 elements (1 tile width, padded) per stick
    const auto accessor = TensorAccessor(tensor_args, dst_addr, output_stick_size);

    // Write output sticks (per tile-row)
    // Output width is reduced to 1 tile (32 elements padded)
    // Each tile-row produces 32 output sticks
    constexpr uint32_t TILE_HEIGHT = 32;
    uint32_t stick_id = 0;

    for (uint32_t ht = 0; ht < Ht; ++ht) {
        // Wait for 1 tile from untilize (representing 32 sticks of width 32)
        cb_wait_front(cb_out_rm, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_out_rm);

        // Write 32 output sticks for this tile-row
        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint64_t noc_addr = accessor.get_noc_addr(stick_id);
            noc_async_write(l1_read_addr, noc_addr, output_stick_size);
            l1_read_addr += output_stick_size;
            stick_id++;
        }
        noc_async_write_barrier();

        // Pop the consumed tile
        cb_pop_front(cb_out_rm, 1);
    }
}
