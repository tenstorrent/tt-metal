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
    constexpr uint32_t cb_out_rm = tt::CBIndex::c_16;  // Output RM sticks

    // Create tensor accessor for output
    const auto accessor = TensorAccessor(tensor_args, dst_addr, output_stick_size);

    // Write output sticks (per tile-row)
    // Each tile-row produces 32 output sticks (one reduced tile)
    constexpr uint32_t TILE_HEIGHT = 32;
    uint32_t stick_id = 0;

    for (uint32_t ht = 0; ht < Ht; ++ht) {
        // Wait for untilized data from compute kernel
        cb_wait_front(cb_out_rm, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_out_rm);

        // Write 32 output sticks
        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint64_t noc_addr = accessor.get_noc_addr(stick_id);
            noc_async_write(l1_read_addr, noc_addr, output_stick_size);
            l1_read_addr += output_stick_size;
            stick_id++;
        }
        noc_async_write_barrier();

        // Release CB space
        cb_pop_front(cb_out_rm, 1);
    }
}
