// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for reduce_avg_w_rm operation
// Writes row-major output sticks (width 32) from CB to DRAM

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(1);
    constexpr auto dst_tensor_args = TensorAccessorArgs<2>();  // TensorAccessor args start at index 2

    // Runtime args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    // Setup TensorAccessor for output
    const auto d = TensorAccessor(dst_tensor_args, dst_addr, output_stick_size);

    // Write output sticks - TILE_HEIGHT (32) sticks per block
    // Each block corresponds to one reduced tile that was untilized
    constexpr uint32_t TILE_HEIGHT = 32;
    uint32_t num_blocks = num_sticks / TILE_HEIGHT;
    uint32_t stick_id = start_stick_id;

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Wait for one page of untilized output (32 sticks of width 32)
        cb_wait_front(cb_id_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);

        // Write 32 sticks (one tile height worth) to DRAM
        for (uint32_t h = 0; h < TILE_HEIGHT; h++) {
            uint64_t noc_addr = d.get_noc_addr(stick_id);
            noc_async_write(l1_read_addr, noc_addr, output_stick_size);
            l1_read_addr += output_stick_size;
            stick_id++;
        }

        // Wait for all writes to complete
        noc_async_write_barrier();

        // Release the page
        cb_pop_front(cb_id_out, 1);
    }
}
