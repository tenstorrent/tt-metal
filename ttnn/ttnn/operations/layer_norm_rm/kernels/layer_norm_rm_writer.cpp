// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// Compile-time arguments:
// 0: output_stick_size (W * element_size)
// 1: tile_height (32)
// 2: num_tile_rows (total number of tile-rows to process)
// 3: Wt (tiles along W dimension)
// 4+: TensorAccessorArgs for output

constexpr uint32_t output_stick_size = get_compile_time_arg_val(0);
constexpr uint32_t tile_height = get_compile_time_arg_val(1);
constexpr uint32_t num_tile_rows = get_compile_time_arg_val(2);
constexpr uint32_t Wt = get_compile_time_arg_val(3);

constexpr auto output_accessor_args = TensorAccessorArgs<4>();

void kernel_main() {
    // Runtime arguments
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_output_rm = tt::CBIndex::c_16;

    // Create TensorAccessor for output
    const auto output_accessor = TensorAccessor(output_accessor_args, dst_addr, output_stick_size);

    // Main loop: write output sticks per tile-row
    uint32_t stick_id = 0;
    for (uint32_t tr = 0; tr < num_tile_rows; tr++) {
        // Wait for untilized output (Wt tiles worth of RM sticks)
        cb_wait_front(cb_output_rm, Wt);

        uint32_t l1_read_addr = get_read_ptr(cb_output_rm);

        // Write 32 sticks to DRAM
        for (uint32_t k = 0; k < tile_height; k++) {
            uint64_t dst_noc_addr = output_accessor.get_noc_addr(stick_id);
            noc_async_write(l1_read_addr, dst_noc_addr, output_stick_size);
            l1_read_addr += output_stick_size;
            stick_id++;
        }
        noc_async_write_barrier();

        cb_pop_front(cb_output_rm, Wt);
    }
}
