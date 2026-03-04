// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Writer Kernel
// Writes RM output sticks from cb_out (c_17) to DRAM via TensorAccessor.
// Per tile-row: waits for Wt tiles in cb_out, extracts 32 sticks,
// writes each stick via noc_async_write, then pops cb_out.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

constexpr uint32_t cb_out = tt::CBIndex::c_17;
constexpr uint32_t TILE_HEIGHT = 32;

// Compile-time args
constexpr uint32_t stick_size = get_compile_time_arg_val(0);
constexpr auto output_accessor_args = TensorAccessorArgs<1>();

void kernel_main() {
    // Runtime args
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    uint32_t start_stick_id = get_arg_val<uint32_t>(3);

    const auto output_accessor = TensorAccessor(output_accessor_args, dst_addr, stick_size);

    uint32_t num_tile_rows = num_sticks / TILE_HEIGHT;
    uint32_t stick_id = start_stick_id;

    for (uint32_t tile_row = 0; tile_row < num_tile_rows; ++tile_row) {
        // Wait for Wt tile-sized pages from compute
        cb_wait_front(cb_out, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_out);

        // Write 32 sticks to DRAM
        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint64_t noc_addr = output_accessor.get_noc_addr(stick_id);
            noc_async_write(l1_read_addr, noc_addr, stick_size);
            l1_read_addr += stick_size;
            stick_id++;
        }
        noc_async_write_barrier();

        // Release the pages
        cb_pop_front(cb_out, Wt);
    }
}
