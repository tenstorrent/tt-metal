// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel
//
// Writes untilized RM sticks from CB 16 to DRAM.
// After untilize, CB 16 contains Wt tile-pages worth of data per tile-row,
// laid out as 32 rows of stick_size_bytes each (row-major in the tile page).
//
// Compile-time args:
//   [0] stick_size_bytes: W * element_size (bytes per output RM stick)
//   [1+] TensorAccessorArgs for output
//
// Runtime args:
//   [0] dst_addr: output buffer base address
//   [1] num_sticks: total sticks this core writes
//   [2] start_stick_id: first output stick index

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(0);
    constexpr auto output_tensor_args = TensorAccessorArgs<1>();

    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t Wt = stick_size_bytes / (32 * 2);  // bf16: 2 bytes per element

    // CB index
    constexpr uint32_t cb_out = 16;

    // Runtime args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    if (num_sticks == 0) {
        return;
    }

    const auto output_accessor = TensorAccessor(output_tensor_args, dst_addr, stick_size_bytes);

    uint32_t num_tile_rows = num_sticks / TILE_HEIGHT;
    uint32_t stick_id = start_stick_id;

    for (uint32_t tr = 0; tr < num_tile_rows; tr++) {
        // Wait for Wt tile-pages from compute (untilized output)
        cb_wait_front(cb_out, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_out);

        // Write 32 sticks to DRAM
        for (uint32_t s = 0; s < TILE_HEIGHT; s++) {
            uint64_t noc_addr = get_noc_addr(stick_id, output_accessor);
            noc_async_write(l1_read_addr, noc_addr, stick_size_bytes);
            l1_read_addr += stick_size_bytes;
            stick_id++;
        }

        noc_async_write_barrier();

        // Pop Wt pages from CB 16
        cb_pop_front(cb_out, Wt);
    }
}
