// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel
// Reads RM output from CB 16 (tile-sized pages, Wt pages per tile-row)
// and writes 32 full sticks per tile-row back to DRAM.
//
// After pack_untilize, CB 16 contains Wt tile-pages where the full row-major
// data for 32 rows (each of width W=Wt*32 elements) is laid out as:
//   row 0: row 1: ... row 31:  (each row is stick_size bytes, contiguous)
// Total size = 32 * stick_size = Wt * tile_size.
//
// Compile-time args:
//   0: stick_size        - W * 2 bytes
//   1: num_sticks_total  - N*C*H total sticks
//   2+: TensorAccessorArgs(output)
//
// Runtime args:
//   0: output_addr - output tensor buffer address

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t num_sticks_total = get_compile_time_arg_val(1);

    // TensorAccessor for output (starts at compile-time arg index 2)
    constexpr auto output_args = TensorAccessorArgs<2>();

    // ---- Runtime args ----
    uint32_t output_addr = get_arg_val<uint32_t>(0);

    // Construct output TensorAccessor (interleaved, page_size = stick_size)
    const auto output_accessor = TensorAccessor(output_args, output_addr, stick_size);

    // ---- CB indices ----
    constexpr uint32_t cb_output_rm = 16;

    // Compute tile_size and Wt from compile-time info
    // tile_size = 2048 for bfloat16 32x32 tiles
    constexpr uint32_t tile_size = 2048;
    constexpr uint32_t Wt = (stick_size * 32) / tile_size;  // stick_size * 32 rows = Wt * tile_size

    uint32_t num_tile_rows = num_sticks_total / 32;
    uint32_t stick_id = 0;

    for (uint32_t ht = 0; ht < num_tile_rows; ht++) {
        // Wait for Wt tile-pages in CB 16 (produced by untilize)
        cb_wait_front(cb_output_rm, Wt);

        // L1 base address for this tile-row's data
        uint32_t base_l1 = get_read_ptr(cb_output_rm);

        // After pack_untilize, the data is stored as 32 contiguous rows of stick_size each.
        // Row i is at offset i * stick_size from base_l1.
        for (uint32_t row = 0; row < 32; row++) {
            uint32_t l1_read_addr = base_l1 + row * stick_size;
            uint64_t noc_addr = get_noc_addr(stick_id, output_accessor);
            noc_async_write(l1_read_addr, noc_addr, stick_size);
            stick_id++;
        }
        noc_async_write_barrier();

        cb_pop_front(cb_output_rm, Wt);
    }
}
