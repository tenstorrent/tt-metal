// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Writer Kernel
//
// Streams Wt output tiles from c_16 to DRAM for each tile-row.
//
// Compile-time args (via TensorAccessorArgs):
//   [0+] : TensorAccessorArgs for output tensor
//
// Runtime args:
//   [0] output_addr       : uint32  Output buffer base address
//   [1] num_rows_per_core : uint32  Number of tile-rows for this core
//   [2] Wt               : uint32  Width in tiles
//   [3] tile_offset      : uint32  Starting tile index in the output buffer

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t tile_offset = get_arg_val<uint32_t>(3);

    // Compile-time args: TensorAccessor for output
    constexpr auto output_args = TensorAccessorArgs<0>();

    constexpr uint32_t cb_output = 16;  // c_16

    const uint32_t tile_bytes = get_local_cb_interface(cb_output).fifo_page_size;

    // Create TensorAccessor for output
    const auto output_accessor = TensorAccessor(output_args, output_addr, tile_bytes);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        for (uint32_t col = 0; col < Wt; ++col) {
            uint32_t tile_idx = tile_offset + row * Wt + col;
            cb_wait_front(cb_output, 1);
            uint32_t l1_read_addr = get_read_ptr(cb_output);
            noc_async_write_page(tile_idx, output_accessor, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_output, 1);
        }
    }
}
