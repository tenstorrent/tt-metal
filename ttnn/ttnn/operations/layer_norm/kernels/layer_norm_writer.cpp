// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm - Writer Kernel
// Runs on NCRISC (RISCV_1), writes data from L1 circular buffers to DRAM via NOC1.
//
// Responsibilities:
//   Per tile-row (Ht iterations): drain Wt output tiles from cb_output to DRAM

#include "api/dataflow/dataflow_api.h"

// Compile-time args
constexpr uint32_t Ht = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr auto output_tensor_args = TensorAccessorArgs<2>();

// CB index
constexpr uint32_t cb_output = 16;

void kernel_main() {
    // Runtime args
    const uint32_t output_addr = get_arg_val<uint32_t>(0);

    const uint32_t tile_bytes = get_tile_size(cb_output);

    // Build TensorAccessor for output writes
    const auto output_accessor = TensorAccessor(output_tensor_args, output_addr, tile_bytes);

    // Main loop: for each tile-row, write Wt output tiles
    uint32_t tile_id = 0;
    for (uint32_t ht = 0; ht < Ht; ++ht) {
        cb_wait_front(cb_output, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_output);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            noc_async_write_page(tile_id, output_accessor, l1_read_addr);
            l1_read_addr += tile_bytes;
            tile_id++;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_output, Wt);
    }
}
