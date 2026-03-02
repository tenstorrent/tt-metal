// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm - Writer Kernel
// Runs on RISCV_1 (NCRISC), writes output tiles to DRAM via NOC1.
//
// Compile-time args:
//   [0+] output_accessor_args : TensorAccessorArgs for output tensor
//
// Runtime args:
//   [0] output_addr   : uint32_t - Output tensor base address
//   [1] num_rows      : uint32_t - Number of tile-rows
//   [2] Wt            : uint32_t - Width in tiles
//   [3] start_tile_id : uint32_t - Starting output tile index

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    // ========== Compile-time args ==========
    auto output_accessor_args = TensorAccessorArgs<0, 0>();

    // ========== Runtime args ==========
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(3);

    // ========== CB indices ==========
    constexpr uint32_t cb_out = 16;

    // ========== Setup TensorAccessor ==========
    const uint32_t output_page_size = get_tile_size(cb_out);
    auto output_accessor = TensorAccessor(output_accessor_args, output_addr, output_page_size);

    // ========== Write output tiles row by row ==========
    uint32_t tile_id = start_tile_id;

    for (uint32_t row = 0; row < num_rows; ++row) {
        // Wait for compute to push Wt output tiles
        cb_wait_front(cb_out, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_out);

        // Write Wt tiles for this row
        for (uint32_t w = 0; w < Wt; ++w) {
            uint64_t noc_addr = output_accessor.get_noc_addr(tile_id);
            noc_async_write(l1_read_addr, noc_addr, output_page_size);
            l1_read_addr += output_page_size;
            tile_id++;
        }

        // Wait for all writes to complete, then pop tiles
        noc_async_write_barrier();
        cb_pop_front(cb_out, Wt);
    }
}
