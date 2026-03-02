// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm - Reader Kernel
// Runs on RISCV_0 (BRISC), reads input tiles from DRAM via NOC0.
//
// Compile-time args:
//   [0] gamma_has_value  : uint32_t - 1 if gamma is provided
//   [1] beta_has_value   : uint32_t - 1 if beta is provided
//   [2+] input_accessor_args : TensorAccessorArgs for input
//
// Runtime args:
//   [0] input_addr    : uint32_t - Input tensor base address
//   [1] gamma_addr    : uint32_t - Gamma base address (0 if absent)
//   [2] beta_addr     : uint32_t - Beta base address (0 if absent)
//   [3] num_rows      : uint32_t - Number of tile-rows (N*C*Ht)
//   [4] Wt            : uint32_t - Width in tiles
//   [5] start_tile_id : uint32_t - Starting tile index
//   [6] eps_u32       : uint32_t - Epsilon packed as IEEE 754 float bits

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t gamma_has_value = get_compile_time_arg_val(0);
    constexpr uint32_t beta_has_value = get_compile_time_arg_val(1);

    // TensorAccessor args for input start at CT index 2
    auto input_accessor_args = TensorAccessorArgs<2, 0>();
    // For interleaved DRAM, this consumes 1 CT arg and 0 CRT args

    // ========== Runtime args ==========
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_rows = get_arg_val<uint32_t>(3);
    const uint32_t Wt = get_arg_val<uint32_t>(4);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(5);
    const uint32_t eps_u32 = get_arg_val<uint32_t>(6);

    // ========== CB indices ==========
    constexpr uint32_t cb_input = 0;

    // ========== Setup TensorAccessor ==========
    const uint32_t input_page_size = get_tile_size(cb_input);
    auto input_accessor = TensorAccessor(input_accessor_args, input_addr, input_page_size);

    // ========== Read input tiles row by row ==========
    uint32_t tile_id = start_tile_id;

    for (uint32_t row = 0; row < num_rows; ++row) {
        // Reserve space for Wt tiles in the input CB
        cb_reserve_back(cb_input, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_input);

        // Read Wt tiles for this row
        for (uint32_t w = 0; w < Wt; ++w) {
            uint64_t noc_addr = input_accessor.get_noc_addr(tile_id);
            noc_async_read(noc_addr, l1_write_addr, input_page_size);
            l1_write_addr += input_page_size;
            tile_id++;
        }

        // Wait for all reads to complete, then push tiles
        noc_async_read_barrier();
        cb_push_back(cb_input, Wt);
    }
}
