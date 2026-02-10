// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Reader Kernel
// Runs on RISCV_0 (BRISC), reads data from DRAM via NOC0
//
// This kernel:
// 1. Generates reduce scaler tile (1/W) -> c_1
// 2. Generates epsilon scalar tile -> c_7
// 3. Reads gamma sticks and replicates to 32 rows -> c_3
// 4. Reads beta sticks and replicates to 32 rows -> c_4
// 5. For each tile-row: reads 32 input sticks -> c_0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/scalar_helpers.hpp"

void kernel_main() {
    // ========== COMPILE-TIME ARGUMENTS ==========
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t is_float32 = get_compile_time_arg_val(1);
    constexpr auto input_tensor_args = TensorAccessorArgs<2>();
    constexpr auto gamma_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();
    constexpr auto beta_tensor_args = TensorAccessorArgs<gamma_tensor_args.next_compile_time_args_offset()>();

    // ========== RUNTIME ARGUMENTS ==========
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_sticks = get_arg_val<uint32_t>(3);
    const uint32_t Wt = get_arg_val<uint32_t>(4);
    const uint32_t block_width_size = get_arg_val<uint32_t>(5);
    const uint32_t reduce_scaler = get_arg_val<uint32_t>(6);
    const uint32_t eps_scalar = get_arg_val<uint32_t>(7);
    const uint32_t gamma_num_sticks = get_arg_val<uint32_t>(8);
    const uint32_t gamma_stick_size = get_arg_val<uint32_t>(9);

    // ========== CB INDICES ==========
    constexpr uint32_t cb_input_rm = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_1;
    constexpr uint32_t cb_gamma_rm = tt::CBIndex::c_3;
    constexpr uint32_t cb_beta_rm = tt::CBIndex::c_4;
    constexpr uint32_t cb_eps_scalar = tt::CBIndex::c_7;
    constexpr uint32_t tile_height = 32;

    // ========== TENSOR ACCESSORS ==========
    const auto input_accessor = TensorAccessor(input_tensor_args, input_addr, stick_size);
    const auto gamma_accessor = TensorAccessor(gamma_tensor_args, gamma_addr, gamma_stick_size);
    const auto beta_accessor = TensorAccessor(beta_tensor_args, beta_addr, gamma_stick_size);

    // ========== ONE-TIME SETUP ==========

    // 1. Generate reduce scaler tile (1/W) -> c_1
    dataflow_kernel_lib::generate_reduce_scaler(cb_reduce_scaler, reduce_scaler);

    // 2. Generate epsilon scalar tile -> c_7
    if constexpr (is_float32) {
        dataflow_kernel_lib::generate_bcast_scalar(cb_eps_scalar, eps_scalar);
    } else {
        dataflow_kernel_lib::generate_bcast_scalar_bfloat16(cb_eps_scalar, eps_scalar);
    }

    // 3. Read gamma sticks and replicate to 32 rows -> c_3
    // Gamma is shape [W] = single row of W elements.
    // We need to write the same row 32 times into the CB to fill one tile-height block.
    {
        cb_reserve_back(cb_gamma_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_gamma_rm);

        // Read the single gamma row and replicate 32 times
        // First read the gamma row (stick 0) into the first row position
        uint64_t gamma_noc_addr = gamma_accessor.get_noc_addr(0);

        for (uint32_t k = 0; k < tile_height; k++) {
            noc_async_read(gamma_noc_addr, l1_write_addr, gamma_stick_size);
            l1_write_addr += gamma_stick_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma_rm, Wt);
    }

    // 4. Read beta sticks and replicate to 32 rows -> c_4
    {
        cb_reserve_back(cb_beta_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_beta_rm);

        uint64_t beta_noc_addr = beta_accessor.get_noc_addr(0);

        for (uint32_t k = 0; k < tile_height; k++) {
            noc_async_read(beta_noc_addr, l1_write_addr, gamma_stick_size);
            l1_write_addr += gamma_stick_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_beta_rm, Wt);
    }

    // ========== PER TILE-ROW LOOP ==========
    // Process num_sticks / tile_height tile-rows
    uint32_t num_tile_rows = num_sticks / tile_height;
    uint32_t stick_id = 0;

    for (uint32_t i = 0; i < num_tile_rows; i++) {
        // Resolve 32 NoC addresses for sticks in this tile-row
        uint64_t base_src_noc_addr[tile_height];
        for (uint32_t j = 0; j < tile_height; j++) {
            base_src_noc_addr[j] = input_accessor.get_noc_addr(stick_id);
            stick_id++;
        }

        // Read all Wt tiles worth of sticks (one full tile-row width)
        cb_reserve_back(cb_input_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_input_rm);

        for (uint32_t k = 0; k < tile_height; k++) {
            noc_async_read(base_src_noc_addr[k], l1_write_addr, block_width_size);
            l1_write_addr += block_width_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_rm, Wt);
    }
}
