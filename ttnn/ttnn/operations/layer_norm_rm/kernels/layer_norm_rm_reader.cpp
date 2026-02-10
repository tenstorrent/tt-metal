// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/scalar_helpers.hpp"

// Compile-time arguments:
// 0: stick_size (W * element_size)
// 1: gamma_beta_stick_size (same as stick_size)
// 2+: TensorAccessorArgs for input
// N+: TensorAccessorArgs for gamma
// M+: TensorAccessorArgs for beta

constexpr uint32_t stick_size = get_compile_time_arg_val(0);
constexpr uint32_t gamma_beta_stick_size = get_compile_time_arg_val(1);

constexpr auto input_accessor_args = TensorAccessorArgs<2>();
constexpr auto gamma_accessor_args = TensorAccessorArgs<input_accessor_args.next_compile_time_args_offset()>();
constexpr auto beta_accessor_args = TensorAccessorArgs<gamma_accessor_args.next_compile_time_args_offset()>();

void kernel_main() {
    // Runtime arguments
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_sticks = get_arg_val<uint32_t>(3);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(4);
    const uint32_t Wt = get_arg_val<uint32_t>(5);
    const uint32_t reduce_scaler = get_arg_val<uint32_t>(6);
    const uint32_t eps_scalar = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_input_rm = tt::CBIndex::c_0;
    constexpr uint32_t cb_gamma_rm = tt::CBIndex::c_2;
    constexpr uint32_t cb_beta_rm = tt::CBIndex::c_4;
    constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_6;
    constexpr uint32_t cb_eps_scalar = tt::CBIndex::c_7;

    constexpr uint32_t tile_height = 32;

    // Create TensorAccessors
    const auto input_accessor = TensorAccessor(input_accessor_args, src_addr, stick_size);

    // Phase 0: One-time setup

    // Generate reduce scaler tile (1/W) in c_6
    dataflow_kernel_lib::generate_reduce_scaler(cb_reduce_scaler, reduce_scaler);

    // Generate epsilon scalar tile in c_7 (matches input dtype format)
    // eps_scalar is packed as (bf16 << 16 | bf16) for bfloat16 input
    dataflow_kernel_lib::generate_bcast_scalar_bfloat16(cb_eps_scalar, eps_scalar);

    // Read gamma if provided
    if (gamma_addr != 0) {
        const auto gamma_accessor = TensorAccessor(gamma_accessor_args, gamma_addr, gamma_beta_stick_size);

        cb_reserve_back(cb_gamma_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_gamma_rm);

        // Read the single gamma row from DRAM
        uint64_t gamma_noc_addr = get_noc_addr(0, gamma_accessor);
        noc_async_read(gamma_noc_addr, l1_write_addr, gamma_beta_stick_size);
        noc_async_read_barrier();

        // Replicate the gamma row 31 more times to fill a tile-row (32 sticks)
        // First stick is already at l1_write_addr, copy it to the next 31 positions
        uint32_t src_addr_l1 = l1_write_addr;
        for (uint32_t k = 1; k < tile_height; k++) {
            uint32_t dst_addr_l1 = l1_write_addr + k * gamma_beta_stick_size;
            // Use NOC to copy within L1 (read from local L1)
            uint64_t src_noc = get_noc_addr(src_addr_l1);
            noc_async_read(src_noc, dst_addr_l1, gamma_beta_stick_size);
        }
        noc_async_read_barrier();

        cb_push_back(cb_gamma_rm, Wt);
    }

    // Read beta if provided
    if (beta_addr != 0) {
        const auto beta_accessor = TensorAccessor(beta_accessor_args, beta_addr, gamma_beta_stick_size);

        cb_reserve_back(cb_beta_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_beta_rm);

        // Read the single beta row from DRAM
        uint64_t beta_noc_addr = get_noc_addr(0, beta_accessor);
        noc_async_read(beta_noc_addr, l1_write_addr, gamma_beta_stick_size);
        noc_async_read_barrier();

        // Replicate the beta row 31 more times
        uint32_t src_addr_l1 = l1_write_addr;
        for (uint32_t k = 1; k < tile_height; k++) {
            uint32_t dst_addr_l1 = l1_write_addr + k * gamma_beta_stick_size;
            uint64_t src_noc = get_noc_addr(src_addr_l1);
            noc_async_read(src_noc, dst_addr_l1, gamma_beta_stick_size);
        }
        noc_async_read_barrier();

        cb_push_back(cb_beta_rm, Wt);
    }

    // Phase 1: Main loop - read input sticks per tile-row
    uint32_t stick_id = 0;
    for (uint32_t tr = 0; tr < num_tile_rows; tr++) {
        // Pre-compute NOC addresses for all 32 sticks in this tile-row
        uint64_t base_src_noc_addr[tile_height];
        for (uint32_t j = 0; j < tile_height; j++) {
            base_src_noc_addr[j] = get_noc_addr(stick_id, input_accessor);
            stick_id++;
        }

        // Read all 32 sticks into c_0 (Wt tiles worth of RM data)
        cb_reserve_back(cb_input_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_input_rm);

        for (uint32_t k = 0; k < tile_height; k++) {
            noc_async_read(base_src_noc_addr[k], l1_write_addr, stick_size);
            l1_write_addr += stick_size;
        }
        noc_async_read_barrier();

        cb_push_back(cb_input_rm, Wt);
    }
}
