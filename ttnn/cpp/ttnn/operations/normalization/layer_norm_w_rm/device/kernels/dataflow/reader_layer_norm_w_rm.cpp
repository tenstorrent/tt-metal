// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"

void kernel_main() {
    // ============================================================
    // Compile-time args
    // ============================================================
    constexpr uint32_t input_stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t packed_scaler_value = get_compile_time_arg_val(1);
    constexpr uint32_t packed_epsilon_value = get_compile_time_arg_val(2);
    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t gamma_stick_size = get_compile_time_arg_val(5);
    constexpr uint32_t beta_stick_size = get_compile_time_arg_val(6);
    constexpr auto input_tensor_args = TensorAccessorArgs<7>();
    constexpr auto gamma_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();
    constexpr auto beta_tensor_args = TensorAccessorArgs<gamma_tensor_args.next_compile_time_args_offset()>();

    // ============================================================
    // Runtime args
    // ============================================================
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);

    // ============================================================
    // CB IDs
    // ============================================================
    constexpr uint32_t cb_in_rm = tt::CBIndex::c_0;      // Input RM sticks
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;     // Scaler tile (1/W)
    constexpr uint32_t cb_epsilon = tt::CBIndex::c_7;    // Epsilon scalar tile
    constexpr uint32_t cb_gamma_rm = tt::CBIndex::c_10;  // Gamma RM sticks
    constexpr uint32_t cb_beta_rm = tt::CBIndex::c_12;   // Beta RM sticks

    // ============================================================
    // Phase 0: Initialize scalers and read gamma/beta (once at program start)
    // ============================================================

    // Generate scaler tile (1/W) for mean and variance calculations
    generate_reduce_scaler(cb_scaler, packed_scaler_value);

    // Generate epsilon scalar tile for numerical stability in rsqrt
    generate_reduce_scaler(cb_epsilon, packed_epsilon_value);

    // Read gamma (once at program start)
    // Gamma tensor has shape [1, ..., 1, W] with 32 rows after padding
    {
        const auto gamma_accessor = TensorAccessor(gamma_tensor_args, gamma_addr, gamma_stick_size);
        constexpr uint32_t TILE_HEIGHT = 32;

        cb_reserve_back(cb_gamma_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_gamma_rm);

        // Read 32 sticks for gamma
        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint64_t noc_addr = gamma_accessor.get_noc_addr(s);
            noc_async_read(noc_addr, l1_write_addr, gamma_stick_size);
            l1_write_addr += gamma_stick_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma_rm, Wt);
    }

    // Read beta (once at program start)
    // Beta tensor has shape [1, ..., 1, W] with 32 rows after padding
    {
        const auto beta_accessor = TensorAccessor(beta_tensor_args, beta_addr, beta_stick_size);
        constexpr uint32_t TILE_HEIGHT = 32;

        cb_reserve_back(cb_beta_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_beta_rm);

        // Read 32 sticks for beta
        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint64_t noc_addr = beta_accessor.get_noc_addr(s);
            noc_async_read(noc_addr, l1_write_addr, beta_stick_size);
            l1_write_addr += beta_stick_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_beta_rm, Wt);
    }

    // ============================================================
    // Phase 1: Read input sticks (per tile-row)
    // ============================================================
    const auto input_accessor = TensorAccessor(input_tensor_args, src_addr, input_stick_size);

    // Each tile-row consists of 32 sticks
    constexpr uint32_t TILE_HEIGHT = 32;
    uint32_t stick_id = 0;

    for (uint32_t ht = 0; ht < Ht; ++ht) {
        // Reserve Wt pages
        cb_reserve_back(cb_in_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_in_rm);

        // Read 32 sticks for this tile-row
        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint64_t noc_addr = input_accessor.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_write_addr, input_stick_size);
            l1_write_addr += input_stick_size;
            stick_id++;
        }
        noc_async_read_barrier();

        // Signal data ready for compute
        cb_push_back(cb_in_rm, Wt);
    }
}
