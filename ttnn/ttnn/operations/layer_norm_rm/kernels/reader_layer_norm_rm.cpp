// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for layer_norm_rm.
// Reads RM input sticks from DRAM, generates scaler and epsilon tiles,
// and optionally reads gamma/beta RM sticks.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include <tt-metalium/constants.hpp>

void kernel_main() {
    // ================================================================
    // Compile-time args
    // ================================================================
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);

    // TensorAccessor args for input, gamma, beta (all declared unconditionally)
    constexpr auto input_args = TensorAccessorArgs<4>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    // ================================================================
    // Runtime args
    // ================================================================
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(2);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr = get_arg_val<uint32_t>(4);
    const uint32_t scaler_packed = get_arg_val<uint32_t>(5);
    const uint32_t eps_packed = get_arg_val<uint32_t>(6);

    if (num_tile_rows == 0) {
        return;
    }

    // ================================================================
    // CB IDs
    // ================================================================
    constexpr uint32_t cb_in_rm = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_eps = tt::CBIndex::c_7;
    constexpr uint32_t cb_gamma_rm = tt::CBIndex::c_19;
    constexpr uint32_t cb_beta_rm = tt::CBIndex::c_20;

    constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;

    // ================================================================
    // Startup: Generate scaler tile (1.0f) into cb_scaler
    // ================================================================
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(1.0f);

    // ================================================================
    // Startup: Generate epsilon tile into cb_eps
    // ================================================================
    generate_bcast_unary_scalar(cb_eps, eps_packed);

    // ================================================================
    // Startup: Read gamma RM sticks (optional)
    // ================================================================
    if constexpr (has_gamma) {
        const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, stick_size);
        cb_reserve_back(cb_gamma_rm, Wt);
        uint32_t gamma_write_addr = get_write_ptr(cb_gamma_rm);
        for (uint32_t t = 0; t < Wt; t++) {
            dataflow_kernel_lib::zero_faces<DataFormat::Float16_b, false>(gamma_write_addr + t * 2048);
        }
        uint64_t gamma_noc_addr = gamma_accessor.get_noc_addr(0);
        noc_async_read(gamma_noc_addr, gamma_write_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_gamma_rm, Wt);
    }

    // ================================================================
    // Startup: Read beta RM sticks (optional)
    // ================================================================
    if constexpr (has_beta) {
        const auto beta_accessor = TensorAccessor(beta_args, beta_addr, stick_size);
        cb_reserve_back(cb_beta_rm, Wt);
        uint32_t beta_write_addr = get_write_ptr(cb_beta_rm);
        for (uint32_t t = 0; t < Wt; t++) {
            dataflow_kernel_lib::zero_faces<DataFormat::Float16_b, false>(beta_write_addr + t * 2048);
        }
        uint64_t beta_noc_addr = beta_accessor.get_noc_addr(0);
        noc_async_read(beta_noc_addr, beta_write_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_beta_rm, Wt);
    }

    // ================================================================
    // Per tile-row: read 32 RM sticks into cb_in_rm
    // ================================================================
    const auto input_accessor = TensorAccessor(input_args, src_addr, stick_size);

    for (uint32_t tr = 0; tr < num_tile_rows; tr++) {
        cb_reserve_back(cb_in_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_in_rm);

        uint32_t base_stick = (start_tile_row + tr) * TILE_H;
        for (uint32_t s = 0; s < TILE_H; s++) {
            uint32_t page_id = base_stick + s;
            uint64_t noc_addr = input_accessor.get_noc_addr(page_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in_rm, Wt);
    }
}
