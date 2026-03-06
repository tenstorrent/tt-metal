// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_gamma = 1;
constexpr uint32_t cb_beta = 2;
constexpr uint32_t cb_reduce_scaler = 8;
constexpr uint32_t cb_eps = 9;

void kernel_main() {
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);
    constexpr uint32_t input_acc_idx = 4;
    constexpr auto input_args = TensorAccessorArgs<input_acc_idx>();

    uint32_t arg_idx = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t N = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_stick_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t scaler_packed = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eps_packed = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t beta_addr = get_arg_val<uint32_t>(arg_idx++);

    if (N == 0) {
        return;
    }

    const auto input_accessor = TensorAccessor(input_args, src_addr, stick_size);

    constexpr uint32_t sticks_per_tile_row = 32;

    // Generate reduce scaler tile (1/W) - program lifetime constant
    union {
        float f;
        uint32_t u;
    } scaler_conv;
    scaler_conv.u = scaler_packed;
    dataflow_kernel_lib::prepare_reduce_scaler<cb_reduce_scaler>(scaler_conv.f);

    // Generate epsilon tile - program lifetime constant
    union {
        float f;
        uint32_t u;
    } eps_conv;
    eps_conv.u = eps_packed;
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_conv.f);

    // Read gamma tiles if present (Wt tiles, program lifetime)
    if constexpr (has_gamma) {
        constexpr uint32_t tile_size = get_tile_size(cb_gamma);
        cb_reserve_back(cb_gamma, Wt);
        uint32_t gamma_l1_write_addr = get_write_ptr(cb_gamma);
        for (uint32_t t = 0; t < Wt; t++) {
            uint64_t gamma_noc_addr = get_noc_addr(t, gamma_addr);
            noc_async_read(gamma_noc_addr, gamma_l1_write_addr, tile_size);
            gamma_l1_write_addr += tile_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, Wt);
    }

    // Read beta tiles if present (Wt tiles, program lifetime)
    if constexpr (has_beta) {
        constexpr uint32_t tile_size = get_tile_size(cb_beta);
        cb_reserve_back(cb_beta, Wt);
        uint32_t beta_l1_write_addr = get_write_ptr(cb_beta);
        for (uint32_t t = 0; t < Wt; t++) {
            uint64_t beta_noc_addr = get_noc_addr(t, beta_addr);
            noc_async_read(beta_noc_addr, beta_l1_write_addr, tile_size);
            beta_l1_write_addr += tile_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_beta, Wt);
    }

    // Main loop: read N tile-rows of RM sticks into cb_in
    for (uint32_t tr = 0; tr < N; tr++) {
        cb_reserve_back(cb_in, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_in);

        for (uint32_t s = 0; s < sticks_per_tile_row; s++) {
            uint64_t noc_addr = input_accessor.get_noc_addr(start_stick_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            start_stick_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, Wt);
    }
}
