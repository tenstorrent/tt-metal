// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Stub reader kernel for layernorm_fused_rm
// Reads input RM sticks, gamma, beta and generates scaler/epsilon tiles

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    // TensorAccessor compile-time args
    // Each TensorAccessorArgs takes 2 args for ROW_MAJOR interleaved
    constexpr auto src_args = TensorAccessorArgs<2>();
    constexpr auto gamma_args = TensorAccessorArgs<4>();  // 2 + 2 (src has 2 args)
    constexpr auto beta_args = TensorAccessorArgs<6>();   // 2 + 2 + 2 (src + gamma)

    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(3);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(4);
    const uint32_t scaler_packed = get_arg_val<uint32_t>(5);
    const uint32_t epsilon_packed = get_arg_val<uint32_t>(6);

    // Create TensorAccessors
    const auto s = TensorAccessor(src_args, src_addr, stick_size);
    const auto g = TensorAccessor(gamma_args, gamma_addr, stick_size);
    const auto b = TensorAccessor(beta_args, beta_addr, stick_size);

    // CB indices
    constexpr uint32_t cb_in_rm = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_eps = tt::CBIndex::c_3;
    constexpr uint32_t cb_gamma_rm = tt::CBIndex::c_4;
    constexpr uint32_t cb_beta_rm = tt::CBIndex::c_5;

    // STUB: Read gamma and beta once at start (persistent)
    // Gamma and beta are 1D tensors (one stick each)
    cb_reserve_back(cb_gamma_rm, 1);
    uint32_t gamma_l1_addr = get_write_ptr(cb_gamma_rm);
    noc_async_read(g.get_noc_addr(0), gamma_l1_addr, stick_size);
    noc_async_read_barrier();
    cb_push_back(cb_gamma_rm, 1);

    cb_reserve_back(cb_beta_rm, 1);
    uint32_t beta_l1_addr = get_write_ptr(cb_beta_rm);
    noc_async_read(b.get_noc_addr(0), beta_l1_addr, stick_size);
    noc_async_read_barrier();
    cb_push_back(cb_beta_rm, 1);

    // STUB: Generate scaler tile (1/W) - just push garbage for now
    cb_reserve_back(cb_scaler, 1);
    cb_push_back(cb_scaler, 1);

    // STUB: Generate epsilon tile - just push garbage for now
    cb_reserve_back(cb_eps, 1);
    cb_push_back(cb_eps, 1);

    // STUB: Read input rows and push to cb_in_rm
    // Each tile row is 32 sticks tall, we read them all at once
    // CB page = one stick, so we push 32 pages per tile row
    uint32_t stick_id = start_stick_id;
    for (uint32_t tile_row = 0; tile_row < num_tile_rows; tile_row++) {
        // Read 32 sticks (one tile row height)
        cb_reserve_back(cb_in_rm, 32);
        uint32_t l1_addr = get_write_ptr(cb_in_rm);

        for (uint32_t stick_in_tile_row = 0; stick_in_tile_row < 32; stick_in_tile_row++) {
            noc_async_read(s.get_noc_addr(stick_id), l1_addr, stick_size);
            stick_id++;
            l1_addr += stick_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in_rm, 32);
    }
}
