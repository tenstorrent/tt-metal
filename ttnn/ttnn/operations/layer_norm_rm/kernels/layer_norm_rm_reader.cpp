// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm — Reader Kernel (BRISC / NOC0)
//
// Multi-core: each core reads its assigned slice of input rows.
//
// 1. Prepare reduce scaler (1/W) into cb_scaler
// 2. Prepare epsilon tile into cb_eps
// 3. Read gamma sticks into cb_gamma_rm (if has_gamma) — ROW granularity
// 4. Read beta sticks into cb_beta_rm (if has_beta)   — ROW granularity
// 5. Read input sticks into cb_rm_in                   — TILE granularity

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // ── Compile-time args ──
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);

    // TensorAccessor compile-time args — always declared unconditionally, chained offsets
    constexpr auto input_ta_args = TensorAccessorArgs<4>();
    [[maybe_unused]] constexpr auto gamma_ta_args = TensorAccessorArgs<input_ta_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_ta_args = TensorAccessorArgs<gamma_ta_args.next_compile_time_args_offset()>();

    // ── Runtime args ──
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t scaler_bits = get_arg_val<uint32_t>(3);
    const uint32_t eps_bits = get_arg_val<uint32_t>(4);
    const uint32_t num_rows = get_arg_val<uint32_t>(5);
    const uint32_t start_row = get_arg_val<uint32_t>(6);

    // Reinterpret uint32 bits as float
    union {
        uint32_t u;
        float f;
    } scaler_conv, eps_conv;
    scaler_conv.u = scaler_bits;
    eps_conv.u = eps_bits;
    const float scaler_f = scaler_conv.f;
    const float eps_f = eps_conv.f;

    // ── CB indices ──
    constexpr uint32_t cb_rm_in = 0;
    constexpr uint32_t cb_gamma_rm = 1;
    constexpr uint32_t cb_beta_rm = 2;
    constexpr uint32_t cb_scaler = 8;
    constexpr uint32_t cb_eps = 9;

    // ── Step 1: Prepare reduce scaler (1/W) ──
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_f);

    // ── Step 2: Prepare epsilon tile ──
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_f);

    // ── Step 3: Read gamma sticks (ROW granularity, 1 row) ──
    if constexpr (has_gamma) {
        const auto gamma_accessor = TensorAccessor(gamma_ta_args, gamma_addr);
        dataflow_kernel_lib::read_sticks_for_tilize<cb_gamma_rm, dataflow_kernel_lib::TilizeGranularity::ROW>(
            gamma_accessor, 1, row_bytes);
    }

    // ── Step 4: Read beta sticks (ROW granularity, 1 row) ──
    if constexpr (has_beta) {
        const auto beta_accessor = TensorAccessor(beta_ta_args, beta_addr);
        dataflow_kernel_lib::read_sticks_for_tilize<cb_beta_rm, dataflow_kernel_lib::TilizeGranularity::ROW>(
            beta_accessor, 1, row_bytes);
    }

    // ── Step 5: Read input sticks (TILE granularity, per-core slice) ──
    const auto input_accessor = TensorAccessor(input_ta_args, input_addr);
    dataflow_kernel_lib::read_sticks_for_tilize<cb_rm_in>(input_accessor, num_rows, row_bytes, start_row);
}
