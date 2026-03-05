// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel
// Reads row-major input sticks from DRAM into CB 0 (batching 32 sticks per Wt pages).
// Prepares reduce scaler (1/W) in CB 8 and epsilon scaler in CB 9.
// Reads gamma (CB 1) and beta (CB 2) as single sticks for asymmetric tilize + ROW broadcast.
//
// Compile-time args:
//   0: stick_size        - W * 2 bytes
//   1: num_sticks_total  - N*C*H total input sticks
//   2: num_gamma_sticks  - 1 (single stick for asymmetric tilize)
//   3: Wt               - W / 32
//   4: epsilon_bits      - epsilon as uint32 (bit-cast from float)
//   5+: TensorAccessorArgs(input)  - compile-time accessor args for input
//   N+: TensorAccessorArgs(gamma)  - compile-time accessor args for gamma
//   M+: TensorAccessorArgs(beta)   - compile-time accessor args for beta
//
// Runtime args:
//   0: input_addr  - input tensor buffer address
//   1: gamma_addr  - gamma tensor buffer address
//   2: beta_addr   - beta tensor buffer address

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t num_sticks_total = get_compile_time_arg_val(1);
    constexpr uint32_t num_gamma_sticks = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t epsilon_bits = get_compile_time_arg_val(4);

    // TensorAccessors for input, gamma, beta
    // Input accessor starts at compile-time arg index 5
    constexpr auto input_args = TensorAccessorArgs<5>();
    constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    // ---- Runtime args ----
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    uint32_t beta_addr = get_arg_val<uint32_t>(2);

    // Construct TensorAccessors (interleaved, page_size = stick_size)
    const auto input_accessor = TensorAccessor(input_args, input_addr, stick_size);
    const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, stick_size);
    const auto beta_accessor = TensorAccessor(beta_args, beta_addr, stick_size);

    // ---- CB indices ----
    constexpr uint32_t cb_input_rm = 0;
    constexpr uint32_t cb_gamma_rm = 1;
    constexpr uint32_t cb_beta_rm = 2;
    constexpr uint32_t cb_reduce_scaler = 8;
    constexpr uint32_t cb_eps_scaler = 9;

    // ---- Step 1: Prepare reduce scaler (1/W) in CB 8 ----
    // W = Wt * 32
    constexpr float inv_W = 1.0f / (float)(Wt * 32);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_reduce_scaler>(inv_W);

    // ---- Step 2: Prepare epsilon scaler in CB 9 ----
    uint32_t eps_tmp = epsilon_bits;
    float epsilon_f;
    __builtin_memcpy(&epsilon_f, &eps_tmp, sizeof(float));
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps_scaler>(epsilon_f);

    // ---- Step 3: Read gamma (1 stick) into CB 1 ----
    // Gamma shape is [1,1,1,W] = 1 stick. Asymmetric tilize (1 stick) + ROW broadcast.
    {
        cb_reserve_back(cb_gamma_rm, 1);
        uint32_t l1_addr = get_write_ptr(cb_gamma_rm);
        uint64_t noc_addr = get_noc_addr(0, gamma_accessor);
        noc_async_read(noc_addr, l1_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_gamma_rm, 1);
    }

    // ---- Step 4: Read beta (1 stick) into CB 2 ----
    // Beta shape is [1,1,1,W] = 1 stick. Asymmetric tilize (1 stick) + ROW broadcast.
    {
        cb_reserve_back(cb_beta_rm, 1);
        uint32_t l1_addr = get_write_ptr(cb_beta_rm);
        uint64_t noc_addr = get_noc_addr(0, beta_accessor);
        noc_async_read(noc_addr, l1_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_beta_rm, 1);
    }

    // ---- Step 5: Stream input sticks into CB 0 ----
    // Batch 32 sticks per tile-row (one tile-row = 32 sticks of W elements each).
    uint32_t stick_id = 0;
    uint32_t num_tile_rows = num_sticks_total / 32;
    for (uint32_t ht = 0; ht < num_tile_rows; ht++) {
        cb_reserve_back(cb_input_rm, 32);
        uint32_t l1_write_addr = get_write_ptr(cb_input_rm);

        for (uint32_t s = 0; s < 32; s++) {
            uint64_t noc_addr = get_noc_addr(stick_id, input_accessor);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_rm, 32);
    }
}
