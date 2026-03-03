// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel
// Runs on RISCV_0 (BRISC), reads RM sticks from DRAM via NOC0.
//
// Compile-time args:
//   [0]      stick_size           - bytes per RM input stick (W * sizeof(bfloat16))
//   [1..]    TensorAccessorArgs(input)
//   [N+0]    gamma_stick_size     - bytes per gamma stick
//   [N+1..]  TensorAccessorArgs(gamma)
//   [M+0]    beta_stick_size      - bytes per beta stick
//   [M+1..]  TensorAccessorArgs(beta)
//
// Runtime args:
//   [0] src_addr        - input buffer DRAM address
//   [1] gamma_addr      - gamma buffer DRAM address
//   [2] beta_addr       - beta buffer DRAM address
//   [3] num_rows        - total tile-rows to process
//   [4] Wt              - tiles per row (W / 32)
//   [5] start_stick_id  - first stick id for this core
//   [6] scaler_value    - 1/W as float bits (reinterpreted)
//   [7] eps_value       - epsilon as float bits (reinterpreted)
//
// CB usage:
//   c_0  (cb_input_rm)  - push Wt pages per tile-row (32 RM sticks per row)
//   c_1  (cb_gamma_rm)  - push Wt pages once at start (32 gamma sticks)
//   c_2  (cb_beta_rm)   - push Wt pages once at start (32 beta sticks)
//   c_8  (cb_scaler)    - push 1 tile (1/W scaler, never popped)
//   c_9  (cb_eps)       - push 1 tile (epsilon scaler, never popped)

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t TILE_HEIGHT = 32;

// CB indices
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_gamma_rm = 1;
constexpr uint32_t cb_beta_rm = 2;
constexpr uint32_t cb_scaler = 8;
constexpr uint32_t cb_eps = 9;

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr auto input_args = TensorAccessorArgs<1>();

    constexpr uint32_t gamma_stick_size = get_compile_time_arg_val(input_args.next_compile_time_args_offset());
    constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset() + 1>();

    constexpr uint32_t beta_stick_size = get_compile_time_arg_val(gamma_args.next_compile_time_args_offset());
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset() + 1>();

    // ---- Runtime args ----
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_rows = get_arg_val<uint32_t>(3);
    const uint32_t Wt = get_arg_val<uint32_t>(4);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(5);
    const uint32_t scaler_bits = get_arg_val<uint32_t>(6);
    const uint32_t eps_bits = get_arg_val<uint32_t>(7);

    // ---- TensorAccessors ----
    const auto input_accessor = TensorAccessor(input_args, src_addr, stick_size);
    const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, gamma_stick_size);
    const auto beta_accessor = TensorAccessor(beta_args, beta_addr, beta_stick_size);

    // ---- 1. Prepare scaler tile (1/W) in cb_scaler ----
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_f);

    // ---- 2. Prepare epsilon tile in cb_eps ----
    float eps_f = __builtin_bit_cast(float, eps_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_f);

    // ---- 3. Read gamma sticks (once) into cb_gamma_rm ----
    // Gamma shape is (1,1,1,W) but RM page = 1 stick = W elements.
    // For tilize, we need 32 sticks (tile height) in cb_gamma_rm = Wt pages.
    // Only the first stick has real data; remaining 31 sticks are padding (zeros).
    // We read 1 real stick, rest are already zeroed by the CB allocation.
    cb_reserve_back(cb_gamma_rm, Wt);
    uint32_t gamma_l1_addr = get_write_ptr(cb_gamma_rm);
    // Read the single gamma stick (stick_id = 0)
    uint64_t gamma_noc_addr = get_noc_addr(0, gamma_accessor);
    noc_async_read(gamma_noc_addr, gamma_l1_addr, gamma_stick_size);
    noc_async_read_barrier();
    cb_push_back(cb_gamma_rm, Wt);

    // ---- 4. Read beta sticks (once) into cb_beta_rm ----
    cb_reserve_back(cb_beta_rm, Wt);
    uint32_t beta_l1_addr = get_write_ptr(cb_beta_rm);
    uint64_t beta_noc_addr = get_noc_addr(0, beta_accessor);
    noc_async_read(beta_noc_addr, beta_l1_addr, beta_stick_size);
    noc_async_read_barrier();
    cb_push_back(cb_beta_rm, Wt);

    // ---- 5. Read input sticks per tile-row ----
    // For each tile-row: read 32 RM sticks into cb_input_rm, push Wt pages
    uint32_t stick_id = start_stick_id;
    for (uint32_t row = 0; row < num_rows; ++row) {
        cb_reserve_back(cb_input_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_input_rm);
        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint64_t noc_addr = get_noc_addr(stick_id, input_accessor);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_rm, Wt);
    }
}
