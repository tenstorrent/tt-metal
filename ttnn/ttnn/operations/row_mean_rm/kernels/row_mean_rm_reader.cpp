// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// row_mean_rm - Reader Kernel
// Reads RM sticks from DRAM via NOC0, prepares scaler tile (1/W).
//
// Compile-time args:
//   [0]      stick_size           - bytes per RM input stick (W * sizeof(bfloat16))
//   [1..]    TensorAccessorArgs(input)
//
// Runtime args:
//   [0] src_addr        - input buffer DRAM address
//   [1] num_rows        - total tile-rows to process
//   [2] Wt              - tiles per row (W / 32)
//   [3] start_stick_id  - first stick id for this core
//   [4] scaler_value    - 1/W as float bits (reinterpreted)
//
// CB usage:
//   c_0  (cb_input_rm)  - push Wt pages per tile-row (32 RM sticks per row)
//   c_8  (cb_scaler)    - push 1 tile (1/W scaler, never popped)

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t TILE_HEIGHT = 32;

constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_scaler = 8;

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr auto input_args = TensorAccessorArgs<1>();

    // ---- Runtime args ----
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(3);
    const uint32_t scaler_bits = get_arg_val<uint32_t>(4);

    // ---- TensorAccessor ----
    const auto input_accessor = TensorAccessor(input_args, src_addr, stick_size);

    // ---- 1. Prepare scaler tile (1/W) in cb_scaler ----
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_f);

    // ---- 2. Read input sticks per tile-row ----
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
