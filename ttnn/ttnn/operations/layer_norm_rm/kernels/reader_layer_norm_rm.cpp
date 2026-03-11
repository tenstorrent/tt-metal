// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Reader Kernel
// Reads RM sticks from DRAM via TensorAccessor.
// Per tile-row: reads same 32 sticks N times (N passes).
// At program start: fills scaler CB with 1/W, epsilon CB with eps.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t c_0 = 0;  // RM input staging for tilize
constexpr uint32_t c_2 = 2;  // Reduce scaler 1/W
constexpr uint32_t c_3 = 3;  // Epsilon scalar

// Compile-time args
constexpr uint32_t stick_size = get_compile_time_arg_val(0);
constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
constexpr uint32_t has_beta = get_compile_time_arg_val(2);
constexpr auto input_tensor_args = TensorAccessorArgs<3>();

// Number of passes per tile-row: currently 2 (mean + subtract)
constexpr uint32_t NUM_PASSES = 2;

void kernel_main() {
    // Runtime args
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);
    uint32_t gamma_addr = get_arg_val<uint32_t>(3);
    uint32_t beta_addr = get_arg_val<uint32_t>(4);
    uint32_t scaler_packed = get_arg_val<uint32_t>(5);
    uint32_t eps_packed = get_arg_val<uint32_t>(6);

    const auto input_accessor = TensorAccessor(input_tensor_args, src_addr, stick_size);

    constexpr uint32_t W = stick_size / 2;  // bfloat16 = 2 bytes per element
    constexpr uint32_t Wt = W / 32;         // tiles per row

    // Prepare scaler CB (1/W) at program start
    float scaler_val = 1.0f / static_cast<float>(W);
    dataflow_kernel_lib::prepare_reduce_scaler<c_2>(scaler_val);

    // Prepare epsilon CB at program start
    // Decode eps from packed bf16: extract upper 16 bits, convert to float
    uint16_t eps_bf16 = static_cast<uint16_t>(eps_packed >> 16);
    union {
        uint32_t u;
        float f;
    } eps_conv;
    eps_conv.u = static_cast<uint32_t>(eps_bf16) << 16;
    dataflow_kernel_lib::prepare_reduce_scaler<c_3>(eps_conv.f);

    // Main loop: for each tile-row, read 32 sticks NUM_PASSES times
    uint32_t stick_id = start_stick_id;
    constexpr uint32_t STICKS_PER_TILE_ROW = 32;

    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        for (uint32_t pass = 0; pass < NUM_PASSES; ++pass) {
            cb_reserve_back(c_0, Wt);
            uint32_t l1_write_addr = get_write_ptr(c_0);
            for (uint32_t s = 0; s < STICKS_PER_TILE_ROW; ++s) {
                uint64_t noc_addr = input_accessor.get_noc_addr(stick_id + s);
                noc_async_read(noc_addr, l1_write_addr, stick_size);
                l1_write_addr += stick_size;
            }
            noc_async_read_barrier();
            cb_push_back(c_0, Wt);
        }
        stick_id += STICKS_PER_TILE_ROW;
    }
}
