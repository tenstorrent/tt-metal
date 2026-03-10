// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Reader Kernel
// Reads RM sticks from DRAM into c_0 (Wt tile-sized pages per block of 32 sticks).
// On first invocation, generates constant tiles:
//   c_8:  reduce scaler (1/W) via prepare_reduce_scaler
//   c_10: epsilon scalar tile via prepare_reduce_scaler
// If gamma/beta present: reads their TILE_LAYOUT pages into c_1/c_2 once at start.
//   Gamma/beta are pre-tilized on the host (entry point does ttnn.tilize()).

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t c_0 = 0;    // RM input sticks staging
constexpr uint32_t c_1 = 1;    // Gamma tiles (optional, TILE_LAYOUT)
constexpr uint32_t c_2 = 2;    // Beta tiles (optional, TILE_LAYOUT)
constexpr uint32_t c_8 = 8;    // Reduce scaler (1/W)
constexpr uint32_t c_10 = 10;  // Epsilon scalar tile

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);  // W * 2 bytes
    constexpr uint32_t Wt = get_compile_time_arg_val(1);          // W / 32
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);
    constexpr auto input_tensor_args = TensorAccessorArgs<4>();  // Input TensorAccessor at index 4
    constexpr auto gamma_tensor_args = TensorAccessorArgs<5>();  // Gamma TensorAccessor at index 5
    constexpr auto beta_tensor_args = TensorAccessorArgs<6>();   // Beta TensorAccessor at index 6

    // Runtime args
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(1);  // tile-rows to process
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);           // first RM stick ID
    const uint32_t gamma_addr = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr = get_arg_val<uint32_t>(4);

    // Derive W from stick_size: W = stick_size / 2 (bf16 is 2 bytes per element)
    constexpr uint32_t W = stick_size / 2;

    // bf16 tile page size: 32x32 elements * 2 bytes = 2048 bytes
    constexpr uint32_t bf16_tile_size = 32 * 32 * 2;

    // Create TensorAccessor for input (RM sticks)
    const auto input_accessor = TensorAccessor(input_tensor_args, input_addr, stick_size);

    // ========== One-time: generate constant tiles ==========
    // Reduce scaler: 1/W
    constexpr float inv_W = 1.0f / static_cast<float>(W);
    dataflow_kernel_lib::prepare_reduce_scaler<c_8>(inv_W);

    // Epsilon tile
    constexpr float epsilon = 1e-5f;
    dataflow_kernel_lib::prepare_reduce_scaler<c_10>(epsilon);

    // ========== One-time: read gamma/beta tile pages if present ==========
    // Gamma and beta are pre-tilized on the host to TILE_LAYOUT.
    // Shape after tilize: (1,1,32,W) with Wt tiles. Read Wt tile pages into c_1/c_2.
    if constexpr (has_gamma) {
        const auto gamma_accessor = TensorAccessor(gamma_tensor_args, gamma_addr, bf16_tile_size);
        cb_reserve_back(c_1, Wt);
        uint32_t l1_write_addr = get_write_ptr(c_1);
        for (uint32_t t = 0; t < Wt; t++) {
            uint64_t noc_addr = gamma_accessor.get_noc_addr(t);
            noc_async_read(noc_addr, l1_write_addr, bf16_tile_size);
            l1_write_addr += bf16_tile_size;
        }
        noc_async_read_barrier();
        cb_push_back(c_1, Wt);
    }

    if constexpr (has_beta) {
        const auto beta_accessor = TensorAccessor(beta_tensor_args, beta_addr, bf16_tile_size);
        cb_reserve_back(c_2, Wt);
        uint32_t l1_write_addr = get_write_ptr(c_2);
        for (uint32_t t = 0; t < Wt; t++) {
            uint64_t noc_addr = beta_accessor.get_noc_addr(t);
            noc_async_read(noc_addr, l1_write_addr, bf16_tile_size);
            l1_write_addr += bf16_tile_size;
        }
        noc_async_read_barrier();
        cb_push_back(c_2, Wt);
    }

    // ========== Per-block loop: read 32 RM sticks into c_0 ==========
    for (uint32_t block = 0; block < num_rows_per_core; block++) {
        // Reserve Wt pages in c_0 (tile-sized pages)
        cb_reserve_back(c_0, Wt);
        uint32_t l1_write_addr = get_write_ptr(c_0);

        // Read 32 RM sticks
        for (uint32_t stick = 0; stick < 32; stick++) {
            uint64_t noc_addr = input_accessor.get_noc_addr(start_stick_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            start_stick_id++;
        }
        noc_async_read_barrier();

        // Push Wt pages (32 sticks = Wt tile-sized pages of data)
        cb_push_back(c_0, Wt);
    }
}
