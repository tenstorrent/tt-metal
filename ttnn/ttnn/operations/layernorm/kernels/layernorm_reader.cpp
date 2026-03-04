// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Reader Kernel
// Reads 32 RM sticks per tile-row into cb_in (c_0).
// Generates reduce scaler tile (1/W) into cb_reduce_scaler (c_2).
// Generates epsilon tile into cb_eps (c_8).

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t cb_in = tt::CBIndex::c_0;
constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_2;
constexpr uint32_t cb_eps = tt::CBIndex::c_8;
constexpr uint32_t TILE_HEIGHT = 32;

// Compile-time args
constexpr uint32_t stick_size = get_compile_time_arg_val(0);
constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
constexpr uint32_t has_beta = get_compile_time_arg_val(2);
constexpr auto input_accessor_args = TensorAccessorArgs<3>();

void kernel_main() {
    // Runtime args
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    uint32_t start_stick_id = get_arg_val<uint32_t>(3);
    uint32_t gamma_addr = get_arg_val<uint32_t>(4);
    uint32_t beta_addr = get_arg_val<uint32_t>(5);
    uint32_t eps_uint32 = get_arg_val<uint32_t>(6);

    // Input tensor accessor
    const auto input_accessor = TensorAccessor(input_accessor_args, src_addr, stick_size);

    // Compute W from stick_size (bfloat16 = 2 bytes per element)
    constexpr uint32_t element_size = 2;
    constexpr uint32_t W = stick_size / element_size;

    // Generate reduce scaler tile: 1/W for mean computation (AVG reduce)
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_reduce_scaler,
        ckernel::PoolType::AVG,
        ckernel::ReduceDim::REDUCE_ROW,
        W>();

    // Generate epsilon tile
    float eps;
    // Reinterpret eps_uint32 bits as float
    union {
        uint32_t u;
        float f;
    } eps_conv;
    eps_conv.u = eps_uint32;
    eps = eps_conv.f;
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps);

    // Number of tile-rows
    uint32_t num_tile_rows = num_sticks / TILE_HEIGHT;

    // Main loop: read 32 RM sticks per tile-row into cb_in
    uint32_t stick_id = start_stick_id;
    for (uint32_t tile_row = 0; tile_row < num_tile_rows; ++tile_row) {
        cb_reserve_back(cb_in, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_in);

        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint64_t noc_addr = input_accessor.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();

        cb_push_back(cb_in, Wt);
    }
}
