// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Reader Kernel
// Reads 32 RM sticks per tile-row into cb_in (c_0).
// On first tile-row, also reads gamma/beta as RM sticks into cb_gamma/cb_beta.
// Generates reduce scaler tile and epsilon tile.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/tensor_accessor_args.h"

constexpr uint32_t cb_in = tt::CBIndex::c_0;
constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_2;
constexpr uint32_t cb_gamma = tt::CBIndex::c_6;
constexpr uint32_t cb_beta = tt::CBIndex::c_7;
constexpr uint32_t cb_eps = tt::CBIndex::c_8;

constexpr uint32_t TILE_HEIGHT = 32;

// Compile-time args
constexpr uint32_t stick_size = get_compile_time_arg_val(0);
constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
constexpr uint32_t has_beta = get_compile_time_arg_val(2);

// Input tensor accessor always starts at index 3
constexpr auto input_accessor_args = TensorAccessorArgs<3>();

// NOTE: Gamma/beta accessor args are NOT declared here to avoid template
// instantiation of TensorAccessorArgs at out-of-range compile-time arg indices.
// When has_gamma=0 and has_beta=0, there are only 4 compile-time args (indices 0-3).
// Instantiating TensorAccessorArgs<4> would cause a compile error because
// get_ct_arg<4>() exceeds the array bounds.
// Instead, we use the InterleavedAddrGen directly for gamma/beta reads.

void kernel_main() {
    // Runtime args
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    uint32_t start_stick_id = get_arg_val<uint32_t>(3);
    uint32_t gamma_addr = get_arg_val<uint32_t>(4);
    uint32_t beta_addr = get_arg_val<uint32_t>(5);
    uint32_t eps_value_uint = get_arg_val<uint32_t>(6);

    // Input tensor accessor
    const auto input_accessor = TensorAccessor(input_accessor_args, src_addr, stick_size);

    // Number of tile-rows
    uint32_t num_tile_rows = num_sticks / TILE_HEIGHT;

    // Generate reduce scaler tile (1/W for mean computation)
    {
        float W_float = static_cast<float>(Wt * 32);
        float scaler = 1.0f / W_float;
        dataflow_kernel_lib::prepare_reduce_scaler<cb_reduce_scaler>(scaler);
    }

    // Generate epsilon tile
    {
        union {
            uint32_t u;
            float f;
        } eps_conv;
        eps_conv.u = eps_value_uint;
        dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_conv.f);
    }

    // Read gamma if present -- use simple interleaved address gen
    // Gamma is a [1, W] RM tensor with 1 page of stick_size bytes
    if (has_gamma && gamma_addr != 0) {
        const InterleavedAddrGen<true> gamma_addrgen = {
            .bank_base_address = gamma_addr,
            .page_size = stick_size,
        };
        cb_reserve_back(cb_gamma, Wt);
        uint32_t gamma_l1_addr = get_write_ptr(cb_gamma);
        uint64_t gamma_noc_addr = get_noc_addr(0, gamma_addrgen);
        noc_async_read(gamma_noc_addr, gamma_l1_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_gamma, Wt);
    }

    // Read beta if present
    if (has_beta && beta_addr != 0) {
        const InterleavedAddrGen<true> beta_addrgen = {
            .bank_base_address = beta_addr,
            .page_size = stick_size,
        };
        cb_reserve_back(cb_beta, Wt);
        uint32_t beta_l1_addr = get_write_ptr(cb_beta);
        uint64_t beta_noc_addr = get_noc_addr(0, beta_addrgen);
        noc_async_read(beta_noc_addr, beta_l1_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_beta, Wt);
    }

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
