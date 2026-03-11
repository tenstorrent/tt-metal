// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Reader Kernel
// Reads RM sticks from DRAM via TensorAccessor, generates reduce scaler and
// epsilon tile, optionally reads gamma/beta tiles. Sends each tile-row
// num_passes times (multi-pass streaming) into CB c_0.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t cb_input_rm = 0;       // c_0
constexpr uint32_t cb_reduce_scaler = 2;  // c_2
constexpr uint32_t cb_eps_scalar = 3;     // c_3
constexpr uint32_t cb_gamma = 4;          // c_4
constexpr uint32_t cb_beta = 5;           // c_5

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t epsilon_u32 = get_compile_time_arg_val(1);
    constexpr auto input_accessor_args = TensorAccessorArgs<2>();

    // Runtime args
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(1);
    const uint32_t num_sticks = get_arg_val<uint32_t>(2);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr = get_arg_val<uint32_t>(4);

    if (num_sticks == 0) {
        return;
    }

    // TensorAccessor for input
    const auto input_accessor = TensorAccessor(input_accessor_args, input_addr, stick_size);

    // Derive W and Wt from stick_size: stick_size = W * 2 (bf16), Wt = W / 32
    constexpr uint32_t W = stick_size / 2;
    constexpr uint32_t Wt = W / 32;     // tiles per row
    constexpr uint32_t num_passes = 3;  // 3-pass: pass1=mean, pass2=var+rsqrt, pass3=normalize

    // Number of tile-rows (each tile-row = 32 sticks)
    const uint32_t nblocks = num_sticks / 32;

    // Generate reduce scaler tile (1/W) into c_2 at program start
    // This is needed for reduce SUM to compute mean = sum * (1/W)
    dataflow_kernel_lib::prepare_reduce_scaler<cb_reduce_scaler>(1.0f / static_cast<float>(W));

    // Generate epsilon tile into c_3 at program start
    // Reinterpret epsilon_u32 back to float
    union {
        uint32_t u;
        float f;
    } eps_conv;
    eps_conv.u = epsilon_u32;
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps_scalar>(eps_conv.f);

    // Read tile-rows, sending each one num_passes times
    for (uint32_t block = 0; block < nblocks; block++) {
        for (uint32_t pass = 0; pass < num_passes; pass++) {
            // Reserve Wt pages in c_0 (tilize convention: 32 sticks = Wt tile-pages)
            cb_reserve_back(cb_input_rm, Wt);
            uint32_t l1_write_addr = get_write_ptr(cb_input_rm);

            // Read 32 sticks for this tile-row
            for (uint32_t stick = 0; stick < 32; stick++) {
                uint32_t stick_id = start_stick_id + block * 32 + stick;
                uint64_t noc_addr = input_accessor.get_noc_addr(stick_id);
                noc_async_read(noc_addr, l1_write_addr, stick_size);
                l1_write_addr += stick_size;
            }
            noc_async_read_barrier();

            cb_push_back(cb_input_rm, Wt);
        }
    }
}
