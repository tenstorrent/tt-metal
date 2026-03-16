// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Reader Kernel
// Reads RM sticks from DRAM, generates reduce scaler and epsilon,
// loads gamma/beta if present.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);

    // TensorAccessor args for input (starts at CT arg index 4)
    constexpr auto input_args = TensorAccessorArgs<4>();

    // Gamma tensor accessor (always occupies a slot, even if absent)
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    // Beta tensor accessor (always occupies a slot, even if absent)
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    // ========== Runtime args ==========
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t scaler_bits = get_arg_val<uint32_t>(3);
    const uint32_t eps_bits = get_arg_val<uint32_t>(4);

    // ========== Constants ==========
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_gamma_rm = 3;
    constexpr uint32_t cb_beta_rm = 4;
    constexpr uint32_t cb_scaler = 8;
    constexpr uint32_t cb_eps = 9;

    // Stick size = W * sizeof(bf16) = Wt * 32 * 2 = Wt * 64
    constexpr uint32_t stick_size = Wt * 32 * 2;
    // Tile size for bf16 32x32 tile
    constexpr uint32_t tile_size = 32 * 32 * 2;

    // Create input TensorAccessor with stick_size as page_size (RM tensor pages = sticks)
    const auto input_accessor = TensorAccessor(input_args, input_addr, stick_size);

    // ========== Generate scaler and epsilon tiles (once, before main loop) ==========
    // Runtime args contain raw float32 bit patterns. Use union to convert.
    union {
        uint32_t u;
        float f;
    } scaler_conv, eps_conv;
    scaler_conv.u = scaler_bits;
    eps_conv.u = eps_bits;
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_conv.f);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_conv.f);

    // ========== Load gamma/beta if present (once, before main loop) ==========
    if constexpr (has_gamma) {
        const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, stick_size);
        // Gamma shape is (1,1,1,W) -> 1 stick. Replicate it 32 times for tilize.
        // The CB has Wt pages of tile_size each. We write 32 copies of the stick
        // into the Wt tile-sized pages so tilize produces tiles with identical rows.
        cb_reserve_back(cb_gamma_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_gamma_rm);
        for (uint32_t row = 0; row < 32; row++) {
            // Read the single gamma stick (page 0) into current position
            noc_async_read_page(0, gamma_accessor, l1_write_addr);
            l1_write_addr += stick_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma_rm, Wt);
    }

    if constexpr (has_beta) {
        const auto beta_accessor = TensorAccessor(beta_args, beta_addr, stick_size);
        cb_reserve_back(cb_beta_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_beta_rm);
        for (uint32_t row = 0; row < 32; row++) {
            noc_async_read_page(0, beta_accessor, l1_write_addr);
            l1_write_addr += stick_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_beta_rm, Wt);
    }

    // ========== Main loop: read input RM sticks ==========
    uint32_t stick_id = 0;
    for (uint32_t ht = 0; ht < Ht; ht++) {
        // Each tile-row = 32 sticks. Reserve Wt tile-sized pages.
        cb_reserve_back(cb_in, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_in);

        // Read 32 sticks sequentially into the reserved CB space
        for (uint32_t row = 0; row < 32; row++) {
            noc_async_read_page(stick_id, input_accessor, l1_write_addr);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, Wt);
    }
}
