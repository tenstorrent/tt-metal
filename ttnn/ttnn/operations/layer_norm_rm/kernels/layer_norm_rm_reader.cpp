// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel
// Reads RM input sticks from DRAM via TensorAccessor into c_0.
// Generates reduce scaler tile (1/W) into c_2 and epsilon tile into c_7.
// Optionally reads gamma/beta sticks (32 sticks) into c_27/c_28.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

void kernel_main() {
    // ========== COMPILE-TIME ARGS ==========
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);  // W * 2 bytes
    constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t has_beta = get_compile_time_arg_val(2);
    constexpr auto input_accessor_args = TensorAccessorArgs<3>();

    // Gamma/beta accessor args follow input accessor args (conditionally)
    // The offset chain: input starts at 3, gamma starts after input, beta starts after gamma
    constexpr uint32_t gamma_cta_start = input_accessor_args.next_compile_time_args_offset();

    // ========== RUNTIME ARGS ==========
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(3);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(4);
    const uint32_t beta_addr = get_arg_val<uint32_t>(5);
    const uint32_t eps_bits = get_arg_val<uint32_t>(6);  // epsilon as bit-cast uint32_t

    // Early exit for idle cores
    if (num_sticks == 0) {
        return;
    }

    // ========== TENSOR ACCESSOR ==========
    const auto input_accessor = TensorAccessor(input_accessor_args, src_addr, stick_size);

    // ========== CB CONSTANTS ==========
    constexpr uint32_t cb_input_rm = 0;       // c_0: input RM sticks
    constexpr uint32_t cb_reduce_scaler = 2;  // c_2: reduce scaler (1/W)
    constexpr uint32_t cb_eps = 7;            // c_7: epsilon tile (persistent)
    constexpr uint32_t cb_gamma_rm = 27;      // c_27: gamma RM sticks
    constexpr uint32_t cb_beta_rm = 28;       // c_28: beta RM sticks

    // ========== GENERATE REDUCE SCALER (1/W) INTO c_2 ==========
    // W = Wt * 32 (total elements in width dimension)
    const uint32_t W = Wt * 32;
    const float scaler = 1.0f / static_cast<float>(W);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_reduce_scaler>(scaler);

    // ========== GENERATE EPSILON TILE INTO c_7 ==========
    // Epsilon is passed as bit-cast uint32_t, reinterpret as float
    union {
        uint32_t u;
        float f;
    } eps_union;
    eps_union.u = eps_bits;
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_union.f);

    // ========== READ GAMMA STICKS INTO c_27 (once, if present) ==========
    if constexpr (has_gamma) {
        constexpr auto gamma_accessor_args = TensorAccessorArgs<gamma_cta_start>();
        const auto gamma_accessor = TensorAccessor(gamma_accessor_args, gamma_addr, stick_size);

        cb_reserve_back(cb_gamma_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_gamma_rm);

        // Gamma is (1,1,1,W) -> 1 row of data, but we read 32 sticks (tile-row)
        // Sticks 1-31 are padding (zeros), only stick 0 has data
        for (uint32_t s = 0; s < 32; s++) {
            noc_async_read_page(s, gamma_accessor, l1_write_addr);
            l1_write_addr += stick_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma_rm, Wt);
    }

    // ========== READ BETA STICKS INTO c_28 (once, if present) ==========
    if constexpr (has_beta) {
        // Beta accessor offset depends on whether gamma accessor args were included
        constexpr uint32_t beta_cta_start_with_gamma =
            TensorAccessorArgs<gamma_cta_start>().next_compile_time_args_offset();
        constexpr uint32_t beta_cta_start_actual = has_gamma ? beta_cta_start_with_gamma : gamma_cta_start;
        constexpr auto beta_accessor_args = TensorAccessorArgs<beta_cta_start_actual>();
        const auto beta_accessor = TensorAccessor(beta_accessor_args, beta_addr, stick_size);

        cb_reserve_back(cb_beta_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_beta_rm);

        // Beta is (1,1,1,W) -> 1 row of data, read 32 sticks
        for (uint32_t s = 0; s < 32; s++) {
            noc_async_read_page(s, beta_accessor, l1_write_addr);
            l1_write_addr += stick_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_beta_rm, Wt);
    }

    // ========== MAIN LOOP: READ INPUT STICKS ==========
    // Each block = 32 sticks = 1 tile-row = Wt tile-sized pages
    // Reader batches 32 sticks per block, pushes Wt pages
    const uint32_t num_blocks = num_sticks / 32;

    uint32_t stick_id = start_stick_id;

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Reserve Wt pages in c_0 (tile-sized pages containing RM sticks)
        cb_reserve_back(cb_input_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_input_rm);

        // Read 32 sticks into the reserved space
        for (uint32_t s = 0; s < 32; s++) {
            noc_async_read_page(stick_id, input_accessor, l1_write_addr);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();

        // Push Wt pages (32 sticks = Wt tile-sized pages)
        cb_push_back(cb_input_rm, Wt);
    }
}
