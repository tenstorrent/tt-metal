// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(1);

    // Tensor accessor chaining: input, gamma, beta
    constexpr auto input_ta_args = TensorAccessorArgs<2>();
    constexpr auto gamma_ta_args = TensorAccessorArgs<input_ta_args.next_compile_time_args_offset()>();
    constexpr auto beta_ta_args = TensorAccessorArgs<gamma_ta_args.next_compile_time_args_offset()>();

    // Runtime args
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rm_pages = get_arg_val<uint32_t>(1);
    const uint32_t input_start_page = get_arg_val<uint32_t>(2);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(3);
    const uint32_t gamma_num_pages = get_arg_val<uint32_t>(4);
    const uint32_t beta_addr = get_arg_val<uint32_t>(5);
    const uint32_t beta_num_pages = get_arg_val<uint32_t>(6);

    // CB indices
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_gamma_rm = 3;
    constexpr uint32_t cb_beta_rm = 4;
    constexpr uint32_t cb_scaler = 25;
    constexpr uint32_t cb_eps = 26;

    // Page size for all RM tensors (W * 2 bytes for bf16)
    constexpr uint32_t page_size = W * 2;

    // Create tensor accessors
    const auto input_accessor = TensorAccessor(input_ta_args, input_addr, page_size);
    const auto gamma_accessor = TensorAccessor(gamma_ta_args, gamma_addr, page_size);
    const auto beta_accessor = TensorAccessor(beta_ta_args, beta_addr, page_size);

    // Step 1: Prepare scaler tile (1/W) for reduce
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scaler,
        ckernel::PoolType::AVG,
        ckernel::ReduceDim::REDUCE_ROW,
        tt::constants::TILE_WIDTH,
        W>();

    // Step 2: Prepare epsilon tile (fill entire tile with eps for use with add<SCALAR>)
    {
        float eps_f = __builtin_bit_cast(float, eps_bits);
        // Convert float to packed bf16 pair
        union {
            float f;
            uint32_t bits;
        } conv;
        conv.f = eps_f;
        uint16_t eps_bf16 = static_cast<uint16_t>(conv.bits >> 16);
        uint32_t eps_packed = (static_cast<uint32_t>(eps_bf16) << 16) | eps_bf16;

        cb_reserve_back(cb_eps, 1);
        uint32_t eps_addr = get_write_ptr(cb_eps);
        // Tile layout: 4 faces of 128 uint32 words each (bf16: 16 rows × 8 packed cols)
        volatile tt_l1_ptr uint32_t* eps_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(eps_addr);
        for (uint32_t i = 0; i < 512; i++) {
            eps_ptr[i] = eps_packed;
        }
        cb_push_back(cb_eps, 1);
    }

    // Step 3: Read 1 gamma page -> cb_gamma_rm
    cb_reserve_back(cb_gamma_rm, 1);
    uint32_t l1_write_addr = get_write_ptr(cb_gamma_rm);
    noc_async_read_page(0, gamma_accessor, l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(cb_gamma_rm, 1);

    // Step 4: Read 1 beta page -> cb_beta_rm
    cb_reserve_back(cb_beta_rm, 1);
    l1_write_addr = get_write_ptr(cb_beta_rm);
    noc_async_read_page(0, beta_accessor, l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(cb_beta_rm, 1);

    // Step 5: Main loop - read 32 RM pages per tile row
    uint32_t page_id = input_start_page;
    uint32_t pages_remaining = num_rm_pages;
    while (pages_remaining > 0) {
        uint32_t pages_this_block = (pages_remaining >= 32) ? 32 : pages_remaining;
        for (uint32_t i = 0; i < pages_this_block; i++) {
            cb_reserve_back(cb_in, 1);
            l1_write_addr = get_write_ptr(cb_in);
            noc_async_read_page(page_id, input_accessor, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in, 1);
            page_id++;
        }
        pages_remaining -= pages_this_block;
    }
}
