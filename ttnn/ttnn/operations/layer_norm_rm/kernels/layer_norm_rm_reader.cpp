// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel
// Stage 4: Reads RM sticks from DRAM, fills scaler/eps CBs,
//          reads optional gamma/beta sticks for tilize by compute.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr auto input_accessor_args = TensorAccessorArgs<1>();
    constexpr uint32_t has_gamma_ct_idx = input_accessor_args.next_compile_time_args_offset();
    constexpr uint32_t has_gamma = get_compile_time_arg_val(has_gamma_ct_idx);
    constexpr uint32_t has_beta = get_compile_time_arg_val(has_gamma_ct_idx + 1);

    // ========== CB indices ==========
    constexpr uint32_t cb_rm_input = 0;  // RM sticks from DRAM
    constexpr uint32_t cb_scaler = 8;    // Reduce scaler (1/W)
    constexpr uint32_t cb_eps = 9;       // Epsilon constant

    // ========== Runtime args ==========
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);
    const uint32_t Wt = get_arg_val<uint32_t>(3);
    const uint32_t W = get_arg_val<uint32_t>(4);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(5);
    const uint32_t beta_addr = get_arg_val<uint32_t>(6);
    const uint32_t eps_packed = get_arg_val<uint32_t>(7);

    // ========== Setup TensorAccessor for input ==========
    const auto input_accessor = TensorAccessor(input_accessor_args, src_addr, stick_size);

    // ========== Fill scaler CB (c_8) with 1/W for reduce_row ==========
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(1.0f / static_cast<float>(W));

    // ========== Fill epsilon CB (c_9) with eps constant ==========
    cb_reserve_back(cb_eps, 1);
    fill_with_val_bfloat16(cb_eps, eps_packed);
    cb_push_back(cb_eps, 1);

    // ========== Optional: Read gamma sticks into c_0 for compute tilize ==========
    if constexpr (has_gamma) {
        constexpr uint32_t gamma_accessor_start = has_gamma_ct_idx + 2;
        constexpr auto gamma_accessor_args = TensorAccessorArgs<gamma_accessor_start>();
        const auto gamma_accessor = TensorAccessor(gamma_accessor_args, gamma_addr, stick_size);

        // Reserve Wt pages, read 1 gamma stick (shape 1,1,1,W), push Wt pages
        // Only Row0 of the tilized result matters for ROW broadcast
        cb_reserve_back(cb_rm_input, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_rm_input);
        uint64_t gamma_noc_addr = gamma_accessor.get_noc_addr(0);
        noc_async_read(gamma_noc_addr, l1_write_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_rm_input, Wt);
    }

    // ========== Optional: Read beta sticks into c_0 for compute tilize ==========
    if constexpr (has_beta) {
        // Beta accessor starts after gamma accessor (if gamma present) or after has_beta flag
        constexpr uint32_t beta_accessor_start =
            has_gamma ? TensorAccessorArgs<has_gamma_ct_idx + 2>().next_compile_time_args_offset()
                      : has_gamma_ct_idx + 2;
        constexpr auto beta_accessor_args = TensorAccessorArgs<beta_accessor_start>();
        const auto beta_accessor = TensorAccessor(beta_accessor_args, beta_addr, stick_size);

        // Reserve Wt pages, read 1 beta stick (shape 1,1,1,W), push Wt pages
        cb_reserve_back(cb_rm_input, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_rm_input);
        uint64_t beta_noc_addr = beta_accessor.get_noc_addr(0);
        noc_async_read(beta_noc_addr, l1_write_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_rm_input, Wt);
    }

    // ========== Main loop: read 32 RM sticks per block ==========
    uint32_t stick_id = start_stick_id;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Reserve Wt pages (tile-sized) in cb_rm_input
        cb_reserve_back(cb_rm_input, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_rm_input);

        // Read 32 sticks (one tile-row)
        for (uint32_t stick = 0; stick < 32; ++stick) {
            uint64_t noc_addr = input_accessor.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();

        // Push Wt pages (32 sticks = Wt tiles worth of data)
        cb_push_back(cb_rm_input, Wt);
    }
}
