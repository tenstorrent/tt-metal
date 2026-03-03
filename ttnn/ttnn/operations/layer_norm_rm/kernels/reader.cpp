// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel
//
// Reads RM sticks from DRAM into CB 0 for tilize.
// Prepares scaler and epsilon tiles for reduce operations.
// Reads gamma/beta sticks (32 repeated copies) into CB 6/7 for tilize by compute.
//
// Compile-time args:
//   [0] stick_size_bytes
//   [1] gamma_stick_size
//   [2] has_gamma
//   [3] has_beta
//   [4]   TensorAccessorArgs for input (1 arg for interleaved)
//   [5]   TensorAccessorArgs for gamma (if has_gamma, 1 arg)
//   [5/6] TensorAccessorArgs for beta (if has_beta, 1 arg)
//
// Runtime args:
//   [0] src_addr
//   [1] gamma_addr
//   [2] beta_addr
//   [3] num_sticks (total RM sticks this core processes)
//   [4] start_stick_id
//   [5] scaler_value (unused - compute scaler in kernel from stick_size)
//   [6] eps_value (unused - hardcoded 1e-5f)

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // ===== Compile-time args =====
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t gamma_stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);

    // Input tensor accessor starts at CTA index 4
    constexpr auto input_tensor_args = TensorAccessorArgs<4>();
    // Gamma tensor accessor follows input (at next_compile_time_args_offset)
    constexpr uint32_t gamma_cta_offset = input_tensor_args.next_compile_time_args_offset();
    // Beta tensor accessor follows gamma (or input if no gamma)

    // Wt = number of tiles per row = stick_size / (32 * 2) for bf16
    constexpr uint32_t Wt = stick_size_bytes / (32 * 2);
    constexpr uint32_t TILE_HEIGHT = 32;

    // CB indices
    constexpr uint32_t cb_in_rm = 0;
    constexpr uint32_t cb_scaler = 2;
    constexpr uint32_t cb_eps = 3;
    constexpr uint32_t cb_gamma_rm = 6;
    constexpr uint32_t cb_beta_rm = 7;

    // ===== Runtime args =====
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_sticks = get_arg_val<uint32_t>(3);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(4);
    // Args 5,6 reserved for scaler_value and eps_value (computed locally)

    // Early exit for idle cores
    if (num_sticks == 0) {
        return;
    }

    // Create tensor accessor for input
    const auto input_accessor = TensorAccessor(input_tensor_args, src_addr, stick_size_bytes);

    // ===== One-time setup: prepare scaler and epsilon tiles =====
    {
        constexpr uint32_t W = stick_size_bytes / 2;
        constexpr float scaler_float = 1.0f / W;
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_float);
    }

    // Prepare epsilon tile
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(1e-5f);

    // ===== One-time: read gamma and beta sticks =====
    // Gamma/beta have shape (1,1,1,W) = 1 stick of gamma_stick_size bytes.
    // We read 32 copies of this stick into the staging CB to form a tilizable block.
    if constexpr (has_gamma) {
        constexpr auto gamma_tensor_args = TensorAccessorArgs<gamma_cta_offset>();
        const auto gamma_accessor = TensorAccessor(gamma_tensor_args, gamma_addr, gamma_stick_size);

        cb_reserve_back(cb_gamma_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_gamma_rm);

        // Read the single gamma stick, then copy it 32 times
        // First read stick 0 from DRAM
        uint64_t gamma_noc_addr = get_noc_addr(0, gamma_accessor);
        noc_async_read(gamma_noc_addr, l1_write_addr, gamma_stick_size);
        noc_async_read_barrier();

        // Copy the first stick to the remaining 31 positions
        uint32_t src_l1_addr = l1_write_addr;
        for (uint32_t s = 1; s < TILE_HEIGHT; s++) {
            uint32_t dst_l1_addr = l1_write_addr + s * gamma_stick_size;
            // Use local L1 copy (noc_async_read from own L1)
            uint64_t src_noc_addr = get_noc_addr(src_l1_addr);
            noc_async_read(src_noc_addr, dst_l1_addr, gamma_stick_size);
        }
        noc_async_read_barrier();

        cb_push_back(cb_gamma_rm, Wt);
    }

    if constexpr (has_beta) {
        // Beta CTA offset depends on whether gamma is present
        constexpr uint32_t beta_cta_offset =
            has_gamma ? TensorAccessorArgs<gamma_cta_offset>().next_compile_time_args_offset() : gamma_cta_offset;
        constexpr auto beta_tensor_args = TensorAccessorArgs<beta_cta_offset>();
        const auto beta_accessor = TensorAccessor(beta_tensor_args, beta_addr, gamma_stick_size);

        cb_reserve_back(cb_beta_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_beta_rm);

        // Read the single beta stick, then copy it 32 times
        uint64_t beta_noc_addr = get_noc_addr(0, beta_accessor);
        noc_async_read(beta_noc_addr, l1_write_addr, gamma_stick_size);
        noc_async_read_barrier();

        uint32_t src_l1_addr = l1_write_addr;
        for (uint32_t s = 1; s < TILE_HEIGHT; s++) {
            uint32_t dst_l1_addr = l1_write_addr + s * gamma_stick_size;
            uint64_t src_noc_addr = get_noc_addr(src_l1_addr);
            noc_async_read(src_noc_addr, dst_l1_addr, gamma_stick_size);
        }
        noc_async_read_barrier();

        cb_push_back(cb_beta_rm, Wt);
    }

    // ===== Per-tile-row loop: read RM sticks into cb_in_rm =====
    uint32_t num_tile_rows = num_sticks / TILE_HEIGHT;
    uint32_t stick_id = start_stick_id;

    for (uint32_t tr = 0; tr < num_tile_rows; tr++) {
        cb_reserve_back(cb_in_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_in_rm);

        for (uint32_t s = 0; s < TILE_HEIGHT; s++) {
            uint64_t noc_addr = get_noc_addr(stick_id, input_accessor);
            noc_async_read(noc_addr, l1_write_addr, stick_size_bytes);
            l1_write_addr += stick_size_bytes;
            stick_id++;
        }

        noc_async_read_barrier();
        cb_push_back(cb_in_rm, Wt);
    }
}
