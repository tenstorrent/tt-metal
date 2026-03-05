// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel
//
// Reads RM input sticks from DRAM using TensorAccessor.
// Generates scaler tile (1/W) into cb_scaler and epsilon tile into cb_eps.
// For affine transform: reads gamma/beta tilized tiles into cb_gamma/cb_beta.
//
// Compile-time args:
//   [0]  stick_size       - W * element_size bytes
//   [1+] TensorAccessorArgs(input)
//   [N]  has_gamma        - 0 or 1
//   [N+1] has_beta        - 0 or 1
//   [N+2+] TensorAccessorArgs(gamma) if has_gamma
//   [M+] TensorAccessorArgs(beta) if has_beta
//
// Runtime args:
//   [0] src_addr           - Input buffer base address
//   [1] num_blocks         - Number of tile-rows for this core
//   [2] start_stick_id     - First stick (row) index
//   [3] gamma_addr         - Gamma buffer base address (0 if no gamma)
//   [4] beta_addr          - Beta buffer base address (0 if no beta)
//   [5] eps_value          - Epsilon as bit-cast uint32
//   [6] mean_scaler_value  - 1/W as bit-cast uint32

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t cb_in_rm = 0;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_eps = 3;
constexpr uint32_t cb_gamma = 4;
constexpr uint32_t cb_beta = 5;

// Helper to read Wt tilized tile pages from DRAM into a CB using TensorAccessor.
template <typename TensorAccessorT>
void read_tiles_into_cb(const TensorAccessorT& accessor, uint32_t cb_id, uint32_t num_tiles, uint32_t tile_bytes) {
    cb_reserve_back(cb_id, num_tiles);
    uint32_t l1_addr = get_write_ptr(cb_id);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        uint64_t noc_addr = accessor.get_noc_addr(t);
        noc_async_read(noc_addr, l1_addr, tile_bytes);
        l1_addr += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_tiles);
}

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr auto input_tensor_args = TensorAccessorArgs<1>();
    constexpr uint32_t flags_idx = input_tensor_args.next_compile_time_args_offset();
    constexpr uint32_t has_gamma = get_compile_time_arg_val(flags_idx);
    constexpr uint32_t has_beta = get_compile_time_arg_val(flags_idx + 1);

    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr = get_arg_val<uint32_t>(4);
    const uint32_t eps_bits = get_arg_val<uint32_t>(5);
    const uint32_t scaler_bits = get_arg_val<uint32_t>(6);

    // Derived constants
    constexpr uint32_t tile_size_bytes = get_tile_size(cb_in_rm);
    constexpr uint32_t Wt = stick_size / (32 * 2);

    // Create TensorAccessor for input (RM tensor: page = 1 stick = stick_size bytes)
    const auto input_accessor = TensorAccessor(input_tensor_args, src_addr, stick_size);

    // Early exit for idle cores
    if (num_blocks == 0) {
        return;
    }

    // === Prepare scaler tile (1/W) into cb_scaler ===
    union {
        uint32_t u;
        float f;
    } scaler_union;
    scaler_union.u = scaler_bits;
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_union.f);

    // === Prepare epsilon tile into cb_eps ===
    union {
        uint32_t u;
        float f;
    } eps_union;
    eps_union.u = eps_bits;
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_union.f);

    // === Read gamma/beta tilized tiles from DRAM ===
    // Gamma and beta are pre-tilized on host side. Each has Wt tile pages.
    // They are persistent: pushed once here, consumed repeatedly by compute.
    if constexpr (has_gamma && has_beta) {
        // Both gamma and beta present
        constexpr auto gamma_tensor_args = TensorAccessorArgs<flags_idx + 2>();
        constexpr uint32_t beta_start = gamma_tensor_args.next_compile_time_args_offset();
        constexpr auto beta_tensor_args = TensorAccessorArgs<beta_start>();

        const auto gamma_accessor = TensorAccessor(gamma_tensor_args, gamma_addr, tile_size_bytes);
        const auto beta_accessor = TensorAccessor(beta_tensor_args, beta_addr, tile_size_bytes);

        read_tiles_into_cb(gamma_accessor, cb_gamma, Wt, tile_size_bytes);
        read_tiles_into_cb(beta_accessor, cb_beta, Wt, tile_size_bytes);
    } else if constexpr (has_gamma) {
        // Only gamma
        constexpr auto gamma_tensor_args = TensorAccessorArgs<flags_idx + 2>();
        const auto gamma_accessor = TensorAccessor(gamma_tensor_args, gamma_addr, tile_size_bytes);
        read_tiles_into_cb(gamma_accessor, cb_gamma, Wt, tile_size_bytes);
    } else if constexpr (has_beta) {
        // Only beta (no gamma)
        constexpr auto beta_tensor_args = TensorAccessorArgs<flags_idx + 2>();
        const auto beta_accessor = TensorAccessor(beta_tensor_args, beta_addr, tile_size_bytes);
        read_tiles_into_cb(beta_accessor, cb_beta, Wt, tile_size_bytes);
    }

    // === Main loop: read RM sticks per tile-row ===
    uint32_t stick_id = start_stick_id;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        cb_reserve_back(cb_in_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_in_rm);

        for (uint32_t stick = 0; stick < 32; ++stick) {
            uint64_t noc_addr = input_accessor.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in_rm, Wt);
    }
}
