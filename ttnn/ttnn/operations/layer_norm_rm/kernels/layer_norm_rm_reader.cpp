// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel
//
// Reads RM sticks from DRAM into cb_rm_in (tilize input).
// Prepares reduce scaler (1/W) in cb_reduce_scaler.
// Prepares epsilon tile in cb_eps.
// Optionally reads gamma/beta tiles into cb_gamma/cb_beta.
//
// Compile-time arg layout:
//   [0]: stick_size
//   [1]: gamma_accessor_ct_start (0 if no gamma)
//   [2]: beta_accessor_ct_start (0 if no beta)
//   [3+]: TensorAccessorArgs(input)
//   [gamma_start+]: TensorAccessorArgs(gamma) if has_gamma
//   [beta_start+]:  TensorAccessorArgs(beta) if has_beta

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t cb_rm_in = 0;
constexpr uint32_t cb_reduce_scaler = 2;
constexpr uint32_t cb_eps = 3;
constexpr uint32_t cb_gamma = 25;
constexpr uint32_t cb_beta = 26;

// Tile size for bfloat16 tiles
constexpr uint32_t TILE_SIZE = 2048;

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t gamma_accessor_ct_start = get_compile_time_arg_val(1);
    constexpr uint32_t beta_accessor_ct_start = get_compile_time_arg_val(2);
    // TensorAccessorArgs for input tensor start at index 3
    constexpr auto input_accessor_args = TensorAccessorArgs<3>();

    // Runtime args
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_blocks = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);
    uint32_t Wt = get_arg_val<uint32_t>(3);
    uint32_t has_gamma = get_arg_val<uint32_t>(4);
    uint32_t has_beta = get_arg_val<uint32_t>(5);
    uint32_t gamma_addr = get_arg_val<uint32_t>(6);
    uint32_t beta_addr = get_arg_val<uint32_t>(7);
    uint32_t eps_packed = get_arg_val<uint32_t>(8);

    // Input TensorAccessor
    const auto input_accessor = TensorAccessor(input_accessor_args, src_addr, stick_size);

    // ============ 1. Prepare reduce scaler: 1/W ============
    float scaler_val = 1.0f / static_cast<float>(Wt * 32);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_reduce_scaler>(scaler_val);

    // ============ 2. Prepare epsilon tile ============
    uint16_t eps_bf16 = static_cast<uint16_t>(eps_packed >> 16);
    uint32_t eps_f32_bits = static_cast<uint32_t>(eps_bf16) << 16;
    union {
        uint32_t u;
        float f;
    } eps_conv;
    eps_conv.u = eps_f32_bits;
    float eps_float = eps_conv.f;
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_float);

    // ============ 3. Read gamma/beta tiles ============
    // Gamma and beta are TILE_LAYOUT tensors (tilized on host side).
    // Each is (1,1,32,W) padded = Wt tiles. Read Wt tiles into the CB.
    if constexpr (gamma_accessor_ct_start > 0) {
        if (has_gamma) {
            constexpr auto gamma_accessor_args = TensorAccessorArgs<gamma_accessor_ct_start>();
            const auto gamma_accessor = TensorAccessor(gamma_accessor_args, gamma_addr, TILE_SIZE);

            cb_reserve_back(cb_gamma, Wt);
            uint32_t gamma_l1_addr = get_write_ptr(cb_gamma);
            for (uint32_t t = 0; t < Wt; ++t) {
                uint64_t noc_addr = gamma_accessor.get_noc_addr(t);
                noc_async_read(noc_addr, gamma_l1_addr, TILE_SIZE);
                gamma_l1_addr += TILE_SIZE;
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma, Wt);
        }
    }

    if constexpr (beta_accessor_ct_start > 0) {
        if (has_beta) {
            constexpr auto beta_accessor_args = TensorAccessorArgs<beta_accessor_ct_start>();
            const auto beta_accessor = TensorAccessor(beta_accessor_args, beta_addr, TILE_SIZE);

            cb_reserve_back(cb_beta, Wt);
            uint32_t beta_l1_addr = get_write_ptr(cb_beta);
            for (uint32_t t = 0; t < Wt; ++t) {
                uint64_t noc_addr = beta_accessor.get_noc_addr(t);
                noc_async_read(noc_addr, beta_l1_addr, TILE_SIZE);
                beta_l1_addr += TILE_SIZE;
            }
            noc_async_read_barrier();
            cb_push_back(cb_beta, Wt);
        }
    }

    // ============ 4. Main loop: read RM sticks ============
    uint32_t stick_id = start_stick_id;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        cb_reserve_back(cb_rm_in, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_rm_in);

        for (uint32_t stick = 0; stick < 32; ++stick) {
            uint64_t noc_addr = input_accessor.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();

        cb_push_back(cb_rm_in, Wt);
    }
}
