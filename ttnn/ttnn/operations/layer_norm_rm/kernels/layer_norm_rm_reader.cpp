// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for layer_norm_rm — streams the input tensor 3 times (one full sweep
// per pass), plus one-shot reads for the scaler tile(s) and optional gamma/beta
// sticks.
//
// Per pass × per tile-row:
//   RM input:    push 32 sticks of padded_row_bytes into cb_input_sticks.
//   TILE input:  push Wt tiles into cb_input_tiles directly.
//
// The scaler tile pair (full + partial when partial_w) is pushed once at
// startup and never popped by reduce<>; the compute kernel pops it at exit.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t total_tile_rows = get_compile_time_arg_val(1);
    constexpr uint32_t padded_row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t input_row_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t inv_W_bits = get_compile_time_arg_val(4);
    constexpr uint32_t has_partial_w = get_compile_time_arg_val(5);
    constexpr uint32_t partial_w = get_compile_time_arg_val(6);
    constexpr uint32_t is_rm_input = get_compile_time_arg_val(7);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(8);
    constexpr uint32_t has_beta = get_compile_time_arg_val(9);

    constexpr auto input_args = TensorAccessorArgs<10>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    uint32_t beta_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_input_sticks = 0;
    constexpr uint32_t cb_input_tiles = 1;
    constexpr uint32_t cb_gamma_sticks = 2;
    constexpr uint32_t cb_beta_sticks = 3;
    constexpr uint32_t cb_scaler = 4;
    constexpr uint32_t TILE_H = 32;

    // ---------------- Scaler ----------------
    {
        float inv_W = __builtin_bit_cast(float, inv_W_bits);
        if constexpr (has_partial_w) {
            dataflow_kernel_lib::prepare_partial_reduce_scalers<
                cb_scaler,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                partial_w>(inv_W);
        } else {
            dataflow_kernel_lib::
                prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(inv_W);
        }
    }

    // ---------------- One-shot gamma / beta ----------------
    if constexpr (has_gamma) {
        const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, padded_row_bytes);
        cb_reserve_back(cb_gamma_sticks, 1);
        uint32_t l1_addr = get_write_ptr(cb_gamma_sticks);
        uint64_t noc_addr = gamma_accessor.get_noc_addr(0);
        noc_async_read(noc_addr, l1_addr, input_row_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_gamma_sticks, 1);
    }

    if constexpr (has_beta) {
        const auto beta_accessor = TensorAccessor(beta_args, beta_addr, padded_row_bytes);
        cb_reserve_back(cb_beta_sticks, 1);
        uint32_t l1_addr = get_write_ptr(cb_beta_sticks);
        uint64_t noc_addr = beta_accessor.get_noc_addr(0);
        noc_async_read(noc_addr, l1_addr, input_row_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_beta_sticks, 1);
    }

    // ---------------- Three passes over the input ----------------
    // Order: tr-outer, pass-inner — matches the compute kernel's per-tile-row
    // loop that runs all 3 passes for one row before moving to the next.
    if constexpr (is_rm_input) {
        // RM input: stream sticks for each tile-row × 3 passes.
        const auto input_accessor = TensorAccessor(input_args, input_addr, padded_row_bytes);
        for (uint32_t tr = 0; tr < total_tile_rows; ++tr) {
            uint32_t row_base = tr * TILE_H;
            for (uint32_t pass = 0; pass < 3; ++pass) {
                for (uint32_t row = 0; row < TILE_H; ++row) {
                    cb_reserve_back(cb_input_sticks, 1);
                    uint32_t l1_addr = get_write_ptr(cb_input_sticks);
                    uint64_t noc_addr = input_accessor.get_noc_addr(row_base + row);
                    noc_async_read(noc_addr, l1_addr, input_row_bytes);
                    noc_async_read_barrier();
                    cb_push_back(cb_input_sticks, 1);
                }
            }
        }
    } else {
        // TILE input: stream tiles for each tile-row × 3 passes.
        constexpr uint32_t tile_bytes_v = get_tile_size(cb_input_tiles);
        const auto input_accessor = TensorAccessor(input_args, input_addr, tile_bytes_v);
        for (uint32_t tr = 0; tr < total_tile_rows; ++tr) {
            for (uint32_t pass = 0; pass < 3; ++pass) {
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    uint32_t tile_id = tr * Wt + wt;
                    cb_reserve_back(cb_input_tiles, 1);
                    uint32_t l1_addr = get_write_ptr(cb_input_tiles);
                    noc_async_read_tile(tile_id, input_accessor, l1_addr);
                    noc_async_read_barrier();
                    cb_push_back(cb_input_tiles, 1);
                }
            }
        }
    }
}
