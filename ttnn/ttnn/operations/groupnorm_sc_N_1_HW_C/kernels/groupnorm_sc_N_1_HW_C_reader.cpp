// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for groupnorm_sc_N_1_HW_C (single-core GroupNorm over (N, 1, HW, C)).
//
// Startup:
//   - push the reduce scaler tile once: 1/sqrt(HW * Cg). REDUCE_SCALAR applies
//     the scaler twice (row then col), so SUM * (1/sqrt(N))^2 = mean.
//   - read gamma / beta (Wt tiles each, host-tilized (1,1,1,C) -> 1 x Wt tile
//     row) once; they persist in their CBs for the whole kernel (HeldBulk).
//
// Per (n, g) group the compute kernel makes THREE streaming passes over the
// same Ht x Wg tile slab (mean, variance, normalize), so the reader streams
// the slab three times. Tile index: n*Ht*Wt + r*Wt + g*Wg + c.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t Wg = get_compile_time_arg_val(2);
    constexpr uint32_t G = get_compile_time_arg_val(3);
    constexpr uint32_t N = get_compile_time_arg_val(4);
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(5) != 0;
    constexpr bool HAS_BETA = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t inv_sqrt_n_bits = get_compile_time_arg_val(7);

    // Accessors declared unconditionally, chained offsets (placeholders when absent).
    constexpr auto input_args = TensorAccessorArgs<8>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_input_tiles = 0;
    constexpr uint32_t cb_gamma_tiles = 1;
    constexpr uint32_t cb_beta_tiles = 2;
    constexpr uint32_t cb_scaler = 8;

    // Scaler: non-standard value (1/sqrt(HW*Cg) combines the SCALAR double-apply
    // with the 1/N mean factor) -> prepare_reduce_scaler, pool-type-aware overload.
    const float inv_sqrt_n = __builtin_bit_cast(float, inv_sqrt_n_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_SCALAR>(
        inv_sqrt_n);

    const uint32_t tile_bytes = get_tile_size(cb_input_tiles);
    const auto input = TensorAccessor(input_args, input_addr, tile_bytes);

    // gamma / beta: Wt tiles each, read once, pushed in bulk; never re-read.
    if constexpr (HAS_GAMMA) {
        const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma_tiles);
        const auto gamma = TensorAccessor(gamma_args, gamma_addr, gamma_tile_bytes);
        cb_reserve_back(cb_gamma_tiles, Wt);
        uint32_t l1_addr = get_write_ptr(cb_gamma_tiles);
        for (uint32_t t = 0; t < Wt; ++t) {
            noc_async_read_tile(t, gamma, l1_addr);
            l1_addr += gamma_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma_tiles, Wt);
    }
    if constexpr (HAS_BETA) {
        const uint32_t beta_tile_bytes = get_tile_size(cb_beta_tiles);
        const auto beta = TensorAccessor(beta_args, beta_addr, beta_tile_bytes);
        cb_reserve_back(cb_beta_tiles, Wt);
        uint32_t l1_addr = get_write_ptr(cb_beta_tiles);
        for (uint32_t t = 0; t < Wt; ++t) {
            noc_async_read_tile(t, beta, l1_addr);
            l1_addr += beta_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_beta_tiles, Wt);
    }

    // Group loop: 3 streaming passes over the Ht x Wg slab per group.
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t group_base = n * Ht * Wt + g * Wg;
            for (uint32_t pass = 0; pass < 3; ++pass) {
                for (uint32_t r = 0; r < Ht; ++r) {
                    for (uint32_t c = 0; c < Wg; ++c) {
                        const uint32_t tile_id = group_base + r * Wt + c;
                        cb_reserve_back(cb_input_tiles, 1);
                        noc_async_read_tile(tile_id, input, get_write_ptr(cb_input_tiles));
                        noc_async_read_barrier();
                        cb_push_back(cb_input_tiles, 1);
                    }
                }
            }
        }
    }
}
