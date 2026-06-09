// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for toy_variance.
//
// Variance via Var(x) = E[(x - E[x])^2] is a two-pass algorithm:
//   Pass 1: stream x  → compute mean
//   Pass 2: stream x  → compute (x - mean)^2 → mean
//
// The reader streams the input tensor TWICE through cb_in. Each pass pushes
// tiles in the order the streaming reduce expects:
//   for b in [0, NUM_BLOCKS): for ht in [0, Ht):
//     for wt in [0, BLOCK_SIZE): tile_id = ht*Wt + b*BLOCK_SIZE + wt
//
// The scaler tile (1/N for SUM-reduce-as-mean) is pushed once at startup;
// reduce<> waits on it but never pops, so the same tile serves both passes.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(3);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(4);  // 1/N as fp32 bits
    constexpr uint32_t has_partial_w = get_compile_time_arg_val(5);
    constexpr uint32_t partial_w = get_compile_time_arg_val(6);  // valid positions in last W-tile
    constexpr auto src_args = TensorAccessorArgs<7>();

    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_scaler = 2;

    // Scaler = 1/N → SUM reduce produces means directly. For non-tile-aligned
    // W, also emit a partial scaler tile that zeros out positions beyond
    // partial_w; the compute kernel selects it for the last W-tile of the
    // last block via ReducePartialScaler::last_tile_at(1).
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    if constexpr (has_partial_w) {
        dataflow_kernel_lib::prepare_partial_reduce_scalers<
            cb_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            partial_w>(scaler_f);
    } else {
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            scaler_f);
    }

    uint32_t tile_bytes = get_tile_size(cb_in);
    const auto accessor = TensorAccessor(src_args, src_addr, tile_bytes);

    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                for (uint32_t wt = 0; wt < BLOCK_SIZE; ++wt) {
                    uint32_t tile_id = ht * Wt + b * BLOCK_SIZE + wt;
                    cb_reserve_back(cb_in, 1);
                    uint32_t l1_write_addr = get_write_ptr(cb_in);
                    noc_async_read_tile(tile_id, accessor, l1_write_addr);
                    noc_async_read_barrier();
                    cb_push_back(cb_in, 1);
                }
            }
        }
    }
}
