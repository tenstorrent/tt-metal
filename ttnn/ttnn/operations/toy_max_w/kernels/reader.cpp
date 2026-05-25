// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for toy_max_w.
//
// Single streaming pass: pushes input tiles in the order accumulate_reduce
// expects:
//   for b in [0, NUM_BLOCKS): for ht in [0, Ht):
//     for wt in [0, BLOCK_SIZE): tile_id = ht*Wt + b*BLOCK_SIZE + wt
//
// For MAX, the standard reduce scaler is 1.0 — calculate_and_prepare_* picks
// the right value from the pool type. When W is not tile-aligned, the
// partial-scaler variant emits two tiles (full at index 0, partial at index 1);
// for MAX the partial tile zeros the valid positions and fills padded
// positions with -inf so they never win the max.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(3);
    constexpr uint32_t has_partial_w = get_compile_time_arg_val(4);
    constexpr uint32_t partial_w = get_compile_time_arg_val(5);  // valid positions in last W-tile
    constexpr auto src_args = TensorAccessorArgs<6>();

    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_scaler = 2;

    if constexpr (has_partial_w) {
        dataflow_kernel_lib::calculate_and_prepare_partial_reduce_scalers<
            cb_scaler,
            ckernel::PoolType::MAX,
            ckernel::ReduceDim::REDUCE_ROW,
            partial_w>();
    } else {
        dataflow_kernel_lib::
            calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    }

    uint32_t tile_bytes = get_tile_size(cb_in);
    const auto accessor = TensorAccessor(src_args, src_addr, tile_bytes);

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
