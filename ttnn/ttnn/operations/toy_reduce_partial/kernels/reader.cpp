// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Unified reader for toy_reduce_partial.
//
// Generates scaler tile(s) and reads input tiles. Uses pool-type-aware
// prepare_partial_reduce_scalers overload which automatically selects the
// correct scaler layout (col-0 for matmul path, row-0 for reduce_tile path).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);

    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(1);
    constexpr uint32_t has_partial = get_compile_time_arg_val(2);
    constexpr uint32_t partial_dim = get_compile_time_arg_val(3);  // partial_w or partial_h
    constexpr uint32_t reduce_row_mode = get_compile_time_arg_val(4);
    constexpr uint32_t pool_type_sum = get_compile_time_arg_val(5);
    constexpr auto src_args = TensorAccessorArgs<6>();

    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_scaler = 2;

    constexpr auto reduce_dim = reduce_row_mode ? ckernel::ReduceDim::REDUCE_ROW : ckernel::ReduceDim::REDUCE_COL;

    float scaler_f = __builtin_bit_cast(float, scaler_bits);

    if constexpr (pool_type_sum) {
        // Pool-type-aware: auto-dispatches to col-0 fill for SUM REDUCE_ROW (matmul path)
        if constexpr (has_partial) {
            dataflow_kernel_lib::
                prepare_partial_reduce_scalers<cb_scaler, ckernel::PoolType::SUM, reduce_dim, partial_dim>(scaler_f);
        } else {
            dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, reduce_dim>(scaler_f);
        }
    } else {
        // MAX always uses reduce_tile layout — dimension-aware overload handles axis dispatch
        if constexpr (has_partial) {
            dataflow_kernel_lib::prepare_partial_reduce_scalers<cb_scaler, reduce_dim, partial_dim>(scaler_f);
        } else {
            dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_f);
        }
    }

    // Stream input tiles
    uint32_t tile_bytes = get_tile_size(cb_in);
    const auto accessor = TensorAccessor(src_args, src_addr, tile_bytes);

    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        cb_reserve_back(cb_in, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_in);
        noc_async_read_tile(i, accessor, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
