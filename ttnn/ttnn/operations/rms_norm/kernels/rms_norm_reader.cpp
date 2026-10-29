// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm reader: generates the reduce scaler(s) once, then loads R x Wt tile blocks of x.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t block_row_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t input_tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t partial_w = get_compile_time_arg_val(5);
    constexpr auto input_args = TensorAccessorArgs<6>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t core_rows = get_arg_val<uint32_t>(2);

    using dataflow_kernel_lib::PoolType;
    using dataflow_kernel_lib::ReduceDim;
    if constexpr (partial_w != 0) {
        dataflow_kernel_lib::prepare_partial_reduce_scalers<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW, partial_w>(
            1.0f);
    } else {
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f);
    }

    const auto input_accessor = TensorAccessor(input_args, input_addr, input_tile_bytes);

    // Nominal block quantum keeps the CB ring aligned; a ragged last block reads fewer tiles.
    constexpr uint32_t block_tiles = block_row_tiles * tensor_w_tiles;
    for (uint32_t block_row = 0; block_row < core_rows; block_row += block_row_tiles) {
        const uint32_t rows = (core_rows - block_row) < block_row_tiles ? (core_rows - block_row) : block_row_tiles;
        const uint32_t first_tile = (row_start + block_row) * tensor_w_tiles;
        const uint32_t num_tiles = rows * tensor_w_tiles;

        cb_reserve_back(cb_input_tiles, block_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_input_tiles);
        for (uint32_t tile = 0; tile < num_tiles; ++tile) {
            noc_async_read(input_accessor.get_noc_addr(first_tile + tile), l1_write_addr, input_tile_bytes);
            l1_write_addr += input_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_tiles, block_tiles);
    }
}
