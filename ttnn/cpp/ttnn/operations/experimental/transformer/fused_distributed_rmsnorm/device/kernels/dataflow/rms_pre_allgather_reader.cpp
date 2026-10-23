// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel reads the layernorm inputs from interleaved dram.
 */

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/debug/assert.h"

void kernel_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_scalar_cb = get_compile_time_arg_val(1);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(2);
    constexpr uint32_t block_size = get_compile_time_arg_val(3);
    constexpr auto input_args = TensorAccessorArgs<4>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);  // Source address in dram
    const uint32_t tile_row_start = get_arg_val<uint32_t>(1);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(2);

    const uint32_t input_tile_bytes = get_tile_size(input_cb);

    const auto input_accessor = TensorAccessor(input_args, input_addr);

    Noc noc;
    CircularBuffer cb_input(input_cb);

    // Generate constant tiles for reduce scalar
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<reduce_scalar_cb, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        uint32_t input_tile_idx = tile_row * num_tile_cols;
        // read input tiles
        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            cb_input.reserve_back(block_size);
            uint32_t input_wr_offset = 0;
            for (uint32_t r = 0; r < block_size && col_tile + r < num_tile_cols; r++) {
                noc.async_read(
                    input_accessor,
                    cb_input,
                    input_tile_bytes,
                    {.page_id = input_tile_idx},
                    {.offset_bytes = input_wr_offset});
                input_wr_offset += input_tile_bytes;
                input_tile_idx++;
            }
            noc.async_read_barrier();
            cb_input.push_back(block_size);
        }
    }
}
