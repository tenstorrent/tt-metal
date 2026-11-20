// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISCV_0 of every core, on NOC_0: the softmax's reduce scalers, as ttnn.softmax's reader prepares them; then each
// block's input tiles, addressed by page id, so the input may be interleaved or sharded. The writer brings the
// weight in on the other NoC, so the first block's reads start at once.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t input_cb_id = get_named_compile_time_arg_val("input_cb");
    constexpr uint32_t max_scaler_cb_id = get_named_compile_time_arg_val("max_scaler_cb");
    constexpr uint32_t sum_scaler_cb_id = get_named_compile_time_arg_val("sum_scaler_cb");
    constexpr auto input_args = TensorAccessorArgs<0>();

    const uint32_t input_address = get_arg_val<uint32_t>(0);
    const uint32_t first_block = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);

    const Noc noc;
    DataflowBuffer input_cb(input_cb_id);
    const auto input = TensorAccessor(input_args, input_address);
    constexpr uint32_t input_tile_bytes = get_tile_size(input_cb_id);

    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<max_scaler_cb_id, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<sum_scaler_cb_id, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    for (uint32_t block = first_block; block < first_block + num_blocks; ++block) {
        input_cb.reserve_back(hidden_tiles);
        for (uint32_t tile = 0; tile < hidden_tiles; ++tile) {
            noc.async_read(
                input,
                input_cb,
                input_tile_bytes,
                {.page_id = block * hidden_tiles + tile},
                {.offset_bytes = tile * input_tile_bytes});
        }
        noc.async_read_barrier();
        input_cb.push_back(hidden_tiles);
    }
}
