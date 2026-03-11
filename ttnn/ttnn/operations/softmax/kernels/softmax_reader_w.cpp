// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax Reader Kernel (dim=-1, width reduction)
// Sequential tile reads: Wt tiles per row, prepare reduce scaler in c_2

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t cb_input = 0;
constexpr uint32_t cb_scaler = 2;

void kernel_main() {
    // Compile-time args
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(0);
    constexpr auto tensor_args = TensorAccessorArgs<1>();

    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    if (num_tiles == 0) {
        return;
    }

    // Prepare reduce scaler tile in c_2 (1.0 for SUM/MAX)
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(1.0f);

    // Set up tensor accessor for reading input tiles
    const uint32_t page_size = get_tile_size(cb_input);
    const auto accessor = TensorAccessor(tensor_args, src_addr, page_size);

    // Read all assigned tiles sequentially
    uint32_t tile_id = start_id;
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_reserve_back(cb_input, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_input);
        noc_async_read_tile(tile_id, accessor, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_input, 1);
        ++tile_id;
    }
}
