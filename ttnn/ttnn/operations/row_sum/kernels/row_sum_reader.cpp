// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row Sum - Reader Kernel
// Prepares the reduce scaler, then reads input data from DRAM.
// Supports both tiled (tile streaming) and row-major (read_sticks_for_tilize) inputs.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

using namespace ckernel;

void kernel_main() {
    // Compile-time args
    constexpr bool input_is_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t arg3 = get_compile_time_arg_val(3);  // num_input_tiles (tiled) or H (RM)
    constexpr uint32_t arg4 = get_compile_time_arg_val(4);  // 0 (tiled) or input_row_bytes (RM)

    // TensorAccessor compile-time args start at index 5
    constexpr auto accessor_args = TensorAccessorArgs<5>();

    // Runtime args
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);

    // Step 1: Prepare scaler tile (1.0 for SUM)
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>();

    // Step 2: Read input data
    if constexpr (input_is_rm) {
        // Row-major: use read_sticks_for_tilize helper (TILE granularity)
        constexpr uint32_t H = arg3;
        constexpr uint32_t row_bytes = arg4;
        const auto accessor = TensorAccessor(accessor_args, input_addr);
        dataflow_kernel_lib::read_sticks_for_tilize<cb_in>(accessor, H, row_bytes, start_page);
    } else {
        // Tiled: stream tiles one at a time
        constexpr uint32_t num_tiles = arg3;
        const auto accessor = TensorAccessor(accessor_args, input_addr);

        for (uint32_t i = start_page; i < start_page + num_tiles; i++) {
            cb_reserve_back(cb_in, 1);
            uint32_t l1_addr = get_write_ptr(cb_in);
            noc_async_read_tile(i, accessor, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in, 1);
        }
    }
}
