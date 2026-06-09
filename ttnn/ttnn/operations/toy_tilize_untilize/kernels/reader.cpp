// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_rm_in = 0;
    constexpr uint32_t row_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t use_row_granularity = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t total_num_rows = get_arg_val<uint32_t>(1);

    const auto accessor = TensorAccessor(src_args, src_addr);

    if constexpr (use_row_granularity) {
        dataflow_kernel_lib::read_sticks_for_tilize<cb_rm_in, dataflow_kernel_lib::TilizeGranularity::ROW>(
            accessor, total_num_rows, row_bytes);
    } else {
        dataflow_kernel_lib::read_sticks_for_tilize<cb_rm_in, dataflow_kernel_lib::TilizeGranularity::TILE>(
            accessor, total_num_rows, row_bytes);
    }
}
