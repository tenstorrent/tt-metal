// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// layer_norm_rm writer (BRISC).
//
// Per work-item: drain Wt tile-pages from cb_output_tiles back to DRAM as 32
// row-major sticks. write_sticks_after_untilize is the partner-side of the
// reader's TILE-granularity read; both use tile-sized CB pages.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_output_tiles = 16;
}  // namespace

void kernel_main() {
    constexpr uint32_t row_bytes = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(1);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(2);

    const auto accessor = TensorAccessor(dst_args, output_addr);

    for (uint32_t i = 0; i < num_tile_rows; ++i) {
        const uint32_t start_page = (start_tile_row + i) * 32u;
        dataflow_kernel_lib::write_sticks_after_untilize<cb_output_tiles>(
            accessor, /*total_num_rows=*/32u, row_bytes, start_page);
    }
}
