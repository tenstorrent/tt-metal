// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Interleaved reader of the multi-core untilize whose tile range follows the common row count:
// this core's tile rows times the tiles per row, so a new height rewrites one word.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/untilize/device/kernels/untilize_row_split.hpp"

void kernel_main() {
    const uint32_t core_idx = get_arg(args::core_idx);
    const uint32_t num_tile_rows = get_arg(args::num_tile_rows);
    constexpr uint32_t tiles_per_row = get_arg(args::tiles_per_row);
    constexpr uint32_t num_cores = get_arg(args::num_cores);

    const auto [row_start, row_end] = untilize_row_range(core_idx, num_tile_rows, num_cores);
    const uint32_t start_id = row_start * tiles_per_row;
    const uint32_t end_page_id = row_end * tiles_per_row;

    Noc noc;
    DataflowBuffer dfb_in(dfb::in);

    const uint32_t tile_bytes = dfb_in.get_tile_size();

    const auto s = TensorAccessor(tensor::src);

    for (uint32_t page_id = start_id; page_id < end_page_id; ++page_id) {
        dfb_in.reserve_back(1);
        noc.async_read(s, dfb_in, tile_bytes, {.page_id = page_id, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in.push_back(1);
    }
}
