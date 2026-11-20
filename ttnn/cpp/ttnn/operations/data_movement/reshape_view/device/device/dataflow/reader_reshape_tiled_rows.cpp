// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the tiled reshape that keeps the last dim. For each output tile of this core it loads, in
// order, the input tile of each run of rows (hostdevcommon/rows_common.hpp). The row counts and the
// output tile count are common runtime args, so the binary and the program serve every row count.

#include <stdint.h>

#include "experimental/kernel_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"

#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/hostdevcommon/rows_common.hpp"

using tt::data_movement::common::enhanced_noc_async_read;
namespace rows = ttnn::prim::detail::reshape_rows;

void kernel_main() {
    const uint32_t core_idx = get_arg(args::core_idx);
    const uint32_t num_output_pages = get_arg(args::num_output_pages);
    const uint32_t in_rows = get_arg(args::in_rows);
    const uint32_t out_rows = get_arg(args::out_rows);

    constexpr uint32_t Tile_size_bytes = get_arg(args::Tile_size_bytes);
    constexpr uint32_t tiles_per_row = get_arg(args::tiles_per_row);
    constexpr uint32_t num_cores = get_arg(args::num_cores);

    const rows::Rows r{
        in_rows,
        ttnn::runtime_length::div_up(in_rows, rows::kTileRows),
        out_rows,
        ttnn::runtime_length::div_up(out_rows, rows::kTileRows),
        tiles_per_row};
    const uint32_t start = ttnn::runtime_length::balanced_start(core_idx, num_output_pages, num_cores);
    const uint32_t end = ttnn::runtime_length::balanced_end(core_idx, num_output_pages, num_cores);

    const auto input_addr_gen = TensorAccessor(tensor::input);

    Noc noc;
    DataflowBuffer dfb_in_tiles(dfb::in_tiles);
    for (uint32_t out_page = start; out_page < end; ++out_page) {
        rows::for_each_run(r, out_page, [&](const rows::Run& run) {
            dfb_in_tiles.reserve_back(1);
            const uint32_t write_addr = dfb_in_tiles.get_write_ptr();
            enhanced_noc_async_read<Tile_size_bytes, true>(
                noc, input_addr_gen.get_noc_addr(run.in_page), write_addr, Tile_size_bytes);
            noc.async_read_barrier();
            dfb_in_tiles.push_back(1);
        });
    }
}
