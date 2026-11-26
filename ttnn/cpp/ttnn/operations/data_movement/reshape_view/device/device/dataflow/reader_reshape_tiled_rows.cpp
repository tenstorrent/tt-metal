// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the tiled reshape that keeps the last dim (see reshape_tiled_rows.hpp): reads the input tiles the runs
// of its output tiles come from, once per change of input tile, in the order the writer consumes them. The
// shape arrives as runtime args, so the binary does not depend on it.

#include <stdint.h>

#include "experimental/kernel_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"

#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/device/dataflow/reshape_tiled_rows.hpp"

using tt::data_movement::common::enhanced_noc_async_read;
namespace rows = ttnn::prim::reshape_rows;

void kernel_main() {
    const uint32_t start_output_page = get_arg(args::start_output_page);
    const uint32_t end_output_page = get_arg(args::end_output_page);
    const rows::Geometry g{
        .in_rows = get_arg(args::in_rows),
        .out_rows = get_arg(args::out_rows),
        .c_tiles = get_arg(args::c_tiles),
        .in_row_tiles = get_arg(args::in_row_tiles),
        .out_row_tiles = get_arg(args::out_row_tiles),
    };
    constexpr uint32_t Tile_size_bytes = get_arg(args::Tile_size_bytes);

    const auto input_addr_gen = TensorAccessor(tensor::input);
    Noc noc;
    DataflowBuffer dfb_in_tiles(dfb::in_tiles);

    // The input tile the previous run read, kept across output tiles (see rows::output_tile_at).
    uint32_t previous = UINT32_MAX;
    for (uint32_t k = start_output_page; k < end_output_page; ++k) {
        rows::for_each_run(rows::output_tile_at(k, g), g, [&](uint32_t in_tile, uint32_t, uint32_t, uint32_t) {
            if (in_tile == previous) {
                return;
            }
            previous = in_tile;
            dfb_in_tiles.reserve_back(1);
            const uint32_t write_addr = dfb_in_tiles.get_write_ptr();
            enhanced_noc_async_read<Tile_size_bytes, true>(
                noc, input_addr_gen.get_noc_addr(in_tile), write_addr, Tile_size_bytes);
            noc.async_read_barrier();
            dfb_in_tiles.push_back(1);
        });
    }
}
