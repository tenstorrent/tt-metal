// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the tiled reshape that keeps the last dim (see reshape_tiled_rows.hpp): assembles each output tile in
// an L1 scratch page from the runs of the input tiles the reader pushes, then writes it out. An input tile stays at
// the DFB's front until a run needs another, so consecutive output tiles can share it. The shape arrives as
// runtime args, so the binary does not depend on it.

#include <stdint.h>

#include "experimental/kernel_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"

#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/device/dataflow/reshape_tiled_rows.hpp"

using namespace tt::data_movement::common;
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
    constexpr uint32_t element_sz_bytes = get_arg(args::element_sz_bytes);
    constexpr uint32_t run_row_bytes = rows::FACE_ROWS * element_sz_bytes;

    const auto output_addrgen = TensorAccessor(tensor::output);
    Noc noc;
    DataflowBuffer dfb_in_tiles(dfb::in_tiles);
    Scratchpad<uint8_t> working(scratch::working);
    const auto working_mem = working.local_mem();

    // The input tile at the DFB's front, kept across output tiles like the reader's (see rows::output_tile_at).
    uint32_t previous = UINT32_MAX;
    uint32_t input_base_addr = 0;
    for (uint32_t k = start_output_page; k < end_output_page; ++k) {
        const uint32_t out_page = rows::output_tile_at(k, g);
        rows::for_each_run(out_page, g, [&](uint32_t in_tile, uint32_t in_row, uint32_t out_row, uint32_t n) {
            if (in_tile != previous) {
                if (previous != UINT32_MAX) {
                    noc.async_write_barrier();
                    dfb_in_tiles.pop_front(1);
                }
                dfb_in_tiles.wait_front(1);
                input_base_addr = dfb_in_tiles.get_read_ptr();
                previous = in_tile;
            }
            for (uint32_t face_col = 0; face_col < 2; ++face_col) {
                const auto out_mem = working_mem + rows::row_offset(out_row, face_col) * element_sz_bytes;
                const uint32_t in_addr = input_base_addr + rows::row_offset(in_row, face_col) * element_sz_bytes;
                tt_memmove<false, true, false, Tile_size_bytes>(
                    noc, out_mem, CoreLocalMem<uint8_t>(in_addr), n * run_row_bytes);
            }
        });
        // Rows past out_rows keep whatever the scratch page held, as with the page-map writer; reshape.cpp fills
        // tile padding only on an explicit pad_value or for block-float dtypes.
        noc.async_write_barrier();
        enhanced_noc_async_write<Tile_size_bytes, true>(
            noc, working_mem, output_addrgen.get_noc_addr(out_page), Tile_size_bytes);
        noc.async_write_barrier();
    }
    if (previous != UINT32_MAX) {
        dfb_in_tiles.pop_front(1);
    }
}
