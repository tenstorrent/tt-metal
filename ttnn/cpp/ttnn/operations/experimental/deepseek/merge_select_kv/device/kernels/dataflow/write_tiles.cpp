// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TILED_OUTPUT: writes this core's row of tiles (its 32 output rows, tilized) to output tiles
// [first_tile, first_tile + tiles_per_row).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t tiles_per_row = get_arg(args::tiles_per_row);
    const uint32_t first_tile = get_arg(args::first_tile);

    const auto output = TensorAccessor(tensor::output);  // TILE, interleaved
    DataflowBuffer tiled_dfb(dfb::tiled);                // consumer <- tilize_rows

    Noc noc;
    tiled_dfb.wait_front(tiles_per_row);
    const uint32_t tile_bytes = tiled_dfb.get_entry_size();
    for (uint32_t t = 0; t < tiles_per_row; ++t) {
        noc.async_write(tiled_dfb, output, tile_bytes, {.offset_bytes = t * tile_bytes}, {.page_id = first_tile + t});
    }
    noc.async_write_barrier();
    tiled_dfb.pop_front(tiles_per_row);
}
