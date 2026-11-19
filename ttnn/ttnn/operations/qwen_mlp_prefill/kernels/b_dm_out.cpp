// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "rect.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t tile_bytes = CT_ARG("tile_bytes_bf16");
    constexpr uint32_t b_tiles = CT_ARG("b_tiles_per_core");
    constexpr uint32_t chips = CT_ARG("chips");
    const uint32_t chip = RT_ARG("chip");
    const uint32_t first_column = RT_ARG("index") * b_tiles;
    const auto y = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_y")>(), RT_ARG("y_addr"), tile_bytes);
    CircularBuffer y_segment(CT_ARG("cb_y"));

    for (uint32_t m = 0; m < CT_ARG("rows"); m++) {
        if (m % chips == chip) {
            y_segment.wait_front(b_tiles);
            for (uint32_t t = 0; t < b_tiles; t++) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(y_segment.get_read_ptr() + t * tile_bytes),
                    y,
                    tile_bytes,
                    {},
                    {.page_id = m * CT_ARG("h_tiles") + first_column + t});
            }
            noc.async_writes_flushed();
            y_segment.pop_front(b_tiles);
        }
    }
    noc.async_write_barrier();
}
