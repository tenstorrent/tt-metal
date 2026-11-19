// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "rect.hpp"

#include "weights.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t tile_bytes = CT_ARG("tile_bytes_bf16");
    constexpr uint32_t i_tiles = CT_ARG("i_tiles");
    constexpr uint32_t b_tiles = CT_ARG("b_tiles_per_core");
    constexpr uint32_t chips = CT_ARG("chips");
    constexpr uint32_t a_active = CT_ARG("a_active");
    const Rect a_grid = RECT("a_grid");
    const uint32_t chip = RT_ARG("chip");
    const uint32_t first_column = RT_ARG("index") * b_tiles;
    CircularBuffer a(CT_ARG("cb_a"));
    CircularBuffer h_segment(CT_ARG("cb_h"));
    Semaphore<> a_free[2] = {Semaphore<>(CT_ARG("sem_a_free")), Semaphore<>(CT_ARG("sem_a_free") + 1)};
    Semaphore<> a_valid[2] = {Semaphore<>(CT_ARG("sem_a_valid")), Semaphore<>(CT_ARG("sem_a_valid") + 1)};

    const auto h = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_h")>(), RT_ARG("h_addr"), tile_bytes);
    const auto w_down =
        TensorAccessor(TensorAccessorArgs<CT_ARG("ta_w_down")>(), RT_ARG("w_down_addr"), CT_ARG("tile_bytes_bfp4"));
    load_resident_weights(
        noc, w_down, CT_ARG("cb_wd"), CT_ARG("tile_bytes_bfp4"), i_tiles, CT_ARG("h_tiles"), first_column, b_tiles);

    for (uint32_t m = 0; m < CT_ARG("rows"); m++) {
        const uint32_t slot = m & 1;
        if (m % chips == chip) {
            h_segment.reserve_back(b_tiles);
            for (uint32_t t = 0; t < b_tiles; t++) {
                noc.async_read(
                    h,
                    CoreLocalMem<uint32_t>(h_segment.get_write_ptr() + t * tile_bytes),
                    tile_bytes,
                    {.page_id = m * CT_ARG("h_tiles") + first_column + t},
                    {});
            }
            noc.async_read_barrier();
            h_segment.push_back(b_tiles);
        }
        a.reserve_back(i_tiles);
        a_grid.signal(noc, a_free[slot], 1);
        a_valid[slot].wait_min(a_active * (m / 2 + 1));
        a.push_back(i_tiles);
    }
}
