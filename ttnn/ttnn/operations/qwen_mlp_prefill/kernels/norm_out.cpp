// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "rect.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t row_tiles = CT_ARG("h_tiles");
    constexpr uint32_t row_bytes = row_tiles * CT_ARG("tile_bytes_act");
    constexpr uint32_t chunk_bytes = CT_ARG("mcast_chunk_tiles") * CT_ARG("tile_bytes_act");
    constexpr uint32_t depth = CT_ARG("n_depth");
    const Rect a_grid = RECT("a_grid");
    const uint32_t a_slots = arena_base(CT_ARG("cb_norm_scaler"));
    CircularBuffer out(CT_ARG("cb_norm_out"));

    for (uint32_t m = RT_ARG("index"); m < CT_ARG("rows"); m += CT_ARG("norm_cores")) {
        const uint32_t slot = m % depth;
        out.wait_front(row_tiles);

        Semaphore<>(CT_ARG("sem_n_free") + slot).wait_min(CT_ARG("a_active") * (m / depth + 1));

        for (uint32_t offset = 0; offset < row_bytes; offset += chunk_bytes) {
            a_grid.write(noc, out.get_read_ptr() + offset, a_slots + slot * row_bytes + offset, chunk_bytes);
        }
        noc.async_writes_flushed();
        Semaphore<> n_valid(CT_ARG("sem_n_valid") + slot);
        a_grid.signal(noc, n_valid, 1);
        noc.async_write_barrier();

        out.pop_front(row_tiles);
    }
}
