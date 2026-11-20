// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "rect.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t row_tiles = CT_ARG("n_row_tiles");
    constexpr uint32_t row_bytes = row_tiles * CT_ARG("tile_bytes_act");
    constexpr uint32_t chunk_bytes = CT_ARG("mcast_chunk_tiles") * CT_ARG("tile_bytes_act");
    constexpr uint32_t depth = CT_ARG("n_depth");
    constexpr uint32_t slots_per_core = depth / CT_ARG("norm_cores");
    const Rect a_rects[2] = {RECT("a_main"), RECT("a_top")};
    const uint32_t base = arena_base(CT_ARG("cb_norm_out"));
    CircularBuffer out(CT_ARG("cb_norm_out"));

    for (uint32_t m = RT_ARG("index"), row = 0; m < CT_ARG("rows"); m += CT_ARG("norm_cores"), row++) {
        const uint32_t slot = m % depth;
        out.wait_front(row_tiles);

        Semaphore<>(CT_ARG("sem_n_free") + row % slots_per_core)
            .wait_min(CT_ARG("a_active") * (row / slots_per_core + 1));

        for (uint32_t offset = 0; offset < row_bytes; offset += chunk_bytes) {
            const uint32_t bytes = row_bytes - offset < chunk_bytes ? row_bytes - offset : chunk_bytes;
            for (const Rect& rect : a_rects) {
                rect.write(noc, out.get_read_ptr() + offset, base + slot * row_bytes + offset, bytes);
            }
        }
        noc.async_writes_flushed();
        Semaphore<> n_valid(CT_ARG("sem_n_valid") + slot);
        for (const Rect& rect : a_rects) {
            rect.increment(noc, n_valid, 1);
        }
        noc.async_write_barrier();

        out.pop_front(row_tiles);
    }
    noc.async_atomic_barrier();
}
