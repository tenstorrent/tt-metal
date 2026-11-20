// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "rect.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t chunk_tiles = CT_ARG("norm_chunk_tiles");
    constexpr uint32_t chunk_bytes = chunk_tiles * CT_ARG("tile_bytes_act");
    constexpr uint32_t chunks_per_row = CT_ARG("h_tiles") / chunk_tiles;
    constexpr uint32_t row_bytes = CT_ARG("n_row_tiles") * CT_ARG("tile_bytes_act");
    constexpr uint32_t depth = CT_ARG("n_depth");
    const Rect a_rect = RECT("a_rect");
    const uint32_t base = arena_base(CT_ARG("cb_n"));
    CircularBuffer out(CT_ARG("cb_norm_out"));
    CircularBuffer rstd(CT_ARG("cb_norm_rstd_out"));

    for (uint32_t m = 0; m < CT_ARG("rows"); m++) {
        const uint32_t slot = m % depth;
        const uint32_t slot_address = base + slot * row_bytes;

        Semaphore<>(CT_ARG("sem_n_free") + slot).wait_min(CT_ARG("a_active") * (m / depth + 1));

        for (uint32_t chunk = 0; chunk < chunks_per_row; chunk++) {
            out.wait_front(chunk_tiles);
            a_rect.write_including_source(noc, out.get_read_ptr(), slot_address + chunk * chunk_bytes, chunk_bytes);
            noc.async_writes_flushed();
            out.pop_front(chunk_tiles);
        }
        rstd.wait_front(1);
        a_rect.write_including_source(
            noc, rstd.get_read_ptr(), slot_address + chunks_per_row * chunk_bytes, CT_ARG("tile_bytes_act"));
        noc.async_writes_flushed();
        rstd.pop_front(1);

        Semaphore<> n_valid(CT_ARG("sem_n_valid") + slot);
        a_rect.increment_except(noc, n_valid, 1, RT_ARG("self_x"), RT_ARG("self_y"));
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
