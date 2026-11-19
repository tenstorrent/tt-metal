// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "rect.hpp"

#include "reduce_scaler.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t tile_bytes = CT_ARG("tile_bytes_bf16");
    constexpr uint32_t row_tiles = CT_ARG("h_tiles");
    constexpr uint32_t row_bytes = row_tiles * tile_bytes;
    constexpr uint32_t chunk_bytes = CT_ARG("mcast_chunk_tiles") * tile_bytes;
    constexpr uint32_t norm_cores = CT_ARG("norm_cores");
    constexpr uint32_t a_active = CT_ARG("a_active");
    const Rect a_grid = RECT("a_grid");
    const uint32_t a_slots = arena_base(CT_ARG("cb_norm_scaler"));

    const auto h = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_h")>(), RT_ARG("h_addr"), tile_bytes);
    const auto w_norm = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_w_norm")>(), RT_ARG("w_norm_addr"), tile_bytes);

    CircularBuffer in(CT_ARG("cb_norm_in"));
    CircularBuffer weight(CT_ARG("cb_norm_weight"));
    CircularBuffer out(CT_ARG("cb_norm_out"));
    Semaphore<> n_free[2] = {Semaphore<>(CT_ARG("sem_n_free")), Semaphore<>(CT_ARG("sem_n_free") + 1)};
    Semaphore<> n_valid[2] = {Semaphore<>(CT_ARG("sem_n_valid")), Semaphore<>(CT_ARG("sem_n_valid") + 1)};

    fill_reduce_scaler(noc, CT_ARG("cb_norm_scaler"), tile_bytes);
    weight.reserve_back(row_tiles);
    for (uint32_t t = 0; t < row_tiles; t++) {
        noc.async_read(
            w_norm, CoreLocalMem<uint32_t>(weight.get_write_ptr() + t * tile_bytes), tile_bytes, {.page_id = t}, {});
    }
    noc.async_read_barrier();
    weight.push_back(row_tiles);

    for (uint32_t m = RT_ARG("index"); m < CT_ARG("rows"); m += norm_cores) {
        const uint32_t slot = m & 1;
        in.reserve_back(row_tiles);
        for (uint32_t t = 0; t < row_tiles; t++) {
            noc.async_read(
                h,
                CoreLocalMem<uint32_t>(in.get_write_ptr() + t * tile_bytes),
                tile_bytes,
                {.page_id = m * row_tiles + t},
                {});
        }
        noc.async_read_barrier();
        in.push_back(row_tiles);

        out.wait_front(row_tiles);
        n_free[slot].wait_min(a_active * (m / 2 + 1));
        for (uint32_t offset = 0; offset < row_bytes; offset += chunk_bytes) {
            a_grid.write(noc, out.get_read_ptr() + offset, a_slots + slot * row_bytes + offset, chunk_bytes);
        }
        noc.async_writes_flushed();
        a_grid.signal(noc, n_valid[slot], 1);
        noc.async_write_barrier();
        out.pop_front(row_tiles);
    }
}
