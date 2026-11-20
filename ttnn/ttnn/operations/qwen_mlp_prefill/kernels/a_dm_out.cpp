// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "rect.hpp"

#include "weights.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t a_tiles = CT_ARG("a_tiles_per_core");
    constexpr uint32_t tile_bytes = CT_ARG("tile_bytes_act");
    constexpr uint32_t a_row_bytes = CT_ARG("i_tiles") * tile_bytes;
    constexpr uint32_t depth = CT_ARG("a_depth");
    constexpr uint32_t gather_depth = CT_ARG("gather_depth");
    constexpr uint32_t own_bytes = a_tiles * tile_bytes;
    constexpr uint32_t gather_bytes = CT_ARG("gather_tiles") * tile_bytes;
    const Rect b_grid = RECT("b_rect");
    const Rect row_sources{
        RT_ARG("row_x_start"), RT_ARG("row_y"), RT_ARG("collector_x") - 1, RT_ARG("row_y"), RT_ARG("row_cores") - 1};
    const uint32_t row_cores = RT_ARG("row_cores");
    const bool collector = RT_ARG("is_collector") == 1;
    const uint32_t gather = arena_base(CT_ARG("cb_gather"));
    const uint32_t b_slots = arena_base(CT_ARG("cb_n")) + RT_ARG("row_first") * own_bytes;
    CircularBuffer a_out(CT_ARG("cb_a_out"));

    const auto w_gate_up = TensorAccessor(
        TensorAccessorArgs<CT_ARG("ta_w_gate_up")>(), RT_ARG("w_gate_up_addr"), CT_ARG("tile_bytes_bfp4"));
    SharedWeights<decltype(w_gate_up)> shared_weights{
        noc,
        w_gate_up,
        CircularBuffer(CT_ARG("cb_wg")),
        Semaphore<>(CT_ARG("sem_a_weights")),
        {RT_ARG("partner_x"), RT_ARG("partner_y")},
        CT_ARG("tile_bytes_bfp4"),
        CT_ARG("gate_up_tiles"),
        2 * a_tiles * RT_ARG("index"),
        2 * a_tiles,
        CT_ARG("weight_chunk_rows"),
        CT_ARG("h_tiles") / CT_ARG("weight_chunk_rows"),
        CT_ARG("group"),
        CT_ARG("groups"),
        CT_ARG("weight_reads_in_flight")};
    shared_weights.start();
    shared_weights.finish();

    for (uint32_t m = 0; m < CT_ARG("rows"); m++) {
        const uint32_t slot = m % depth;
        const uint32_t gather_slot = m % gather_depth;
        const uint32_t gather_use = m / gather_depth;
        const uint32_t destination = gather + gather_slot * gather_bytes + RT_ARG("position") * own_bytes;
        a_out.wait_front(a_tiles);

        if (!collector) {
            Semaphore<>(CT_ARG("sem_gather_free") + gather_slot).wait_min(gather_use);
            noc.async_write(
                CoreLocalMem<uint32_t>(a_out.get_read_ptr()),
                UnicastEndpoint{},
                own_bytes,
                {},
                {.noc_x = RT_ARG("collector_x"), .noc_y = RT_ARG("row_y"), .addr = destination});
            noc.async_writes_flushed();
            Semaphore<>(CT_ARG("sem_gather_valid") + gather_slot).up(noc, RT_ARG("collector_x"), RT_ARG("row_y"), 1);
        } else {
            noc.async_write(
                CoreLocalMem<uint32_t>(a_out.get_read_ptr()),
                UnicastEndpoint{},
                own_bytes,
                {},
                {.noc_x = RT_ARG("collector_x"), .noc_y = RT_ARG("row_y"), .addr = destination});
            Semaphore<>(CT_ARG("sem_gather_valid") + gather_slot).wait_min((row_cores - 1) * (gather_use + 1));

            Semaphore<>(CT_ARG("sem_a_free") + slot).wait_min(CT_ARG("b_cores") * (m / depth + 1));

            noc.async_writes_flushed();
            b_grid.write(noc, gather + gather_slot * gather_bytes, b_slots + slot * a_row_bytes, row_cores * own_bytes);
            noc.async_writes_flushed();

            Semaphore<> a_valid(CT_ARG("sem_a_valid") + slot);
            b_grid.increment(noc, a_valid, 1);

            noc.async_write_barrier();

            Semaphore<> gather_free(CT_ARG("sem_gather_free") + gather_slot);
            row_sources.increment(noc, gather_free, 1);
        }
        a_out.pop_front(a_tiles);
    }
    noc.async_atomic_barrier();
}
