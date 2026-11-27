// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_expand writer (BRISC, NoC1) — regime R1 of op_design.md.
//
// This core owns output tiles [start_t, start_t + n_t) in column-major order
// (col = t / Ht, ht = t % Ht). Each column segment of that range is served by one
// broadcast tile from cb_bcast_tiles: await it, write it whole (2048 B) to every output
// page (img*Ht + ht)*Wt + wt of the segment with no barrier between writes, flush, pop.
// One write barrier before return (BB-03).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_bcast_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t reader_write_eighths = get_compile_time_arg_val(4);
    constexpr auto output_args = TensorAccessorArgs<5>();

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_t = get_arg_val<uint32_t>(0);
    const uint32_t n_t = get_arg_val<uint32_t>(1);
    if (n_t == 0) {
        return;
    }

    const auto output_accessor = TensorAccessor(output_args, output_addr, page_size);

    const uint32_t end_t = start_t + n_t;
    uint32_t t = start_t;
    while (t < end_t) {
        const uint32_t col = t / Ht;
        const uint32_t col_end = (col + 1) * Ht;
        const uint32_t seg_end = end_t < col_end ? end_t : col_end;
        // The reader writes the last (seg_len * reader_write_eighths) / 8 tiles of the segment.
        const uint32_t my_end = seg_end - ((seg_end - t) * reader_write_eighths) / 8;
        const uint32_t img = col / Wt;
        const uint32_t wt = col - img * Wt;
        uint32_t page = (img * Ht + (t - col * Ht)) * Wt + wt;

        {
            DeviceZoneScopedN("G6E_WAIT_BCAST");
            cb_wait_front(cb_bcast_tiles, 1);
        }
        {
            DeviceZoneScopedN("G6E_WRITE");
            const uint32_t src = get_read_ptr(cb_bcast_tiles);
            for (; t < my_end; ++t, page += Wt) {
                noc_async_write(src, output_accessor.get_noc_addr(page), page_size);
            }
            t = seg_end;
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_bcast_tiles, 1);
    }
    {
        DeviceZoneScopedN("G6E_DRAIN");
        noc_async_write_barrier();
    }
}
