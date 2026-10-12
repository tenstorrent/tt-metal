// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr auto cb_id_dst = tt::CBIndex::c_2;

    Noc noc;
    DataflowBuffer dfb_dst(cb_id_dst);

#if DST_SHARDED
    // Output is sharded in place; the wait is only a readiness handshake. Pop to
    // leave the CB balanced.
    dfb_dst.wait_front(num_pages);
    dfb_dst.pop_front(num_pages);
#else
    constexpr auto dst_args = TensorAccessorArgs<0, 0>();
    const auto dst = TensorAccessor(dst_args, dst_addr);

    uint32_t end_id = start_id + num_pages;
#if RM_INTERLEAVED
    constexpr uint32_t kWriteBurst = WRITE_BURST;
    const uint32_t chunks_per_row = get_arg_val<uint32_t>(3);
    const uint32_t chunk_size = get_arg_val<uint32_t>(4);
    const uint32_t last_chunk_size = get_arg_val<uint32_t>(5);
    const uint32_t rows_per_tile = get_arg_val<uint32_t>(6);
    const uint32_t total_rows = get_arg_val<uint32_t>(7);
    const uint32_t page_bytes = get_local_cb_interface(cb_id_dst).fifo_page_size;
    const uint32_t depth = get_local_cb_interface(cb_id_dst).fifo_num_pages;
    uint32_t drained_in_cycle = 0;
    uint32_t block = start_id;
    uint32_t chunk = 0;
    while (block < end_id) {
        const uint32_t pages_left = (end_id - block - 1) * chunks_per_row + (chunks_per_row - chunk);
        uint32_t n = kWriteBurst;
        const uint32_t until_wrap = depth - drained_in_cycle;
        if (n > pages_left) {
            n = pages_left;
        }
        if (n > until_wrap) {
            n = until_wrap;
        }
        dfb_dst.wait_front(n);
        uint32_t b = block;
        uint32_t c = chunk;
        for (uint32_t k = 0; k < n; ++k) {
            uint32_t base_page = b * rows_per_tile;
            uint32_t remaining = total_rows - base_page;
            uint32_t actual_rows = (rows_per_tile < remaining) ? rows_per_tile : remaining;
            uint32_t bytes = (c == chunks_per_row - 1) ? last_chunk_size : chunk_size;
            for (uint32_t r = 0; r < actual_rows; ++r) {
                noc.async_write(
                    dfb_dst,
                    dst,
                    bytes,
                    {.offset_bytes = k * page_bytes + r * bytes},
                    {.page_id = base_page + r, .offset_bytes = c * chunk_size});
            }
            c++;
            if (c == chunks_per_row) {
                c = 0;
                b++;
            }
        }
        noc.async_writes_flushed();
        dfb_dst.pop_front(n);
        drained_in_cycle += n;
        if (drained_in_cycle == depth) {
            drained_in_cycle = 0;
        }
        block = b;
        chunk = c;
    }
    noc.async_write_barrier();
#else
    // Flush once per WRITE_BURST instead of once per tile, so a few writes share the
    // round trip. The output CB stays shallower than the input CB: one burst plus a
    // second burst of slots so pack can fill the next group while these writes drain.
    constexpr uint32_t kWriteBurst = WRITE_BURST;
    const uint32_t page_bytes = get_local_cb_interface(cb_id_dst).fifo_page_size;
    const uint32_t depth = get_local_cb_interface(cb_id_dst).fifo_num_pages;
    uint32_t drained_in_cycle = 0;
    for (uint32_t i = start_id; i < end_id;) {
        uint32_t n = kWriteBurst;
        const uint32_t remaining = end_id - i;
        const uint32_t until_wrap = depth - drained_in_cycle;
        if (n > remaining) {
            n = remaining;
        }
        if (n > until_wrap) {
            n = until_wrap;
        }
        dfb_dst.wait_front(n);
        for (uint32_t k = 0; k < n; ++k) {
            noc.async_write(dfb_dst, dst, page_bytes, {.offset_bytes = k * page_bytes}, {.page_id = i + k});
        }
        noc.async_writes_flushed();
        dfb_dst.pop_front(n);
        drained_in_cycle += n;
        if (drained_in_cycle == depth) {
            drained_in_cycle = 0;
        }
        i += n;
    }
    noc.async_write_barrier();
#endif
#endif
}
