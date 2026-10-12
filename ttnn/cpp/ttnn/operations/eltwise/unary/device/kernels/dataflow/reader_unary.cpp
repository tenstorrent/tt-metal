// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr auto cb_id_src = tt::CBIndex::c_0;

    Noc noc;
    DataflowBuffer dfb_src(cb_id_src);

#if SRC_SHARDED
    dfb_src.reserve_back(num_pages);
    dfb_src.push_back(num_pages);
#else
    constexpr auto src_args = TensorAccessorArgs<0, 0>();
    const auto src = TensorAccessor(src_args, src_addr);

    uint32_t end_id = start_id + num_pages;
    // Post READ_BURST pages, then one barrier. The burst ramps 1, 1, 2, 4, ... up to
    // READ_BURST so the first page reaches compute at one-page latency instead of waiting
    // behind a full first burst from every core. For a power-of-two burst the ramp sums to
    // one burst, so the first full burst starts on a CB-depth boundary. A burst of 1 stays
    // one page in flight.
    constexpr uint32_t kReadBurst = READ_BURST;
    uint32_t ramp = 1;
    uint32_t first_burst = 1;
    auto next_burst = [&]() -> uint32_t {
        const uint32_t n = ramp;
        if (first_burst) {
            first_burst = 0;
        } else if (ramp < kReadBurst) {
            ramp = (ramp * 2 < kReadBurst) ? ramp * 2 : kReadBurst;
        }
        return n;
    };
#if RM_INTERLEAVED
    // A page is one (block, chunk), which is several short row reads on a narrow row and one
    // tile-sized read when the row is a full tile.
    const uint32_t chunks_per_row = get_arg_val<uint32_t>(3);
    const uint32_t chunk_size = get_arg_val<uint32_t>(4);
    const uint32_t last_chunk_size = get_arg_val<uint32_t>(5);
    const uint32_t rows_per_tile = get_arg_val<uint32_t>(6);
    const uint32_t total_rows = get_arg_val<uint32_t>(7);
    const uint32_t page_bytes = get_local_cb_interface(cb_id_src).fifo_page_size;
    const uint32_t depth = get_local_cb_interface(cb_id_src).fifo_num_pages;
    uint32_t filled_in_cycle = 0;
    uint32_t block = start_id;
    uint32_t chunk = 0;
    while (block < end_id) {
        const uint32_t pages_left = (end_id - block - 1) * chunks_per_row + (chunks_per_row - chunk);
        uint32_t n = next_burst();
        const uint32_t until_wrap = depth - filled_in_cycle;
        if (n > pages_left) {
            n = pages_left;
        }
        if (n > until_wrap) {
            n = until_wrap;
        }
        dfb_src.reserve_back(n);
        uint32_t b = block;
        uint32_t c = chunk;
        for (uint32_t k = 0; k < n; ++k) {
            uint32_t base_page = b * rows_per_tile;
            uint32_t remaining = total_rows - base_page;
            uint32_t actual_rows = (rows_per_tile < remaining) ? rows_per_tile : remaining;
            uint32_t bytes = (c == chunks_per_row - 1) ? last_chunk_size : chunk_size;
            for (uint32_t r = 0; r < actual_rows; ++r) {
                noc.async_read(
                    src,
                    dfb_src,
                    bytes,
                    {.page_id = base_page + r, .offset_bytes = c * chunk_size},
                    {.offset_bytes = k * page_bytes + r * bytes});
            }
            c++;
            if (c == chunks_per_row) {
                c = 0;
                b++;
            }
        }
        noc.async_read_barrier();
        dfb_src.push_back(n);
        filled_in_cycle += n;
        if (filled_in_cycle == depth) {
            filled_in_cycle = 0;
        }
        block = b;
        chunk = c;
    }
#else
    // The CB is two bursts deep, so the next burst can be posted while compute consumes the
    // previous one. A burst never crosses the CB wrap: offsets are contiguous from the write
    // pointer.
    const uint32_t page_bytes = get_local_cb_interface(cb_id_src).fifo_page_size;
    const uint32_t depth = get_local_cb_interface(cb_id_src).fifo_num_pages;
    uint32_t filled_in_cycle = 0;
    for (uint32_t i = start_id; i < end_id;) {
        uint32_t n = next_burst();
        const uint32_t remaining = end_id - i;
        const uint32_t until_wrap = depth - filled_in_cycle;
        if (n > remaining) {
            n = remaining;
        }
        if (n > until_wrap) {
            n = until_wrap;
        }
        dfb_src.reserve_back(n);
        for (uint32_t k = 0; k < n; ++k) {
            noc.async_read(src, dfb_src, page_bytes, {.page_id = i + k}, {.offset_bytes = k * page_bytes});
        }
        noc.async_read_barrier();
        dfb_src.push_back(n);
        filled_in_cycle += n;
        if (filled_in_cycle == depth) {
            filled_in_cycle = 0;
        }
        i += n;
    }
#endif
#endif
}
