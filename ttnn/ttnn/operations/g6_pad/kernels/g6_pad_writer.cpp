// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_pad new_storage_tiles writer (BRISC / NoC1): writes the core's fill pages from its own fill
// page first (they hide the wait for the reader's first block), then every staged page at its
// output page id. Each output page is written exactly once by exactly one core.
//
// write_fill_pages / store_block use raw noc_async_write + TensorAccessor: kernel_lib has no
// interleaved batch writer. Stateful one-packet writes are not used because consecutive
// interleaved pages sit on different banks, i.e. different NoC endpoints.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_pad_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_staged_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_fill_page_writer = get_compile_time_arg_val(1);
    constexpr uint32_t es = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr bool fill_is_zero = get_compile_time_arg_val(4) != 0;
    constexpr auto out_args = TensorAccessorArgs<5>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t c_bits = get_common_arg_val<uint32_t>(1);
    const uint32_t ht_in = get_common_arg_val<uint32_t>(2);
    const uint32_t wt_in = get_common_arg_val<uint32_t>(3);
    const uint32_t lead0_in = get_common_arg_val<uint32_t>(4);
    const uint32_t lead1_in = get_common_arg_val<uint32_t>(5);
    const uint32_t ht_out = get_common_arg_val<uint32_t>(6);
    const uint32_t wt_out = get_common_arg_val<uint32_t>(7);
    const uint32_t lead1_out = get_common_arg_val<uint32_t>(8);
    const uint32_t block_pages = get_common_arg_val<uint32_t>(9);

    const uint32_t staged_start = get_arg_val<uint32_t>(0);
    const uint32_t staged_count = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const uint32_t fill_slice = get_arg_val<uint32_t>(3);
    const uint32_t fill_ht = get_arg_val<uint32_t>(4);
    const uint32_t fill_wt = get_arg_val<uint32_t>(5);
    const uint32_t fill_count = get_arg_val<uint32_t>(6);

    const auto out = TensorAccessor(out_args, out_addr, page_bytes);

    if (fill_count > 0) {
        {
            DeviceZoneScopedN("G6PAD-FILL-PAGE");
            g6_pad::build_fill_page<cb_fill_page_writer, es, fill_is_zero>(0, page_bytes, c_bits);
        }
        DeviceZoneScopedN("G6PAD-FILL-WRITES");
        const uint32_t fill_page = get_write_ptr(cb_fill_page_writer);
        uint32_t o0 = fill_slice / lead1_out;
        uint32_t o1 = fill_slice - o0 * lead1_out;
        uint32_t ht = fill_ht;
        uint32_t wt = fill_wt;
        for (uint32_t f = 0; f < fill_count; ++f) {
            const uint32_t page_id = ((o0 * lead1_out + o1) * ht_out + ht) * wt_out + wt;
            noc_async_write(fill_page, out.get_noc_addr(page_id), page_bytes);
            // Next fill page in output order. Rows of an image slice above Ht_in start at Wt_in;
            // a row with no fill page (Wt_in == Wt_out) is skipped.
            ++wt;
            while (wt == wt_out && f + 1 < fill_count) {
                if (++ht == ht_out) {
                    ht = 0;
                    if (++o1 == lead1_out) {
                        o1 = 0;
                        ++o0;
                    }
                }
                wt = (o0 < lead0_in && o1 < lead1_in && ht < ht_in) ? wt_in : 0;
            }
        }
    }

    g6_pad::StagedWalk walk(staged_start, ht_in, wt_in, lead1_in);
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t done = block_idx * block_pages;
        const uint32_t block_extent = staged_count - done < block_pages ? staged_count - done : block_pages;
        {
            DeviceZoneScopedN("G6PAD-WAIT-STAGED");
            cb_wait_front(cb_staged_tiles, block_pages);
        }
        DeviceZoneScopedN("G6PAD-STORE-BLOCK");
        uint32_t slot = get_read_ptr(cb_staged_tiles);
        for (uint32_t i = 0; i < block_extent; ++i) {
            const uint32_t page_id = ((walk.i0 * lead1_out + walk.i1) * ht_out + walk.ht) * wt_out + walk.wt;
            noc_async_write(slot, out.get_noc_addr(page_id), page_bytes);
            walk.advance(ht_in, wt_in, lead1_in);
            slot += page_bytes;
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_staged_tiles, block_pages);
    }

    {
        DeviceZoneScopedN("G6PAD-FINAL-BARRIER");
        noc_async_write_barrier();
    }
}
