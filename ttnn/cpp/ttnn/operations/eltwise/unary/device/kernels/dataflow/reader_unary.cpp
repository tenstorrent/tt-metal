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
    constexpr uint32_t onepage = 1;
    constexpr auto src_args = TensorAccessorArgs<0, 0>();
    const auto src = TensorAccessor(src_args, src_addr);

    uint32_t end_id = start_id + num_pages;
#if RM_INTERLEAVED
    const uint32_t chunks_per_row = get_arg_val<uint32_t>(3);
    const uint32_t chunk_size = get_arg_val<uint32_t>(4);
    const uint32_t last_chunk_size = get_arg_val<uint32_t>(5);
    const uint32_t rows_per_tile = get_arg_val<uint32_t>(6);
    const uint32_t total_rows = get_arg_val<uint32_t>(7);

    for (uint32_t block = start_id; block < end_id; ++block) {
        uint32_t base_page = block * rows_per_tile;
        uint32_t remaining = total_rows - base_page;
        uint32_t actual_rows = (rows_per_tile < remaining) ? rows_per_tile : remaining;

        for (uint32_t j = 0; j < chunks_per_row; ++j) {
            uint32_t bytes = (j == chunks_per_row - 1) ? last_chunk_size : chunk_size;
            dfb_src.reserve_back(onepage);
            for (uint32_t r = 0; r < actual_rows; ++r) {
                noc.async_read(
                    src,
                    dfb_src,
                    bytes,
                    {.page_id = base_page + r, .offset_bytes = j * chunk_size},
                    {.offset_bytes = r * bytes});
            }
            noc.async_read_barrier();
            dfb_src.push_back(onepage);
        }
    }
#else
    const uint32_t page_bytes = get_local_cb_interface(cb_id_src).fifo_page_size;
#if SHARD_ROTATE
    // DRAM height-sharded. A shard's pages sit in one bank, so reading start_id, start_id + 1, ...
    // keeps every read of a burst on one bank. Pages are instead visited slot by slot across the
    // shards (slot 0 of shard 0, 1, ..., then slot 1, ...), so consecutive reads, and each burst,
    // spread over the banks the way interleaved pages do. This core takes num_pages of that order
    // from slot start_id, shard arg 6. A short last shard has no page at slots >= last_shard_pages.
    const uint32_t shard_pages = get_arg_val<uint32_t>(3);
    const uint32_t num_shards = get_arg_val<uint32_t>(4);
    const uint32_t last_shard_pages = get_arg_val<uint32_t>(5);
    uint32_t slot = start_id;
    uint32_t shard = get_arg_val<uint32_t>(6);
    auto next_page = [&]() -> uint32_t {
        if (shard == num_shards - 1 && slot >= last_shard_pages) {
            shard = 0;
            slot++;
        }
        const uint32_t page = shard * shard_pages + slot;
        if (++shard == num_shards) {
            shard = 0;
            slot++;
        }
        return page;
    };
    // Post READ_BURST pages, then one barrier. The burst ramps 1, 1, 2, 4, ... up to READ_BURST so the
    // first page reaches compute at one-page latency instead of waiting behind a full first burst from
    // every core. For a power-of-two burst the ramp sums to one burst, so the first full burst starts on
    // a CB-depth boundary. The CB is two bursts deep, so the next burst can be posted while compute
    // consumes the previous one. A burst never crosses the CB wrap: offsets are contiguous from the
    // write pointer.
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
    const uint32_t depth = get_local_cb_interface(cb_id_src).fifo_num_pages;
    uint32_t filled_in_cycle = 0;
    for (uint32_t done = 0; done < num_pages;) {
        uint32_t n = next_burst();
        const uint32_t remaining = num_pages - done;
        const uint32_t until_wrap = depth - filled_in_cycle;
        if (n > remaining) {
            n = remaining;
        }
        if (n > until_wrap) {
            n = until_wrap;
        }
        dfb_src.reserve_back(n);
        for (uint32_t k = 0; k < n; ++k) {
            noc.async_read(src, dfb_src, page_bytes, {.page_id = next_page()}, {.offset_bytes = k * page_bytes});
        }
        noc.async_read_barrier();
        dfb_src.push_back(n);
        filled_in_cycle += n;
        if (filled_in_cycle == depth) {
            filled_in_cycle = 0;
        }
        done += n;
    }
#else
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_src.reserve_back(onepage);
        noc.async_read(src, dfb_src, page_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_src.push_back(onepage);
    }
#endif  // SHARD_ROTATE
#endif
#endif
}
