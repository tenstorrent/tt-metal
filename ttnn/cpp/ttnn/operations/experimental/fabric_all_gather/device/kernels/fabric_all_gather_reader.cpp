// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of a link worker (or of a copy core): reads the chip's own shard from the input, then the shards it
// relays from the output, each relayed chunk only once the arrival counter says it has landed, into the chunk CB.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric_all_gather_common.hpp"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t run_pages = get_compile_time_arg_val(3);
    constexpr uint32_t num_banks = get_compile_time_arg_val(4);
    constexpr uint32_t group = get_compile_time_arg_val(5);      // chunks per read barrier (half the CB)
    constexpr uint32_t inc_every = get_compile_time_arg_val(6);  // upstream increments once per this many chunks
    constexpr bool kBatchFromMetadata = get_compile_time_arg_val(7) != 0;
    constexpr bool kPrefixFromMetadata = get_compile_time_arg_val(8) != 0;
    // 1 = interleaved input: a run of pages b, b + NB, ... is contiguous in one bank, one read. 0 = any other
    // DRAM layout (e.g. ND-sharded): read the run page by page through the accessor.
    constexpr bool kInputInterleaved = get_compile_time_arg_val(9) != 0;
    constexpr auto in_args = TensorAccessorArgs<10>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto batch_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto prefix_args = TensorAccessorArgs<batch_args.next_compile_time_args_offset()>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;

    // Common args: [0] input address [1] output address [2] arrival counter address [3] batch metadata address
    // [4] batch slot layers [5] batch slot layer index [6] host slot base (pages) [7] pages per slot
    // [8..15] geometry block (fabric_all_gather_common.hpp).
    constexpr uint32_t kGeom = 8;
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t arrival_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t scratch = get_write_ptr(cb_meta);
    const Geometry geo = read_geometry<kPrefixFromMetadata>(kGeom, page_bytes, scratch, prefix_args);
    uint32_t slot_base = get_common_arg_val<uint32_t>(6);
    if constexpr (kBatchFromMetadata) {
        const auto batch = TensorAccessor(batch_args, get_common_arg_val<uint32_t>(3), sizeof(uint32_t));
        const uint32_t user = read_metadata_word(batch, scratch);
        const uint32_t slot = user * get_common_arg_val<uint32_t>(4) + get_common_arg_val<uint32_t>(5);
        slot_base = slot * get_common_arg_val<uint32_t>(7);
    }

    // Per-core args: [0] first bank [1] bank stride [2] walk rotation [3] entries, then entries (rank | part << 16);
    // entry 0 is this chip's own shard (read from the input), the rest are relays (read from the output).
    uint32_t a = 0;
    const uint32_t first = get_arg_val<uint32_t>(a++);
    const uint32_t stride = get_arg_val<uint32_t>(a++);
    const uint32_t rot = get_arg_val<uint32_t>(a++);
    const uint32_t num_entries = get_arg_val<uint32_t>(a++);
    const uint32_t entries_idx = a;

    const auto in = TensorAccessor(in_args, in_addr, page_bytes);
    const auto out = TensorAccessor(out_args, out_addr, page_bytes);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);

    const uint32_t full = fag::port_chunks(geo.num_stripes, geo.stripe_pages, num_banks, run_pages, first, stride, 0);
    uint32_t chunks_left = 0;
    for (uint32_t k = 0; k < num_entries; ++k) {
        chunks_left += fag::port_chunks(
            geo.num_stripes,
            geo.stripe_pages,
            num_banks,
            run_pages,
            first,
            stride,
            get_arg_val<uint32_t>(entries_idx + k) >> 16);
    }

    // Chunks are read in batches of up to `group` and pushed together. A batch never crosses the end of the CB
    // (2 x group chunks; `offset` is where the next batch starts), and it is pushed early, before blocking on a relay
    // that has not landed yet: holding read chunks while waiting would deadlock when a shard is shorter than a batch
    // (every chip waits for chunks its neighbour has read but not pushed).
    constexpr uint32_t cb_chunks = 2 * group;
    uint32_t batch = 0, cap = 0, offset = 0, wptr = 0;
    auto flush = [&]() {
        noc_async_read_barrier();
        cb_push_back(cb, run_pages * batch);
        offset += batch;
        if (offset >= cb_chunks) {
            offset -= cb_chunks;
        }
        batch = 0;
    };
    for (uint32_t k = 0; k < num_entries; ++k) {
        const uint32_t entry = get_arg_val<uint32_t>(entries_idx + k);
        const uint32_t rank = entry & 0xFFFF;
        // relay entry k is (part of) upstream entry k - 1; every upstream entry before it is a whole shard
        const uint32_t up_base = k > 0 ? (k - 1) * full : 0;
        const uint32_t entry_end = k * full;  // one past upstream's last chunk of entry k - 1
        const uint32_t prior_ends = fag::entry_end_increments(k > 0 ? k - 1 : 0, full, inc_every);
        fag::for_each_chunk(
            geo.num_stripes,
            geo.stripe_pages,
            num_banks,
            run_pages,
            first,
            stride,
            rot,
            entry >> 16,
            [&](uint32_t stripe, uint32_t page, uint32_t n, uint32_t idx) {
                if (k > 0) {
                    // upstream chunk up_base + idx has landed once the increment of the first increment-carrying chunk
                    // at or after it has (see fag::increments_through)
                    const uint32_t need = fag::increments_through(up_base + idx, entry_end, prior_ends, inc_every);
                    if (batch > 0 && *arrived < need) {
                        flush();
                    }
                    noc_semaphore_wait_min(arrived, need);
                }
                if (batch == 0) {
                    cap = group < cb_chunks - offset ? group : cb_chunks - offset;
                    cb_reserve_back(cb, run_pages * cap);
                    wptr = get_write_ptr(cb);
                }
                const uint32_t dst = wptr + batch * chunk_bytes;
                if (k == 0) {
                    const uint32_t in_page = slot_base + stripe * geo.stripe_pages_max + page;
                    if constexpr (kInputInterleaved) {
                        noc_async_read(in.get_noc_addr(in_page), dst, n * page_bytes);
                    } else {
                        for (uint32_t t = 0; t < n; ++t) {
                            noc_async_read(in.get_noc_addr(in_page + t * num_banks), dst + t * page_bytes, page_bytes);
                        }
                    }
                } else {
                    noc_async_read(out.get_noc_addr(output_page(geo, rank, stripe, page)), dst, n * page_bytes);
                }
                --chunks_left;
                if (++batch == cap || chunks_left == 0) {
                    flush();
                }
            });
    }
}
