// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Copy writer (terms: fabric_all_gather_chunk_walk.hpp): writes this chip's own shard, as its reader fetched it, into
// this chip's own output slot. Same batching as the reader.
//
// Interleaved input: the reader fetched chunks, each is one contiguous write. Non-interleaved input: the reader
// fetched input-order runs (for_each_input_run) and published each run's position on run_cb; their pages are scattered
// over the interleaved output, so each page is its own write, and each finished block is signalled to every link worker
// of this chip.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric_all_gather_common.hpp"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t metadata_cb = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t num_banks = get_compile_time_arg_val(4);
    constexpr uint32_t batch_chunks = get_compile_time_arg_val(5);
    constexpr bool kPrefixFromMetadata = get_compile_time_arg_val(6) != 0;
    constexpr bool kInputInterleaved = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t staged_semaphore_id = get_compile_time_arg_val(8);
    constexpr uint32_t run_cb = get_compile_time_arg_val(9);  // non-interleaved input: the reader's runs
    constexpr auto output_args = TensorAccessorArgs<10>();
    constexpr auto prefix_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr uint32_t chunk_bytes = pages_per_chunk * page_bytes;
    constexpr uint32_t cb_chunks = 2 * batch_chunks;
    constexpr uint32_t block_pages = pages_per_chunk * batch_chunks * kCopyBlockBatches;  // same as the reader

    // Common args: the reader's ([0] input [1] output ... [3..7] slot, then the shard geometry), so both compute the
    // same runs. Per-core args: [0] first bank / copy index [1] bank stride / copy cores [2] rank [3] link workers,
    // then their NoC x, y.
    const uint32_t landing_l1 = get_write_ptr(metadata_cb);
    const ShardGeometry shard = read_shard_geometry<kPrefixFromMetadata>(landing_l1, prefix_args);
    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(1), page_bytes);
    const uint32_t first_bank = get_arg_val<uint32_t>(0);
    const uint32_t bank_stride = get_arg_val<uint32_t>(1);
    const uint32_t rank = get_arg_val<uint32_t>(2);

    uint32_t batch = 0, cb_offset = 0;
    auto pop = [&]() {
        noc_async_write_barrier();
        cb_pop_front(cb, pages_per_chunk * batch);
        cb_offset = (cb_offset + batch) % cb_chunks;
        batch = 0;
    };
    auto done = [&]() {
        const uint32_t batch_cap = batch_chunks < cb_chunks - cb_offset ? batch_chunks : cb_chunks - cb_offset;
        if (++batch == batch_cap) {
            pop();
        }
    };
    if constexpr (kInputInterleaved) {
        for_each_chunk(
            shard,
            num_banks,
            pages_per_chunk,
            first_bank,
            bank_stride,
            kWholeShard,
            [&](uint32_t stripe, uint32_t page, uint32_t num_pages) {
                cb_wait_front(cb, pages_per_chunk * (batch + 1));
                noc_async_write(
                    get_read_ptr(cb) + batch * chunk_bytes,
                    output.get_noc_addr(output_page(shard, rank, stripe, page)),
                    num_pages * page_bytes);
                done();
            });
    } else {
        // the reader publishes each run on run_cb; this core's share is blocks c, c + n, ... of the flat shard
        const uint32_t num_link_workers = get_arg_val<uint32_t>(3);
        const uint32_t staged = get_semaphore(staged_semaphore_id + first_bank);
        const uint32_t total = shard.num_stripes * shard.active_stripe_pages;
        uint32_t my_pages = 0;
        for (uint32_t block = first_bank; block * block_pages < total; block += bank_stride) {
            my_pages += (block + 1) * block_pages < total ? block_pages : total - block * block_pages;
        }
        // The output is interleaved: page p is row p / num_banks of bank p % num_banks. Consecutive pages cycle the
        // banks, so compute each bank's base once instead of an accessor lookup per page.
        uint64_t bank_base[num_banks];
        for (uint32_t b = 0; b < num_banks; ++b) {
            bank_base[b] = output.get_noc_addr(b);
        }
        for (uint32_t done_pages = 0; done_pages < my_pages;) {
            cb_wait_front(run_cb, 1);
            auto* run = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(run_cb));
            const uint32_t flat = run[0], num_pages = run[1] & 0x7FFFFFFFu;
            const bool last_of_block = (run[1] & 0x80000000u) != 0;
            cb_pop_front(run_cb, 1);
            const uint32_t stripe = flat / shard.active_stripe_pages, page = flat % shard.active_stripe_pages;
            cb_wait_front(cb, pages_per_chunk * (batch + 1));
            const uint32_t src = get_read_ptr(cb) + batch * chunk_bytes;
            const uint32_t first = output_page(shard, rank, stripe, page);
            for (uint32_t i = 0; i < num_pages; ++i) {
                const uint32_t o = first + i;
                const uint64_t dst = bank_base[o % num_banks] + (o / num_banks) * page_bytes;
                if constexpr (page_bytes <= NOC_MAX_BURST_SIZE) {
                    noc_async_write_one_packet(src + i * page_bytes, dst, page_bytes);
                } else {  // a page larger than one NoC packet
                    noc_async_write(src + i * page_bytes, dst, page_bytes);
                }
            }
            done_pages += num_pages;
            const uint32_t batch_cap = batch_chunks < cb_chunks - cb_offset ? batch_chunks : cb_chunks - cb_offset;
            if (last_of_block) {
                ++batch;
                pop();  // write barrier: the block has landed before it is signalled
            } else if (++batch == batch_cap) {
                noc_async_writes_flushed();  // the batch has left L1; it only has to land by the block's end
                cb_pop_front(cb, pages_per_chunk * batch);
                cb_offset = (cb_offset + batch) % cb_chunks;
                batch = 0;
            }
            if (last_of_block) {
                for (uint32_t w = 0; w < num_link_workers; ++w) {
                    noc_semaphore_inc(
                        get_noc_addr(get_arg_val<uint32_t>(4 + 2 * w), get_arg_val<uint32_t>(5 + 2 * w), staged), 1);
                }
            }
        }
        noc_async_atomic_barrier();
    }
    if (batch > 0) {
        pop();
    }
}
