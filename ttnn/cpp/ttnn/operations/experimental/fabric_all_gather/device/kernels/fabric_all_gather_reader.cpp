// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of a link worker or copy core (terms: fabric_all_gather_chunk_walk.hpp). Reads its send list into the chunk
// CB: entry 0 is this chip's own shard, relay entry k is read from this chip's output once the arrival counter
// reaches k.
//
// Own shard, interleaved input: read chunk by chunk from the input. Non-interleaved input: the copy cores first write
// it into this chip's output slot (in input order, see for_each_input_run), and the link workers read it from there
// like a relay once every copy core has signalled.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric_all_gather_common.hpp"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t metadata_cb = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t num_banks = get_compile_time_arg_val(4);
    constexpr uint32_t batch_chunks = get_compile_time_arg_val(5);  // the CB holds 2 batches
    constexpr bool kBatchFromMetadata = get_compile_time_arg_val(6) != 0;
    constexpr bool kPrefixFromMetadata = get_compile_time_arg_val(7) != 0;
    constexpr bool kInputInterleaved = get_compile_time_arg_val(8) != 0;
    constexpr bool kCopyCore = get_compile_time_arg_val(9) != 0;
    // link workers: program semaphores staged_semaphore_id + c = blocks of the own shard copy core c has converted
    constexpr uint32_t staged_semaphore_id = get_compile_time_arg_val(10);
    constexpr uint32_t num_copy_cores = get_compile_time_arg_val(11);
    constexpr uint32_t run_cb = get_compile_time_arg_val(12);  // copy cores: each run's (flat start, pages | last)
    constexpr auto input_args = TensorAccessorArgs<13>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto batch_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto prefix_args = TensorAccessorArgs<batch_args.next_compile_time_args_offset()>();
    constexpr uint32_t chunk_bytes = pages_per_chunk * page_bytes;
    constexpr uint32_t cb_chunks = 2 * batch_chunks;
    constexpr uint32_t block_pages = pages_per_chunk * batch_chunks * kCopyBlockBatches;  // copy-core block

    // Common args: [0] input [1] output [2] arrival counter [3] batch index tensor [4] slot layers [5] slot layer
    // [6] slot base page (host) [7] pages per slot, then the shard geometry (fabric_all_gather_common.hpp).
    const uint32_t landing_l1 = get_write_ptr(metadata_cb);
    const ShardGeometry shard = read_shard_geometry<kPrefixFromMetadata>(landing_l1, prefix_args);
    const uint32_t slot_base = read_slot_base<kBatchFromMetadata>(landing_l1, batch_args);
    const auto input = TensorAccessor(input_args, get_common_arg_val<uint32_t>(0), page_bytes);
    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(1), page_bytes);
    auto* arrivals = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_common_arg_val<uint32_t>(2));

    // Per-core args: [0] first bank [1] bank stride [2] entries, then the send list. A copy core has one entry (its
    // own shard); with a non-interleaved input, [0] / [1] are its index / the number of copy cores.
    const uint32_t first_bank = get_arg_val<uint32_t>(0);
    const uint32_t bank_stride = get_arg_val<uint32_t>(1);
    const uint32_t num_entries = get_arg_val<uint32_t>(2);

    // A batch is pushed when it is full, when it reaches the end of the CB (so it stays contiguous) and at the end of
    // every entry (so nothing is held while waiting for the next relay).
    uint32_t cb_offset = 0, batch = 0, batch_cap = 0, write_ptr = 0;
    auto push = [&]() {
        noc_async_read_barrier();
        cb_push_back(cb, pages_per_chunk * batch);
        cb_offset = (cb_offset + batch) % cb_chunks;
        batch = 0;
    };
    auto next_slot = [&]() {  // L1 address of the next chunk slot, reserving a batch when one starts
        if (batch == 0) {
            batch_cap = batch_chunks < cb_chunks - cb_offset ? batch_chunks : cb_chunks - cb_offset;
            cb_reserve_back(cb, pages_per_chunk * batch_cap);
            write_ptr = get_write_ptr(cb);
        }
        return write_ptr + batch * chunk_bytes;
    };
    auto slot_done = [&]() {
        if (++batch == batch_cap) {
            push();
        }
    };

    if constexpr (kCopyCore && !kInputInterleaved) {
        for_each_input_run(
            shard,
            input,
            slot_base,
            page_bytes,
            pages_per_chunk,
            block_pages,
            first_bank,
            bank_stride,
            [&](uint32_t stripe, uint32_t page, uint32_t num_pages, bool last_of_block) {
                noc_async_read(
                    input.get_noc_addr(slot_base + stripe * shard.stripe_pages + page),
                    next_slot(),
                    num_pages * page_bytes);
                cb_reserve_back(run_cb, 1);
                auto* run = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(run_cb));
                run[0] = stripe * shard.active_stripe_pages + page;
                run[1] = num_pages | (last_of_block ? 0x80000000u : 0u);
                cb_push_back(run_cb, 1);
                ++batch;
                if (batch == batch_cap || last_of_block) {  // the writer signals whole blocks
                    push();
                }
            });
        return;
    }

    // Own shard of a non-interleaved input: wait until the copy cores have converted every block the chunk touches.
    auto wait_staged = [&](uint32_t stripe, uint32_t page, uint32_t num_pages) {
        const uint32_t last_flat = stripe * shard.active_stripe_pages + page + (num_pages - 1) * num_banks;
        for (uint32_t c = 0; c < num_copy_cores; ++c) {
            auto* staged = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(staged_semaphore_id + c));
            const uint32_t need = blocks_needed(last_flat, block_pages, c, num_copy_cores);
            if (*staged < need) {
                if (batch > 0) {
                    push();  // never block while holding read chunks
                }
                noc_semaphore_wait_min(staged, need);
            }
        }
    };

    for (uint32_t k = 0; k < num_entries; ++k) {
        const uint32_t entry = get_arg_val<uint32_t>(3 + k);
        if (count_chunks(shard, num_banks, pages_per_chunk, first_bank, bank_stride, entry_half(entry)) == 0) {
            continue;  // nothing on our banks (its sender may already have reset the counter: don't wait)
        }
        const bool from_output = k > 0 || !kInputInterleaved;
        if (k > 0) {
            noc_semaphore_wait_min(arrivals, k);  // upstream's entry k - 1 has landed in our output
        }
        for_each_chunk(
            shard,
            num_banks,
            pages_per_chunk,
            first_bank,
            bank_stride,
            entry_half(entry),
            [&](uint32_t stripe, uint32_t page, uint32_t num_pages) {
                if constexpr (!kInputInterleaved) {
                    if (k == 0) {
                        wait_staged(stripe, page, num_pages);
                    }
                }
                const uint32_t dst = next_slot();
                if (from_output) {
                    noc_async_read(
                        output.get_noc_addr(output_page(shard, entry_rank(entry), stripe, page)),
                        dst,
                        num_pages * page_bytes);
                } else {
                    noc_async_read(
                        input.get_noc_addr(slot_base + stripe * shard.stripe_pages + page),
                        dst,
                        num_pages * page_bytes);
                }
                slot_done();
            });
        if (batch > 0) {
            push();
        }
    }
}
