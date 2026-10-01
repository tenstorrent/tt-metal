// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of a fabric link worker or a local copy core (terms: fabric_all_gather_chunk_walk.hpp). Reads its outgoing
// shards into the chunk CB, one fabric chunk per chunk slot, a CB batch at a time: outgoing shard 0 is this chip's own
// shard; forwarded shard k is read from this chip's output once the shards-arrived counter reaches k.
//
// Own shard, interleaved input: read chunk by chunk from the input. Non-interleaved input: the local copy cores first
// convert it into this chip's output (for_each_contiguous_input_run), and the link workers read it from there like a
// forwarded shard, once the conversion blocks a chunk touches are done.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric_all_gather_common.hpp"

void kernel_main() {
    constexpr uint32_t chunk_cb = get_compile_time_arg_val(0);
    constexpr uint32_t metadata_cb = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_fabric_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t num_dram_banks = get_compile_time_arg_val(4);
    constexpr uint32_t chunks_per_cb_batch = get_compile_time_arg_val(5);  // the chunk CB holds 2 CB batches
    constexpr bool kCacheSlotFromMetadata = get_compile_time_arg_val(6) != 0;
    constexpr bool kValidPrefixFromMetadata = get_compile_time_arg_val(7) != 0;
    constexpr bool kInputInterleaved = get_compile_time_arg_val(8) != 0;
    constexpr bool kLocalCopyCore = get_compile_time_arg_val(9) != 0;
    // link workers: program semaphore blocks_converted_semaphore_id + c = conversion blocks local copy core c has done
    constexpr uint32_t blocks_converted_semaphore_id = get_compile_time_arg_val(10);
    constexpr uint32_t num_local_copy_cores = get_compile_time_arg_val(11);
    constexpr uint32_t input_run_cb =
        get_compile_time_arg_val(12);  // copy cores: (flat_page, num_pages | last) per run
    constexpr auto input_args = TensorAccessorArgs<13>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto cache_slot_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto valid_prefix_args = TensorAccessorArgs<cache_slot_args.next_compile_time_args_offset()>();
    constexpr uint32_t fabric_chunk_bytes = pages_per_fabric_chunk * page_bytes;
    constexpr uint32_t chunk_slots_in_cb = 2 * chunks_per_cb_batch;
    constexpr uint32_t conversion_block_pages =
        pages_per_fabric_chunk * chunks_per_cb_batch * kCbBatchesPerConversionBlock;

    // Common args: [0] input [1] output [2] shards-arrived counter [3] cache slot tensor [4] layers per user [5] layer
    // [6] cache slot first page (host) [7] pages per cache slot, then the chip shard geometry (common.hpp).
    const uint32_t landing_l1 = get_write_ptr(metadata_cb);
    const ChipShardGeometry geometry =
        read_chip_shard_geometry<kValidPrefixFromMetadata>(landing_l1, valid_prefix_args);
    const uint32_t cache_slot_first_page =
        read_cache_slot_first_page<kCacheSlotFromMetadata>(landing_l1, cache_slot_args);
    const auto input = TensorAccessor(input_args, get_common_arg_val<uint32_t>(0), page_bytes);
    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(1), page_bytes);
    auto* shards_arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_common_arg_val<uint32_t>(2));

    // Per-core args: [0] first owned bank [1] owned bank stride [2] number of outgoing shards, then the outgoing
    // shards. A local copy core has one (its own shard); with a non-interleaved input, [0] / [1] are its index / the
    // number of local copy cores instead.
    const uint32_t first_owned_bank = get_arg_val<uint32_t>(0);
    const uint32_t owned_bank_stride = get_arg_val<uint32_t>(1);
    const uint32_t num_outgoing_shards = get_arg_val<uint32_t>(2);
    constexpr uint32_t kOutgoingShardsArg = 3;

    // A CB batch is pushed when it is full, when it reaches the end of the CB (so it stays contiguous in L1) and at the
    // end of every outgoing shard (so nothing is held while waiting for the next forwarded shard).
    uint32_t cb_chunk_offset = 0, chunks_in_batch = 0, batch_capacity = 0, write_ptr = 0;
    auto push_batch = [&]() {
        noc_async_read_barrier();
        cb_push_back(chunk_cb, pages_per_fabric_chunk * chunks_in_batch);
        cb_chunk_offset = (cb_chunk_offset + chunks_in_batch) % chunk_slots_in_cb;
        chunks_in_batch = 0;
    };
    auto next_chunk_slot = [&]() {  // L1 address of the next chunk slot, reserving a CB batch when one starts
        if (chunks_in_batch == 0) {
            batch_capacity = chunks_per_cb_batch < chunk_slots_in_cb - cb_chunk_offset
                                 ? chunks_per_cb_batch
                                 : chunk_slots_in_cb - cb_chunk_offset;
            cb_reserve_back(chunk_cb, pages_per_fabric_chunk * batch_capacity);
            write_ptr = get_write_ptr(chunk_cb);
        }
        return write_ptr + chunks_in_batch * fabric_chunk_bytes;
    };
    auto chunk_slot_filled = [&]() {
        if (++chunks_in_batch == batch_capacity) {
            push_batch();
        }
    };

    if constexpr (kLocalCopyCore && !kInputInterleaved) {
        const uint32_t local_copy_core_index = first_owned_bank;
        const uint32_t num_converting_cores = owned_bank_stride;
        for_each_contiguous_input_run(
            geometry,
            input,
            cache_slot_first_page,
            page_bytes,
            pages_per_fabric_chunk,
            conversion_block_pages,
            local_copy_core_index,
            num_converting_cores,
            [&](uint32_t outer_slice, uint32_t page_in_slice, uint32_t num_pages, bool last_run_of_block) {
                noc_async_read(
                    input.get_noc_addr(
                        cache_slot_first_page + outer_slice * geometry.pages_per_outer_slice + page_in_slice),
                    next_chunk_slot(),
                    num_pages * page_bytes);
                cb_reserve_back(input_run_cb, 1);
                auto* input_run = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(input_run_cb));
                input_run[0] = outer_slice * geometry.valid_pages_per_outer_slice + page_in_slice;  // flat_page
                input_run[1] = num_pages | (last_run_of_block ? 0x80000000u : 0u);
                cb_push_back(input_run_cb, 1);
                ++chunks_in_batch;
                if (chunks_in_batch == batch_capacity ||
                    last_run_of_block) {  // the writer signals whole conversion blocks
                    push_batch();
                }
            });
        return;
    }

    // Own shard of a non-interleaved input: wait until the local copy cores have converted every conversion block the
    // chunk touches (its pages are page_in_slice, + num_dram_banks, ...: the last one is the furthest).
    auto wait_until_converted = [&](uint32_t outer_slice, uint32_t page_in_slice, uint32_t num_pages) {
        const uint32_t last_flat_page =
            outer_slice * geometry.valid_pages_per_outer_slice + page_in_slice + (num_pages - 1) * num_dram_banks;
        for (uint32_t copy_core = 0; copy_core < num_local_copy_cores; ++copy_core) {
            auto* blocks_converted = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(blocks_converted_semaphore_id + copy_core));
            const uint32_t needed_blocks =
                conversion_blocks_needed(last_flat_page, conversion_block_pages, copy_core, num_local_copy_cores);
            if (*blocks_converted < needed_blocks) {
                if (chunks_in_batch > 0) {
                    push_batch();  // never block while holding chunks that were already read
                }
                noc_semaphore_wait_min(blocks_converted, needed_blocks);
            }
        }
    };

    for (uint32_t outgoing_index = 0; outgoing_index < num_outgoing_shards; ++outgoing_index) {
        const uint32_t outgoing_shard = get_arg_val<uint32_t>(kOutgoingShardsArg + outgoing_index);
        if (count_fabric_chunks(
                geometry,
                num_dram_banks,
                pages_per_fabric_chunk,
                first_owned_bank,
                owned_bank_stride,
                outgoing_shard_bank_half(outgoing_shard)) == 0) {
            continue;  // nothing on our banks: don't wait (the sender may already have reset the counter)
        }
        const bool read_from_output = outgoing_index > 0 || !kInputInterleaved;
        if (outgoing_index > 0) {
            noc_semaphore_wait_min(shards_arrived, outgoing_index);  // upstream's outgoing shard k - 1 is in our output
        }
        for_each_fabric_chunk(
            geometry,
            num_dram_banks,
            pages_per_fabric_chunk,
            first_owned_bank,
            owned_bank_stride,
            outgoing_shard_bank_half(outgoing_shard),
            [&](uint32_t outer_slice, uint32_t page_in_slice, uint32_t num_pages) {
                if constexpr (!kInputInterleaved) {
                    if (outgoing_index == 0) {
                        wait_until_converted(outer_slice, page_in_slice, num_pages);
                    }
                }
                const uint32_t chunk_slot = next_chunk_slot();
                if (read_from_output) {
                    noc_async_read(
                        output.get_noc_addr(output_page_index(
                            geometry, outgoing_shard_rank(outgoing_shard), outer_slice, page_in_slice)),
                        chunk_slot,
                        num_pages * page_bytes);
                } else {
                    noc_async_read(
                        input.get_noc_addr(
                            cache_slot_first_page + outer_slice * geometry.pages_per_outer_slice + page_in_slice),
                        chunk_slot,
                        num_pages * page_bytes);
                }
                chunk_slot_filled();
            });
        if (chunks_in_batch > 0) {
            push_batch();
        }
    }
}
