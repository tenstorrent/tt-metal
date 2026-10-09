// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local copy writer (terms: fabric_all_gather_chunk_walk.hpp): writes this chip's own shard, as its reader fetched it,
// into this chip's own output. Same CB batches as the reader.
//
// Interleaved input: the reader fetched fabric chunks, each one physically contiguous write. Non-interleaved input:
// the reader fetched contiguous input runs (for_each_contiguous_input_run) and published each run's position on
// input_run_cb; logically consecutive pages are in different banks of the interleaved output, so each page is its own
// write, and each finished conversion block is signalled to every fabric link worker of this chip.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric_all_gather_common.hpp"

void kernel_main() {
    constexpr uint32_t chunk_cb = get_compile_time_arg_val(0);
    constexpr uint32_t metadata_cb = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_fabric_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t num_dram_banks = get_compile_time_arg_val(4);
    constexpr uint32_t chunks_per_cb_batch = get_compile_time_arg_val(5);
    constexpr bool kValidPrefixFromMetadata = get_compile_time_arg_val(6) != 0;
    constexpr bool kInputInterleaved = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t blocks_converted_semaphore_id = get_compile_time_arg_val(8);
    constexpr uint32_t input_run_cb =
        get_compile_time_arg_val(9);  // non-interleaved input: the reader's contiguous input runs
    constexpr auto output_args = TensorAccessorArgs<10>();
    constexpr auto valid_prefix_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr uint32_t fabric_chunk_bytes = pages_per_fabric_chunk * page_bytes;
    constexpr uint32_t chunk_slots_in_cb = 2 * chunks_per_cb_batch;
    constexpr uint32_t conversion_block_pages =
        pages_per_fabric_chunk * chunks_per_cb_batch * kCbBatchesPerConversionBlock;  // as in the reader

    // Common args: the reader's ([0] input [1] output ... [3..7] cache slot, then the chip shard geometry), so both
    // compute the same chunks and runs. Per-core args: [0] first owned lane / local copy core index [1] owned lane
    // stride / number of local copy cores [2] rank [3] number of fabric link workers, then their NoC x, y.
    const uint32_t landing_l1 = get_write_ptr(metadata_cb);
    const ChipShardGeometry geometry =
        read_chip_shard_geometry<kValidPrefixFromMetadata>(landing_l1, valid_prefix_args);
    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(1), page_bytes);
    const uint32_t first_owned_lane = get_arg_val<uint32_t>(0);
    const uint32_t owned_lane_stride = get_arg_val<uint32_t>(1);
    const uint32_t rank = get_arg_val<uint32_t>(2);

    uint32_t chunks_in_batch = 0, cb_chunk_offset = 0;
    auto pop_batch = [&]() {
        noc_async_write_barrier();
        cb_pop_front(chunk_cb, pages_per_fabric_chunk * chunks_in_batch);
        cb_chunk_offset = (cb_chunk_offset + chunks_in_batch) % chunk_slots_in_cb;
        chunks_in_batch = 0;
    };
    auto chunk_written = [&]() {
        const uint32_t batch_capacity = chunks_per_cb_batch < chunk_slots_in_cb - cb_chunk_offset
                                            ? chunks_per_cb_batch
                                            : chunk_slots_in_cb - cb_chunk_offset;
        if (++chunks_in_batch == batch_capacity) {
            pop_batch();
        }
    };
    if constexpr (kInputInterleaved) {
        for_each_fabric_chunk(
            geometry,
            num_dram_banks,
            pages_per_fabric_chunk,
            first_owned_lane,
            owned_lane_stride,
            kWholeChipShard,
            [&](uint32_t outer_slice, uint32_t page_in_slice, uint32_t num_pages) {
                cb_wait_front(chunk_cb, pages_per_fabric_chunk * (chunks_in_batch + 1));
                noc_async_write(
                    get_read_ptr(chunk_cb) + chunks_in_batch * fabric_chunk_bytes,
                    output.get_noc_addr(output_page_index(geometry, rank, outer_slice, page_in_slice)),
                    num_pages * page_bytes);
                chunk_written();
            });
    } else {
        // the reader publishes each run on input_run_cb; this core's share is conversion blocks c, c + n, ...
        const uint32_t local_copy_core_index = first_owned_lane;
        const uint32_t num_converting_cores = owned_lane_stride;
        const uint32_t num_fabric_link_workers = get_arg_val<uint32_t>(3);
        const uint32_t blocks_converted_addr = get_semaphore(blocks_converted_semaphore_id + local_copy_core_index);
        const uint32_t valid_pages_in_shard = geometry.num_outer_slices * geometry.valid_pages_per_outer_slice;
        uint32_t pages_to_convert = 0;
        for (uint32_t conversion_block = local_copy_core_index;
             conversion_block * conversion_block_pages < valid_pages_in_shard;
             conversion_block += num_converting_cores) {
            pages_to_convert += (conversion_block + 1) * conversion_block_pages < valid_pages_in_shard
                                    ? conversion_block_pages
                                    : valid_pages_in_shard - conversion_block * conversion_block_pages;
        }
        // The output is interleaved: page p is at row p / num_dram_banks of physical bank p % num_dram_banks. So
        // compute each bank's base once instead of an accessor lookup per page.
        uint64_t dram_bank_base[num_dram_banks];
        for (uint32_t bank = 0; bank < num_dram_banks; ++bank) {
            dram_bank_base[bank] = output.get_noc_addr(bank);
        }
        for (uint32_t pages_converted = 0; pages_converted < pages_to_convert;) {
            cb_wait_front(input_run_cb, 1);
            auto* input_run = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(input_run_cb));
            const uint32_t flat_page = input_run[0], num_pages = input_run[1] & 0x7FFFFFFFu;
            const bool last_run_of_block = (input_run[1] & 0x80000000u) != 0;
            cb_pop_front(input_run_cb, 1);
            const uint32_t outer_slice = flat_page / geometry.valid_pages_per_outer_slice,
                           page_in_slice = flat_page % geometry.valid_pages_per_outer_slice;
            cb_wait_front(chunk_cb, pages_per_fabric_chunk * (chunks_in_batch + 1));
            const uint32_t chunk_slot = get_read_ptr(chunk_cb) + chunks_in_batch * fabric_chunk_bytes;
            const uint32_t first_output_page = output_page_index(geometry, rank, outer_slice, page_in_slice);
            for (uint32_t i = 0; i < num_pages; ++i) {
                const uint32_t out_page = first_output_page + i;
                const uint64_t dst_noc_addr =
                    dram_bank_base[out_page % num_dram_banks] + (out_page / num_dram_banks) * page_bytes;
                if constexpr (page_bytes <= NOC_MAX_BURST_SIZE) {
                    noc_async_write_one_packet(chunk_slot + i * page_bytes, dst_noc_addr, page_bytes);
                } else {  // a page larger than one NoC packet
                    noc_async_write(chunk_slot + i * page_bytes, dst_noc_addr, page_bytes);
                }
            }
            pages_converted += num_pages;
            const uint32_t batch_capacity = chunks_per_cb_batch < chunk_slots_in_cb - cb_chunk_offset
                                                ? chunks_per_cb_batch
                                                : chunk_slots_in_cb - cb_chunk_offset;
            if (last_run_of_block) {
                ++chunks_in_batch;
                pop_batch();  // write barrier: the conversion block has landed before it is signalled
            } else if (++chunks_in_batch == batch_capacity) {
                noc_async_writes_flushed();  // the CB batch has left L1; it only has to land by the block's end
                cb_pop_front(chunk_cb, pages_per_fabric_chunk * chunks_in_batch);
                cb_chunk_offset = (cb_chunk_offset + chunks_in_batch) % chunk_slots_in_cb;
                chunks_in_batch = 0;
            }
            if (last_run_of_block) {
                for (uint32_t link_worker = 0; link_worker < num_fabric_link_workers; ++link_worker) {
                    noc_semaphore_inc(
                        get_noc_addr(
                            get_arg_val<uint32_t>(4 + 2 * link_worker),
                            get_arg_val<uint32_t>(5 + 2 * link_worker),
                            blocks_converted_addr),
                        1);
                }
            }
        }
        noc_async_atomic_barrier();
    }
    if (chunks_in_batch > 0) {
        pop_batch();
    }
}
