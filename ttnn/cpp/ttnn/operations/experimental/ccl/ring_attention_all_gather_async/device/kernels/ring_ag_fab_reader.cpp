// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// EXPERIMENT: fabric_all_gather's link-worker reader (fabric_all_gather_reader.cpp) for the ring attention
// all-gather, extended to several inputs (an outgoing shard is that rank's shard of every input, in input order).
// Reads its outgoing shards into the chunk CB, one fabric chunk per chunk slot, a CB batch at a time: outgoing
// shard 0 is this chip's own shard; forwarded shard k is read from this chip's output once the shards-arrived
// counter reaches k.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/operations/experimental/fabric_all_gather/device/kernels/fabric_all_gather_chunk_walk.hpp"

using namespace ttnn::operations::experimental::fabric_all_gather::chunk_walk;

void kernel_main() {
    constexpr uint32_t chunk_cb = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t pages_per_fabric_chunk = get_compile_time_arg_val(2);
    constexpr uint32_t num_dram_banks = get_compile_time_arg_val(3);
    constexpr uint32_t chunks_per_cb_batch = get_compile_time_arg_val(4);  // the chunk CB holds 2 CB batches
    constexpr uint32_t num_inputs = get_compile_time_arg_val(5);
    constexpr auto input_args = TensorAccessorArgs<6>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr uint32_t fabric_chunk_bytes = pages_per_fabric_chunk * page_bytes;
    constexpr uint32_t chunk_slots_in_cb = 2 * chunks_per_cb_batch;

    // Args: [2] shards-arrived counter (this direction's semaphore) [4] number of ranks [5] first owned lane [6] owned
    // lane stride [7] number of outgoing shards, then the outgoing shard ranks, then per input: outer slices, pages per
    // outer slice, input address, output address.
    auto* shards_arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(2));
    const uint32_t num_ranks = get_arg_val<uint32_t>(4);
    const uint32_t first_owned_lane = get_arg_val<uint32_t>(5);
    const uint32_t owned_lane_stride = get_arg_val<uint32_t>(6);
    const uint32_t num_outgoing_shards = get_arg_val<uint32_t>(7);
    constexpr uint32_t kOutgoingShardsArg = 8;
    const uint32_t inputs_arg = kOutgoingShardsArg + num_outgoing_shards;

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

    for (uint32_t outgoing_index = 0; outgoing_index < num_outgoing_shards; ++outgoing_index) {
        const uint32_t rank = get_arg_val<uint32_t>(kOutgoingShardsArg + outgoing_index);
        if (outgoing_index > 0) {
            noc_semaphore_wait_min(shards_arrived, outgoing_index);  // upstream's outgoing shard k - 1 is in our output
        }
        for (uint32_t input_index = 0; input_index < num_inputs; ++input_index) {
            const uint32_t input_arg = inputs_arg + 4 * input_index;
            const uint32_t pages_per_outer_slice = get_arg_val<uint32_t>(input_arg + 1);
            const ChipShardGeometry geometry{
                get_arg_val<uint32_t>(input_arg), pages_per_outer_slice, pages_per_outer_slice, num_ranks};
            const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(input_arg + 2), page_bytes);
            const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(input_arg + 3), page_bytes);
            for_each_fabric_chunk(
                geometry,
                num_dram_banks,
                pages_per_fabric_chunk,
                first_owned_lane,
                owned_lane_stride,
                kWholeChipShard,
                [&](uint32_t outer_slice, uint32_t page_in_slice, uint32_t num_pages) {
                    const uint32_t chunk_slot = next_chunk_slot();
                    if (outgoing_index > 0) {
                        noc_async_read(
                            output.get_noc_addr(output_page_index(geometry, rank, outer_slice, page_in_slice)),
                            chunk_slot,
                            num_pages * page_bytes);
                    } else {
                        noc_async_read(
                            input.get_noc_addr(outer_slice * geometry.pages_per_outer_slice + page_in_slice),
                            chunk_slot,
                            num_pages * page_bytes);
                    }
                    chunk_slot_filled();
                });
        }
        if (chunks_in_batch > 0) {
            push_batch();
        }
    }
}
