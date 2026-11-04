// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "sparse_push_constants.hpp"
#include "sparse_push_packet_builder.hpp"

#include <algorithm>
#include <cstdint>

using namespace sparse_push;

// One of the two senders of a (link, direction) fabric channel on an owner chip. For each requester the channel
// serves, it reads the requester's plan table, walks the bitmap over the blocks this chip owns, and hands the rows the
// requester needs to the writer in packets (PacketBuilder), which sends them into the requester's compact buffer.

namespace {
constexpr uint32_t kTablePageBytes = plan_table::kPageBytes;
}  // namespace

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t pages_per_packet = get_compile_time_arg_val(1);
    constexpr uint32_t banks = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_slot = get_compile_time_arg_val(3);
    constexpr bool has_batch_index_tensor = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t num_devices = get_compile_time_arg_val(5);
    constexpr uint32_t stripe_chunk = get_compile_time_arg_val(6);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t table_pages = get_compile_time_arg_val(8);  // per requester
    constexpr uint32_t out_pages = get_compile_time_arg_val(9);
    // Which of the requester's needed rows this push sends. The plan splits them at the block holding the chunk
    // start (header word kWordSplitBlock):
    //   kPrefix: rows of the blocks before the split. Earlier chunks wrote them, so they can be pushed before this
    //            chunk's KV exists, e.g. beside the previous layer's attention.
    //   kRest:   the split block onwards, which includes this chunk's rows. It runs once the layer has written its
    //            KV, and continues each owner's run where kPrefix stopped (possibly mid-packet), so the two together
    //            fill exactly the slots a kAll push fills.
    //   kAll:    every needed row in one push, when nothing was prefetched.
    constexpr auto part = static_cast<PushPart>(get_compile_time_arg_val(10));
    constexpr uint32_t num_slots = get_compile_time_arg_val(11);
    static_assert(stripe_chunk % plan_table::kRowsPerWord == 0 && stripe_chunk <= kMaxStripeChunk);
    constexpr auto input_args = TensorAccessorArgs<12>();
    constexpr auto table_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr uint32_t after_table = table_args.next_compile_time_args_offset();
    constexpr auto batch_args = TensorAccessorArgs<has_batch_index_tensor ? after_table : 0>();
    constexpr uint32_t owner_off = plan_table::kOwnerOff;
    constexpr uint32_t bitmap_off = plan_table::kBitmapOff;
    constexpr uint32_t owner_prefix_off = plan_table::kOwnerPrefixOff;
    constexpr uint32_t words_per_block = stripe_chunk / plan_table::kRowsPerWord;

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t table_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t batch_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t num_layers = get_common_arg_val<uint32_t>(3);
    const uint32_t layer_idx = get_common_arg_val<uint32_t>(4);
    const uint32_t host_slot = get_common_arg_val<uint32_t>(5);
    const uint32_t link = get_arg_val<uint32_t>(0);
    const uint32_t dir = get_arg_val<uint32_t>(1);
    const uint32_t my_rank = get_arg_val<uint32_t>(2);
    const uint32_t sender_index = get_arg_val<uint32_t>(3);  // among this channel's senders
    const uint32_t senders = get_arg_val<uint32_t>(4);
    const uint32_t links = get_arg_val<uint32_t>(5);
    constexpr uint32_t dest_chip_ids[] = DEST_CHIP_ID;
    constexpr uint32_t dest_mesh_ids[] = DEST_MESH_ID;

    const auto input = TensorAccessor(input_args, input_addr, page_bytes);
    const auto table = TensorAccessor(table_args, table_addr, kTablePageBytes);

    // Cache slot to push: the host's, or (trace-safe) the user id read on device * num_layers + layer_idx.
    cb_reserve_back(kUserId, 1);
    const uint32_t user_id_l1 = get_write_ptr(kUserId);
    uint32_t slot = host_slot;
    if constexpr (has_batch_index_tensor) {
        const auto batch = TensorAccessor(batch_args, batch_addr);
        noc_async_read(batch.get_noc_addr(0), user_id_l1, sizeof(uint32_t));
        noc_async_read_barrier();
        invalidate_l1_cache();
        slot = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(user_id_l1) * num_layers + layer_idx;
    }
    ASSERT(slot < num_slots);  // watcher builds only: the host checks only its own batch_index
    const uint32_t slot_base = slot * pages_per_slot;

    // kTable holds the requester's table header and per-owner counts, then a window of its bitmap blocks.
    cb_reserve_back(kTable, 1);
    const uint32_t header_l1 = get_write_ptr(kTable);
    const uint32_t blocks_l1 = header_l1 + bitmap_off * sizeof(uint32_t);
    auto* header = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(header_l1);
    auto* owner_rows = header + owner_off;                // [owner]: rows the requester needs from it
    auto* owner_prefix_rows = header + owner_prefix_off;  // [owner]: of those, rows in the prefix (kPrefix)

    // Whether this sender's (link, direction) channel carries `requester`'s rows; called once per requester, in
    // order. The direction is the fabric's next hop, as in the writer, so the completion increments follow the data.
    // This chip's own rows need no fabric and go to direction 0. Requesters on one direction alternate between the
    // links.
    uint32_t nth_on_dir = 0;
    auto serves = [&](uint32_t requester) {
        bool on_dir = dir == 0;
        if (requester != my_rank) {
            on_dir = static_cast<uint32_t>(tt::tt_fabric::get_next_hop_router_direction(
                         dest_mesh_ids[requester], dest_chip_ids[requester])) == dir;
        }
        return on_dir && nth_on_dir++ % links == link;
    };

    // How many of this chip's blocks (global my_rank, my_rank + N, ...) lie below global block `global_block`. The
    // split need not fall on a whole row of blocks, so owners below split % N get one more prefix block.
    auto owned_below = [&](uint32_t global_block) {
        return global_block > my_rank ? (global_block - my_rank + num_devices - 1) / num_devices : 0;
    };

    // Byte offset, in a requester's table, of local block `block`'s bitmap words (global block my_rank + block * N).
    auto bitmap_offset = [&](uint32_t block) {
        return (bitmap_off + (my_rank + block * num_devices) * words_per_block) * sizeof(uint32_t);
    };

    PacketBuilder<page_bytes, pages_per_packet, banks, out_pages, decltype(input)> packets(
        input, slot_base, sender_index, senders);
    for (uint32_t requester = 0; requester < num_devices; ++requester) {
        if (!serves(requester)) {
            continue;
        }
        const uint32_t table_page = requester * table_pages;  // the requester's table starts here
        noc_async_read(table.get_noc_addr(table_page), header_l1, plan_table::kHeaderWords * sizeof(uint32_t));
        noc_async_read_barrier();
        invalidate_l1_cache();
        // Watcher builds only: the plan must have been built for this push's geometry.
        ASSERT(header[plan_table::kWordMagic] == plan_table::kMagic);
        ASSERT(
            header[plan_table::kWordNumDevices] == num_devices && header[plan_table::kWordStripeChunk] == stripe_chunk);
        ASSERT(header[plan_table::kWordPagesPerPacket] == pages_per_packet && header[plan_table::kWordBanks] == banks);
        ASSERT(part != PushPart::kRest || header[plan_table::kWordHasSplit] != 0);

        // The requester's compact buffer holds one padded run per owner, in owner order, ours after the lower ones.
        uint32_t run_base = 0;
        for (uint32_t owner = 0; owner < my_rank; ++owner) {
            run_base += padded_run_pages<pages_per_packet, banks>(owner_rows[owner]);
        }
        const uint32_t my_rows = owner_rows[my_rank];
        const uint32_t prefix_rows = owner_prefix_rows[my_rank];
        const uint32_t part_rows = part == PushPart::kAll      ? my_rows
                                   : part == PushPart::kPrefix ? prefix_rows
                                                               : my_rows - prefix_rows;
        if (part_rows == 0) {
            continue;
        }
        packets.begin_run(requester, run_base, part == PushPart::kRest ? prefix_rows : 0);

        // This chip owns global blocks my_rank, my_rank + N, ...: its local block j is global block my_rank + j * N
        // and holds local rows [j * stripe_chunk, (j + 1) * stripe_chunk). The plan's bitmap is valid only below
        // active_blocks; kPrefix takes the blocks below the split block, kRest the others.
        const uint32_t plan_blocks = header[plan_table::kWordActiveBlocks];  // blocks the plan wrote
        const uint32_t active_blocks = std::min(plan_blocks, num_blocks);
        const uint32_t prefix_blocks = owned_below(header[plan_table::kWordSplitBlock]);
        const uint32_t first_block = part == PushPart::kRest ? prefix_blocks : 0;
        const uint32_t end_block = part == PushPart::kPrefix ? std::min(owned_below(active_blocks), prefix_blocks)
                                                             : owned_below(active_blocks);

        // Reads local block `block`'s bitmap words into kTable window slot `window`. A block's words (at most 64 B)
        // can start anywhere, so the read covers them in whole kDramReadAlign units from the boundary at or below
        // them (at most kWindowBytes), in two reads when the words cross a table page. Reading no further than the
        // words keeps the last block of the last requester's table inside the table.
        auto read_block_bitmap = [&](uint32_t block, uint32_t window) {
            const uint32_t words_start = bitmap_offset(block);
            const uint32_t window_start = words_start & ~(kDramReadAlign - 1);
            const uint32_t window_bytes =
                (words_start - window_start + words_per_block * sizeof(uint32_t) + kDramReadAlign - 1) &
                ~(kDramReadAlign - 1);
            const uint32_t window_page = table_page + window_start / kTablePageBytes;
            const uint32_t page_offset = window_start % kTablePageBytes;
            const uint32_t first_bytes = std::min(kTablePageBytes - page_offset, window_bytes);
            const uint32_t window_l1 = blocks_l1 + window * kWindowBytes;
            noc_async_read(table.get_noc_addr(window_page) + page_offset, window_l1, first_bytes);
            if (first_bytes < window_bytes) {
                noc_async_read(
                    table.get_noc_addr(window_page + 1), window_l1 + first_bytes, window_bytes - first_bytes);
            }
        };

        // Walk this chip's blocks in order. Their bitmaps are fetched kBlocksPerBatch at a time behind one barrier.
        for (uint32_t block = first_block; block < end_block;) {
            const uint32_t batch_end = std::min(block + kBlocksPerBatch, end_block);
            for (uint32_t bitmap_block = block; bitmap_block < batch_end; ++bitmap_block) {
                read_block_bitmap(bitmap_block, bitmap_block - block);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t window = 0; block < batch_end; ++block, ++window) {
                const uint32_t in_window = bitmap_offset(block) % kDramReadAlign;
                auto* block_bits =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(blocks_l1 + window * kWindowBytes + in_window);
                for (uint32_t word_index = 0; word_index < words_per_block; ++word_index) {
                    uint32_t word_bits = block_bits[word_index];
                    while (word_bits) {
                        // Lowest set bit first: rows in ascending order.
                        const uint32_t row =
                            block * stripe_chunk + word_index * plan_table::kRowsPerWord + __builtin_ctz(word_bits);
                        word_bits &= word_bits - 1;
                        packets.add_row(row);
                    }
                }
            }
        }
        packets.end_run();
    }
    packets.finish();
}
