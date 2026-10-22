// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sparse KV push reader, driven by the all-gathered plan tables (sparse_push_plan_device_operation.hpp). For each
// requester routed through this sender, the owner reads the requester's per-owner counts (giving its run base in the
// requester's compact buffer) and scans the requester's bitmap over the blocks this chip owns, in order, turning set
// bits into local cache rows. Row k of a run goes to packet k / P (P rows per packet), whose rows sit `banks` pages
// apart in bank lane (k / P) % banks, matching the plan op's rewritten indices.
//
// Parts (push_part): 0 every needed row; 1 rows in blocks before the plan's split block (the chunk start: the prefix,
// pushed by a sparse_sdpa with a fused push); 2 the rest, continuing each run where part 1 ended.

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "sparse_push_route.hpp"

#include <cstdint>

namespace {
constexpr uint32_t kData = 0, kMeta = 1, kTable = 2, kScalar = 3;  // CB ids (sparse_push_program.hpp)
constexpr uint32_t kBatch = 4;                                     // packets per read barrier
constexpr uint32_t kTablePageBytes = 4096;                         // plan table pages
constexpr uint32_t kWindowBytes = 128;   // 64 B aligned read window covering one block's bitmap words
constexpr uint32_t kBlocksPerFill = 16;  // blocks fetched per bitmap read barrier

inline uint32_t lowest_bit(uint32_t x) {
    // de Bruijn count-trailing-zeros (x != 0)
    constexpr uint8_t kDeBruijn[32] = {0,  1,  28, 2,  29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4,  8,
                                       31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6,  11, 5,  10, 9};
    return kDeBruijn[((x & (0u - x)) * 0x077CB531u) >> 27];
}
}  // namespace

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t pages_per_packet = get_compile_time_arg_val(1);
    constexpr uint32_t banks = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_slot = get_compile_time_arg_val(3);
    constexpr bool batch_from_meta = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t num_devices = get_compile_time_arg_val(5);
    constexpr uint32_t stripe_chunk = get_compile_time_arg_val(6);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t table_pages = get_compile_time_arg_val(8);  // per requester
    constexpr uint32_t owner_off = get_compile_time_arg_val(9);
    constexpr uint32_t bitmap_off = get_compile_time_arg_val(10);
    constexpr uint32_t owner_prefix_off = get_compile_time_arg_val(11);
    constexpr uint32_t push_part = get_compile_time_arg_val(12);
    static_assert(push_part <= 2, "push_part is 0 (all), 1 (prefix) or 2 (remainder)");
    constexpr auto input_args = TensorAccessorArgs<13>();
    constexpr auto table_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr uint32_t after_table = table_args.next_compile_time_args_offset();
    constexpr auto batch_args = TensorAccessorArgs<batch_from_meta ? after_table : 0>();
    constexpr uint32_t words_per_block = stripe_chunk / 32;
    constexpr uint32_t unit = pages_per_packet * banks;

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t table_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t batch_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t num_layers = get_common_arg_val<uint32_t>(3);
    const uint32_t layer_idx = get_common_arg_val<uint32_t>(4);
    const uint32_t scalar_slot = get_common_arg_val<uint32_t>(5);
    const uint32_t link = get_arg_val<uint32_t>(0);
    const uint32_t dir = get_arg_val<uint32_t>(1);
    const uint32_t my_rank = get_arg_val<uint32_t>(2);
    const uint32_t sub = get_arg_val<uint32_t>(3);
    const uint32_t sub_count = get_arg_val<uint32_t>(4);
    const uint32_t links = get_arg_val<uint32_t>(5);
    constexpr uint32_t dest_chip_ids[] = DEST_CHIP_ID;
    constexpr uint32_t dest_mesh_ids[] = DEST_MESH_ID;

    const auto input = TensorAccessor(input_args, input_addr, page_bytes);
    const auto table = TensorAccessor(table_args, table_addr, kTablePageBytes);

    cb_reserve_back(kScalar, 1);
    const uint32_t scalar_l1 = get_write_ptr(kScalar);
    auto* scalar = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scalar_l1);
    uint32_t slot = scalar_slot;
    if constexpr (batch_from_meta) {
        const auto batch = TensorAccessor(batch_args, batch_addr);
        noc_async_read(batch.get_noc_addr(0), scalar_l1, sizeof(uint32_t));
        noc_async_read_barrier();
        invalidate_l1_cache();
        slot = scalar[0] * num_layers + layer_idx;
    }
    const uint32_t slot_base = slot * pages_per_slot;

    // kTable holds the requester's table header and per-owner counts, then a window of its bitmap blocks.
    cb_reserve_back(kTable, 1);
    const uint32_t header_l1 = get_write_ptr(kTable);
    const uint32_t blocks_l1 = header_l1 + 512;
    auto* header = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(header_l1);
    auto* counts = header + owner_off;
    auto* prefix_counts = header + owner_prefix_off;

    uint32_t batched = 0;
    auto flush_batch = [&]() {
        if (batched) {
            noc_async_read_barrier();
            cb_push_back(kData, batched);
            cb_push_back(kMeta, batched);
            batched = 0;
        }
    };
    // Packets of a channel are dealt round robin to its sub_count senders (every sender walks all of them).
    uint32_t packet_turn = 0;
    uint32_t nth_on_dir = 0;
    for (uint32_t dest = 0; dest < num_devices; ++dest) {
        const bool on_dir = dest == my_rank ? dir == 0
                                            : sparse_push::route<MESH_ROWS, MESH_COLS>(
                                                  my_rank,
                                                  dest,
                                                  static_cast<uint32_t>(tt::tt_fabric::get_next_hop_router_direction(
                                                      dest_mesh_ids[dest], dest_chip_ids[dest])))
                                                      .dir == dir;
        // Requesters on this direction alternate between links.
        if (!on_dir || nth_on_dir++ % links != link) {
            continue;
        }
        const uint32_t first_page = dest * table_pages;
        noc_async_read(table.get_noc_addr(first_page), header_l1, (owner_prefix_off + 32) * 4);
        noc_async_read_barrier();
        invalidate_l1_cache();
        uint32_t base = 0;
        for (uint32_t o = 0; o < my_rank; ++o) {
            base += (counts[o] + unit - 1) / unit * unit;
        }
        const uint32_t before = prefix_counts[my_rank];
        const uint32_t part_rows =
            push_part == 0 ? counts[my_rank] : (push_part == 1 ? before : counts[my_rank] - before);
        if (part_rows == 0) {
            continue;
        }

        // Rank of this part's first row in the run; the remainder may start mid-packet.
        const uint32_t first_rank = push_part == 2 ? before : 0;
        uint32_t packet = first_rank / pages_per_packet, lane0 = first_rank % pages_per_packet, n = 0;
        bool take = false;  // this sender's packet
        uint32_t data_l1 = 0;
        auto open_packet = [&]() {
            take = packet_turn == sub;
            if (take) {
                cb_reserve_back(kData, batched + 1);
                cb_reserve_back(kMeta, batched + 1);
                data_l1 = get_write_ptr(kData) + batched * pages_per_packet * page_bytes;
            }
        };
        auto close_packet = [&]() {
            if (take) {
                auto* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kMeta) + batched * 16);
                meta[0] = dest;
                meta[1] = base + (packet % banks) + (packet / banks) * unit + lane0 * banks;
                meta[2] = n;
                if (++batched == kBatch) {
                    flush_batch();
                }
            }
            ++packet;
            n = 0;
            lane0 = 0;
            packet_turn = packet_turn + 1 == sub_count ? 0 : packet_turn + 1;
        };
        open_packet();

        // This chip's blocks: my_rank, my_rank + N, ...; block jb holds local rows [jb * sc, (jb + 1) * sc).
        const uint32_t active_blocks = header[3] < num_blocks ? header[3] : num_blocks;
        uint32_t my_blocks = active_blocks > my_rank ? (active_blocks - my_rank + num_devices - 1) / num_devices : 0;
        // This owner's blocks before the split block (the chunk start's block): my_rank + j * N < split. The split is
        // chunk aligned only when the chunk start is (a padded/rotated chunk starts mid-slab), so it is NOT a whole
        // block row of every owner: owners below split % N have one more prefix block.
        const uint32_t split_block = header[4];
        const uint32_t split_j = split_block > my_rank ? (split_block - my_rank + num_devices - 1) / num_devices : 0;
        uint32_t jb_first = 0;
        if constexpr (push_part == 1) {
            my_blocks = my_blocks < split_j ? my_blocks : split_j;
        } else if constexpr (push_part == 2) {
            jb_first = split_j;
        }
        for (uint32_t jb0 = jb_first; jb0 < my_blocks; jb0 += kBlocksPerFill) {
            const uint32_t fill = my_blocks - jb0 < kBlocksPerFill ? my_blocks - jb0 : kBlocksPerFill;
            for (uint32_t f = 0; f < fill; ++f) {
                const uint32_t word = bitmap_off + (my_rank + (jb0 + f) * num_devices) * words_per_block;
                const uint32_t start = (word * 4) & ~63u;
                const uint32_t page = first_page + start / kTablePageBytes;
                const uint32_t in_page = start % kTablePageBytes;
                const uint32_t first =
                    kTablePageBytes - in_page < kWindowBytes ? kTablePageBytes - in_page : kWindowBytes;
                const uint32_t dst = blocks_l1 + f * kWindowBytes;
                noc_async_read(table.get_noc_addr(page) + in_page, dst, first);
                if (first < kWindowBytes) {
                    noc_async_read(table.get_noc_addr(page + 1), dst + first, kWindowBytes - first);
                }
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t f = 0; f < fill; ++f) {
                const uint32_t jb = jb0 + f;
                const uint32_t word = bitmap_off + (my_rank + jb * num_devices) * words_per_block;
                auto* w =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(blocks_l1 + f * kWindowBytes + (word * 4) % 64);
                for (uint32_t k = 0; k < words_per_block; ++k) {
                    uint32_t bits = w[k];
                    while (bits) {
                        const uint32_t row = jb * stripe_chunk + k * 32 + lowest_bit(bits);
                        bits &= bits - 1;
                        if (take) {
                            noc_async_read(input.get_noc_addr(slot_base + row), data_l1 + n * page_bytes, page_bytes);
                        }
                        if (lane0 + ++n == pages_per_packet) {
                            close_packet();
                            open_packet();
                        }
                    }
                }
            }
        }
        if (n) {
            close_packet();
        }
    }
    flush_batch();
    // Terminator.
    cb_reserve_back(kData, 1);
    cb_reserve_back(kMeta, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kMeta))[2] = 0;
    cb_push_back(kData, 1);
    cb_push_back(kMeta, 1);
}
