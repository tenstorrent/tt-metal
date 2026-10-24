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
//
// With a hot table (sparse_push_hot), rows it marks for multicast are not pushed per requester: each owner's multicast
// rows of the part go out once, as packets of consecutive hot slots (the writer sends each on its direction's branch,
// so the four directions' senders together reach every chip): hot packet g is sent by sender (g / links) % sub_count
// of link g % links in every direction. Every run starts past the hot region and holds only the requester's other
// (cold) rows, packed: those of the current part (blocks from the split block on) first, then those of the prefix
// part. So the remainder starts at rank 0 and the prefix at the requester's cold current-part rows from this owner.

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "sparse_push_route.hpp"

#include <cstdint>

namespace {
constexpr uint32_t kData = 0, kMeta = 1, kTable = 2, kScalar = 3, kHotMine = 5;  // CB ids (sparse_push_program.hpp)
constexpr uint32_t kMulticast = 0xFFFFFFFFu;                                     // packet destination: every chip
constexpr uint32_t kBatch = 4;                                                   // packets per read barrier
constexpr uint32_t kTablePageBytes = 4096;                                       // plan table pages
constexpr uint32_t kWindowBytes = 128;   // 64 B aligned read window covering one block's bitmap words
constexpr uint32_t kBlocksPerFill = 64;  // blocks fetched per bitmap read barrier

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
    constexpr bool has_hot = get_compile_time_arg_val(13) != 0;
    constexpr uint32_t hot_bases_off = get_compile_time_arg_val(14);  // hot table: [owner * 2 + part] sub-run base
    constexpr auto input_args = TensorAccessorArgs<15>();
    constexpr auto table_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto hot_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr uint32_t after_table = hot_args.next_compile_time_args_offset();
    constexpr auto batch_args = TensorAccessorArgs<batch_from_meta ? after_table : 0>();
    constexpr uint32_t words_per_block = stripe_chunk / 32;
    constexpr uint32_t unit = pages_per_packet * banks;

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t table_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t batch_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t num_layers = get_common_arg_val<uint32_t>(3);
    const uint32_t layer_idx = get_common_arg_val<uint32_t>(4);
    const uint32_t scalar_slot = get_common_arg_val<uint32_t>(5);
    const uint32_t hot_addr = get_common_arg_val<uint32_t>(6);
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
    const auto hot_table = TensorAccessor(hot_args, hot_addr, kTablePageBytes);

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

    // kTable: a window of bitmap blocks, then the header and per-owner counts of every requester this sender serves,
    // then (the remainder) each one's first block window. Headers and those windows are read in one batch each, not
    // one DRAM round trip per requester.
    constexpr uint32_t kHeaderBytes = (owner_prefix_off + 32) * 4;
    static_assert(kHeaderBytes % 64 == 0, "header slots stay 64 B aligned");
    cb_reserve_back(kTable, 1);
    const uint32_t blocks_l1 = get_write_ptr(kTable);
    const uint32_t headers_l1 = blocks_l1 + kBlocksPerFill * kWindowBytes;
    const uint32_t first_windows_l1 = headers_l1 + num_devices * kHeaderBytes;

    uint32_t batched = 0;
    auto flush_batch = [&]() {
        if (batched) {
            noc_async_read_barrier();
            cb_push_back(kData, batched);
            cb_push_back(kMeta, batched);
            batched = 0;
        }
    };
    // The requesters this sender serves: those whose first hop is this direction, alternating between links.
    uint32_t served[num_devices];
    uint32_t n_served = 0;
    for (uint32_t dest = 0, nth_on_dir = 0; dest < num_devices; ++dest) {
        const bool on_dir = dest == my_rank ? dir == 0
                                            : sparse_push::route<MESH_ROWS, MESH_COLS>(
                                                  my_rank,
                                                  dest,
                                                  static_cast<uint32_t>(tt::tt_fabric::get_next_hop_router_direction(
                                                      dest_mesh_ids[dest], dest_chip_ids[dest])))
                                                      .dir == dir;
        if (on_dir && nth_on_dir++ % links == link) {
            served[n_served++] = dest;
        }
    }
    // Bitmap window of block jb (local block index) of a requester's table (or of the hot table): 64 B aligned,
    // possibly split by a page.
    auto read_window_of = [&](const auto& tables, uint32_t first_page, uint32_t jb, uint32_t dst) {
        const uint32_t word = bitmap_off + (my_rank + jb * num_devices) * words_per_block;
        const uint32_t start = (word * 4) & ~63u;
        const uint32_t page = first_page + start / kTablePageBytes;
        const uint32_t in_page = start % kTablePageBytes;
        const uint32_t first = kTablePageBytes - in_page < kWindowBytes ? kTablePageBytes - in_page : kWindowBytes;
        noc_async_read(tables.get_noc_addr(page) + in_page, dst, first);
        if (first < kWindowBytes) {
            noc_async_read(tables.get_noc_addr(page + 1), dst + first, kWindowBytes - first);
        }
    };
    auto read_window = [&](uint32_t first_page, uint32_t jb, uint32_t dst) {
        read_window_of(table, first_page, jb, dst);
    };
    // This owner's blocks of a requester's table: my_rank, my_rank + N, ... below its active blocks; and those
    // before the split block (the chunk start's block): my_rank + j * N < split. The split is chunk aligned only when
    // the chunk start is (a padded/rotated chunk starts mid-slab), so it is NOT a whole block row of every owner:
    // owners below split % N have one more prefix block.
    auto owned_blocks = [&](const volatile tt_l1_ptr uint32_t* header) {
        const uint32_t active_blocks = header[3] < num_blocks ? header[3] : num_blocks;
        return active_blocks > my_rank ? (active_blocks - my_rank + num_devices - 1) / num_devices : 0;
    };
    auto split_blocks = [&](const volatile tt_l1_ptr uint32_t* header) {
        const uint32_t split_block = header[4];
        return split_block > my_rank ? (split_block - my_rank + num_devices - 1) / num_devices : 0;
    };
    for (uint32_t i = 0; i < n_served; ++i) {
        noc_async_read(table.get_noc_addr(served[i] * table_pages), headers_l1 + i * kHeaderBytes, kHeaderBytes);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    // The first current-part window of every requester: the remainder starts there and, with a hot table, the prefix
    // part's ranks start behind the current part's cold rows.
    constexpr bool prefetch_current = push_part == 2 || (has_hot && push_part == 1);
    if constexpr (prefetch_current) {
        for (uint32_t i = 0; i < n_served; ++i) {
            auto* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(headers_l1 + i * kHeaderBytes);
            const uint32_t jb = split_blocks(h);
            if (jb < owned_blocks(h)) {
                read_window(served[i] * table_pages, jb, first_windows_l1 + i * kWindowBytes);
            }
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
    }
    // The hot table: its header (hot region slots, this owner's sub-run bases) and this owner's hot words, kept for
    // the whole push (words_per_block per local block).
    uint32_t hot_slots = 0, hot_base[2] = {0, 0}, hot_blocks = 0, hot_split_j = 0;
    uint32_t* hot_mine = nullptr;
    if constexpr (has_hot) {
        cb_reserve_back(kHotMine, 1);
        hot_mine = reinterpret_cast<uint32_t*>(get_write_ptr(kHotMine));
        noc_async_read(hot_table.get_noc_addr(0), blocks_l1, kHeaderBytes);
        noc_async_read_barrier();
        invalidate_l1_cache();
        auto* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(blocks_l1);
        hot_slots = h[1];
        hot_base[0] = h[hot_bases_off + my_rank * 2];
        hot_base[1] = h[hot_bases_off + my_rank * 2 + 1];
        hot_blocks = owned_blocks(h);
        hot_split_j = split_blocks(h);
        for (uint32_t jb0 = 0; jb0 < hot_blocks; jb0 += kBlocksPerFill) {
            const uint32_t fill = hot_blocks - jb0 < kBlocksPerFill ? hot_blocks - jb0 : kBlocksPerFill;
            for (uint32_t f = 0; f < fill; ++f) {
                read_window_of(hot_table, 0, jb0 + f, blocks_l1 + f * kWindowBytes);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t f = 0; f < fill; ++f) {
                const uint32_t word = bitmap_off + (my_rank + (jb0 + f) * num_devices) * words_per_block;
                auto* w =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(blocks_l1 + f * kWindowBytes + (word * 4) % 64);
                for (uint32_t k = 0; k < words_per_block; ++k) {
                    hot_mine[(jb0 + f) * words_per_block + k] = w[k];
                }
            }
        }
    }
    auto hot_word = [&](uint32_t jb, uint32_t k) -> uint32_t {
        if constexpr (has_hot) {
            return jb < hot_blocks ? hot_mine[jb * words_per_block + k] : 0u;
        } else {
            return 0u;
        }
    };

    // Packets: runs of consecutive slots within one P-slot group. A run's slot = base + bank lane of its rank.
    bool take = false;  // this sender's packet
    uint32_t data_l1 = 0, n = 0, start = 0;
    auto open_packet = [&](bool mine) {
        take = mine;
        if (take) {
            cb_reserve_back(kData, batched + 1);
            cb_reserve_back(kMeta, batched + 1);
            data_l1 = get_write_ptr(kData) + batched * pages_per_packet * page_bytes;
        }
    };
    auto emit = [&](uint32_t dest, uint32_t base) {
        if (take) {
            auto* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kMeta) + batched * 16);
            const uint32_t group = start / pages_per_packet;
            meta[0] = dest;
            meta[1] = base + (group % banks) + (group / banks) * unit + (start % pages_per_packet) * banks;
            meta[2] = n;
            if (++batched == kBatch) {
                flush_batch();
            }
        }
        n = 0;
    };

    // Packets of a channel are dealt round robin to its sub_count senders (every sender walks all of them).
    uint32_t packet_turn = 0;
    for (uint32_t i = 0; i < n_served; ++i) {
        const uint32_t dest = served[i];
        const uint32_t first_page = dest * table_pages;
        auto* header = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(headers_l1 + i * kHeaderBytes);
        auto* counts = header + owner_off;
        auto* prefix_counts = header + owner_prefix_off;
        uint32_t base = hot_slots;
        for (uint32_t o = 0; o < my_rank; ++o) {
            base += (counts[o] + unit - 1) / unit * unit;
        }
        const uint32_t before = prefix_counts[my_rank];
        const uint32_t part_rows =
            push_part == 0 ? counts[my_rank] : (push_part == 1 ? before : counts[my_rank] - before);
        if (part_rows == 0) {
            continue;
        }

        auto close = [&]() {
            emit(dest, base);
            packet_turn = packet_turn + 1 == sub_count ? 0 : packet_turn + 1;
        };
        // This chip's blocks: my_rank, my_rank + N, ...; block jb holds local rows [jb * sc, (jb + 1) * sc). The
        // prefix part is blocks [0, prefix_end), the current part [split_j, my_blocks).
        const uint32_t my_blocks = owned_blocks(header);
        const uint32_t split_j = split_blocks(header);
        const uint32_t prefix_end = my_blocks < split_j ? my_blocks : split_j;
        // Bitmap words of blocks [jb_begin, jb_end), a fill at a time: f(jb, words). The first window of the current
        // part was prefetched with the headers.
        auto for_each_block = [&](uint32_t jb_begin, uint32_t jb_end, auto&& f) {
            for (uint32_t jb0 = jb_begin; jb0 < jb_end; jb0 += kBlocksPerFill) {
                const uint32_t fill = jb_end - jb0 < kBlocksPerFill ? jb_end - jb0 : kBlocksPerFill;
                const bool prefetched = prefetch_current && jb0 == split_j;
                for (uint32_t b = prefetched ? 1 : 0; b < fill; ++b) {
                    read_window(first_page, jb0 + b, blocks_l1 + b * kWindowBytes);
                }
                noc_async_read_barrier();
                invalidate_l1_cache();
                for (uint32_t b = 0; b < fill; ++b) {
                    const uint32_t jb = jb0 + b;
                    const uint32_t word = bitmap_off + (my_rank + jb * num_devices) * words_per_block;
                    const uint32_t window =
                        prefetched && b == 0 ? first_windows_l1 + i * kWindowBytes : blocks_l1 + b * kWindowBytes;
                    f(jb, reinterpret_cast<volatile tt_l1_ptr uint32_t*>(window + (word * 4) % 64));
                }
            }
        };
        // Push the cold rows of blocks [jb_begin, jb_end) from rank `rank` on (multicast rows have no slot here).
        uint32_t rank = 0;
        auto push_blocks = [&](uint32_t jb_begin, uint32_t jb_end) {
            for_each_block(jb_begin, jb_end, [&](uint32_t jb, volatile tt_l1_ptr uint32_t* w) {
                for (uint32_t k = 0; k < words_per_block; ++k) {
                    for (uint32_t bits = w[k] & ~hot_word(jb, k); bits != 0; bits &= bits - 1) {
                        if (n == 0) {
                            start = rank;
                            open_packet(packet_turn == sub);
                        }
                        if (take) {
                            const uint32_t row = jb * stripe_chunk + k * 32 + lowest_bit(bits);
                            noc_async_read(input.get_noc_addr(slot_base + row), data_l1 + n * page_bytes, page_bytes);
                        }
                        ++n;
                        if (++rank % pages_per_packet == 0) {
                            close();
                        }
                    }
                }
            });
        };
        if constexpr (has_hot) {
            // Current part first, then the prefix part.
            if constexpr (push_part == 1) {
                for_each_block(split_j, my_blocks, [&](uint32_t jb, volatile tt_l1_ptr uint32_t* w) {
                    for (uint32_t k = 0; k < words_per_block; ++k) {
                        rank += __builtin_popcount(w[k] & ~hot_word(jb, k));
                    }
                });
            } else {
                push_blocks(split_j, my_blocks);
            }
            if constexpr (push_part != 2) {
                push_blocks(0, prefix_end);
            }
        } else {
            // Prefix part first, then the current part (the remainder may start mid-packet).
            if constexpr (push_part != 2) {
                push_blocks(0, prefix_end);
            }
            if constexpr (push_part != 1) {
                rank = before;
                push_blocks(split_j, my_blocks);
            }
        }
        if (n) {
            close();
        }
    }

    // This owner's hot rows of the part, as packets of consecutive hot slots: one multicast reaches every chip.
    if constexpr (has_hot) {
        uint32_t group = 0;  // hot packets of this owner so far, over both parts
        for (uint32_t part = push_part == 2 ? 1 : 0; part <= (push_part == 1 ? 0u : 1u); ++part) {
            const uint32_t jb_begin = part == 0 ? 0 : hot_split_j;
            const uint32_t jb_end = part == 0 ? (hot_split_j < hot_blocks ? hot_split_j : hot_blocks) : hot_blocks;
            uint32_t h = 0;  // rank in the (owner, part) sub-run
            for (uint32_t jb = jb_begin; jb < jb_end; ++jb) {
                for (uint32_t k = 0; k < words_per_block; ++k) {
                    for (uint32_t bits = hot_mine[jb * words_per_block + k]; bits != 0; bits &= bits - 1) {
                        if (n == 0) {
                            start = h;
                            open_packet(group % links == link && (group / links) % sub_count == sub);
                        }
                        if (take) {
                            const uint32_t row = jb * stripe_chunk + k * 32 + lowest_bit(bits);
                            noc_async_read(input.get_noc_addr(slot_base + row), data_l1 + n * page_bytes, page_bytes);
                        }
                        ++n;
                        if (++h % pages_per_packet == 0) {
                            emit(kMulticast, hot_base[part]);
                            ++group;
                        }
                    }
                }
            }
            if (n) {
                emit(kMulticast, hot_base[part]);
                ++group;
            }
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
