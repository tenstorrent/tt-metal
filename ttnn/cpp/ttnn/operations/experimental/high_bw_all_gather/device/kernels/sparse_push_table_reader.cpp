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
//
// The reader keeps DRAM busy: the bitmap windows of the next fill are read while it walks the current one, and each
// packet's rows carry the transaction id of its packet slot, so a packet goes to the writer as soon as its own rows
// are in, with the next ones already in flight.

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "sparse_push_route.hpp"

#include <cstdint>

namespace {
constexpr uint32_t kData = 0, kMeta = 1, kTable = 2, kScalar = 3, kHotMine = 5;  // CB ids (sparse_push_program.hpp)
constexpr uint32_t kPacketSlots = 8;                 // pages of kData and kMeta (sparse_push_program.hpp)
constexpr uint32_t kMetaEntryBytes = 16;             // kMeta page
constexpr uint32_t kMulticast = 0xFFFFFFFFu;         // packet destination: every chip
constexpr uint32_t kTablePageBytes = 4096;           // plan table pages
constexpr uint32_t kWindowBytes = 128;               // 64 B aligned read window covering one block's bitmap words
constexpr uint32_t kBlocksPerFill = 32;              // blocks per window fill; two fills in flight
constexpr uint32_t kWindowTrid = 14;                 // read transaction ids of the two window fills (14, 15)
constexpr uint32_t kMaxUnpushed = kPacketSlots / 2;  // finished packets held back while their rows are in flight
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
    constexpr uint32_t packet_bytes = pages_per_packet * page_bytes;

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

    // kTable: two fills of bitmap windows, then the header and per-owner counts of every requester this sender
    // serves, then (the prefix with a hot table) each one's first current-part window. Headers and those windows are
    // read in one batch each, not one DRAM round trip per requester.
    constexpr uint32_t kHeaderBytes = (owner_prefix_off + 32) * 4;
    static_assert(kHeaderBytes % 64 == 0, "header slots stay 64 B aligned");
    cb_reserve_back(kTable, 1);
    const uint32_t windows_l1 = get_write_ptr(kTable);
    const uint32_t headers_l1 = windows_l1 + 2 * kBlocksPerFill * kWindowBytes;
    const uint32_t first_windows_l1 = headers_l1 + num_devices * kHeaderBytes;

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
    // possibly split by a page. The block's words sit at window_word(jb) words into it.
    auto read_window =
        [&](const auto& tables, uint32_t first_page, uint32_t jb, uint32_t dst) __attribute__((always_inline)) {
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
    auto window_words = [&](uint32_t window_l1, uint32_t jb) __attribute__((always_inline)) {
        const uint32_t word = bitmap_off + (my_rank + jb * num_devices) * words_per_block;
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(window_l1 + (word * 4) % 64);
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
    // The hot table: its header (hot region slots, this owner's sub-run bases) and this owner's hot words, kept for
    // the whole push (words_per_block per local block).
    uint32_t hot_slots = 0, hot_base[2] = {0, 0}, hot_blocks = 0, hot_split_j = 0;
    uint32_t* hot_mine = nullptr;
    if constexpr (has_hot) {
        noc_async_read(hot_table.get_noc_addr(0), windows_l1, kHeaderBytes);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    if constexpr (has_hot) {
        cb_reserve_back(kHotMine, 1);
        hot_mine = reinterpret_cast<uint32_t*>(get_write_ptr(kHotMine));
        auto* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(windows_l1);
        hot_slots = h[1];
        hot_base[0] = h[hot_bases_off + my_rank * 2];
        hot_base[1] = h[hot_bases_off + my_rank * 2 + 1];
        hot_blocks = owned_blocks(h);
        hot_split_j = split_blocks(h);
        // All of this owner's hot windows, two fills in flight.
        auto issue = [&](uint32_t jb0) __attribute__((always_inline)) {
            const uint32_t buf = (jb0 / kBlocksPerFill) & 1;
            noc_async_read_set_trid(kWindowTrid + buf);
            for (uint32_t jb = jb0; jb < hot_blocks && jb < jb0 + kBlocksPerFill; ++jb) {
                read_window(hot_table, 0, jb, windows_l1 + (buf * kBlocksPerFill + jb - jb0) * kWindowBytes);
            }
        };
        if (hot_blocks) {
            issue(0);
        }
        for (uint32_t jb0 = 0; jb0 < hot_blocks; jb0 += kBlocksPerFill) {
            const uint32_t buf = (jb0 / kBlocksPerFill) & 1;
            if (jb0 + kBlocksPerFill < hot_blocks) {
                issue(jb0 + kBlocksPerFill);
            }
            noc_async_read_barrier_with_trid(kWindowTrid + buf);
            for (uint32_t jb = jb0; jb < hot_blocks && jb < jb0 + kBlocksPerFill; ++jb) {
                auto* w = window_words(windows_l1 + (buf * kBlocksPerFill + jb - jb0) * kWindowBytes, jb);
                for (uint32_t k = 0; k < words_per_block; ++k) {
                    hot_mine[jb * words_per_block + k] = w[k];
                }
            }
        }
    }
    // This owner's hot words of local block jb (none past the hot table's active blocks).
    const uint32_t no_hot[words_per_block] = {};
    auto hot_words = [&](uint32_t jb) __attribute__((always_inline)) -> const uint32_t* {
        if constexpr (has_hot) {
            return jb < hot_blocks ? hot_mine + jb * words_per_block : no_hot;
        } else {
            return no_hot;
        }
    };

    // Per served requester: its run base, the rank of the part's first row in the run, and the block ranges to walk
    // in run order (segments: requester, first and end block).
    uint32_t run_base[num_devices], first_rank[num_devices];
    uint32_t seg_req[2 * num_devices], seg_begin[2 * num_devices], seg_end[2 * num_devices];
    uint32_t n_segs = 0;
    // With a hot table the prefix part's ranks start behind the requester's cold current-part rows from this owner:
    // count them from its current-part windows (the first one prefetched for every requester).
    constexpr bool count_current = has_hot && push_part == 1;
    if constexpr (count_current) {
        for (uint32_t i = 0; i < n_served; ++i) {
            auto* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(headers_l1 + i * kHeaderBytes);
            const uint32_t jb = split_blocks(h);
            if (jb < owned_blocks(h)) {
                read_window(table, served[i] * table_pages, jb, first_windows_l1 + i * kWindowBytes);
            }
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
    }
    for (uint32_t i = 0; i < n_served; ++i) {
        auto* header = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(headers_l1 + i * kHeaderBytes);
        auto* counts = header + owner_off;
        const uint32_t before = header[owner_prefix_off + my_rank];
        const uint32_t part_rows =
            push_part == 0 ? counts[my_rank] : (push_part == 1 ? before : counts[my_rank] - before);
        if (part_rows == 0) {
            continue;
        }
        uint32_t base = hot_slots;
        for (uint32_t o = 0; o < my_rank; ++o) {
            base += (counts[o] + unit - 1) / unit * unit;
        }
        run_base[i] = base;
        // This chip's blocks: my_rank, my_rank + N, ...; block jb holds local rows [jb * sc, (jb + 1) * sc). The
        // prefix part is blocks [0, prefix_end), the current part [split_j, my_blocks).
        const uint32_t my_blocks = owned_blocks(header);
        const uint32_t split_j = split_blocks(header);
        const uint32_t prefix_end = my_blocks < split_j ? my_blocks : split_j;
        auto add = [&](uint32_t b, uint32_t e) __attribute__((always_inline)) {
            if (b < e) {
                seg_req[n_segs] = i;
                seg_begin[n_segs] = b;
                seg_end[n_segs++] = e;
            }
        };
        uint32_t rank = 0;
        if constexpr (has_hot) {
            // Current part first, then the prefix part.
            if constexpr (push_part == 1) {
                for (uint32_t jb = split_j; jb < my_blocks; ++jb) {
                    uint32_t window_l1 = first_windows_l1 + i * kWindowBytes;
                    if (jb != split_j) {  // a chunk start that is not block aligned: a second current block
                        window_l1 = windows_l1;
                        read_window(table, served[i] * table_pages, jb, window_l1);
                        noc_async_read_barrier();
                        invalidate_l1_cache();
                    }
                    auto* w = window_words(window_l1, jb);
                    const uint32_t* hw = hot_words(jb);
                    for (uint32_t k = 0; k < words_per_block; ++k) {
                        rank += __builtin_popcount(w[k] & ~hw[k]);
                    }
                }
            } else {
                add(split_j, my_blocks);
            }
            if constexpr (push_part != 2) {
                add(0, prefix_end);
            }
        } else {
            // Prefix part first, then the current part (the remainder may start mid-packet).
            if constexpr (push_part != 2) {
                add(0, prefix_end);
            } else {
                rank = before;
            }
            if constexpr (push_part != 1) {
                add(split_j, my_blocks);
            }
        }
        first_rank[i] = rank;
    }

    // Packets: runs of consecutive slots within one P-slot group. A run's slot = base + bank lane of its rank.
    // Packet q (of this sender) takes slot q % kPacketSlots of kData and kMeta, and its row reads transaction id
    // slot + 1; finished packets go to the writer in order, once their reads are in.
    const uint32_t data_base = get_write_ptr(kData), meta_base = get_write_ptr(kMeta);
    uint32_t produced = 0, pushed = 0;
    bool take = false;  // this sender's packet
    uint32_t data_l1 = 0, n = 0, start = 0, read_trid = 0;
    auto push_ready = [&](bool all) __attribute__((always_inline)) {
        while (pushed < produced) {
            const uint32_t trid = pushed % kPacketSlots + 1;
            if (!all && produced - pushed <= kMaxUnpushed &&
                !ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid)) {
                break;
            }
            noc_async_read_barrier_with_trid(trid);
            cb_push_back(kData, 1);
            cb_push_back(kMeta, 1);
            ++pushed;
        }
    };
    auto open_packet = [&](bool mine) __attribute__((always_inline)) {
        take = mine;
        if (take) {
            cb_reserve_back(kData, produced - pushed + 1);
            cb_reserve_back(kMeta, produced - pushed + 1);
            const uint32_t s = produced % kPacketSlots;
            data_l1 = data_base + s * packet_bytes;
            read_trid = s + 1;
            noc_async_read_set_trid(read_trid);
        }
    };
    auto emit = [&](uint32_t dest, uint32_t base) __attribute__((always_inline)) {
        if (take) {
            auto* meta =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_base + (produced % kPacketSlots) * kMetaEntryBytes);
            const uint32_t group = start / pages_per_packet;
            meta[0] = dest;
            meta[1] = base + (group % banks) + (group / banks) * unit + (start % pages_per_packet) * banks;
            meta[2] = n;
            ++produced;
            push_ready(false);
        }
        n = 0;
    };
    auto read_row = [&](uint32_t row) __attribute__((always_inline)) {
        if (take) {
            noc_async_read(input.get_noc_addr(slot_base + row), data_l1 + n * page_bytes, page_bytes);
        }
        ++n;
    };

    // The walk: segments a fill of windows at a time, the next fill in flight. Packets of a channel are dealt round
    // robin to its sub_count senders (every sender walks all of them).
    uint32_t pf_seg = 0, pf_jb = n_segs ? seg_begin[0] : 0;
    uint32_t fill_seg[2] = {0, 0}, fill_jb[2] = {0, 0}, fill_n[2] = {0, 0};
    auto issue_fill = [&](uint32_t buf) __attribute__((always_inline)) {
        fill_n[buf] = 0;
        if (pf_seg == n_segs) {
            return;
        }
        const uint32_t e = seg_end[pf_seg];
        const uint32_t cnt = e - pf_jb < kBlocksPerFill ? e - pf_jb : kBlocksPerFill;
        fill_seg[buf] = pf_seg;
        fill_jb[buf] = pf_jb;
        fill_n[buf] = cnt;
        noc_async_read_set_trid(kWindowTrid + buf);
        const uint32_t first_page = served[seg_req[pf_seg]] * table_pages;
        for (uint32_t b = 0; b < cnt; ++b) {
            read_window(table, first_page, pf_jb + b, windows_l1 + (buf * kBlocksPerFill + b) * kWindowBytes);
        }
        noc_async_read_set_trid(read_trid);  // an open packet's rows keep its id
        pf_jb += cnt;
        if (pf_jb == e && ++pf_seg < n_segs) {
            pf_jb = seg_begin[pf_seg];
        }
    };
    uint32_t packet_turn = 0, cur_req = num_devices, dest = 0, base = 0, rank = 0;
    issue_fill(0);
    for (uint32_t buf = 0; fill_n[buf] != 0; buf ^= 1) {
        issue_fill(buf ^ 1);
        noc_async_read_barrier_with_trid(kWindowTrid + buf);
        const uint32_t i = seg_req[fill_seg[buf]];
        if (i != cur_req) {
            if (n) {
                emit(dest, base);
                packet_turn = packet_turn + 1 == sub_count ? 0 : packet_turn + 1;
            }
            cur_req = i;
            dest = served[i];
            base = run_base[i];
            rank = first_rank[i];
        }
        for (uint32_t b = 0; b < fill_n[buf]; ++b) {
            const uint32_t jb = fill_jb[buf] + b;
            auto* w = window_words(windows_l1 + (buf * kBlocksPerFill + b) * kWindowBytes, jb);
            const uint32_t* hw = hot_words(jb);
            for (uint32_t k = 0; k < words_per_block; ++k) {
                for (uint32_t bits = w[k] & ~hw[k]; bits != 0; bits &= bits - 1) {
                    if (n == 0) {
                        start = rank;
                        open_packet(packet_turn == sub);
                    }
                    read_row(jb * stripe_chunk + k * 32 + static_cast<uint32_t>(__builtin_ctz(bits)));
                    if (++rank % pages_per_packet == 0) {
                        emit(dest, base);
                        packet_turn = packet_turn + 1 == sub_count ? 0 : packet_turn + 1;
                    }
                }
            }
        }
    }
    if (n) {
        emit(dest, base);
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
                        read_row(jb * stripe_chunk + k * 32 + static_cast<uint32_t>(__builtin_ctz(bits)));
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
    push_ready(true);
    noc_async_read_set_trid(0);
    // Terminator.
    cb_reserve_back(kData, 1);
    cb_reserve_back(kMeta, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_base + (produced % kPacketSlots) * kMetaEntryBytes)[2] = 0;
    cb_push_back(kData, 1);
    cb_push_back(kMeta, 1);
}
