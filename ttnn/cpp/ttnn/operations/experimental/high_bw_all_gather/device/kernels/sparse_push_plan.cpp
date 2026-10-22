// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// on-device plan for the sparse KV push (see sparse_push_plan_device_operation.hpp for the table layout).
// One requester chip, `cores` cores, three phases separated by barriers:
//   1. scatter: each core sets the bits of its index segments in its own L1 bitmap. Bits live in 16-word chunks
//      (512 rows); a chunk is zeroed on first touch and marked in the core's dirty map, so untouched chunks are
//      never zeroed or read. Each core also records the largest row it saw.
//   2. reduce: every core reads all dirty maps and extents; the active blocks (up to the largest row) are split over
//      the cores, and each core ORs the dirty chunks of its range from every core and fills that range's in-block
//      word prefixes and block counts;
//   3. rewrite: each core pulls every other core's reduced range (its image now holds the full active table),
//      derives the per-owner block prefixes and compact-buffer bases, writes its share of table pages, and rewrites
//      its index segments to compact slots, sharing the rewrite with NCRISC (REWRITE_HELPER).
// Work units are (row, segment) pairs of seg_len indices, numbered row-major.

#include "api/dataflow/dataflow_api.h"

#include <cstdint>

namespace {
constexpr uint32_t kImage = 0, kRow = 1, kPre = 2, kScratch = 3, kDirty = 4, kAllDirty = 5, kRow2 = 6, kSync = 7;
constexpr uint32_t kSentinel = 0xFFFFFFFFu;
constexpr uint32_t kTablePageBytes = 4096;
constexpr uint32_t kChunkWords = 16;  // bitmap words per dirty chunk
constexpr uint32_t kChunkShift = 9;   // rows per chunk = 16 * 32

inline uint32_t popcount(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    return (x * 0x01010101u) >> 24;
}
}  // namespace

void kernel_main() {
    constexpr uint32_t num_devices = get_compile_time_arg_val(0);
    constexpr uint32_t stripe_chunk = get_compile_time_arg_val(1);
    constexpr uint32_t words_per_block = stripe_chunk / 32;
    constexpr uint32_t bitmap_off = get_compile_time_arg_val(2);  // words
    constexpr uint32_t prefix_off = get_compile_time_arg_val(3);  // words; in-block word prefixes, one byte each
    constexpr uint32_t counts_off = get_compile_time_arg_val(4);  // words; block counts
    constexpr uint32_t owner_off = get_compile_time_arg_val(5);   // words; per-owner row counts
    constexpr uint32_t table_pages = get_compile_time_arg_val(6);
    constexpr uint32_t pages_per_packet = get_compile_time_arg_val(7);
    constexpr uint32_t banks = get_compile_time_arg_val(8);
    constexpr uint32_t topk = get_compile_time_arg_val(9);
    constexpr uint32_t cores = get_compile_time_arg_val(10);
    constexpr uint32_t block_align = get_compile_time_arg_val(11);
    constexpr uint32_t arrive_sem_id = get_compile_time_arg_val(12);
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(13);
    constexpr uint32_t magic = get_compile_time_arg_val(14);
    constexpr uint32_t seg_len = get_compile_time_arg_val(15);       // indices per work unit
    constexpr uint32_t dirty_words = get_compile_time_arg_val(16);   // dirty map words (after a 16-word header)
    constexpr uint32_t ready_sem_id = get_compile_time_arg_val(17);  // BRISC -> NCRISC: table and bases are ready
    constexpr uint32_t num_blocks = get_compile_time_arg_val(18);
    constexpr uint32_t or_go_sem_id = get_compile_time_arg_val(19);      // BRISC -> NCRISC: chunk copies have landed
    constexpr uint32_t or_done_sem_id = get_compile_time_arg_val(20);    // NCRISC -> BRISC: its half of the OR is done
    constexpr uint32_t owner_prefix_off = get_compile_time_arg_val(21);  // words; per-owner rows before the split
    constexpr bool has_split = get_compile_time_arg_val(22) != 0;
    constexpr auto indices_args = TensorAccessorArgs<23>();
    constexpr auto table_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto compact_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto split_args = TensorAccessorArgs<has_split ? compact_args.next_compile_time_args_offset() : 0>();
    constexpr uint32_t unit = pages_per_packet * banks;
    constexpr uint32_t row_bytes = topk * sizeof(uint32_t);
    constexpr uint32_t seg_bytes = seg_len * sizeof(uint32_t);
    constexpr uint32_t segs_per_row = topk / seg_len;
    constexpr uint32_t dirty_slot_words = 16 + dirty_words;

    size_t arg = 0;
    const uint32_t indices_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t table_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t compact_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t core_idx = get_arg_val<uint32_t>(arg++);
    // BRISC scatters [unit_start, unit_start + unit_count) and rewrites the first `rewrite_split` of them; NCRISC
    // (REWRITE_HELPER) rewrites the rest.
    const uint32_t unit_start = get_arg_val<uint32_t>(arg++);
    const uint32_t unit_count = get_arg_val<uint32_t>(arg++);
    const uint32_t rewrite_split = get_arg_val<uint32_t>(arg++);
    const uint32_t split_addr = has_split ? get_arg_val<uint32_t>(arg++) : 0;
    // Common runtime args: every core's (x, y), core 0 first.
    auto core_noc = [&](uint32_t c, uint32_t l1) {
        return get_noc_addr(get_common_arg_val<uint32_t>(2 * c), get_common_arg_val<uint32_t>(2 * c + 1), l1);
    };

    const auto indices = TensorAccessor(indices_args, indices_addr, row_bytes);
    const auto table = TensorAccessor(table_args, table_addr, kTablePageBytes);
    const auto compact = TensorAccessor(compact_args, compact_addr, row_bytes);

    cb_reserve_back(kImage, 1);
#ifdef REWRITE_HELPER
    cb_reserve_back(kRow2, 1);
    const uint32_t row_l1 = get_write_ptr(kRow2);
#else
    cb_reserve_back(kRow, 1);
    const uint32_t row_l1 = get_write_ptr(kRow);
#endif
    cb_reserve_back(kPre, 1);
    cb_reserve_back(kScratch, 1);
    cb_reserve_back(kDirty, 1);
    cb_reserve_back(kAllDirty, 1);
    cb_reserve_back(kSync, 1);
    auto* sync = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kSync));  // NCRISC's OR chunk range
    const uint32_t image_l1 = get_write_ptr(kImage);
    const uint32_t scratch_l1 = get_write_ptr(kScratch);
    const uint32_t dirty_l1 = get_write_ptr(kDirty);
    const uint32_t all_dirty_l1 = get_write_ptr(kAllDirty);
    // Plain (non-volatile) pointers: every NOC transfer into these buffers is followed by a barrier and
    // invalidate_l1_cache() before the core reads them.
    auto* image = reinterpret_cast<uint32_t*>(image_l1);
    uint32_t* bits = image + bitmap_off;
    auto* prefix = reinterpret_cast<uint8_t*>(image + prefix_off);
    uint32_t* counts = image + counts_off;
    auto* pre = reinterpret_cast<uint32_t*>(get_write_ptr(kPre));
    uint32_t* base = pre + num_blocks;  // compact-buffer run base per owner, shared with NCRISC
    auto* seg = reinterpret_cast<uint32_t*>(row_l1);
    auto* scratch = reinterpret_cast<uint32_t*>(scratch_l1);
    auto* dirty_slot = reinterpret_cast<uint32_t*>(dirty_l1);  // [0] = largest row + 1 (0 = none), dirty map at [16]
    uint32_t* dirty = dirty_slot + 16;
    auto* all_dirty = reinterpret_cast<uint32_t*>(all_dirty_l1);

    const uint32_t arrive_sem = get_semaphore(arrive_sem_id);
    const uint32_t go_sem = get_semaphore(go_sem_id);
    uint32_t phase = 0;
    auto barrier = [&]() {
        ++phase;
        noc_semaphore_inc(core_noc(0, arrive_sem), 1);
        noc_async_atomic_barrier();
        if (core_idx == 0) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_sem), cores * phase);
            for (uint32_t c = 0; c < cores; ++c) {
                noc_semaphore_inc(core_noc(c, go_sem), 1);
            }
            noc_async_atomic_barrier();
        }
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem), phase);
        invalidate_l1_cache();
    };
    auto seg_noc = [&](const auto& accessor, uint32_t u) {
        return accessor.get_noc_addr(u / segs_per_row) + (u % segs_per_row) * seg_bytes;
    };
    // Rewrite segments [u0, u1): slot = base[owner] + bank lane of the row's rank among this owner's rows.
    auto rewrite = [&](uint32_t u0, uint32_t u1) {
        for (uint32_t u = u0; u < u1; ++u) {
            noc_async_write_barrier();  // the previous segment's write has left the buffer
            noc_async_read(seg_noc(indices, u), row_l1, seg_bytes);
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t k = 0; k < seg_len; ++k) {
                const uint32_t v = seg[k];
                if (v == kSentinel) {
                    continue;
                }
                const uint32_t blk = v / stripe_chunk;
                const uint32_t w = v >> 5;
                const uint32_t rank = pre[blk] + prefix[w] + popcount(bits[w] & ((1u << (v & 31)) - 1u));
                const uint32_t j = rank / pages_per_packet, i = rank % pages_per_packet;
                seg[k] = base[blk % num_devices] + (j % banks) + (j / banks) * unit + i * banks;
            }
            noc_async_write(row_l1, seg_noc(compact, u), seg_bytes);
        }
        noc_async_write_barrier();
    };
    // Copy image bytes [off, off + bytes) to the same offsets of the table, page by page.
    auto write_span = [&](uint32_t off, uint32_t bytes) {
        for (uint32_t end = off + bytes; off < end;) {
            const uint32_t in_page = off % kTablePageBytes;
            const uint32_t n = end - off < kTablePageBytes - in_page ? end - off : kTablePageBytes - in_page;
            noc_async_write(image_l1 + off, table.get_noc_addr(off / kTablePageBytes) + in_page, n);
            off += n;
        }
    };
    auto is_dirty = [&](uint32_t c, uint32_t ch) {
        return (all_dirty[c * dirty_slot_words + 16 + (ch >> 5)] >> (ch & 31)) & 1u;
    };
    // OR every core's copy of chunks [ch_a, ch_b), whose dirty copies sit in scratch from slot `slot` on, in the
    // (chunk, core) order the reads were issued.
    auto or_chunks = [&](uint32_t ch_a, uint32_t ch_b, uint32_t slot) {
        for (uint32_t ch = ch_a; ch < ch_b; ++ch) {
            uint32_t acc[kChunkWords];
            for (uint32_t i = 0; i < kChunkWords; ++i) {
                acc[i] = 0;
            }
            for (uint32_t c = 0; c < cores; ++c) {
                if (is_dirty(c, ch)) {
                    const uint32_t* src = scratch + slot++ * kChunkWords;
                    for (uint32_t i = 0; i < kChunkWords; ++i) {
                        acc[i] |= src[i];
                    }
                }
            }
            uint32_t* dst = bits + ch * kChunkWords;
            for (uint32_t i = 0; i < kChunkWords; ++i) {
                dst[i] = acc[i];
            }
        }
    };
    auto wait_flag = [](volatile tt_l1_ptr uint32_t* flag) {
        while (*flag == 0) {
            invalidate_l1_cache();
        }
        invalidate_l1_cache();
    };
    auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_sem_id));
    auto* or_go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(or_go_sem_id));
    auto* or_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(or_done_sem_id));
#ifdef REWRITE_HELPER
    wait_flag(or_go);
    or_chunks(sync[0], sync[1], sync[2]);
    *or_done = 1;
    wait_flag(ready);
    rewrite(unit_start + rewrite_split, unit_start + unit_count);
    return;
#endif

    // 1. Scatter.
    for (uint32_t w = 0; w < dirty_words; ++w) {
        dirty[w] = 0;
    }
    uint32_t row_end = 0;  // largest row seen + 1
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        noc_async_read(seg_noc(indices, u), row_l1, seg_bytes);
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t k = 0; k < seg_len; ++k) {
            const uint32_t v = seg[k];
            if (v == kSentinel) {
                continue;
            }
            const uint32_t c = v >> kChunkShift;
            const uint32_t flag = 1u << (c & 31);
            if (!(dirty[c >> 5] & flag)) {
                dirty[c >> 5] |= flag;
                uint32_t* chunk = bits + c * kChunkWords;
                for (uint32_t i = 0; i < kChunkWords; ++i) {
                    chunk[i] = 0;
                }
            }
            bits[v >> 5] |= 1u << (v & 31);
            row_end = v + 1 > row_end ? v + 1 : row_end;
        }
    }
    dirty_slot[0] = row_end;
    barrier();

    // 2. Reduce. Everyone reads every core's dirty map and extent.
    for (uint32_t c = 0; c < cores; ++c) {
        noc_async_read(core_noc(c, dirty_l1), all_dirty_l1 + c * dirty_slot_words * 4, dirty_slot_words * 4);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    uint32_t active_rows = 0;
    for (uint32_t c = 0; c < cores; ++c) {
        const uint32_t e = all_dirty[c * dirty_slot_words];
        active_rows = e > active_rows ? e : active_rows;
    }
    const uint32_t active_blocks = (active_rows + stripe_chunk - 1) / stripe_chunk;
    const uint32_t blocks_per_core =
        ((active_blocks + cores - 1) / cores + block_align - 1) / block_align * block_align;
    const uint32_t b0 = core_idx * blocks_per_core < active_blocks ? core_idx * blocks_per_core : active_blocks;
    const uint32_t b1 = b0 + blocks_per_core < active_blocks ? b0 + blocks_per_core : active_blocks;
    const uint32_t w0 = b0 * words_per_block, w1 = b1 * words_per_block;
    if (b1 == b0) {
        sync[0] = sync[1] = sync[2] = 0;
        *or_go = 1;
    } else {
        // My word range in whole chunks; w0 is chunk aligned (block_align * words_per_block is a multiple of 16).
        const uint32_t ch0 = w0 / kChunkWords, ch1 = (w1 + kChunkWords - 1) / kChunkWords;
        uint32_t slots = 0;
        for (uint32_t ch = ch0; ch < ch1; ++ch) {
            for (uint32_t c = 0; c < cores; ++c) {
                if (is_dirty(c, ch)) {
                    noc_async_read(
                        core_noc(c, image_l1 + (bitmap_off + ch * kChunkWords) * 4),
                        scratch_l1 + slots++ * kChunkWords * 4,
                        kChunkWords * 4);
                }
            }
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        // Split the chunks by dirty copies between BRISC and NCRISC.
        uint32_t split = ch0, split_slot = 0;
        while (split < ch1 && split_slot * 2 < slots) {
            for (uint32_t c = 0; c < cores; ++c) {
                split_slot += is_dirty(c, split);
            }
            ++split;
        }
        sync[0] = split;
        sync[1] = ch1;
        sync[2] = split_slot;
        *or_go = 1;
        or_chunks(ch0, split, 0);
        wait_flag(or_done);
        for (uint32_t b = b0; b < b1; ++b) {
            uint32_t n = 0;
            for (uint32_t k = 0; k < words_per_block; ++k) {
                const uint32_t w = b * words_per_block + k;
                prefix[w] = static_cast<uint8_t>(n);
                n += popcount(bits[w]);
            }
            counts[b] = n;
        }
        // The owners' push reads the bitmap from the gathered tables; counts go along for inspection.
        write_span((bitmap_off + w0) * 4, (w1 - w0) * 4);
        write_span((counts_off + b0) * 4, (b1 - b0) * 4);
    }
    barrier();

    // Assemble what this core's rewrite needs: every active block count, and the reduced bitmap words and in-block
    // prefixes of the chunks its own indices touched (a chunk lies inside one reducer's range).
    for (uint32_t c = 0; c < cores; ++c) {
        const uint32_t cb0 = c * blocks_per_core < active_blocks ? c * blocks_per_core : active_blocks;
        const uint32_t cb1 = cb0 + blocks_per_core < active_blocks ? cb0 + blocks_per_core : active_blocks;
        if (c != core_idx && cb1 > cb0) {
            noc_async_read(
                core_noc(c, image_l1 + (counts_off + cb0) * 4), image_l1 + (counts_off + cb0) * 4, (cb1 - cb0) * 4);
        }
    }
    // Runs of consecutive dirty chunks within one reducer's range are fetched with one read per field.
    const uint32_t range_words = blocks_per_core * words_per_block;
    uint32_t run_start = 0, run_len = 0, run_owner = 0;
    auto flush_run = [&]() {
        if (run_len && run_owner != core_idx) {
            const uint32_t bit_l1 = image_l1 + (bitmap_off + run_start * kChunkWords) * 4;
            const uint32_t pre_l1 = image_l1 + prefix_off * 4 + run_start * kChunkWords;
            noc_async_read(core_noc(run_owner, bit_l1), bit_l1, run_len * kChunkWords * 4);
            noc_async_read(core_noc(run_owner, pre_l1), pre_l1, run_len * kChunkWords);
        }
        run_len = 0;
    };
    for (uint32_t dw = 0; dw < dirty_words; ++dw) {
        uint32_t m = dirty[dw];
        while (m) {
            const uint32_t low = m & (0u - m);
            m ^= low;
            const uint32_t ch = dw * 32 + popcount(low - 1);
            const uint32_t r = ch * kChunkWords / range_words;
            if (run_len && (r != run_owner || ch != run_start + run_len)) {
                flush_run();
            }
            if (run_len == 0) {
                run_start = ch;
                run_owner = r;
            }
            ++run_len;
        }
    }
    flush_run();
    noc_async_read_barrier();
    invalidate_l1_cache();

    // 3. Per-owner exclusive block prefixes and per-owner totals of this requester's rows.
    uint32_t owner_total[num_devices];
    for (uint32_t o = 0; o < num_devices; ++o) {
        owner_total[o] = 0;
    }
    // The split block: rows before the chunk start form the prefix part of each owner's run.
    uint32_t split_block = active_blocks;
    if constexpr (has_split) {
        const auto split = TensorAccessor(split_args, split_addr);
        noc_async_read(split.get_noc_addr(0), row_l1, sizeof(uint32_t));
        noc_async_read_barrier();
        invalidate_l1_cache();
        split_block = seg[0] / stripe_chunk;
    }
    uint32_t owner_prefix[num_devices];
    for (uint32_t b = 0, o = 0; b < active_blocks; ++b) {
        if (b == split_block) {
            for (uint32_t k = 0; k < num_devices; ++k) {
                owner_prefix[k] = owner_total[k];
            }
        }
        pre[b] = owner_total[o];
        owner_total[o] += counts[b];
        o = o + 1 == num_devices ? 0 : o + 1;
    }
    for (uint32_t o = 0, acc = 0; o < num_devices; ++o) {
        base[o] = acc;
        acc += (owner_total[o] + unit - 1) / unit * unit;
        image[owner_off + o] = owner_total[o];
        image[owner_prefix_off + o] = split_block < active_blocks ? owner_prefix[o] : owner_total[o];
    }
    image[4] = split_block;
    image[0] = magic;
    image[1] = num_devices;
    image[2] = stripe_chunk;
    image[3] = active_blocks;  // the push scans only blocks below this; later table words may be stale
    if (core_idx == 0) {
        write_span(0, bitmap_off * 4);  // header and per-owner counts
    }

    *ready = 1;  // NCRISC may start its share of the rewrite
    rewrite(unit_start, unit_start + rewrite_split);
}
