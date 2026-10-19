// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PROTOTYPE on-device plan for the sparse KV push (see sparse_push_plan_device_operation.hpp for the table layout).
// One requester chip, `cores` cores, three phases separated by barriers:
//   1. scatter: each core sets the bits of its index segments in its own L1 bitmap (the table image's bitmap field);
//   2. reduce:  each core ORs one block range across every core's bitmap, then fills that range's in-block word
//               prefixes and block counts in its image;
//   3. assemble + rewrite: each core copies every other core's reduced ranges into its image (all images are now
//      the full table), derives the per-owner block prefixes and compact-buffer bases, writes its share of table
//      pages, and rewrites its index segments to compact slots.
// Work units are (row, segment) pairs of seg_len indices, numbered row-major.

#include "api/dataflow/dataflow_api.h"

#include <cstdint>

namespace {
constexpr uint32_t kImage = 0, kRow = 1, kPre = 2, kScratch = 3;
constexpr uint32_t kSentinel = 0xFFFFFFFFu;
constexpr uint32_t kTablePageBytes = 4096;

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
    constexpr uint32_t bitmap_words = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t bitmap_off = get_compile_time_arg_val(4);  // words
    constexpr uint32_t prefix_off = get_compile_time_arg_val(5);  // words; in-block word prefixes, one byte each
    constexpr uint32_t counts_off = get_compile_time_arg_val(6);  // words; block counts
    constexpr uint32_t owner_off = get_compile_time_arg_val(7);   // words; per-owner row counts
    constexpr uint32_t table_pages = get_compile_time_arg_val(8);
    constexpr uint32_t pages_per_packet = get_compile_time_arg_val(9);
    constexpr uint32_t banks = get_compile_time_arg_val(10);
    constexpr uint32_t topk = get_compile_time_arg_val(11);
    constexpr uint32_t cores = get_compile_time_arg_val(12);
    constexpr uint32_t blocks_per_core = get_compile_time_arg_val(13);
    constexpr uint32_t arrive_sem_id = get_compile_time_arg_val(14);
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(15);
    constexpr uint32_t magic = get_compile_time_arg_val(16);
    constexpr uint32_t seg_len = get_compile_time_arg_val(17);  // indices per work unit
    constexpr auto indices_args = TensorAccessorArgs<18>();
    constexpr auto table_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto compact_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr uint32_t unit = pages_per_packet * banks;
    constexpr uint32_t row_bytes = topk * sizeof(uint32_t);
    constexpr uint32_t seg_bytes = seg_len * sizeof(uint32_t);
    constexpr uint32_t segs_per_row = topk / seg_len;
    constexpr uint32_t zeros_bytes = MEM_ZEROS_SIZE;

    size_t arg = 0;
    const uint32_t indices_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t table_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t compact_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t core_idx = get_arg_val<uint32_t>(arg++);
    const uint32_t unit_start = get_arg_val<uint32_t>(arg++);
    const uint32_t unit_count = get_arg_val<uint32_t>(arg++);
    // Common runtime args: every core's (x, y), core 0 first.
    auto core_noc = [&](uint32_t c, uint32_t l1) {
        return get_noc_addr(get_common_arg_val<uint32_t>(2 * c), get_common_arg_val<uint32_t>(2 * c + 1), l1);
    };
    const uint32_t lead_x = get_common_arg_val<uint32_t>(0);
    const uint32_t lead_y = get_common_arg_val<uint32_t>(1);

    const auto indices = TensorAccessor(indices_args, indices_addr, row_bytes);
    const auto table = TensorAccessor(table_args, table_addr, kTablePageBytes);
    const auto compact = TensorAccessor(compact_args, compact_addr, row_bytes);

    cb_reserve_back(kImage, 1);
    cb_reserve_back(kRow, 1);
    cb_reserve_back(kPre, 1);
    cb_reserve_back(kScratch, 1);
    const uint32_t image_l1 = get_write_ptr(kImage);
    const uint32_t row_l1 = get_write_ptr(kRow);
    const uint32_t scratch_l1 = get_write_ptr(kScratch);
    // Plain (non-volatile) pointers: every NOC transfer into these buffers is followed by a barrier and
    // invalidate_l1_cache() before the core reads them.
    auto* image = reinterpret_cast<uint32_t*>(image_l1);
    uint32_t* bits = image + bitmap_off;
    auto* prefix = reinterpret_cast<uint8_t*>(image + prefix_off);
    uint32_t* counts = image + counts_off;
    auto* pre = reinterpret_cast<uint32_t*>(get_write_ptr(kPre));
    auto* seg = reinterpret_cast<uint32_t*>(row_l1);
    auto* scratch = reinterpret_cast<uint32_t*>(scratch_l1);

    const uint32_t arrive_sem = get_semaphore(arrive_sem_id);
    const uint32_t go_sem = get_semaphore(go_sem_id);
    uint32_t phase = 0;
    auto barrier = [&]() {
        ++phase;
        noc_semaphore_inc(get_noc_addr(lead_x, lead_y, arrive_sem), 1);
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

#ifdef SPARSE_PUSH_PLAN_TIMING
    // Phase end stamps (core 0, wall clock), stored in header words 3..8.
    uint32_t stamps[7];
    stamps[0] = get_timestamp_32b();
#define PLAN_STAMP(i) stamps[i] = get_timestamp_32b()
#else
#define PLAN_STAMP(i)
#endif

    // 1. Scatter. The bitmap is zeroed from the hardware zeros region while the first segment loads.
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bitmap_words * 4; off += zeros_bytes) {
        const uint32_t n = bitmap_words * 4 - off < zeros_bytes ? bitmap_words * 4 - off : zeros_bytes;
        noc_async_read(zeros, image_l1 + bitmap_off * 4 + off, n);
    }
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        noc_async_read(seg_noc(indices, u), row_l1, seg_bytes);
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t k = 0; k < seg_len; ++k) {
            const uint32_t v = seg[k];
            if (v != kSentinel) {
                bits[v >> 5] |= 1u << (v & 31);
            }
        }
    }
    noc_async_read_barrier();  // zero fill, when this core has no segments
    barrier();
    PLAN_STAMP(1);

    // 2. Reduce my block range: fetch every other core's copy of it at once, then OR.
    const uint32_t b0 = core_idx * blocks_per_core < num_blocks ? core_idx * blocks_per_core : num_blocks;
    const uint32_t b1 = b0 + blocks_per_core < num_blocks ? b0 + blocks_per_core : num_blocks;
    const uint32_t w0 = b0 * words_per_block, w1 = b1 * words_per_block;
    if (b1 > b0) {
        const uint32_t range_words = w1 - w0;
        constexpr uint32_t slot_words = blocks_per_core * words_per_block;
        for (uint32_t c = 0, s = 0; c < cores; ++c) {
            if (c != core_idx) {
                noc_async_read(
                    core_noc(c, image_l1 + (bitmap_off + w0) * 4), scratch_l1 + s++ * slot_words * 4, range_words * 4);
            }
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        PLAN_STAMP(6);
        // Word-outer so each word is accumulated in a register; loads of different copies are independent.
        for (uint32_t w = 0; w < range_words; ++w) {
            const uint32_t* src = scratch + w;
            uint32_t a0 = bits[w0 + w], a1 = 0, a2 = 0, a3 = 0;
            uint32_t s = 0;
            for (; s + 4 <= cores - 1; s += 4) {
                a0 |= src[s * slot_words];
                a1 |= src[(s + 1) * slot_words];
                a2 |= src[(s + 2) * slot_words];
                a3 |= src[(s + 3) * slot_words];
            }
            for (; s < cores - 1; ++s) {
                a0 |= src[s * slot_words];
            }
            bits[w0 + w] = a0 | a1 | a2 | a3;
        }
        for (uint32_t b = b0; b < b1; ++b) {
            uint32_t n = 0;
            for (uint32_t k = 0; k < words_per_block; ++k) {
                const uint32_t w = b * words_per_block + k;
                prefix[w] = static_cast<uint8_t>(n);
                n += popcount(bits[w]);
            }
            counts[b] = n;
        }
    }
    barrier();
    PLAN_STAMP(2);

    // 3. Assemble the full table in this core's image.
    for (uint32_t c = 0; c < cores; ++c) {
        const uint32_t cb0 = c * blocks_per_core < num_blocks ? c * blocks_per_core : num_blocks;
        const uint32_t cb1 = cb0 + blocks_per_core < num_blocks ? cb0 + blocks_per_core : num_blocks;
        if (c == core_idx || cb1 == cb0) {
            continue;
        }
        const uint32_t cw0 = cb0 * words_per_block, cw1 = cb1 * words_per_block;
        const uint32_t spans[3][2] = {
            {(bitmap_off + cw0) * 4, (cw1 - cw0) * 4},
            {prefix_off * 4 + cw0, cw1 - cw0},
            {(counts_off + cb0) * 4, (cb1 - cb0) * 4}};
        for (const auto& s : spans) {
            noc_async_read(core_noc(c, image_l1 + s[0]), image_l1 + s[0], s[1]);
        }
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    PLAN_STAMP(3);

    // Per-owner exclusive block prefixes and per-owner totals of this requester's rows.
    uint32_t owner_total[num_devices];
    for (uint32_t o = 0; o < num_devices; ++o) {
        owner_total[o] = 0;
    }
    for (uint32_t b = 0, o = 0; b < num_blocks; ++b) {
        pre[b] = owner_total[o];
        owner_total[o] += counts[b];
        o = o + 1 == num_devices ? 0 : o + 1;
    }
    uint32_t base[num_devices];
    for (uint32_t o = 0, acc = 0; o < num_devices; ++o) {
        base[o] = acc;
        acc += (owner_total[o] + unit - 1) / unit * unit;
        image[owner_off + o] = owner_total[o];
    }
    image[0] = magic;
    image[1] = num_devices;
    image[2] = stripe_chunk;
    for (uint32_t p = core_idx; p < table_pages; p += cores) {
        noc_async_write(image_l1 + p * kTablePageBytes, table.get_noc_addr(p), kTablePageBytes);
    }

    // Rewrite my segments: slot = base[owner] + bank lane of the row's rank among this owner's rows.
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
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
#ifdef SPARSE_PUSH_PLAN_TIMING
    PLAN_STAMP(4);
    barrier();  // every core has rewritten its segments
    PLAN_STAMP(5);
    if (core_idx == 0) {
        for (uint32_t i = 0; i < 7; ++i) {
            image[3 + i] = stamps[i];
        }
        noc_async_write(image_l1, table.get_noc_addr(0), kTablePageBytes);
        noc_async_write_barrier();
    }
#endif
}
