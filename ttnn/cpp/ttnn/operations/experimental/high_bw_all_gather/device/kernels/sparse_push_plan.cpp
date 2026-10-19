// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PROTOTYPE on-device plan for the sparse KV push (see sparse_push_plan_device_operation.hpp for the table layout).
// One requester chip, `cores` cores, three phases separated by barriers:
//   1. scatter: each core sets the bits of its index rows in its own L1 bitmap (the table image's bitmap field);
//   2. reduce:  each core ORs one block range across every core's bitmap, then fills that range's in-block word
//               prefixes and block counts in its image;
//   3. assemble + rewrite: each core copies every other core's reduced ranges into its image (all images are now
//      the full table), derives the per-owner block prefixes and compact-buffer bases, writes its share of table
//      pages, and rewrites its index rows to compact slots.

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
    constexpr auto indices_args = TensorAccessorArgs<17>();
    constexpr auto table_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto compact_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr uint32_t unit = pages_per_packet * banks;
    constexpr uint32_t row_bytes = topk * sizeof(uint32_t);

    size_t arg = 0;
    const uint32_t indices_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t table_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t compact_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t core_idx = get_arg_val<uint32_t>(arg++);
    const uint32_t row_start = get_arg_val<uint32_t>(arg++);
    const uint32_t row_count = get_arg_val<uint32_t>(arg++);
    const uint32_t lead_x = get_arg_val<uint32_t>(arg++);
    const uint32_t lead_y = get_arg_val<uint32_t>(arg++);
    const uint32_t core_xy_arg = arg;  // cores (x, y) pairs follow

    const auto indices = TensorAccessor(indices_args, indices_addr, row_bytes);
    const auto table = TensorAccessor(table_args, table_addr, kTablePageBytes);
    const auto compact = TensorAccessor(compact_args, compact_addr, row_bytes);

    cb_reserve_back(kImage, 1);
    cb_reserve_back(kRow, 1);
    cb_reserve_back(kPre, 1);
    cb_reserve_back(kScratch, 1);
    const uint32_t image_l1 = get_write_ptr(kImage);
    const uint32_t row_l1 = get_write_ptr(kRow);
    auto* image = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(image_l1);
    auto* bits = image + bitmap_off;
    auto* prefix = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(image + prefix_off);
    auto* counts = image + counts_off;
    auto* pre = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kPre));
    auto* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_l1);

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
                noc_semaphore_inc(
                    get_noc_addr(
                        get_arg_val<uint32_t>(core_xy_arg + 2 * c),
                        get_arg_val<uint32_t>(core_xy_arg + 2 * c + 1),
                        go_sem),
                    1);
            }
            noc_async_atomic_barrier();
        }
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem), phase);
    };

    // 1. Scatter.
    for (uint32_t w = 0; w < bitmap_words; ++w) {
        bits[w] = 0;
    }
    for (uint32_t r = row_start; r < row_start + row_count; ++r) {
        noc_async_read(indices.get_noc_addr(r), row_l1, row_bytes);
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t k = 0; k < topk; ++k) {
            const uint32_t v = row[k];
            if (v != kSentinel) {
                bits[v >> 5] |= 1u << (v & 31);
            }
        }
    }
    barrier();

    // 2. Reduce my block range.
    const uint32_t b0 = core_idx * blocks_per_core < num_blocks ? core_idx * blocks_per_core : num_blocks;
    const uint32_t b1 = b0 + blocks_per_core < num_blocks ? b0 + blocks_per_core : num_blocks;
    const uint32_t w0 = b0 * words_per_block, w1 = b1 * words_per_block;
    if (b1 > b0) {
        const uint32_t scratch_l1 = get_write_ptr(kScratch);
        auto* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1);
        const uint32_t range_bytes = (w1 - w0) * 4;
        for (uint32_t c = 0; c < cores; ++c) {
            if (c == core_idx) {
                continue;
            }
            noc_async_read(
                get_noc_addr(
                    get_arg_val<uint32_t>(core_xy_arg + 2 * c),
                    get_arg_val<uint32_t>(core_xy_arg + 2 * c + 1),
                    image_l1 + (bitmap_off + w0) * 4),
                scratch_l1,
                range_bytes);
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t w = w0; w < w1; ++w) {
                bits[w] |= scratch[w - w0];
            }
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

    // 3. Assemble the full table in this core's image.
    for (uint32_t c = 0; c < cores; ++c) {
        const uint32_t cb0 = c * blocks_per_core < num_blocks ? c * blocks_per_core : num_blocks;
        const uint32_t cb1 = cb0 + blocks_per_core < num_blocks ? cb0 + blocks_per_core : num_blocks;
        if (c == core_idx || cb1 == cb0) {
            continue;
        }
        const uint32_t x = get_arg_val<uint32_t>(core_xy_arg + 2 * c),
                       y = get_arg_val<uint32_t>(core_xy_arg + 2 * c + 1);
        const uint32_t cw0 = cb0 * words_per_block, cw1 = cb1 * words_per_block;
        const uint32_t spans[3][2] = {
            {(bitmap_off + cw0) * 4, (cw1 - cw0) * 4},
            {prefix_off * 4 + cw0, cw1 - cw0},
            {(counts_off + cb0) * 4, (cb1 - cb0) * 4}};
        for (const auto& s : spans) {
            noc_async_read(get_noc_addr(x, y, image_l1 + s[0]), image_l1 + s[0], s[1]);
        }
    }
    noc_async_read_barrier();
    invalidate_l1_cache();

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

    // Rewrite my rows: slot = base[owner] + bank lane of the row's rank among this owner's rows.
    for (uint32_t r = row_start; r < row_start + row_count; ++r) {
        noc_async_write_barrier();  // the previous row's write has left the row buffer
        noc_async_read(indices.get_noc_addr(r), row_l1, row_bytes);
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t k = 0; k < topk; ++k) {
            const uint32_t v = row[k];
            if (v == kSentinel) {
                continue;
            }
            const uint32_t blk = v / stripe_chunk;
            const uint32_t w = v >> 5;
            const uint32_t rank = pre[blk] + prefix[w] + popcount(bits[w] & ((1u << (v & 31)) - 1u));
            const uint32_t j = rank / pages_per_packet, i = rank % pages_per_packet;
            row[k] = base[blk % num_devices] + (j % banks) + (j / banks) * unit + i * banks;
        }
        noc_async_write(row_l1, compact.get_noc_addr(r), row_bytes);
    }
    noc_async_write_barrier();
}
