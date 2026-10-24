// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Hot rows of the sparse KV push (see sparse_push_hot_device_operation.hpp), on one chip, `cores` cores with five
// workers each (sparse_push_hot_work.hpp):
//   0. load the core's slice of the top-k indices and every requester's bitmap words of this core's rank range,
//      then this chip's own; meanwhile core c < N reads requester c's active extent and core N this chip's own table
//      header (its owner counts: the run layout);                                                    [exchange 1]
//      then zero the words past each table's active extent;
//   1. hot words of the range (rows at least `threshold` requesters need), then their ranks per (owner, part);
//                                                                                                     [exchange 2]
//   2. per (owner, part) prefixes over the ranges (core o < N: owner o's) [exchange 3], then the hot region's
//      sub-run bases;
//   3. the range's multicast rows (hot rows within their sub-run's cap) and this chip's cold rows (needed, not
//      multicast), then the cold rows' ranks per (owner, part);                                      [exchange 4]
//   4. the cold prefixes (core o < N: owner o's);                                                    [exchange 5]
//   5. rewrite the core's indices (all workers) and write them out; write the range's multicast mask and, on core
//      0, the header.
// An exchange gathers every core's part on core 0, which multicasts it.

#include "api/dataflow/dataflow_api.h"
#include "sparse_push_hot_work.hpp"

#include <cstdint>

using namespace sparse_push_hot;

void kernel_main() {
    constexpr uint32_t num_devices = get_compile_time_arg_val(0);
    constexpr uint32_t stripe_chunk = get_compile_time_arg_val(1);
    constexpr uint32_t bitmap_off = get_compile_time_arg_val(2);   // words
    constexpr uint32_t table_pages = get_compile_time_arg_val(3);  // per requester
    constexpr uint32_t pages_per_packet = get_compile_time_arg_val(4);
    constexpr uint32_t banks = get_compile_time_arg_val(5);
    constexpr uint32_t cores = get_compile_time_arg_val(6);
    constexpr uint32_t threshold = get_compile_time_arg_val(7);
    constexpr uint32_t prefix_cap = get_compile_time_arg_val(8);
    constexpr uint32_t current_cap = get_compile_time_arg_val(9);
    constexpr uint32_t range_blocks = get_compile_time_arg_val(10);  // a multiple of num_devices and of 16
    constexpr uint32_t magic = get_compile_time_arg_val(11);
    constexpr uint32_t bases_off = get_compile_time_arg_val(12);  // words of the hot table header
    constexpr uint32_t topk = get_compile_time_arg_val(13);
    constexpr uint32_t owner_off = get_compile_time_arg_val(14);  // words of the plan table header
    constexpr uint32_t num_blocks = get_compile_time_arg_val(15);
    constexpr auto indices_args = TensorAccessorArgs<16>();
    constexpr auto compact_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<compact_args.next_compile_time_args_offset()>();
    constexpr auto tables_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto hot_args = TensorAccessorArgs<tables_args.next_compile_time_args_offset()>();
    constexpr auto split_args = TensorAccessorArgs<hot_args.next_compile_time_args_offset()>();
    constexpr uint32_t words_per_block = stripe_chunk / 32;
    constexpr uint32_t range_words = range_blocks * words_per_block;
    constexpr uint32_t page_bytes = 4096;
    constexpr uint32_t max_ranges = (num_blocks + range_blocks - 1) / range_blocks;
    constexpr uint32_t kHeaderTrid = 1;
    static_assert(cores > num_devices, "cores 0..N read the table headers");

    size_t arg = 0;
    const uint32_t indices_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t compact_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t table_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tables_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t hot_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t split_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t core_idx = get_arg_val<uint32_t>(arg++);
    const uint32_t i0 = get_arg_val<uint32_t>(arg++);  // this core's indices: flat [i0, i0 + n_idx)
    const uint32_t n_idx = get_arg_val<uint32_t>(arg++);
    // Common runtime args: every core's (x, y), core 0 first, then the multicast rectangle of all of them.
    auto core_noc = [&](uint32_t c, uint32_t l1) {
        return get_noc_addr(get_common_arg_val<uint32_t>(2 * c), get_common_arg_val<uint32_t>(2 * c + 1), l1);
    };
    auto mcast_noc = [&](uint32_t l1) {
        return get_noc_multicast_addr(
            get_common_arg_val<uint32_t>(2 * cores),
            get_common_arg_val<uint32_t>(2 * cores + 1),
            get_common_arg_val<uint32_t>(2 * cores + 2),
            get_common_arg_val<uint32_t>(2 * cores + 3),
            l1);
    };

    const auto indices = TensorAccessor(indices_args, indices_addr, topk * sizeof(uint32_t));
    const auto compact = TensorAccessor(compact_args, compact_addr, topk * sizeof(uint32_t));
    const auto own_table = TensorAccessor(table_args, table_addr, page_bytes);
    const auto tables = TensorAccessor(tables_args, tables_addr, page_bytes);
    const auto hot_table = TensorAccessor(hot_args, hot_addr, page_bytes);

    const uint32_t idx_l1 = get_write_ptr(kIdx);
    const uint32_t compact_l1 = get_write_ptr(kCompact);
    const uint32_t tables_l1 = get_write_ptr(kTables);
    const uint32_t hot_l1 = get_write_ptr(kHot);
    const uint32_t rank16_l1 = get_write_ptr(kRank16);
    const uint32_t ranges_l1 = get_write_ptr(kRanges);
    const uint32_t misc_l1 = get_write_ptr(kMisc);
    const uint32_t out_l1 = get_write_ptr(kOut);
    const uint32_t cold_l1 = get_write_ptr(kCold);
    constexpr uint32_t rank_cold = cold_rank16(num_blocks * words_per_block);
    constexpr RangesLayout ranges_at = ranges_layout(max_ranges, num_devices);
    constexpr uint32_t range_stride = ranges_at.range_stride;
    const uint32_t rankc_l1 = rank16_l1 + rank_cold * 2;
    const uint32_t prefixes_l1 = ranges_l1 + ranges_at.prefixes * 4;  // transposed: [owner * 2 + part][range]
    const uint32_t rangesc_l1 = ranges_l1 + ranges_at.cold * 4;
    const uint32_t cold_prefixes_l1 = prefixes_l1 + ranges_at.cold * 4;
    auto* idx = reinterpret_cast<uint32_t*>(idx_l1);
    auto* compact_idx = reinterpret_cast<uint32_t*>(compact_l1);
    auto* scratch = reinterpret_cast<uint32_t*>(tables_l1);
    auto* hot = reinterpret_cast<uint32_t*>(hot_l1);
    auto* rank16 = reinterpret_cast<uint16_t*>(rank16_l1);
    auto* counts = reinterpret_cast<uint32_t*>(ranges_l1);  // [range][owner * 2 + part]
    auto* prefixes = reinterpret_cast<uint32_t*>(prefixes_l1);
    auto* misc = reinterpret_cast<uint32_t*>(misc_l1);
    auto* out = reinterpret_cast<uint32_t*>(out_l1);
    auto* cold = reinterpret_cast<uint32_t*>(cold_l1);
    auto* rankc = reinterpret_cast<uint16_t*>(rankc_l1);
    auto* cold_counts = reinterpret_cast<uint32_t*>(rangesc_l1);
    auto* cold_prefixes = reinterpret_cast<uint32_t*>(cold_prefixes_l1);

    const uint32_t sem_step = get_semaphore(kSemStep);
    auto sem_done = [](uint32_t w) { return static_cast<uint32_t>(get_semaphore(kSemDone + w)); };
    auto wait_all = [&](uint32_t v) {
        for (uint32_t w = 0; w < kWorkers; ++w) {
            wait_at_least(sem_done(w), v);
        }
    };
    // Every worker's share of the hot words of this core's ranges, then (once all are in) its owners' ranks.
    auto count_and_rank = [&](uint32_t worker) {
        const uint32_t ranges = misc[kMiscRanges], blocks = misc[kMiscActiveBlocks];
        for (uint32_t r = core_idx; r < ranges; r += cores) {
            // One range at a time: the scratch holds one range of every table. Multi-range cores reload it (BRISC,
            // between kCounted rounds) -- not needed at the GLM shapes (ranges <= cores).
            const uint32_t b1 = (r + 1) * range_blocks < blocks ? (r + 1) * range_blocks : blocks;
            const uint32_t n = (b1 - r * range_blocks) * words_per_block;
            count_hot<num_devices>(
                scratch,
                range_words,
                r * range_words,
                share_begin(n, worker),
                share_begin(n, worker + 1),
                threshold,
                hot);
        }
        post(sem_done(worker), kCounted);
        wait_all(kCounted);
        for (uint32_t r = core_idx; r < ranges; r += cores) {
            const uint32_t b0 = r * range_blocks;
            part_ranks<num_devices, words_per_block>(
                hot,
                rank16,
                counts + r * num_devices * 2,
                b0,
                b0 + range_blocks < blocks ? b0 + range_blocks : blocks,
                misc[kMiscSplitBlock],
                worker);
        }
        post(sem_done(worker), kRanked);
    };
    // Every worker's share of the multicast and cold words of this core's range, then (once all are in) its owners'
    // cold ranks.
    auto mask_and_rank = [&](uint32_t worker) {
        const uint32_t ranges = misc[kMiscRanges], blocks = misc[kMiscActiveBlocks];
        const uint32_t split_block = misc[kMiscSplitBlock];
        for (uint32_t r = core_idx; r < ranges; r += cores) {
            const uint32_t b1 = (r + 1) * range_blocks < blocks ? (r + 1) * range_blocks : blocks;
            const uint32_t n = (b1 - r * range_blocks) * words_per_block;
            mask_range<num_devices, words_per_block, range_blocks, range_stride, prefix_cap, current_cap>(
                hot,
                rank16,
                prefixes,
                scratch + num_devices * range_words,
                r * range_words,
                share_begin(n, worker),
                share_begin(n, worker + 1),
                split_block,
                cold);
        }
        post(sem_done(worker), kMasked);
        wait_all(kMasked);
        for (uint32_t r = core_idx; r < ranges; r += cores) {
            const uint32_t b0 = r * range_blocks;
            part_ranks<num_devices, words_per_block>(
                cold,
                rankc,
                cold_counts + r * num_devices * 2,
                b0,
                b0 + range_blocks < blocks ? b0 + range_blocks : blocks,
                split_block,
                worker);
        }
        post(sem_done(worker), kColdRanked);
    };
    auto rewrite = [&](uint32_t worker) {
        rewrite_hot<
            num_devices,
            stripe_chunk,
            range_blocks,
            pages_per_packet,
            banks,
            range_stride,
            rank_cold,
            ranges_at.cold>(
            idx,
            compact_idx,
            share_begin(n_idx, worker),
            share_begin(n_idx, worker + 1),
            hot,
            cold,
            rank16,
            prefixes,
            misc,
            misc[kMiscSplitBlock]);
        post(sem_done(worker), kStepRewrite);
    };
    // Each run of this core's indices that lies in one index row: (row, column, position in idx, length).
    auto for_each_run = [&](auto&& f) {
        for (uint32_t i = i0, end = i0 + n_idx; i < end;) {
            const uint32_t r = i / topk, col = i % topk;
            const uint32_t len = topk - col < end - i ? topk - col : end - i;
            f(r, col, i - i0, len);
            i += len;
        }
    };
    // Write rewritten positions [p0, p1) to the compact indices, once every worker has rewritten.
    auto write_back = [&](uint32_t p0, uint32_t p1) {
        wait_all(kStepRewrite);
        for_each_run([&](uint32_t r, uint32_t col, uint32_t pos, uint32_t len) {
            const uint32_t a = pos > p0 ? pos : p0, b = pos + len < p1 ? pos + len : p1;
            if (a < b) {
                noc_async_write(compact_l1 + a * 4, compact.get_noc_addr(r) + (col + a - pos) * 4, (b - a) * 4);
            }
        });
        noc_async_write_barrier();
    };

#ifdef HOT_HELPER
    constexpr uint32_t worker = 1;
    wait_at_least(sem_step, kStepCount);
    count_and_rank(worker);
    wait_at_least(sem_step, kStepCold);
    mask_and_rank(worker);
    wait_at_least(sem_step, kStepRewrite);
    rewrite(worker);
    write_back(n_idx / 2, n_idx);
#else
    constexpr uint32_t worker = 0;
    const uint32_t arrive_sem = get_semaphore(kSemArrive);
    const uint32_t go_sem = get_semaphore(kSemGo);
    // Exchange k: every core has written its part to the same offsets on core 0 and counts in there; core 0
    // multicasts the gathered spans, then (linked) the go value.
    auto exchange = [&](uint32_t k, auto&& spans) {
        noc_async_write_barrier();
        noc_semaphore_inc(core_noc(0, arrive_sem), 1);
        noc_async_atomic_barrier();
        if (core_idx == 0) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_sem), cores * k);
            if constexpr (cores > 1) {
                spans([&](uint32_t l1, uint32_t bytes) {
                    noc_async_write_multicast(l1, mcast_noc(l1), (bytes + 15) / 16 * 16, cores - 1, true);
                });
                noc_async_writes_flushed();
                misc[kMiscGo] = k;
                noc_semaphore_set_multicast(misc_l1 + kMiscGo * 4, mcast_noc(go_sem), cores - 1);
            }
        } else {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem), k);
        }
        invalidate_l1_cache();
    };
    auto to_root = [&](uint32_t l1, uint32_t bytes) { noc_async_write(l1, core_noc(0, l1), (bytes + 15) / 16 * 16); };
    // Read bytes [off, off + bytes) of a table starting at page page0 of `pages` to l1.
    auto read_table = [&](const auto& pages, uint32_t page0, uint32_t off, uint32_t bytes, uint32_t l1) {
        for (uint32_t end = off + bytes; off < end;) {
            const uint32_t in_page = off % page_bytes;
            const uint32_t n = end - off < page_bytes - in_page ? end - off : page_bytes - in_page;
            noc_async_read(pages.get_noc_addr(page0 + off / page_bytes) + in_page, l1, n);
            off += n;
            l1 += n;
        }
    };
    // Copy l1 bytes to hot table bytes [off, off + bytes), page by page.
    auto write_hot = [&](uint32_t l1, uint32_t off, uint32_t bytes) {
        for (uint32_t end = off + bytes; off < end;) {
            const uint32_t in_page = off % page_bytes;
            const uint32_t n = end - off < page_bytes - in_page ? end - off : page_bytes - in_page;
            noc_async_write(l1, hot_table.get_noc_addr(off / page_bytes) + in_page, n);
            off += n;
            l1 += n;
        }
    };

    // 0. Load. Every rank range's words are read whole (the active extent is not known yet) and the stale words past
    // each table's active blocks zeroed once it is.
    for_each_run([&](uint32_t r, uint32_t col, uint32_t pos, uint32_t len) {
        noc_async_read(indices.get_noc_addr(r) + col * 4, idx_l1 + pos * 4, len * 4);
    });
    const auto split = TensorAccessor(split_args, split_addr);
    noc_async_read(split.get_noc_addr(0), misc_l1 + kMiscSplit * 4, sizeof(uint32_t));
    // The headers: one read per core rather than every core reading all of them (they share a DRAM bank), ahead of
    // the range reads and waited for alone.
    constexpr uint32_t own_header_words = owner_off + num_devices;
    if (core_idx <= num_devices) {
        noc_async_read_set_trid(kHeaderTrid);
        noc_async_read(
            core_idx < num_devices ? tables.get_noc_addr(core_idx * table_pages) : own_table.get_noc_addr(0),
            out_l1,
            own_header_words * 4);
        noc_async_read_set_trid(0);
    }
    const uint32_t w0 = core_idx * range_words;
    const uint32_t range_end =
        w0 + range_words < num_blocks * words_per_block ? w0 + range_words : num_blocks * words_per_block;
    if (core_idx < max_ranges) {
        for (uint32_t c = 0; c < num_devices; ++c) {
            read_table(
                tables, c * table_pages, (bitmap_off + w0) * 4, (range_end - w0) * 4, tables_l1 + c * range_words * 4);
        }
        read_table(
            own_table, 0, (bitmap_off + w0) * 4, (range_end - w0) * 4, tables_l1 + num_devices * range_words * 4);
    }
    if (core_idx <= num_devices) {
        noc_async_read_barrier_with_trid(kHeaderTrid);
        misc[kMiscLimits + core_idx] = out[3] * words_per_block;
        noc_async_write(misc_l1 + (kMiscLimits + core_idx) * 4, core_noc(0, misc_l1 + (kMiscLimits + core_idx) * 4), 4);
        if (core_idx == num_devices) {
            // The plan's run layout: this chip's run of rows from owner o starts behind the padded runs of owners < o.
            constexpr uint32_t unit = pages_per_packet * banks;
            uint32_t base = 0;
            for (uint32_t o = 0; o < num_devices; ++o) {
                misc[kMiscRunBases + o] = base;
                base += (out[owner_off + o] + unit - 1) / unit * unit;
            }
            to_root(misc_l1 + kMiscRunBases * 4, num_devices * 4);
        }
    }
    exchange(1, [&](auto&& f) {
        f(misc_l1 + kMiscLimits * 4, (num_devices + 1) * 4);
        f(misc_l1 + kMiscRunBases * 4, num_devices * 4);
    });
    uint32_t active_blocks = 0;
    for (uint32_t c = 0; c < num_devices; ++c) {
        const uint32_t ab = misc[kMiscLimits + c] / words_per_block;
        active_blocks = ab > active_blocks ? ab : active_blocks;
    }
    const uint32_t split_block = misc[kMiscSplit] / stripe_chunk;
    const uint32_t ranges = (active_blocks + range_blocks - 1) / range_blocks;
    const uint32_t active_words = active_blocks * words_per_block;
    misc[kMiscActiveBlocks] = active_blocks;
    misc[kMiscRanges] = ranges;
    misc[kMiscSplitBlock] = split_block;
    ASSERT(ranges <= cores);  // one range per core: the scratch holds one range
    noc_async_read_barrier();
    invalidate_l1_cache();
    if (core_idx < ranges) {
        for (uint32_t c = 0; c <= num_devices; ++c) {
            const uint32_t limit = misc[kMiscLimits + c];
            for (uint32_t w = limit > w0 ? limit : w0; w < range_end; ++w) {
                scratch[c * range_words + w - w0] = 0;
            }
        }
    }

    // 1. Hot words and their ranks.
    post(sem_step, kStepCount);
    count_and_rank(worker);
    wait_all(kRanked);
    for (uint32_t r = core_idx; r < ranges; r += cores) {
        const uint32_t w0 = r * range_words;
        const uint32_t w1 = w0 + range_words < active_words ? w0 + range_words : active_words;
        to_root(hot_l1 + w0 * 4, (w1 - w0) * 4);
        to_root(rank16_l1 + w0 * 2, (w1 - w0) * 2);
        to_root(ranges_l1 + r * num_devices * 2 * 4, num_devices * 2 * 4);
    }
    exchange(2, [&](auto&& f) {
        f(hot_l1, active_words * 4);
        f(rank16_l1, active_words * 2);
        f(ranges_l1, ranges * num_devices * 2 * 4);
    });

    // 2. Prefixes over the ranges (core o < N: owner o's), then the hot region's sub-run bases: owner-major, part 0
    // then part 1, each padded to whole bank-lane units.
    // Core o < N's prefixes of owner o (columns 2o and 2o + 1 of the transposed table) and totals, gathered on core 0.
    auto owner_prefixes_to_root = [&](uint32_t table_l1, uint32_t totals_word) {
        noc_async_write(
            table_l1 + core_idx * 2 * range_stride * 4,
            core_noc(0, table_l1 + core_idx * 2 * range_stride * 4),
            2 * range_stride * 4);
        noc_async_write(
            misc_l1 + (totals_word + core_idx * 2) * 4, core_noc(0, misc_l1 + (totals_word + core_idx * 2) * 4), 8);
    };
    if (core_idx < num_devices) {
        owner_prefixes<num_devices, range_stride, false>(counts, ranges, core_idx, prefixes, misc + kMiscTotals);
        owner_prefixes_to_root(prefixes_l1, kMiscTotals);
    }
    exchange(3, [&](auto&& f) {
        f(prefixes_l1, 2 * num_devices * range_stride * 4);
        f(misc_l1 + kMiscTotals * 4, 2 * num_devices * 4);
    });
    constexpr uint32_t unit = pages_per_packet * banks;
    uint32_t hot_slots = 0;
    for (uint32_t o = 0; o < num_devices; ++o) {
        for (uint32_t part = 0; part < 2; ++part) {
            const uint32_t cap = part ? current_cap : prefix_cap;
            const uint32_t n = misc[kMiscTotals + o * 2 + part] < cap ? misc[kMiscTotals + o * 2 + part] : cap;
            misc[kMiscBases + o * 2 + part] = hot_slots;
            hot_slots += (n + unit - 1) / unit * unit;
        }
    }
    misc[kMiscHotSlots] = hot_slots;

    // 3. Multicast and cold rows, the cold ranks.
    post(sem_step, kStepCold);
    mask_and_rank(worker);
    wait_all(kColdRanked);
    for (uint32_t r = core_idx; r < ranges; r += cores) {
        const uint32_t w0 = r * range_words;
        const uint32_t w1 = w0 + range_words < active_words ? w0 + range_words : active_words;
        to_root(cold_l1 + w0 * 4, (w1 - w0) * 4);
        to_root(rankc_l1 + w0 * 2, (w1 - w0) * 2);
        to_root(rangesc_l1 + r * num_devices * 2 * 4, num_devices * 2 * 4);
        // The range's multicast mask, for the owners' push.
        write_hot(hot_l1 + w0 * 4, (bitmap_off + w0) * 4, (w1 - w0) * 4);
    }
    exchange(4, [&](auto&& f) {
        f(cold_l1, active_words * 4);
        f(rankc_l1, active_words * 2);
        f(rangesc_l1, ranges * num_devices * 2 * 4);
    });

    // 4. Cold prefixes (core o < N: owner o's).
    if (core_idx < num_devices) {
        owner_prefixes<num_devices, range_stride, true>(
            cold_counts, ranges, core_idx, cold_prefixes, misc + kMiscColdTotals);
        owner_prefixes_to_root(cold_prefixes_l1, kMiscColdTotals);
    }
    exchange(5, [&](auto&& f) { f(cold_prefixes_l1, 2 * num_devices * range_stride * 4); });
    // Each sub-run's run behind the hot region.
    for (uint32_t o = 0; o < num_devices; ++o) {
        misc[kMiscColdBases + o * 2] = misc[kMiscColdBases + o * 2 + 1] = hot_slots + misc[kMiscRunBases + o];
    }

    // 5. Rewrite and the header.
    post(sem_step, kStepRewrite);
    rewrite(worker);
    if (core_idx == 0) {
        auto* header = reinterpret_cast<uint32_t*>(out_l1);
        for (uint32_t w = 0; w < bitmap_off; ++w) {
            header[w] = 0;
        }
        header[0] = magic;
        header[1] = hot_slots;
        header[2] = threshold;
        header[3] = active_blocks;
        header[4] = split_block;
        for (uint32_t k = 0; k < 2 * num_devices; ++k) {
            header[bases_off + k] = misc[kMiscBases + k];
        }
        write_hot(out_l1, 0, bitmap_off * 4);
    }
    write_back(0, n_idx / 2);
#endif
}
