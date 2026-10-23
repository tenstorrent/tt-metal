// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// On-device plan for the sparse KV push (see sparse_push_plan_device_operation.hpp for the table layout).
// One requester chip, `cores` cores; on each core five workers (BRISC, NCRISC, three TRISCs, see
// sparse_push_plan_work.hpp) split the L1 loops and BRISC drives the steps and the NOC:
//   0. load the core's slice of the indices into L1; zero the bitmap and the dirty-chunk bytes;
//   1. scatter: each worker sets the bits of its bitmap words (word % 5) and marks the 512-row chunks it touched;
//      BRISC packs the marks into a dirty map and multicasts it with the largest row to every core;   [barrier]
//   2. reduce-scatter, chunk-cyclic so the sink and local-window chunks every core touches spread over all cores:
//      core c reduces chunks c + k * cores; BRISC and NCRISC read every core's dirty copy of them, all workers OR
//      them, BRISC multicasts the reduced chunks so every core holds the whole bitmap;                 [barrier]
//   3. block prefixes: core c fills the in-block word prefixes and block counts of its block range and multicasts
//      them;                                                                                              [barrier]
//   4. per-owner block prefixes and compact-buffer bases, every core for itself, split by owner over the workers;
//   5. rewrite the core's indices to compact slots (all workers) and write them out.
// The bitmap is zero before the scatter, so chunks no core touched stay zero everywhere without being sent.

#include "api/dataflow/dataflow_api.h"
#include "sparse_push_plan_work.hpp"

#include <cstdint>

using namespace sparse_push_plan;

void kernel_main() {
    constexpr uint32_t num_devices = get_compile_time_arg_val(0);
    constexpr uint32_t stripe_chunk = get_compile_time_arg_val(1);
    constexpr uint32_t bitmap_off = get_compile_time_arg_val(2);        // words
    constexpr uint32_t prefix_off = get_compile_time_arg_val(3);        // words; in-block word prefixes, one byte each
    constexpr uint32_t counts_off = get_compile_time_arg_val(4);        // words; block counts
    constexpr uint32_t owner_off = get_compile_time_arg_val(5);         // words; per-owner row counts
    constexpr uint32_t owner_prefix_off = get_compile_time_arg_val(6);  // words; per-owner rows before the split
    constexpr uint32_t pages_per_packet = get_compile_time_arg_val(7);
    constexpr uint32_t banks = get_compile_time_arg_val(8);
    constexpr uint32_t cores = get_compile_time_arg_val(9);
    constexpr uint32_t magic = get_compile_time_arg_val(10);
    constexpr uint32_t total_chunks = get_compile_time_arg_val(11);  // 512-row chunks of the bitmap
    constexpr bool has_split = get_compile_time_arg_val(14) != 0;
    constexpr uint32_t bitmap_words = get_compile_time_arg_val(15);
    constexpr uint32_t bucket_cap = get_compile_time_arg_val(16);    // indices per (from, to) scatter bucket
    constexpr uint32_t range_blocks = get_compile_time_arg_val(17);  // blocks per rank range, a multiple of num_devices
    constexpr uint32_t topk = get_compile_time_arg_val(18);
    constexpr auto indices_args = TensorAccessorArgs<19>();
    constexpr auto table_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto compact_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto split_args = TensorAccessorArgs<has_split ? compact_args.next_compile_time_args_offset() : 0>();
    constexpr uint32_t words_per_block = stripe_chunk / 32;
    constexpr uint32_t array_words = dirty_array_words(cores);
    constexpr uint32_t unit = pages_per_packet * banks;
    constexpr uint32_t table_page_bytes = 4096;
    constexpr uint32_t block_align = 16;  // keeps every core's word / byte / count range 16 B aligned

    size_t arg = 0;
    const uint32_t indices_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t table_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t compact_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t core_idx = get_arg_val<uint32_t>(arg++);
    const uint32_t i0 = get_arg_val<uint32_t>(arg++);  // this core's indices: flat [i0, i0 + n_idx)
    const uint32_t n_idx = get_arg_val<uint32_t>(arg++);
    const uint32_t split_addr = has_split ? get_arg_val<uint32_t>(arg++) : 0;
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
    const auto table = TensorAccessor(table_args, table_addr, table_page_bytes);
    const auto compact = TensorAccessor(compact_args, compact_addr, topk * sizeof(uint32_t));

    const uint32_t image_l1 = get_write_ptr(kImage);
    const uint32_t idx_l1 = get_write_ptr(kIdx);
    const uint32_t dirty_l1 = get_write_ptr(kDirtyBytes);
    const uint32_t all_dirty_l1 = get_write_ptr(kAllDirty);
    const uint32_t scratch_l1 = get_write_ptr(kScratch);
    const uint32_t misc_l1 = get_write_ptr(kMisc);
    // Plain pointers: every NOC transfer into these buffers is followed by a barrier and a cache invalidation, and
    // every worker hand-off by a fence (post / wait_at_least).
    auto* image = reinterpret_cast<uint32_t*>(image_l1);
    uint32_t* bits = image + bitmap_off;
    uint32_t* counts = image + counts_off;
    auto* idx = reinterpret_cast<uint32_t*>(idx_l1);
    auto* dirty = reinterpret_cast<uint8_t*>(dirty_l1);
    auto* all_dirty = reinterpret_cast<uint32_t*>(all_dirty_l1);
    auto* scratch = reinterpret_cast<uint32_t*>(scratch_l1);
    const uint32_t partials_l1 = get_write_ptr(kRanges);
    auto* partials = reinterpret_cast<uint32_t*>(partials_l1);  // [range][owner]
    const uint32_t rank16_l1 = get_write_ptr(kRank16);
    auto* rank16 = reinterpret_cast<uint16_t*>(rank16_l1);
    auto* misc = reinterpret_cast<uint32_t*>(misc_l1);
    auto* buckets = reinterpret_cast<uint32_t*>(get_write_ptr(kBuckets));
    uint32_t* copies = misc + kMiscCopies;

    const uint32_t sem_step = get_semaphore(kSemStep);
    auto sem_done = [](uint32_t w) { return static_cast<uint32_t>(get_semaphore(kSemDone + w)); };

    uint32_t* mine = all_dirty + kMine * array_words;
    const uint32_t column_l1 = all_dirty_l1 + kColumn * array_words * 4;
    const uint32_t row_ends_l1 = all_dirty_l1 + kRowEnds * array_words * 4;
    // Push this core's dirty copies of reducers [r0, r1)'s chunks to their scratch slots, then add its map word into
    // each reducer's column (the atomic adds land in zeroed words, one writer each).
    auto push = [&](uint32_t r0, uint32_t r1) {
        for (uint32_t r = r0; r < r1; ++r) {
            for (uint32_t m = mine[r]; m != 0; m &= m - 1) {
                const uint32_t k = __builtin_ctz(m);
                const uint32_t ch = r + k * cores;
                noc_async_write(
                    image_l1 + (bitmap_off + ch * kChunkWords) * 4,
                    core_noc(r, scratch_l1 + (k * cores + core_idx) * kChunkWords * 4),
                    kChunkWords * 4);
            }
        }
        for (uint32_t r = r0; r < r1; ++r) {
            if (mine[r] != 0) {
                noc_semaphore_inc(core_noc(r, column_l1 + core_idx * 4), mine[r]);
            }
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
    };
    const uint32_t sem_zeroed = get_semaphore(kSemZeroed);
#ifdef PLAN_HELPER
    auto zeroed = [&]() { wait_at_least(sem_zeroed, 1); };
#else
    auto zeroed = [&]() {
        noc_async_write_barrier();
        post(sem_zeroed, 1);
    };
#endif
    // Scatter, then this worker's share of the reducer-major dirty map.
    auto scatter_and_map = [&](uint32_t worker) {
        misc[kMiscRowEnd + worker] =
            scatter_step(idx, n_idx, buckets, bucket_cap, misc + kMiscBucketed, bits, dirty, worker, sem_done, zeroed);
        post(sem_done(worker), kScattered);
        for (uint32_t w = 0; w < kWorkers; ++w) {
            wait_at_least(sem_done(w), kScattered);
        }
        map_dirty<cores, total_chunks>(dirty, mine, worker);
        post(sem_done(worker), kMapped);
        for (uint32_t w = 0; w < kWorkers; ++w) {
            wait_at_least(sem_done(w), kMapped);
        }
    };
    auto* lists = reinterpret_cast<uint16_t*>(buckets);  // the buckets are spent by the OR
    auto or_step = [&](uint32_t worker) {
        list_copies<cores>(all_dirty + kColumn * array_words, misc[kMiscChunks], lists, copies, worker);
        post(sem_done(worker), kListed);
        for (uint32_t w = 0; w < kWorkers; ++w) {
            wait_at_least(sem_done(w), kListed);
        }
        or_copies<cores>(scratch, lists, copies, misc[kMiscChunks], core_idx, bits, worker);
        post(sem_done(worker), kOred);
    };
    // This core's rank ranges: r = core_idx, core_idx + cores, ...
    auto for_my_ranges = [&](auto&& f) {
        const uint32_t ranges = misc[kMiscRanges], blocks = misc[kMiscActiveBlocks];
        for (uint32_t r = core_idx; r < ranges; r += cores) {
            const uint32_t b0 = r * range_blocks;
            f(r, b0, b0 + range_blocks < blocks ? b0 + range_blocks : blocks);
        }
    };
    auto ranks = [&](uint32_t worker) {
        for_my_ranges([&](uint32_t r, uint32_t b0, uint32_t b1) {
            range_ranks<num_devices, words_per_block>(bits, rank16, counts, partials + r * num_devices, b0, b1, worker);
        });
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
    // Write rewritten positions [p0, p1) of idx to the compact indices, once every worker has rewritten.
    auto write_back = [&](uint32_t p0, uint32_t p1) {
        for (uint32_t w = 0; w < kWorkers; ++w) {
            wait_at_least(sem_done(w), kRewritten);
        }
        for_each_run([&](uint32_t r, uint32_t col, uint32_t pos, uint32_t len) {
            const uint32_t a = pos > p0 ? pos : p0, b = pos + len < p1 ? pos + len : p1;
            if (a < b) {
                noc_async_write(idx_l1 + a * 4, compact.get_noc_addr(r) + (col + a - pos) * 4, (b - a) * 4);
            }
        });
        noc_async_write_barrier();
    };

#ifdef PLAN_HELPER
    constexpr uint32_t worker = 1;
    wait_at_least(sem_step, kStepScatter);
    scatter_and_map(worker);
    push(cores / 2, cores);
    post(sem_done(worker), kPushed);
    wait_at_least(sem_step, kStepOr);
    or_step(worker);
    wait_at_least(sem_step, kStepRange);
    ranks(worker);
    post(sem_done(worker), kStepRange);
    wait_at_least(sem_step, kStepBases);
    bases_and_rewrite<num_devices, stripe_chunk, range_blocks, pages_per_packet, banks>(
        idx, n_idx, bits, rank16, partials, misc, worker, sem_done);
    write_back(n_idx / 2, n_idx);
#else
    constexpr uint32_t worker = 0;
    auto wait_workers = [&](uint32_t step, uint32_t first = 1) {
        for (uint32_t w = first; w < kWorkers; ++w) {
            wait_at_least(sem_done(w), step);
        }
    };
    const uint32_t arrive_sem = get_semaphore(kSemArrive);
    const uint32_t go_sem = get_semaphore(kSemGo);
    // Exchange k: every core has written its part to the same offsets on core 0 and counts in there; core 0
    // multicasts the gathered spans (`spans(f)` calls f(l1, bytes) for each), then the go value. One sender at a
    // time: concurrent multicasts from every core serialize on the NOC.
    auto exchange = [&](uint32_t k, auto&& spans) {
        noc_async_write_barrier();
        noc_semaphore_inc(core_noc(0, arrive_sem), 1);
        noc_async_atomic_barrier();
        if (core_idx == 0) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_sem), cores * k);
            if constexpr (cores > 1) {
                // Linked: the go value lands after the spans on every core, with no ack round trip between.
                spans([&](uint32_t l1, uint32_t bytes) {
                    noc_async_write_multicast(l1, mcast_noc(l1), (bytes + 15) / 16 * 16, cores - 1, true);
                });
                noc_async_writes_flushed();  // the previous go value has left misc
                misc[kMiscGo] = k;
                noc_semaphore_set_multicast(misc_l1 + kMiscGo * 4, mcast_noc(go_sem), cores - 1);
            }
        } else {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem), k);
        }
        invalidate_l1_cache();
    };
    auto to_root = [&](uint32_t l1, uint32_t bytes) { noc_async_write(l1, core_noc(0, l1), (bytes + 15) / 16 * 16); };
    // Zero [addr, addr + bytes): 8 KiB from the zeros region, then copies of that.
    auto zero = [&](uint32_t addr, uint32_t bytes) {
        const uint32_t first = bytes < 8192 ? bytes : 8192;
        for (uint32_t off = 0; off < first; off += MEM_ZEROS_SIZE) {
            noc_async_write(
                MEM_ZEROS_BASE, get_noc_addr(addr + off), first - off < MEM_ZEROS_SIZE ? first - off : MEM_ZEROS_SIZE);
        }
        noc_async_write_barrier();
        for (uint32_t off = first; off < bytes; off += first) {
            noc_async_write(addr, get_noc_addr(addr + off), bytes - off < first ? bytes - off : first);
        }
    };
    // Copy image bytes [off, off + bytes) to the same offsets of the table, page by page.
    auto write_span = [&](uint32_t off, uint32_t bytes) {
        for (uint32_t end = off + bytes; off < end;) {
            const uint32_t in_page = off % table_page_bytes;
            const uint32_t n = end - off < table_page_bytes - in_page ? end - off : table_page_bytes - in_page;
            noc_async_write(image_l1 + off, table.get_noc_addr(off / table_page_bytes) + in_page, n);
            off += n;
        }
    };

    // 0. Load.
    for_each_run([&](uint32_t r, uint32_t col, uint32_t pos, uint32_t len) {
        noc_async_read(indices.get_noc_addr(r) + col * 4, idx_l1 + pos * 4, len * 4);
    });
    if constexpr (has_split) {
        const auto split = TensorAccessor(split_args, split_addr);
        noc_async_read(split.get_noc_addr(0), misc_l1 + kMiscSplit * 4, sizeof(uint32_t));
    }
    zero(image_l1 + bitmap_off * 4, bitmap_words * 4);
    zero(dirty_l1, (total_chunks + 15) / 16 * 16);
    zero(column_l1, array_words * 4);
    noc_async_read_barrier();
    invalidate_l1_cache();

    // 1. Scatter, map the dirty chunks and push the dirty copies to their reducers.
    post(sem_step, kStepScatter);
    scatter_and_map(worker);
    uint32_t row_end = 0;
    for (uint32_t w = 0; w < kWorkers; ++w) {
        row_end = misc[kMiscRowEnd + w] > row_end ? misc[kMiscRowEnd + w] : row_end;
    }
    all_dirty[kRowEnds * array_words + core_idx] = row_end;
    push(0, cores / 2);
    wait_at_least(sem_done(1), kPushed);
    noc_async_write(row_ends_l1 + core_idx * 4, core_noc(0, row_ends_l1 + core_idx * 4), 4);
    exchange(1, [&](auto&& f) { f(row_ends_l1, cores * 4); });

    // 2. Reduce-scatter. Only rows up to the largest one any core saw are active.
    uint32_t active_rows = 0;
    for (uint32_t c = 0; c < cores; ++c) {
        const uint32_t e = all_dirty[kRowEnds * array_words + c];
        active_rows = e > active_rows ? e : active_rows;
    }
    const uint32_t active_blocks = (active_rows + stripe_chunk - 1) / stripe_chunk;
    const uint32_t active_words = active_blocks * words_per_block;
    const uint32_t active_chunks = (active_words + kChunkWords - 1) / kChunkWords;
    const uint32_t chunks = core_idx < active_chunks ? (active_chunks - core_idx + cores - 1) / cores : 0;
    misc[kMiscActiveBlocks] = active_blocks;
    misc[kMiscChunks] = chunks;
    post(sem_step, kStepOr);
    or_step(worker);
    wait_workers(kOred);
    for (uint32_t k = 0; k < chunks; ++k) {
        if (copies[k] != 0) {
            const uint32_t l1 = image_l1 + (bitmap_off + (core_idx + k * cores) * kChunkWords) * 4;
            to_root(l1, kChunkWords * 4);
        }
    }
    exchange(2, [&](auto&& f) { f(image_l1 + bitmap_off * 4, active_words * 4); });

    // 3. Ranks and per-owner partial counts of this core's block ranges.
    const uint32_t ranges = (active_blocks + range_blocks - 1) / range_blocks;
    misc[kMiscRanges] = ranges;
    post(sem_step, kStepRange);
    ranks(worker);
    wait_workers(kStepRange);
    for_my_ranges([&](uint32_t r, uint32_t b0, uint32_t b1) {
        to_root(rank16_l1 + b0 * words_per_block * 2, (b1 - b0) * words_per_block * 2);
        to_root(partials_l1 + r * num_devices * 4, num_devices * 4);
    });
    exchange(3, [&](auto&& f) {
        f(rank16_l1, active_words * 2);
        f(partials_l1, ranges * num_devices * 4);
    });
    for_my_ranges([&](uint32_t r, uint32_t b0, uint32_t b1) {
        // The owners' push reads the bitmap from the gathered tables; counts go along for inspection.
        write_span((bitmap_off + b0 * words_per_block) * 4, (b1 - b0) * words_per_block * 4);
        write_span((counts_off + b0) * 4, (b1 - b0) * 4);
    });

    // 4. Owner bases and 5. rewrite, all workers.
    post(sem_step, kStepBases);
    bases_and_rewrite<num_devices, stripe_chunk, range_blocks, pages_per_packet, banks>(
        idx, n_idx, bits, rank16, partials, misc, worker, sem_done);
    wait_workers(kRewritten);
    if (core_idx == 0) {
        // The split block: rows before the chunk start form the prefix part of each owner's run.
        const uint32_t* totals = misc + kMiscTotals;
        const uint32_t split_block = has_split ? misc[kMiscSplit] / stripe_chunk : active_blocks;
        for (uint32_t o = 0; o < num_devices; ++o) {
            image[owner_off + o] = totals[o];
            // Rows of owner o before the split = its exclusive prefix at its first block from the split on.
            const uint32_t first = split_block + (o + num_devices - split_block % num_devices) % num_devices;
            image[owner_prefix_off + o] = first < active_blocks ? partials[(first / range_blocks) * num_devices + o] +
                                                                      rank16[first * words_per_block]
                                                                : totals[o];
        }
        image[0] = magic;
        image[1] = num_devices;
        image[2] = stripe_chunk;
        image[3] = active_blocks;  // the push scans only blocks below this; later table words may be stale
        image[4] = split_block;
        write_span(0, bitmap_off * 4);  // header and per-owner counts
    }
    write_back(0, n_idx / 2);
#endif
}
