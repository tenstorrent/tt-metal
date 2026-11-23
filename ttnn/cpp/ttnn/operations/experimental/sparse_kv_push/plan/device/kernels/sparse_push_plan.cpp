// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/alignment.h"
#include "api/dataflow/dataflow_api.h"
#include "sparse_push_plan_work.hpp"

#include <algorithm>
#include <cstdint>

#ifdef PLAN_STATS
#include "api/debug/dprint.h"
#endif

// Perf experiments (host env PLAN_ABL=NAME,..., defines PLAN_ABL_NAME):
//   DISPATCH_ONLY: stop after exchange 2 and write the owner-major slots (bitmap words, summary, header words 0-10)
//                  with zero per-owner counts; no ranks, exchange 3, bases, rewrite or compact write-back. Only for
//                  sparse_kv_dispatch, which reads nothing else.
#if defined(PLAN_ABL_DISPATCH_ONLY) && !PLAN_OWNER_MAJOR
#error "PLAN_ABL=DISPATCH_ONLY needs owner_major"
#endif

// Fused into sparse_kv_dispatch (PLAN_FUSED, kvd4: the dispatch kernels call plan_prologue() first): the plan's
// compile-time args follow the dispatch kernel's (PLAN_CT_BASE), its common args follow the dispatch's
// (PLAN_COMMON_BASE: indices, table, chunk start, then the cores' coordinates and the multicast rectangle, then the
// table-ready targets), its buffers sit at PLAN_BUF_OFFSETS in the dispatch CB PLAN_OVERLAY_CB (stage + pool: no
// dispatch traffic uses them before this chip's plan is done), its semaphores at PLAN_SEM_BASE, and its release CB is
// PLAN_RELEASE_CB. A core finds its index from its coordinates.
#ifdef PLAN_FUSED
#if !defined(PLAN_ABL_DISPATCH_ONLY) || PLAN_WORKERS != 5
#error "PLAN_FUSED is the dispatch-only plan on five workers"
#endif
#define PLAN_CT(i) get_compile_time_arg_val(PLAN_CT_BASE + (i))
#define PLAN_TA_BASE PLAN_CT_BASE
#define PLAN_COMMON(i) get_common_arg_val<uint32_t>(PLAN_COMMON_BASE + (i))
#define PLAN_COORDS_OFF 3
#define PLAN_SEM(id) get_semaphore(PLAN_SEM_BASE + (id))
#define PLAN_RELEASE PLAN_RELEASE_CB
#define PLAN_ENTRY inline void plan_prologue()
#else
#define PLAN_CT(i) get_compile_time_arg_val(i)
#define PLAN_TA_BASE 0
#define PLAN_COMMON(i) get_common_arg_val<uint32_t>(i)
#define PLAN_COORDS_OFF 0
#define PLAN_SEM(id) get_semaphore(id)
#define PLAN_RELEASE ::sparse_push_plan::kMisc
#define PLAN_ENTRY void kernel_main()
#endif

// sparse_push_plan: turns one chip's top-k indices into the push's plan table and into compact indices.
//   table:   a bitmap of the needed rows (one bit per cache row), per-owner row counts and prefix counts, the header
//            (layout in sparse_push_constants.hpp, plan_table).
//   compact: every index rewritten to its compact-buffer page, compact_page(run base of its owner, rank), where rank is
//            how many needed rows of the same owner come before it. Ranks need the whole bitmap, hence the steps below.
//
// Every core of the grid takes 1/cores of the indices, and its two data movement processors share the L1 work, half
// each (loops in sparse_push_plan_work.hpp): BRISC (worker 0) drives the steps and does the NoC traffic, NCRISC (worker
// 1, this file as the helper) shares the NoC pushes and the write-back. Within a core, BRISC posts each step and the
// workers report on their
// done semaphores; across cores, exchange(k) gathers on core 0, which multicasts the result back to all.
//
//   0. Load:    read this core's indices; zero its private full-size bitmap and its dirty-chunk flags.
//   1. Scatter: set this core's indices' bits in its own bitmap, without atomics: each worker buckets its indices by
//               the worker that owns their bitmap word, then sets bits only in its own words. Marks the 512-row chunks
//               it touched and the largest row seen; exchange 1 agrees on the largest row any core needs.
//   2. Reduce:  merge the cores' bitmaps. Chunk c is reduced by core c % cores (cyclic, so the chunks every core
//               touches spread out): each core pushes its dirty copies there, the reducer ORs them, and exchange 2
//               gives every core the final bitmap.
//   3. Ranks:   per range of range_blocks blocks (dealt cyclically to cores), each word's rank within its owner and
//               each owner's count; exchange 3 shares them. Each core writes its ranges' bitmap to the table.
//   4. Bases:   per owner, the running offset of each range, the total, and the compact-buffer run base.
//   5. Rewrite: every worker rewrites its share of the indices; BRISC and NCRISC write them back, core 0 the header.

PLAN_ENTRY {
#if defined(PLAN_FUSED) && defined(KVD_ABL_FUSED_NOPLAN)
    return;  // (probe: the fused program's cost without the plan; the caller passes a built table)
#endif
    using namespace sparse_push_plan;
#ifdef PLAN_STATS
    uint32_t plan_t[16] = {};
    auto plan_mark = [&](uint32_t i) {
        plan_t[i] = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    };
#else
    auto plan_mark = [](uint32_t) {};
#endif
    plan_mark(0);
    constexpr uint32_t num_devices = PLAN_CT(0);
    constexpr uint32_t stripe_chunk = PLAN_CT(1);
    constexpr uint32_t pages_per_packet = PLAN_CT(2);
    constexpr uint32_t banks = PLAN_CT(3);
    constexpr uint32_t cores = PLAN_CT(4);
    constexpr uint32_t total_chunks = PLAN_CT(5);  // 512-row chunks of the bitmap
    constexpr auto split_source = static_cast<SplitSource>(PLAN_CT(6));
    constexpr bool has_split = split_source != kNoSplit;
    constexpr uint32_t bitmap_words = PLAN_CT(7);
    constexpr uint32_t bucket_cap = PLAN_CT(8);    // indices per (from, to) scatter bucket
    constexpr uint32_t range_blocks = PLAN_CT(9);  // blocks per rank range, a multiple of num_devices
    constexpr uint32_t topk = PLAN_CT(10);
    // This file runs on both data movement processors: BRISC drives (worker 0), NCRISC helps (worker 1).
    constexpr bool is_helper = PLAN_CT(11) != 0;
    constexpr uint32_t dense_below = PLAN_CT(12);      // dense mode below this chunk start (0: off)
    constexpr uint32_t dense_slot_rows = PLAN_CT(13);  // a chip's slot in the dense layout
    // A second, higher dense threshold for a third output (0: off): see SparsePushPlanParams::dense_below_2.
    constexpr uint32_t dense_below_2 = PLAN_CT(14);
    constexpr uint32_t dense_slot_rows_2 = PLAN_CT(15);
    constexpr bool has_compact2 = dense_below_2 != 0;
    constexpr uint32_t max_rows = bitmap_words * kRowsPerWord;
    constexpr uint32_t range_words = owner_stride(num_devices);  // a range's partials
    constexpr auto indices_args = TensorAccessorArgs<PLAN_TA_BASE + 16>();
    constexpr auto table_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto compact_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto compact2_args = TensorAccessorArgs<compact_args.next_compile_time_args_offset()>();
    constexpr auto split_args =
        // (without a split tensor, any valid accessor's args: the indices'; offset 0 could decode a dispatch kernel's
        // own first arg as accessor flags)
        TensorAccessorArgs<
            split_source == kSplitTensor ? compact2_args.next_compile_time_args_offset() : PLAN_TA_BASE + 16>();
    constexpr uint32_t words_per_block = stripe_chunk / sparse_push::plan_table::kRowsPerWord;
    constexpr uint32_t index_bytes = sizeof(uint32_t);  // one top-k index
    constexpr uint32_t word_bytes = sizeof(uint32_t);   // one bitmap / table / scratch word
    constexpr uint32_t rank_bytes = sizeof(uint16_t);   // one word_ranks entry
    constexpr uint32_t kCoordsPerCore = 2;              // common args: (x, y) per core
    constexpr uint32_t kZeroSeedBytes = 8192;           // zero(): zeroed from MEM_ZEROS, then copied over the rest
    constexpr uint32_t array_words = dirty_array_words(cores);
    constexpr uint32_t table_page_bytes = sparse_push::plan_table::kPageBytes;
    constexpr uint32_t bitmap_off = sparse_push::plan_table::kBitmapOff;

#ifdef PLAN_FUSED
    const uint32_t indices_addr = PLAN_COMMON(0);
    const uint32_t table_addr = PLAN_COMMON(1);
    const uint32_t compact_addr = 0;
    uint32_t core_idx = 0;
    while (core_idx < cores && PLAN_COMMON(PLAN_COORDS_OFF + core_idx) !=
                                   (static_cast<uint32_t>(my_x[0]) | static_cast<uint32_t>(my_y[0]) << 16)) {
        ++core_idx;
    }
    const uint32_t first_index = std::min<uint32_t>(PLAN_TOTAL_IDX, core_idx * PLAN_PER_CORE);
    const uint32_t index_count = std::min<uint32_t>(PLAN_TOTAL_IDX, first_index + PLAN_PER_CORE) - first_index;
    const uint32_t split_arg = has_split ? PLAN_COMMON(2) : 0;
    const uint32_t compact2_addr = 0;
#else
    size_t arg = 0;
    const uint32_t indices_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t table_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t compact_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t core_idx = get_arg_val<uint32_t>(arg++);
    const uint32_t first_index = get_arg_val<uint32_t>(arg++);  // this core's indices: flat [first, first + count)
    const uint32_t index_count = get_arg_val<uint32_t>(arg++);
#endif
    // BRISC and NCRISC (each on its own NoC) split the NoC work in two halves: step 2's reducers and step 5's
    // write-back.
    constexpr uint32_t brisc_reducers = cores / 2;   // BRISC pushes to reducers [0, this), NCRISC to the rest
    const uint32_t brisc_indices = index_count / 2;  // BRISC writes back idx [0, this), NCRISC the rest
#ifndef PLAN_FUSED
    // The chunk start: its tensor's address (kSplitTensor) or the row itself (kSplitValue).
    const uint32_t split_arg = has_split ? get_arg_val<uint32_t>(arg++) : 0;
    const uint32_t compact2_addr = has_compact2 ? get_arg_val<uint32_t>(arg++) : 0;
#endif
    // Common runtime args: every core's (x, y), core 0 first, then the multicast rectangle of all of them.
    // NoC address of `l1` on plan core `core`.
#ifdef PLAN_FUSED
    // (fused: one word a core, x | y << 16, then the rectangle as two words, to keep within the runtime-arg budget)
    auto core_x = [&](uint32_t core) { return PLAN_COMMON(PLAN_COORDS_OFF + core) & 0xFFFFu; };
    auto core_y = [&](uint32_t core) { return PLAN_COMMON(PLAN_COORDS_OFF + core) >> 16; };
    constexpr uint32_t kRectOff = PLAN_COORDS_OFF + cores;
    auto rect = [&](uint32_t i) {  // start x, start y, end x, end y
        const uint32_t v = PLAN_COMMON(kRectOff + i / 2);
        return i % 2 == 0 ? (v & 0xFFFFu) : (v >> 16);
    };
#else
    auto core_x = [&](uint32_t core) { return PLAN_COMMON(PLAN_COORDS_OFF + kCoordsPerCore * core); };
    auto core_y = [&](uint32_t core) { return PLAN_COMMON(PLAN_COORDS_OFF + kCoordsPerCore * core + 1); };
    auto rect = [&](uint32_t i) { return PLAN_COMMON(PLAN_COORDS_OFF + kCoordsPerCore * cores + i); };
#endif
    auto core_noc_addr = [&](uint32_t core, uint32_t l1) { return get_noc_addr(core_x(core), core_y(core), l1); };
    // Multicast address of `l1` on every plan core.
    auto all_cores_mcast_addr = [&](uint32_t l1) {
        // (on NOC1 the rectangle is given end first: a fused plan's root can run on a NOC1 kernel)
        return noc_index == 0 ? get_noc_multicast_addr(rect(0), rect(1), rect(2), rect(3), l1)
                              : get_noc_multicast_addr(rect(2), rect(3), rect(0), rect(1), l1);
    };

    const auto indices = TensorAccessor(indices_args, indices_addr, topk * sizeof(uint32_t));
    // Owner-major tables (PLAN_OWNER_MAJOR): one slot page per owner, the header then that owner's block bitmaps.
    constexpr bool owner_major = PLAN_OWNER_MAJOR != 0;
    const auto table = TensorAccessor(table_args, table_addr, owner_major ? PLAN_SLOT_BYTES : table_page_bytes);
    const auto compact = TensorAccessor(compact_args, compact_addr, topk * sizeof(uint32_t));
    const auto compact2 = TensorAccessor(compact2_args, compact2_addr, topk * sizeof(uint32_t));

#ifdef PLAN_FUSED
    constexpr uint32_t kPlanBufOff[] = {PLAN_BUF_OFFSETS};
    const uint32_t plan_overlay = (get_write_ptr(PLAN_OVERLAY_CB) + 63) & ~63u;
#define PLAN_BUF(cb) (plan_overlay + kPlanBufOff[cb])
#else
#define PLAN_BUF(cb) get_write_ptr(cb)
#endif
    const uint32_t table_image_l1 = PLAN_BUF(kTableImage);
    const uint32_t idx_l1 = PLAN_BUF(kIdx);
    const uint32_t dirty_chunks_l1 = PLAN_BUF(kDirtyChunks);
    const uint32_t dirty_maps_l1 = PLAN_BUF(kDirtyMaps);
    const uint32_t scratch_l1 = PLAN_BUF(kScratch);
    const uint32_t misc_l1 = PLAN_BUF(kMisc);
    // Plain pointers: every NOC transfer into these buffers is followed by a barrier and a cache invalidation, and
    // every worker hand-off by a fence (post / wait_at_least).
    auto* table_image = reinterpret_cast<uint32_t*>(table_image_l1);  // this core's L1 copy of the whole plan table
    uint32_t* bitmap = table_image + bitmap_off;
    auto* idx = reinterpret_cast<uint32_t*>(idx_l1);  // this core's indices, rewritten in place in step 5
    auto* dirty_chunks = reinterpret_cast<uint8_t*>(dirty_chunks_l1);  // [chunk]: 1 if this core set a bit in it
    auto* dirty_maps = reinterpret_cast<uint32_t*>(dirty_maps_l1);     // the DirtyMap arrays
    auto* scratch = reinterpret_cast<uint32_t*>(scratch_l1);  // as a reducer: the chunk copies other cores pushed
    const uint32_t partials_l1 = PLAN_BUF(kPartials);
    auto* partials = reinterpret_cast<uint32_t*>(partials_l1);  // [range][owner]: owner's needed rows in the range
    const uint32_t word_ranks_l1 = PLAN_BUF(kWordRanks);
    // [word]: this owner's needed rows before the word, within its range (fits 16 bits: see range_blocks).
    auto* word_ranks = reinterpret_cast<uint16_t*>(word_ranks_l1);
    auto* misc = reinterpret_cast<uint32_t*>(misc_l1);  // words shared between the workers (kMisc*)
    auto* buckets = reinterpret_cast<uint32_t*>(PLAN_BUF(kBuckets));
    uint32_t* copies = misc + kMiscCopies;

    // Intra-core sync: semaphores with BRISC and NCRISC alone; with the TRISCs (kWorkers 5) words of kMisc, which
    // BRISC zeroes and then releases with a push of the kMisc CB (CB counters restart every launch, the words do not).
    constexpr bool l1_sync = kWorkers > 2;
    const uint32_t sem_step = l1_sync ? misc_l1 + kMiscSync * word_bytes : PLAN_SEM(kSemStep);
    // Waits until BRISC has started step `step`.
    auto wait_for_step = [&](uint32_t step) { wait_at_least(sem_step, step); };
    auto sem_done = [&](uint32_t w) {
        return l1_sync ? misc_l1 + (kMiscSync + 2 + w) * word_bytes : static_cast<uint32_t>(PLAN_SEM(kSemDone + w));
    };
    if constexpr (l1_sync) {
        if constexpr (is_helper) {
            cb_wait_front(PLAN_RELEASE, 1);
        } else {
            for (uint32_t w = 0; w < kMiscSyncWords; ++w) {
                misc[kMiscSync + w] = 0;
            }
            fence();
#ifdef PLAN_FUSED
            misc[kMiscSync + kMiscSyncWords] = core_idx;  // (the TRISCs find their core's index here)
            fence();
#endif
            cb_reserve_back(PLAN_RELEASE, 1);
            cb_push_back(PLAN_RELEASE, 1);
        }
    }

    uint32_t* my_dirty_map = dirty_maps + kMyDirtyMap * array_words;
    const uint32_t column_l1 = dirty_maps_l1 + kColumn * array_words * sizeof(uint32_t);
    const uint32_t row_ends_l1 = dirty_maps_l1 + kRowEnds * array_words * sizeof(uint32_t);
    // Push this core's dirty copies of reducers [first_reducer, end_reducer)'s chunks to their scratch slots, then add
    // its map word into each reducer's column (the atomic adds land in zeroed words, one writer each).
    auto push = [&](uint32_t first_reducer, uint32_t end_reducer) {
        constexpr uint32_t chunk_bytes = kChunkWords * sizeof(uint32_t);
        for (uint32_t reducer = first_reducer; reducer < end_reducer; ++reducer) {
            for (uint32_t dirty_bits = my_dirty_map[reducer]; dirty_bits != 0; dirty_bits &= dirty_bits - 1) {
                const uint32_t reducer_chunk = __builtin_ctz(dirty_bits);  // which of the reducer's chunks
                const uint32_t chunk = reducer + reducer_chunk * cores;    // chunks are dealt cyclically to reducers
                // The reducer keeps every core's copy of its chunk reducer_chunk at scratch slot [reducer_chunk][core].
                const uint32_t slot_l1 = scratch_l1 + (reducer_chunk * cores + core_idx) * chunk_bytes;
                noc_async_write(
                    table_image_l1 + (bitmap_off + chunk * kChunkWords) * sizeof(uint32_t),
                    core_noc_addr(reducer, slot_l1),
                    chunk_bytes);
            }
        }
        for (uint32_t reducer = first_reducer; reducer < end_reducer; ++reducer) {
            if (my_dirty_map[reducer] != 0) {
                noc_semaphore_inc(
                    core_noc_addr(reducer, column_l1 + core_idx * sizeof(uint32_t)), my_dirty_map[reducer]);
            }
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
    };
    // Between bucketing and scatter: BRISC, whose NoC writes zero the bitmap, announces it is zero; the other workers
    // wait for that.
    const uint32_t sem_zeroed = l1_sync ? misc_l1 + (kMiscSync + 1) * word_bytes : PLAN_SEM(kSemZeroed);
    auto sync_zeroed_bitmap = [&]() {
        plan_mark(12);  // bucketed
        if constexpr (is_helper) {
            wait_at_least(sem_zeroed, 1);
        } else {
            noc_async_write_barrier();
            post(sem_zeroed, 1);
        }
    };
    auto scatter_and_map = [&](uint32_t worker) {
        misc[kMiscRowEnd + worker] = scatter_step<max_rows>(
            idx,
            index_count,
            buckets,
            bucket_cap,
            misc + kMiscBucketed,
            bitmap,
            dirty_chunks,
            worker,
            sem_done,
            sync_zeroed_bitmap);
        plan_mark(13);  // scattered (own)
        sync_workers(sem_done, worker, kScattered);
        plan_mark(14);  // both scattered
        map_dirty<cores, total_chunks>(dirty_chunks, my_dirty_map, worker);
        sync_workers(sem_done, worker, kMapped);
    };
    auto* lists = reinterpret_cast<uint16_t*>(buckets);  // the buckets are spent by the OR
    auto or_step = [&](uint32_t worker) {
        list_copies<cores>(dirty_maps + kColumn * array_words, misc[kMiscChunks], lists, copies, worker);
        sync_workers(sem_done, worker, kListed);
        or_copies<cores>(scratch, lists, copies, misc[kMiscChunks], core_idx, bitmap, worker);
        post(sem_done(worker), kOred);  // report only: BRISC waits for it
    };
    // This core's rank ranges (visit_ranges in sparse_push_plan_work.hpp).
    auto for_my_ranges = [&](auto&& visit) {
        visit_ranges<cores, range_blocks>(core_idx, misc[kMiscRanges], misc[kMiscActiveBlocks], visit);
    };
    auto ranks = [&](uint32_t worker) {
        for_my_ranges([&](uint32_t range, uint32_t first_block, uint32_t end_block) {
            range_ranks<num_devices, words_per_block>(
                bitmap, word_ranks, partials + range * range_words, first_block, end_block, worker);
        });
    };

    // Calls visit(index_row, col, pos, len) for each run of this core's indices that lies in one index row: the run
    // starts at column col of that row and at position pos of idx.
    auto for_each_run = [&](auto&& visit) {
        for (uint32_t i = first_index, end = first_index + index_count; i < end;) {
            const uint32_t index_row = i / topk, col = i % topk;
            const uint32_t len = std::min(topk - col, end - i);
            visit(index_row, col, i - first_index, len);
            i += len;
        }
    };
    // Writes positions [first_pos, end_pos) of the L1 indices at src_l1 to the same positions of `out`.
    auto write_positions = [&](uint32_t src_l1, const auto& out, uint32_t first_pos, uint32_t end_pos) {
        for_each_run([&](uint32_t index_row, uint32_t col, uint32_t pos, uint32_t len) {
            // The part of this run inside [first_pos, end_pos).
            const uint32_t from = std::max(pos, first_pos), to = std::min(pos + len, end_pos);
            if (from < to) {
                noc_async_write(
                    src_l1 + from * index_bytes,
                    out.get_noc_addr(index_row) + (col + from - pos) * index_bytes,
                    (to - from) * index_bytes);
            }
        });
    };
    // With dense_below_2 and the chunk start below it: the third output is this worker's share of the indices in the
    // second dense layout, rewritten from the originals before the in-place rewrite below. Otherwise it gets the same
    // indices as the second output (write_back).
    const uint32_t idx2_l1 = has_compact2 ? PLAN_BUF(kIdx2) : 0;
    auto dense2 = [&]() { return has_compact2 && misc[kMiscSplit] < dense_below_2; };
    auto dense2_rewrite_and_write = [&](uint32_t worker) {
        if constexpr (has_compact2) {
            auto* idx2 = reinterpret_cast<uint32_t*>(idx2_l1);
            const uint32_t begin = share_begin(index_count, worker), end = share_begin(index_count, worker + 1);
            for (uint32_t i = begin; i < end; ++i) {
                const uint32_t row = idx[i];
                if (row >= max_rows) {
                    idx2[i] = kSentinel;
                    continue;
                }
                const uint32_t block = row / stripe_chunk;
                idx2[i] = (block % num_devices) * dense_slot_rows_2 + (block / num_devices) * stripe_chunk +
                          row % stripe_chunk;
            }
            write_positions(idx2_l1, compact2, begin, end);
        }
    };
    // Writes the rewritten positions [first_pos, end_pos) of idx back to the compact indices (and to the third output
    // when it takes the same ones), once every worker has rewritten its share. BRISC and NCRISC each take half.
    auto write_back = [&](uint32_t first_pos, uint32_t end_pos) {
        for (uint32_t w = 0; w < kWorkers; ++w) {
            wait_at_least(sem_done(w), kRewritten);
        }
        write_positions(idx_l1, compact, first_pos, end_pos);
        if (has_compact2 && !dense2()) {
            write_positions(idx_l1, compact2, first_pos, end_pos);
        }
        noc_async_write_barrier();
    };

    // Dense mode: every index becomes its row in the dense gather's layout, plain arithmetic, so no step before the
    // rewrite is needed.
    auto dense_rewrite = [&](uint32_t worker) {
        rewrite<num_devices, max_rows, stripe_chunk, range_blocks, pages_per_packet, banks, dense_slot_rows>(
            idx,
            share_begin(index_count, worker),
            share_begin(index_count, worker + 1),
            bitmap,
            word_ranks,
            partials,
            misc + kMiscBase,
            true);
        post(sem_done(worker), kRewritten);
    };

    // NCRISC (worker 1): follows BRISC's steps, does its half, and the second half of the NoC pushes and
    // write-back.
    auto run_helper = [&]() {
        constexpr uint32_t worker = 1;

        // 1. Scatter, map the dirty chunks and push the second half of the dirty copies.
        wait_for_step(kStepScatter);
        if constexpr (dense_below != 0 || has_compact2) {
            invalidate_l1_cache();
        }
        if (dense2()) {
            dense2_rewrite_and_write(worker);
        }
        if constexpr (dense_below != 0) {
            if (is_dense<dense_below>(misc)) {  // dense mode: only the arithmetic rewrite (see run_driver)
                dense_rewrite(worker);
                write_back(brisc_indices, index_count);
                return;
            }
        }
        scatter_and_map(worker);
        push(brisc_reducers, cores);
        post(sem_done(worker), kPushed);  // report only: BRISC waits for it

        // 2. Reduce: OR this core's share of the received copies.
        wait_for_step(kStepReduce);
        or_step(worker);
#ifdef PLAN_ABL_DISPATCH_ONLY
        // the slot builders' copy, split over the workers (BRISC posts kStepRanks on every core after exchange 2)
        wait_for_step(kStepRanks);
        if (num_devices <= cores && core_idx < num_devices) {
            slot_share<num_devices, words_per_block>(bitmap, scratch, core_idx, misc[kMiscActiveBlocks], worker);
        }
        post(sem_done(worker), kRanked);
        return;
#endif

        // 3. Ranks.
        wait_for_step(kStepRanks);
        ranks(worker);
        post(sem_done(worker), kRanked);  // report only: BRISC waits for it

        // 4. Bases and 5. rewrite; then write back the second half of the indices.
        wait_for_step(kStepBases);
        bases_and_rewrite<
            num_devices,
            max_rows,
            stripe_chunk,
            range_blocks,
            pages_per_packet,
            banks,
            dense_below,
            dense_slot_rows>(idx, index_count, bitmap, word_ranks, partials, misc, worker, sem_done);
        write_back(brisc_indices, index_count);
    };

    // BRISC (worker 0): drives the steps, the NoC loads, the cross-core exchanges and the table writes.
    auto run_driver = [&]() {
        constexpr uint32_t worker = 0;
        // BRISC waits for the helper to report `progress`.
        auto wait_workers = [&](uint32_t progress) {
            for (uint32_t w = 1; w < kWorkers; ++w) {
                wait_at_least(sem_done(w), progress);
            }
        };
        auto ceil_div = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };
        // NoC transfers between L1s move whole L1_ALIGNMENT (16 B) units.
        auto l1_transfer_bytes = [](uint32_t bytes) { return align(bytes, L1_ALIGNMENT); };
        const uint32_t arrive_sem = PLAN_SEM(kSemArrive);
        const uint32_t go_sem = PLAN_SEM(kSemGo);
        // Gather on core 0, which multicasts the spans back: concurrent multicasts from every core serialize on the
        // NOC. spans(send) calls send(l1, bytes) for each span to multicast. `round` is 1, 2, 3: the arrive count is
        // never reset, so core 0 waits for cores * round arrivals.
        auto exchange = [&](uint32_t round, auto&& spans) {
            noc_async_write_barrier();
            noc_semaphore_inc(core_noc_addr(0, arrive_sem), 1);
            noc_async_atomic_barrier();
            if (core_idx == 0) {
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_sem), cores * round);
                if constexpr (cores > 1) {
                    // Linked: the go value lands after the spans on every core, with no ack round trip between. A span
                    // is empty when no index on this chip is set (every row masked); the go value still goes out.
                    spans([&](uint32_t l1, uint32_t bytes) {
                        if (bytes != 0) {
                            noc_async_write_multicast(
                                l1, all_cores_mcast_addr(l1), l1_transfer_bytes(bytes), cores - 1, true);
                        }
                    });
                    noc_async_writes_flushed();  // the previous go value has left misc
                    misc[kMiscGo] = round;
                    noc_semaphore_set_multicast(
                        misc_l1 + kMiscGo * word_bytes, all_cores_mcast_addr(go_sem), cores - 1);
                }
            } else {
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem), round);
            }
            invalidate_l1_cache();
        };
        auto to_root = [&](uint32_t l1, uint32_t bytes) {
            noc_async_write(l1, core_noc_addr(0, l1), l1_transfer_bytes(bytes));
        };
        // Zero [addr, addr + bytes): up to kZeroSeedBytes from the zeros region, then copies of that seed.
        auto zero = [&](uint32_t addr, uint32_t bytes) {
            const uint32_t seed = std::min(bytes, kZeroSeedBytes);
            for (uint32_t off = 0; off < seed; off += MEM_ZEROS_SIZE) {
                noc_async_write(
                    MEM_ZEROS_BASE, get_noc_addr(addr + off), std::min<uint32_t>(seed - off, MEM_ZEROS_SIZE));
            }
            noc_async_write_barrier();
            for (uint32_t off = seed; off < bytes; off += seed) {
                noc_async_write(addr, get_noc_addr(addr + off), std::min(bytes - off, seed));
            }
        };
        // Copy table image bytes [off, off + bytes) to the same offsets of the table, page by page.
        auto write_span = [&](uint32_t off, uint32_t bytes) {
            for (uint32_t end = off + bytes; off < end;) {
                const uint32_t in_page = off % table_page_bytes;
                const uint32_t page_bytes_left = std::min(end - off, table_page_bytes - in_page);
                noc_async_write(
                    table_image_l1 + off, table.get_noc_addr(off / table_page_bytes) + in_page, page_bytes_left);
                off += page_bytes_left;
            }
        };

        // Core 0: the header and per-owner counts (filled in table_image), written to the table. In dense mode the
        // dense gather fills the buffer and the push finds no rows: every owner count is 0.
        auto fill_header = [&](bool dense, uint32_t active_blocks, uint32_t split_block) {
            using namespace sparse_push::plan_table;
            if (dense) {
                for (uint32_t owner = 0; owner < num_devices; ++owner) {
                    table_image[kOwnerOff + owner] = 0;
                    table_image[kOwnerPrefixOff + owner] = 0;
                }
            }
            table_image[kWordMagic] = kMagic;
            table_image[kWordNumDevices] = num_devices;
            table_image[kWordStripeChunk] = stripe_chunk;
            table_image[kWordActiveBlocks] = active_blocks;
            table_image[kWordSplitBlock] = split_block;
            table_image[kWordPagesPerPacket] = pages_per_packet;
            table_image[kWordBanks] = banks;
            table_image[kWordHasSplit] = has_split ? 1 : 0;
            table_image[kWordDense] = dense ? 1 : 0;
            table_image[kWordDense2] = (dense || dense2()) ? 1 : 0;
            table_image[kWordHasSummary] = 0;  // (owner-major slots set their own)
        };
        auto write_header = [&](bool dense, uint32_t active_blocks, uint32_t split_block) {
            fill_header(dense, active_blocks, split_block);
            write_span(0, bitmap_off * word_bytes);  // header and per-owner counts (slot 0 when owner-major)
        };

        // Owner-major: this core's owners' slots: the header (every core has every owner's counts), then the owner's
        // blocks' bitmap words, gathered from the full bitmap every core holds since exchange 2.
        auto build_slots = [&](uint32_t active_blocks, uint32_t split_block) {
            fill_header(/*dense=*/false, active_blocks, split_block);
            auto* slot = scratch;
            for (uint32_t w = 0; w < bitmap_off; ++w) {
                slot[w] = table_image[w];
            }
            for (uint32_t owner = core_idx; owner < num_devices; owner += cores) {
                const uint32_t owner_blocks =
                    active_blocks > owner ? (active_blocks - owner + num_devices - 1) / num_devices : 0;
                using namespace sparse_push::plan_table;
                const bool summary = owner_blocks * words_per_block <= kSummaryWords * 32;
                for (uint32_t s = 0; s < kSummaryWords; ++s) {
                    slot[kSummaryOff + s] = 0;
                }
                for (uint32_t block = 0; block < owner_blocks; ++block) {
                    const uint32_t* src = bitmap + (owner + block * num_devices) * words_per_block;
                    for (uint32_t w = 0; w < words_per_block; ++w) {
                        const uint32_t u = block * words_per_block + w;
                        slot[bitmap_off + u] = src[w];
                        if (summary && src[w] != 0) {
                            slot[kSummaryOff + u / 32] |= 1u << (u % 32);
                        }
                    }
                }
                slot[kWordHasSummary] = summary ? kSummaryMagic : 0;
                const uint32_t bytes = (bitmap_off + owner_blocks * words_per_block) * word_bytes;
                noc_async_write(scratch_l1, table.get_noc_addr(owner), (bytes + 63) & ~63u);
                noc_async_write_barrier();
            }
        };
        auto print_stats = [&]() {
#ifdef PLAN_STATS
            DPRINT(
                "PLS core={} t0={} load={} scat={} push={} ex1={} or={} ex2={} ranks={} ex3={} rw={} slot={} end={} "
                "bk={} sc={} scw={}\n",
                core_idx,
                plan_t[0],
                plan_t[1] - plan_t[0],
                plan_t[2] - plan_t[1],
                plan_t[3] - plan_t[2],
                plan_t[4] - plan_t[3],
                plan_t[5] - plan_t[4],
                plan_t[6] - plan_t[5],
                plan_t[7] - plan_t[6],
                plan_t[8] - plan_t[7],
                plan_t[9] - plan_t[8],
                plan_t[10] - plan_t[9],
                plan_t[11] - plan_t[10],
                plan_t[12] - plan_t[1],
                plan_t[13] - plan_t[12],
                plan_t[14] - plan_t[13]);
#endif
        };

        // 0. Load.
        for_each_run([&](uint32_t index_row, uint32_t col, uint32_t pos, uint32_t len) {
            noc_async_read(
                indices.get_noc_addr(index_row) + col * index_bytes, idx_l1 + pos * index_bytes, len * index_bytes);
        });
        if constexpr (split_source == kSplitTensor) {
            const auto split = TensorAccessor(split_args, split_arg);
            noc_async_read(split.get_noc_addr(0), misc_l1 + kMiscSplit * word_bytes, sizeof(uint32_t));
        } else if constexpr (split_source == kSplitValue) {
            misc[kMiscSplit] = split_arg;
        }
        if constexpr (dense_below != 0) {
            // Dense mode skips every step before the rewrite: the indices and the chunk start are all it reads.
            noc_async_read_barrier();
            invalidate_l1_cache();
            if (is_dense<dense_below>(misc)) {
                post(sem_step, kStepScatter);
                if (dense2()) {
                    dense2_rewrite_and_write(worker);
                }
                dense_rewrite(worker);
                wait_workers(kRewritten);
                if (core_idx == 0) {
                    write_header(/*dense=*/true, /*active_blocks=*/0, misc[kMiscSplit] / stripe_chunk);
                }
                write_back(0, brisc_indices);
                return;
            }
        }
        zero(table_image_l1 + bitmap_off * word_bytes, bitmap_words * word_bytes);
        zero(dirty_chunks_l1, l1_transfer_bytes(total_chunks));  // one byte per chunk
        zero(column_l1, array_words * word_bytes);
        noc_async_read_barrier();
        invalidate_l1_cache();
        plan_mark(1);

        // 1. Scatter, map the dirty chunks and push the dirty copies to their reducers.
        post(sem_step, kStepScatter);
        if (dense2()) {
            dense2_rewrite_and_write(worker);
        }
        scatter_and_map(worker);
        plan_mark(2);
        uint32_t my_row_end = 0;  // largest row this core needs + 1
        for (uint32_t w = 0; w < kWorkers; ++w) {
            my_row_end = std::max(my_row_end, misc[kMiscRowEnd + w]);
        }
        dirty_maps[kRowEnds * array_words + core_idx] = my_row_end;
        push(0, brisc_reducers);
        wait_at_least(sem_done(1), kPushed);
        plan_mark(3);
        noc_async_write(
            row_ends_l1 + core_idx * word_bytes, core_noc_addr(0, row_ends_l1 + core_idx * word_bytes), word_bytes);
        exchange(1, [&](auto&& send) { send(row_ends_l1, cores * word_bytes); });
        plan_mark(4);

        // 2. Reduce-scatter. Only rows up to the largest one any core saw are active.
        uint32_t active_rows = 0;
        for (uint32_t core = 0; core < cores; ++core) {
            active_rows = std::max(active_rows, dirty_maps[kRowEnds * array_words + core]);
        }
        const uint32_t active_blocks = ceil_div(active_rows, stripe_chunk);
        const uint32_t active_words = active_blocks * words_per_block;
        const uint32_t active_chunks = ceil_div(active_words, kChunkWords);
        // The reduce chunks this core owns: core_idx, core_idx + cores, ...
        const uint32_t my_chunks = core_idx < active_chunks ? ceil_div(active_chunks - core_idx, cores) : 0;
        misc[kMiscActiveBlocks] = active_blocks;
        misc[kMiscChunks] = my_chunks;
        post(sem_step, kStepReduce);
        or_step(worker);
        wait_workers(kOred);
        plan_mark(5);
        for (uint32_t my_chunk = 0; my_chunk < my_chunks; ++my_chunk) {
            if (copies[my_chunk] != 0) {  // chunks no core touched stay zero on core 0 too
                const uint32_t chunk = core_idx + my_chunk * cores;
                const uint32_t chunk_l1 = table_image_l1 + (bitmap_off + chunk * kChunkWords) * word_bytes;
                to_root(chunk_l1, kChunkWords * word_bytes);
            }
        }
        exchange(2, [&](auto&& send) { send(table_image_l1 + bitmap_off * word_bytes, active_words * word_bytes); });
        plan_mark(6);
#ifdef PLAN_ABL_DISPATCH_ONLY
        const uint32_t split_block = has_split ? misc[kMiscSplit] / stripe_chunk : active_blocks;
        for (uint32_t owner = 0; owner < num_devices; ++owner) {
            table_image[sparse_push::plan_table::kOwnerOff + owner] = 0;
            table_image[sparse_push::plan_table::kOwnerPrefixOff + owner] = 0;
        }
        plan_mark(7);
        plan_mark(8);
        plan_mark(9);
        // owner core_idx's slot on its builder: header here, the words and summary split over the workers (a grid
        // with fewer cores than owners: BRISC builds them all, as the main plan)
        using namespace sparse_push::plan_table;
        if constexpr (num_devices > cores) {
            post(sem_step, kStepRanks);
            wait_workers(kRanked);
            build_slots(active_blocks, split_block);
        } else if (core_idx < num_devices) {
            fill_header(/*dense=*/false, active_blocks, split_block);
            for (uint32_t w = 0; w < kSummaryOff; ++w) {
                scratch[w] = table_image[w];
            }
            for (uint32_t s = 0; s < kSummaryWords; ++s) {
                scratch[kSummaryOff + s] = 0;
            }
        }
        if constexpr (num_devices <= cores) {
            post(sem_step, kStepRanks);
            if (core_idx < num_devices) {
                slot_share<num_devices, words_per_block>(bitmap, scratch, core_idx, active_blocks, worker);
            }
            wait_workers(kRanked);
        }
        if (num_devices <= cores && core_idx < num_devices) {
            const uint32_t owner_blocks =
                active_blocks > core_idx ? (active_blocks - core_idx + num_devices - 1) / num_devices : 0;
            const bool summary = owner_blocks * words_per_block <= kSummaryWords * 32;
            scratch[kWordHasSummary] = summary ? kSummaryMagic : 0;
            const uint32_t bytes = (bitmap_off + owner_blocks * words_per_block) * word_bytes;
            noc_async_write(scratch_l1, table.get_noc_addr(core_idx), (bytes + 63) & ~63u);
            noc_async_write_barrier();
#ifdef PLAN_FUSED
            // the slot is in the table: count in on the dispatch cores that read it (TX cores, planner 0)
            constexpr uint32_t kReadyOff = PLAN_COORDS_OFF + cores + 2;  // [count, x | y << 16 ...]
            for (uint32_t r = 0; r < PLAN_COMMON(kReadyOff); ++r) {
                const uint32_t xy = PLAN_COMMON(kReadyOff + 1 + r);
                noc_semaphore_inc(get_noc_addr(xy & 0xFFFFu, xy >> 16, PLAN_SEM(2)), 1);
            }
            noc_async_atomic_barrier();
#endif
        }
        plan_mark(10);
        plan_mark(11);
        print_stats();
        return;
#endif

        // 3. Ranks and per-owner partial counts of this core's block ranges.
        const uint32_t ranges = ceil_div(active_blocks, range_blocks);
        misc[kMiscRanges] = ranges;
        post(sem_step, kStepRanks);
        ranks(worker);
        wait_workers(kRanked);
        plan_mark(7);
        for_my_ranges([&](uint32_t range, uint32_t first_block, uint32_t end_block) {
            to_root(
                word_ranks_l1 + first_block * words_per_block * rank_bytes,
                (end_block - first_block) * words_per_block * rank_bytes);
            to_root(partials_l1 + range * range_words * word_bytes, range_words * word_bytes);
        });
        exchange(3, [&](auto&& send) {
            send(word_ranks_l1, active_words * rank_bytes);
            send(partials_l1, ranges * range_words * word_bytes);
        });
        plan_mark(8);
        if constexpr (!owner_major) {
            for_my_ranges([&](uint32_t range, uint32_t first_block, uint32_t end_block) {
                write_span(
                    (bitmap_off + first_block * words_per_block) * word_bytes,
                    (end_block - first_block) * words_per_block * word_bytes);
            });
        }

        // 4. Owner bases and 5. rewrite, all workers.
        post(sem_step, kStepBases);
        bases_and_rewrite<
            num_devices,
            max_rows,
            stripe_chunk,
            range_blocks,
            pages_per_packet,
            banks,
            dense_below,
            dense_slot_rows>(idx, index_count, bitmap, word_ranks, partials, misc, worker, sem_done);
        wait_workers(kRewritten);
        plan_mark(9);
        if (owner_major ? core_idx < num_devices : core_idx == 0) {
            const uint32_t* totals = misc + kMiscTotals;
            const uint32_t split_block = has_split ? misc[kMiscSplit] / stripe_chunk : active_blocks;
            using namespace sparse_push::plan_table;
            for (uint32_t owner = 0; owner < num_devices; ++owner) {
                table_image[kOwnerOff + owner] = totals[owner];
                // Rows of the owner before the split = its rank at its first block at or after the split block.
                const uint32_t first_from_split =
                    split_block + (owner + num_devices - split_block % num_devices) % num_devices;
                table_image[kOwnerPrefixOff + owner] =
                    first_from_split < active_blocks
                        ? partials[(first_from_split / range_blocks) * range_words + owner] +
                              word_ranks[first_from_split * words_per_block]
                        : totals[owner];
            }
            if constexpr (owner_major) {
                build_slots(active_blocks, split_block);
            } else {
                write_header(/*dense=*/false, active_blocks, split_block);
            }
        }
        plan_mark(10);
        write_back(0, brisc_indices);
        plan_mark(11);
        print_stats();
    };

    if constexpr (is_helper) {
        run_helper();
    } else {
        run_driver();
    }
}
