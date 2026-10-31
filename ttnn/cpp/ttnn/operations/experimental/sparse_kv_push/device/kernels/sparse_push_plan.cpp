// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "sparse_push_plan_work.hpp"

#include <algorithm>
#include <cstdint>

using namespace sparse_push_plan;

// sparse_push_plan: turns one chip's top-k indices into the push's plan table and into compact indices.
//   table:   a bitmap of the needed rows (one bit per cache row), per-owner row counts and prefix counts, the header
//            (layout in sparse_push_constants.hpp, plan_table).
//   compact: every index rewritten to its compact-buffer page, compact_page(run base of its owner, rank), where rank is
//            how many needed rows of the same owner come before it. Ranks need the whole bitmap, hence the steps below.
//
// Every core of the grid takes 1/cores of the indices, and all five processors of a core share the L1 work, 1/5 each
// (loops in sparse_push_plan_work.hpp): BRISC (worker 0, this file) drives the steps and does the NoC traffic, NCRISC
// (worker 1, this file with PLAN_HELPER) shares the NoC pushes and the write-back, the three TRISCs (workers 2-4,
// sparse_push_plan_compute.cpp) only run L1 loops. Within a core, BRISC posts each step and the workers report on their
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

void kernel_main() {
    constexpr uint32_t num_devices = get_compile_time_arg_val(0);
    constexpr uint32_t stripe_chunk = get_compile_time_arg_val(1);
    constexpr uint32_t pages_per_packet = get_compile_time_arg_val(2);
    constexpr uint32_t banks = get_compile_time_arg_val(3);
    constexpr uint32_t cores = get_compile_time_arg_val(4);
    constexpr uint32_t total_chunks = get_compile_time_arg_val(5);  // 512-row chunks of the bitmap
    constexpr bool has_split = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t bitmap_words = get_compile_time_arg_val(7);
    constexpr uint32_t bucket_cap = get_compile_time_arg_val(8);    // indices per (from, to) scatter bucket
    constexpr uint32_t range_blocks = get_compile_time_arg_val(9);  // blocks per rank range, a multiple of num_devices
    constexpr uint32_t topk = get_compile_time_arg_val(10);
    constexpr uint32_t max_rows = bitmap_words * kRowsPerWord;
    constexpr uint32_t range_words = owner_stride(num_devices);  // a range's partials
    constexpr auto indices_args = TensorAccessorArgs<11>();
    constexpr auto table_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto compact_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto split_args = TensorAccessorArgs<has_split ? compact_args.next_compile_time_args_offset() : 0>();
    constexpr uint32_t words_per_block = stripe_chunk / sparse_push::plan_table::kRowsPerWord;
    constexpr uint32_t index_bytes = sizeof(uint32_t);  // one top-k index
    constexpr uint32_t word_bytes = sizeof(uint32_t);   // one bitmap / table / scratch word
    constexpr uint32_t rank_bytes = sizeof(uint16_t);   // one word_ranks entry
    constexpr uint32_t kCoordsPerCore = 2;              // common args: (x, y) per core
    constexpr uint32_t kZeroSeedBytes = 8192;           // zero(): zeroed from MEM_ZEROS, then copied over the rest
    constexpr uint32_t array_words = dirty_array_words(cores);
    constexpr uint32_t table_page_bytes = sparse_push::plan_table::kPageBytes;
    constexpr uint32_t bitmap_off = sparse_push::plan_table::kBitmapOff;

    size_t arg = 0;
    const uint32_t indices_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t table_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t compact_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t core_idx = get_arg_val<uint32_t>(arg++);
    const uint32_t first_index = get_arg_val<uint32_t>(arg++);  // this core's indices: flat [first, first + count)
    const uint32_t index_count = get_arg_val<uint32_t>(arg++);
    // BRISC and NCRISC (each on its own NoC) split the NoC work in two halves: step 2's reducers and step 5's
    // write-back.
    constexpr uint32_t brisc_reducers = cores / 2;   // BRISC pushes to reducers [0, this), NCRISC to the rest
    const uint32_t brisc_indices = index_count / 2;  // BRISC writes back idx [0, this), NCRISC the rest
    const uint32_t split_addr = has_split ? get_arg_val<uint32_t>(arg++) : 0;
    // Common runtime args: every core's (x, y), core 0 first, then the multicast rectangle of all of them.
    // NoC address of `l1` on plan core `core`.
    auto core_noc_addr = [&](uint32_t core, uint32_t l1) {
        const uint32_t x = get_common_arg_val<uint32_t>(kCoordsPerCore * core);
        const uint32_t y = get_common_arg_val<uint32_t>(kCoordsPerCore * core + 1);
        return get_noc_addr(x, y, l1);
    };
    // Multicast address of `l1` on every plan core.
    auto all_cores_mcast_addr = [&](uint32_t l1) {
        return get_noc_multicast_addr(
            get_common_arg_val<uint32_t>(kCoordsPerCore * cores),      // start x
            get_common_arg_val<uint32_t>(kCoordsPerCore * cores + 1),  // start y
            get_common_arg_val<uint32_t>(kCoordsPerCore * cores + 2),  // end x
            get_common_arg_val<uint32_t>(kCoordsPerCore * cores + 3),  // end y
            l1);
    };

    const auto indices = TensorAccessor(indices_args, indices_addr, topk * sizeof(uint32_t));
    const auto table = TensorAccessor(table_args, table_addr, table_page_bytes);
    const auto compact = TensorAccessor(compact_args, compact_addr, topk * sizeof(uint32_t));

    const uint32_t table_image_l1 = get_write_ptr(kTableImage);
    const uint32_t idx_l1 = get_write_ptr(kIdx);
    const uint32_t dirty_chunks_l1 = get_write_ptr(kDirtyChunks);
    const uint32_t dirty_maps_l1 = get_write_ptr(kDirtyMaps);
    const uint32_t scratch_l1 = get_write_ptr(kScratch);
    const uint32_t misc_l1 = get_write_ptr(kMisc);
    // Plain pointers: every NOC transfer into these buffers is followed by a barrier and a cache invalidation, and
    // every worker hand-off by a fence (post / wait_at_least).
    auto* table_image = reinterpret_cast<uint32_t*>(table_image_l1);  // this core's L1 copy of the whole plan table
    uint32_t* bitmap = table_image + bitmap_off;
    auto* idx = reinterpret_cast<uint32_t*>(idx_l1);  // this core's indices, rewritten in place in step 5
    auto* dirty_chunks = reinterpret_cast<uint8_t*>(dirty_chunks_l1);  // [chunk]: 1 if this core set a bit in it
    auto* dirty_maps = reinterpret_cast<uint32_t*>(dirty_maps_l1);     // the DirtyMap arrays
    auto* scratch = reinterpret_cast<uint32_t*>(scratch_l1);  // as a reducer: the chunk copies other cores pushed
    const uint32_t partials_l1 = get_write_ptr(kPartials);
    auto* partials = reinterpret_cast<uint32_t*>(partials_l1);  // [range][owner]: owner's needed rows in the range
    const uint32_t word_ranks_l1 = get_write_ptr(kWordRanks);
    // [word]: this owner's needed rows before the word, within its range (fits 16 bits: see range_blocks).
    auto* word_ranks = reinterpret_cast<uint16_t*>(word_ranks_l1);
    auto* misc = reinterpret_cast<uint32_t*>(misc_l1);  // words shared between the workers (kMisc*)
    auto* buckets = reinterpret_cast<uint32_t*>(get_write_ptr(kBuckets));
    uint32_t* copies = misc + kMiscCopies;

    const uint32_t sem_step = get_semaphore(kSemStep);
    // Waits until BRISC has started step `step`.
    auto wait_for_step = [&](uint32_t step) { wait_at_least(sem_step, step); };
    auto sem_done = [](uint32_t w) { return static_cast<uint32_t>(get_semaphore(kSemDone + w)); };

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
    const uint32_t sem_zeroed = get_semaphore(kSemZeroed);
#ifdef PLAN_HELPER
    auto sync_zeroed_bitmap = [&]() { wait_at_least(sem_zeroed, 1); };
#else
    auto sync_zeroed_bitmap = [&]() {
        noc_async_write_barrier();
        post(sem_zeroed, 1);
    };
#endif
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
        sync_workers(sem_done, worker, kScattered);
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
    // Writes the rewritten positions [first_pos, end_pos) of idx back to the compact indices, once every worker has
    // rewritten its share. BRISC and NCRISC each take half.
    auto write_back = [&](uint32_t first_pos, uint32_t end_pos) {
        for (uint32_t w = 0; w < kWorkers; ++w) {
            wait_at_least(sem_done(w), kRewritten);
        }
        for_each_run([&](uint32_t index_row, uint32_t col, uint32_t pos, uint32_t len) {
            // The part of this run inside [first_pos, end_pos).
            const uint32_t from = std::max(pos, first_pos), to = std::min(pos + len, end_pos);
            if (from < to) {
                noc_async_write(
                    idx_l1 + from * index_bytes,
                    compact.get_noc_addr(index_row) + (col + from - pos) * index_bytes,
                    (to - from) * index_bytes);
            }
        });
        noc_async_write_barrier();
    };

#ifdef PLAN_HELPER
    // NCRISC (worker 1): follows BRISC's steps, does its 1/5 share, and the second half of the NoC pushes and
    // write-back.
    constexpr uint32_t worker = 1;

    // 1. Scatter, map the dirty chunks and push the second half of the dirty copies.
    wait_for_step(kStepScatter);
    scatter_and_map(worker);
    push(brisc_reducers, cores);
    post(sem_done(worker), kPushed);  // report only: BRISC waits for it

    // 2. Reduce: OR this core's share of the received copies.
    wait_for_step(kStepReduce);
    or_step(worker);

    // 3. Ranks.
    wait_for_step(kStepRanks);
    ranks(worker);
    post(sem_done(worker), kRanked);  // report only: BRISC waits for it

    // 4. Bases and 5. rewrite; then write back the second half of the indices.
    wait_for_step(kStepBases);
    bases_and_rewrite<num_devices, max_rows, stripe_chunk, range_blocks, pages_per_packet, banks>(
        idx, index_count, bitmap, word_ranks, partials, misc, worker, sem_done);
    write_back(brisc_indices, index_count);
#else
    // BRISC (worker 0): drives the steps, the NoC loads, the cross-core exchanges and the table writes.
    constexpr uint32_t worker = 0;
    // BRISC waits for workers 1-4 to report `progress`.
    auto wait_workers = [&](uint32_t progress) {
        for (uint32_t w = 1; w < kWorkers; ++w) {
            wait_at_least(sem_done(w), progress);
        }
    };
    auto ceil_div = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };
    // NoC transfers between L1s move whole L1_ALIGNMENT (16 B) units.
    auto l1_transfer_bytes = [](uint32_t bytes) { return (bytes + L1_ALIGNMENT - 1) / L1_ALIGNMENT * L1_ALIGNMENT; };
    const uint32_t arrive_sem = get_semaphore(kSemArrive);
    const uint32_t go_sem = get_semaphore(kSemGo);
    // Gather on core 0, which multicasts the spans back: concurrent multicasts from every core serialize on the NOC.
    // spans(send) calls send(l1, bytes) for each span to multicast. `round` is 1, 2, 3: the arrive count is never
    // reset, so core 0 waits for cores * round arrivals.
    auto exchange = [&](uint32_t round, auto&& spans) {
        noc_async_write_barrier();
        noc_semaphore_inc(core_noc_addr(0, arrive_sem), 1);
        noc_async_atomic_barrier();
        if (core_idx == 0) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_sem), cores * round);
            if constexpr (cores > 1) {
                // Linked: the go value lands after the spans on every core, with no ack round trip between.
                spans([&](uint32_t l1, uint32_t bytes) {
                    noc_async_write_multicast(l1, all_cores_mcast_addr(l1), l1_transfer_bytes(bytes), cores - 1, true);
                });
                noc_async_writes_flushed();  // the previous go value has left misc
                misc[kMiscGo] = round;
                noc_semaphore_set_multicast(misc_l1 + kMiscGo * word_bytes, all_cores_mcast_addr(go_sem), cores - 1);
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
            noc_async_write(MEM_ZEROS_BASE, get_noc_addr(addr + off), std::min<uint32_t>(seed - off, MEM_ZEROS_SIZE));
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

    // 0. Load.
    for_each_run([&](uint32_t index_row, uint32_t col, uint32_t pos, uint32_t len) {
        noc_async_read(
            indices.get_noc_addr(index_row) + col * index_bytes, idx_l1 + pos * index_bytes, len * index_bytes);
    });
    if constexpr (has_split) {
        const auto split = TensorAccessor(split_args, split_addr);
        noc_async_read(split.get_noc_addr(0), misc_l1 + kMiscSplit * word_bytes, sizeof(uint32_t));
    }
    zero(table_image_l1 + bitmap_off * word_bytes, bitmap_words * word_bytes);
    zero(dirty_chunks_l1, l1_transfer_bytes(total_chunks));  // one byte per chunk
    zero(column_l1, array_words * word_bytes);
    noc_async_read_barrier();
    invalidate_l1_cache();

    // 1. Scatter, map the dirty chunks and push the dirty copies to their reducers.
    post(sem_step, kStepScatter);
    scatter_and_map(worker);
    uint32_t my_row_end = 0;  // largest row this core needs + 1
    for (uint32_t w = 0; w < kWorkers; ++w) {
        my_row_end = std::max(my_row_end, misc[kMiscRowEnd + w]);
    }
    dirty_maps[kRowEnds * array_words + core_idx] = my_row_end;
    push(0, brisc_reducers);
    wait_at_least(sem_done(1), kPushed);
    noc_async_write(
        row_ends_l1 + core_idx * word_bytes, core_noc_addr(0, row_ends_l1 + core_idx * word_bytes), word_bytes);
    exchange(1, [&](auto&& send) { send(row_ends_l1, cores * word_bytes); });

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
    for (uint32_t my_chunk = 0; my_chunk < my_chunks; ++my_chunk) {
        if (copies[my_chunk] != 0) {  // chunks no core touched stay zero on core 0 too
            const uint32_t chunk = core_idx + my_chunk * cores;
            const uint32_t chunk_l1 = table_image_l1 + (bitmap_off + chunk * kChunkWords) * word_bytes;
            to_root(chunk_l1, kChunkWords * word_bytes);
        }
    }
    exchange(2, [&](auto&& send) { send(table_image_l1 + bitmap_off * word_bytes, active_words * word_bytes); });

    // 3. Ranks and per-owner partial counts of this core's block ranges.
    const uint32_t ranges = ceil_div(active_blocks, range_blocks);
    misc[kMiscRanges] = ranges;
    post(sem_step, kStepRanks);
    ranks(worker);
    wait_workers(kRanked);
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
    for_my_ranges([&](uint32_t range, uint32_t first_block, uint32_t end_block) {
        write_span(
            (bitmap_off + first_block * words_per_block) * word_bytes,
            (end_block - first_block) * words_per_block * word_bytes);
    });

    // 4. Owner bases and 5. rewrite, all workers.
    post(sem_step, kStepBases);
    bases_and_rewrite<num_devices, max_rows, stripe_chunk, range_blocks, pages_per_packet, banks>(
        idx, index_count, bitmap, word_ranks, partials, misc, worker, sem_done);
    wait_workers(kRewritten);
    if (core_idx == 0) {
        const uint32_t* totals = misc + kMiscTotals;
        const uint32_t split_block = has_split ? misc[kMiscSplit] / stripe_chunk : active_blocks;
        using namespace sparse_push::plan_table;
        for (uint32_t owner = 0; owner < num_devices; ++owner) {
            table_image[kOwnerOff + owner] = totals[owner];
            // Rows of the owner before the split = its rank at its first block at or after the split block.
            const uint32_t first_from_split =
                split_block + (owner + num_devices - split_block % num_devices) % num_devices;
            table_image[kOwnerPrefixOff + owner] =
                first_from_split < active_blocks ? partials[(first_from_split / range_blocks) * range_words + owner] +
                                                       word_ranks[first_from_split * words_per_block]
                                                 : totals[owner];
        }
        table_image[kWordMagic] = kMagic;
        table_image[kWordNumDevices] = num_devices;
        table_image[kWordStripeChunk] = stripe_chunk;
        table_image[kWordActiveBlocks] = active_blocks;
        table_image[kWordSplitBlock] = split_block;
        table_image[kWordPagesPerPacket] = pages_per_packet;
        table_image[kWordBanks] = banks;
        table_image[kWordHasSplit] = has_split ? 1 : 0;
        write_span(0, bitmap_off * word_bytes);  // header and per-owner counts
    }
    write_back(0, brisc_indices);
#endif
}
