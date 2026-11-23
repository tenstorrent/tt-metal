// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sparse_push_plan on the TRISCs (PLAN_WORKERS=5, the dispatch-only plan): workers 2-4 of a core take their share of
// the L1 loops BRISC and NCRISC run (sparse_push_plan.cpp, run_helper): bucket, scatter, dirty map, copy lists, OR,
// then the slot builders' copy. No NoC: BRISC and NCRISC move the data. Sync through the kMisc words (kMiscSync),
// released by BRISC's push of the kMisc CB.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "sparse_push_plan_work.hpp"

using namespace sparse_push_plan;

void kernel_main() {
#if defined(PLAN_FUSED) && defined(KVD_ABL_FUSED_NOPLAN)
    return;
#endif
    static_assert(kWorkers == kMaxWorkers, "the TRISC workers run only with PLAN_WORKERS=5");
    constexpr uint32_t bitmap_words = get_compile_time_arg_val(0);
    constexpr uint32_t cores = get_compile_time_arg_val(1);
    constexpr uint32_t total_chunks = get_compile_time_arg_val(2);
    constexpr uint32_t bucket_cap = get_compile_time_arg_val(3);
    constexpr uint32_t num_devices = get_compile_time_arg_val(4);
    constexpr uint32_t words_per_block = get_compile_time_arg_val(5);
    constexpr uint32_t max_rows = bitmap_words * kRowsPerWord;
    constexpr uint32_t array_words = dirty_array_words(cores);
    constexpr uint32_t word_bytes = sizeof(uint32_t);
#ifndef PLAN_FUSED
    const uint32_t index_count = get_arg_val<uint32_t>(0);
    const uint32_t core_idx = get_arg_val<uint32_t>(1);
#define PLAN_RELEASE kMisc
#define PLAN_TBUF(cb) get_tile_address(cb, 0)
#else
    // fused into sparse_kv_dispatch (sparse_push_plan.cpp, PLAN_FUSED): buffers in the overlay, the core's index
    // from BRISC (written before the release)
    constexpr uint32_t kPlanBufOff[] = {PLAN_BUF_OFFSETS};
#define PLAN_RELEASE PLAN_RELEASE_CB
#define PLAN_TBUF(cb) (((get_tile_address(PLAN_OVERLAY_CB, 0) + 63) & ~63u) + kPlanBufOff[cb])
#endif
#if defined(TRISC_UNPACK)
    constexpr uint32_t worker = 2;
#elif defined(TRISC_MATH)
    constexpr uint32_t worker = 3;
#else
    constexpr uint32_t worker = 4;
#endif

    // BRISC zeroed the sync words: UNPACK waits for its kMisc push; MATH and PACK block in get_tile_address's mailbox
    // read until UNPACK has passed that wait and posted the address (cb_api.h), so none reads a stale sync word.
    UNPACK((cb_wait_front(PLAN_RELEASE, 1)));
    const uint32_t misc_l1 = PLAN_TBUF(kMisc);
    const uint32_t idx_l1 = PLAN_TBUF(kIdx);
    const uint32_t table_image_l1 = PLAN_TBUF(kTableImage);
    const uint32_t dirty_chunks_l1 = PLAN_TBUF(kDirtyChunks);
    const uint32_t dirty_maps_l1 = PLAN_TBUF(kDirtyMaps);
    const uint32_t scratch_l1 = PLAN_TBUF(kScratch);
    const uint32_t buckets_l1 = PLAN_TBUF(kBuckets);

    auto* misc = reinterpret_cast<uint32_t*>(misc_l1);
#ifdef PLAN_FUSED
    const uint32_t core_idx = reinterpret_cast<volatile uint32_t*>(misc)[kMiscSync + kMiscSyncWords];
    const uint32_t first_index = core_idx * PLAN_PER_CORE < PLAN_TOTAL_IDX ? core_idx * PLAN_PER_CORE : PLAN_TOTAL_IDX;
    const uint32_t index_count =
        (first_index + PLAN_PER_CORE < PLAN_TOTAL_IDX ? first_index + PLAN_PER_CORE : PLAN_TOTAL_IDX) - first_index;
#endif
    auto* idx = reinterpret_cast<uint32_t*>(idx_l1);
    uint32_t* bitmap = reinterpret_cast<uint32_t*>(table_image_l1) + sparse_push::plan_table::kBitmapOff;
    auto* dirty_chunks = reinterpret_cast<uint8_t*>(dirty_chunks_l1);
    auto* dirty_maps = reinterpret_cast<uint32_t*>(dirty_maps_l1);
    const auto* scratch = reinterpret_cast<const uint32_t*>(scratch_l1);
    auto* buckets = reinterpret_cast<uint32_t*>(buckets_l1);
    auto* lists = reinterpret_cast<uint16_t*>(buckets);
    uint32_t* copies = misc + kMiscCopies;

    const uint32_t sem_step = misc_l1 + kMiscSync * word_bytes;
    const uint32_t sem_zeroed = misc_l1 + (kMiscSync + 1) * word_bytes;
    auto sem_done = [&](uint32_t w) { return misc_l1 + (kMiscSync + 2 + w) * word_bytes; };

    // 1. Scatter and map the dirty chunks (as run_helper's scatter_and_map).
    wait_at_least(sem_step, kStepScatter);
    misc[kMiscRowEnd + worker] = scatter_step<max_rows>(
        idx, index_count, buckets, bucket_cap, misc + kMiscBucketed, bitmap, dirty_chunks, worker, sem_done, [&]() {
            wait_at_least(sem_zeroed, 1);
        });
    sync_workers(sem_done, worker, kScattered);
    map_dirty<cores, total_chunks>(dirty_chunks, dirty_maps + kMyDirtyMap * array_words, worker);
    sync_workers(sem_done, worker, kMapped);

    // 2. Reduce: this worker's share of the copy lists, then of the OR (as run_helper's or_step).
    wait_at_least(sem_step, kStepReduce);
    list_copies<cores>(dirty_maps + kColumn * array_words, misc[kMiscChunks], lists, copies, worker);
    sync_workers(sem_done, worker, kListed);
    or_copies<cores>(scratch, lists, copies, misc[kMiscChunks], core_idx, bitmap, worker);
    post(sem_done(worker), kOred);

    // 3. The slot builders' copy (cores < num_devices), as run_helper's.
    wait_at_least(sem_step, kStepRanks);
    if (num_devices <= cores && core_idx < num_devices) {
        slot_share<num_devices, words_per_block>(
            bitmap, reinterpret_cast<uint32_t*>(scratch_l1), core_idx, misc[kMiscActiveBlocks], worker);
    }
    post(sem_done(worker), kRanked);
}
