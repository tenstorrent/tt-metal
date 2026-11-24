// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sparse_push_plan on the three TRISCs of a plan core: workers kFirstTrisc + 0..2 of the plan's L1 loops (overview in
// sparse_push_plan.cpp). They follow BRISC's steps like NCRISC does, but do no NoC work: BRISC and NCRISC load the
// indices, push the dirty copies, run the exchanges and write back. Built once per TRISC with one of TRISC_UNPACK /
// TRISC_MATH / TRISC_PACK defined, from the data movement kernel's compile-time and runtime args.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "sparse_push_plan_work.hpp"

using namespace sparse_push_plan;

#if defined(TRISC_UNPACK)
constexpr uint32_t kWorker = kFirstTrisc + 0;
#elif defined(TRISC_MATH)
constexpr uint32_t kWorker = kFirstTrisc + 1;
#elif defined(TRISC_PACK)
constexpr uint32_t kWorker = kFirstTrisc + 2;
#else
#error "sparse_push_plan_trisc.cpp built for an unknown TRISC"
#endif
static_assert(kFirstTrisc + 3 == kWorkers, "the plan's workers are BRISC, NCRISC and the three TRISCs");

namespace {

// TRISC firmware exposes neither sem_l1_base nor the CB addresses as byte addresses, so read both from this launch's
// kernel config. BRISC advances launch_msg_rd_ptr only after every RISC of the core finishes, so it names this launch.
inline const volatile tt_l1_ptr kernel_config_msg_t* launch_config() {
    const uint32_t rd = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    return &GET_MAILBOX_ADDRESS_DEV(launch[rd])->kernel_config;
}

inline uint32_t semaphore_addr(uint32_t id) {
    const auto* cfg = launch_config();
    return cfg->kernel_config_base[static_cast<int>(ProgrammableCoreType::TENSIX)] +
           cfg->sem_offset[static_cast<int>(ProgrammableCoreType::TENSIX)] + id * L1_ALIGNMENT;
}

inline uint32_t cb_addr(uint32_t cb) {
    const auto* cfg = launch_config();
    const auto* cb_config = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(
        cfg->kernel_config_base[static_cast<int>(ProgrammableCoreType::TENSIX)] + cfg->local_cb_offset);
    return cb_config[cb * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG];  // the CB's L1 byte address
}

}  // namespace

void kernel_main() {
    constexpr uint32_t num_devices = get_compile_time_arg_val(0);
    constexpr uint32_t stripe_chunk = get_compile_time_arg_val(1);
    constexpr uint32_t pages_per_packet = get_compile_time_arg_val(2);
    constexpr uint32_t banks = get_compile_time_arg_val(3);
    constexpr uint32_t cores = get_compile_time_arg_val(4);
    constexpr uint32_t total_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t bitmap_words = get_compile_time_arg_val(7);
    constexpr uint32_t bucket_cap = get_compile_time_arg_val(8);
    constexpr uint32_t range_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t max_rows = bitmap_words * kRowsPerWord;
    constexpr uint32_t range_words = owner_stride(num_devices);
    constexpr uint32_t words_per_block = stripe_chunk / sparse_push::plan_table::kRowsPerWord;
    constexpr uint32_t array_words = dirty_array_words(cores);
    constexpr uint32_t bitmap_off = sparse_push::plan_table::kBitmapOff;
    constexpr uint32_t worker = kWorker;

    const uint32_t core_idx = get_arg_val<uint32_t>(3);
    const uint32_t index_count = get_arg_val<uint32_t>(5);

    auto* table_image = reinterpret_cast<uint32_t*>(cb_addr(kTableImage));
    uint32_t* bitmap = table_image + bitmap_off;
    auto* idx = reinterpret_cast<uint32_t*>(cb_addr(kIdx));
    auto* dirty_chunks = reinterpret_cast<uint8_t*>(cb_addr(kDirtyChunks));
    auto* dirty_maps = reinterpret_cast<uint32_t*>(cb_addr(kDirtyMaps));
    auto* scratch = reinterpret_cast<uint32_t*>(cb_addr(kScratch));
    auto* partials = reinterpret_cast<uint32_t*>(cb_addr(kPartials));
    auto* word_ranks = reinterpret_cast<uint16_t*>(cb_addr(kWordRanks));
    auto* misc = reinterpret_cast<uint32_t*>(cb_addr(kMisc));
    auto* buckets = reinterpret_cast<uint32_t*>(cb_addr(kBuckets));
    auto* lists = reinterpret_cast<uint16_t*>(buckets);  // the buckets are spent by the OR
    uint32_t* copies = misc + kMiscCopies;
    uint32_t* my_dirty_map = dirty_maps + kMyDirtyMap * array_words;

    const uint32_t sem_step = semaphore_addr(kSemStep);
    const uint32_t sem_zeroed = semaphore_addr(kSemZeroed);
    uint32_t sem_done_addr[kWorkers];
    for (uint32_t w = 0; w < kWorkers; ++w) {
        sem_done_addr[w] = semaphore_addr(kSemDone + w);
    }
    auto sem_done = [&](uint32_t w) { return sem_done_addr[w]; };

    // 1. Scatter and map the dirty chunks (BRISC and NCRISC push the copies).
    wait_at_least(sem_step, kStepScatter);
    misc[kMiscRowEnd + worker] = scatter_step<max_rows>(
        idx, index_count, buckets, bucket_cap, misc + kMiscBucketed, bitmap, dirty_chunks, worker, sem_done, [&]() {
            wait_at_least(sem_zeroed, 1);
        });
    sync_workers(sem_done, worker, kScattered);
    map_dirty<cores, total_chunks>(dirty_chunks, my_dirty_map, worker);
    sync_workers(sem_done, worker, kMapped);

    // 2. Reduce: OR this worker's share of the received copies.
    wait_at_least(sem_step, kStepReduce);
    list_copies<cores>(dirty_maps + kColumn * array_words, misc[kMiscChunks], lists, copies, worker);
    sync_workers(sem_done, worker, kListed);
    or_copies<cores>(scratch, lists, copies, misc[kMiscChunks], core_idx, bitmap, worker);
    post(sem_done(worker), kOred);

    // 3. Ranks.
    wait_at_least(sem_step, kStepRanks);
    visit_ranges<cores, range_blocks>(
        core_idx, misc[kMiscRanges], misc[kMiscActiveBlocks], [&](uint32_t range, uint32_t first_block, uint32_t end) {
            range_ranks<num_devices, words_per_block>(
                bitmap, word_ranks, partials + range * range_words, first_block, end, worker);
        });
    post(sem_done(worker), kRanked);

    // 4. Bases and 5. rewrite of this worker's share of the indices.
    wait_at_least(sem_step, kStepBases);
    bases_and_rewrite<num_devices, max_rows, stripe_chunk, range_blocks, pages_per_packet, banks>(
        idx, index_count, bitmap, word_ranks, partials, misc, worker, sem_done);
}
