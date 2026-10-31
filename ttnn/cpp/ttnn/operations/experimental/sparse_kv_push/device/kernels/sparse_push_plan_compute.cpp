// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "sparse_push_plan_work.hpp"

using namespace sparse_push_plan;

// The plan's TRISC workers 2-4: the L1 loops of steps 1-5 only, plain RISC-V code with no Tensix instructions (see the
// overview in sparse_push_plan.cpp). The math TRISC sets up no CB interfaces, so buffer and semaphore addresses come
// from the launch message.

void kernel_main() {
    // The unpack, math and pack TRISCs each run this file, as workers 2, 3 and 4.
#if defined(TRISC_UNPACK)
    constexpr uint32_t worker = 2;
#elif defined(TRISC_MATH)
    constexpr uint32_t worker = 3;
#else
    constexpr uint32_t worker = 4;
#endif
    // Same compile-time args as sparse_push_plan.cpp.
    constexpr uint32_t num_devices = get_compile_time_arg_val(0);
    constexpr uint32_t stripe_chunk = get_compile_time_arg_val(1);
    constexpr uint32_t pages_per_packet = get_compile_time_arg_val(2);
    constexpr uint32_t banks = get_compile_time_arg_val(3);
    constexpr uint32_t cores = get_compile_time_arg_val(4);
    constexpr uint32_t total_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t max_rows = get_compile_time_arg_val(7) * kRowsPerWord;  // bitmap words
    constexpr uint32_t bucket_cap = get_compile_time_arg_val(8);
    constexpr uint32_t range_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t bitmap_off = sparse_push::plan_table::kBitmapOff;
    constexpr uint32_t words_per_block = stripe_chunk / sparse_push::plan_table::kRowsPerWord;
    constexpr uint32_t array_words = dirty_array_words(cores);

    const uint32_t core_idx = get_arg_val<uint32_t>(0);
    const uint32_t index_count = get_arg_val<uint32_t>(1);

    auto* mailboxes = reinterpret_cast<tt_l1_ptr mailboxes_t*>(MEM_MAILBOX_BASE);
    const auto& config = mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config;
    const uint32_t config_base = config.kernel_config_base[ProgrammableCoreType::TENSIX];
    const auto* cb_config = reinterpret_cast<const uint32_t*>(config_base + config.local_cb_offset);
    // A CB's L1 address is the first word of its local CB config.
    auto cb_l1_addr = [&](uint32_t cb) { return cb_config[cb * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG]; };
    // Semaphores sit L1_ALIGNMENT apart from sem_base.
    const uint32_t sem_base = config_base + config.sem_offset[ProgrammableCoreType::TENSIX];
    auto sem_addr = [&](uint32_t id) { return sem_base + id * L1_ALIGNMENT; };

    auto* table_image = reinterpret_cast<uint32_t*>(cb_l1_addr(kTableImage));
    uint32_t* bitmap = table_image + bitmap_off;
    auto* idx = reinterpret_cast<uint32_t*>(cb_l1_addr(kIdx));
    auto* dirty_chunks = reinterpret_cast<uint8_t*>(cb_l1_addr(kDirtyChunks));
    const auto* scratch = reinterpret_cast<const uint32_t*>(cb_l1_addr(kScratch));
    auto* dirty_maps = reinterpret_cast<uint32_t*>(cb_l1_addr(kDirtyMaps));
    uint32_t* my_dirty_map = dirty_maps + kMyDirtyMap * array_words;
    const uint32_t* column = dirty_maps + kColumn * array_words;
    auto* partials = reinterpret_cast<uint32_t*>(cb_l1_addr(kPartials));
    auto* word_ranks = reinterpret_cast<uint16_t*>(cb_l1_addr(kWordRanks));
    auto* misc = reinterpret_cast<uint32_t*>(cb_l1_addr(kMisc));
    uint32_t* copies = misc + kMiscCopies;
    auto* buckets = reinterpret_cast<uint32_t*>(cb_l1_addr(kBuckets));
    auto* lists = reinterpret_cast<uint16_t*>(buckets);  // the buckets are spent by the OR

    const uint32_t sem_step = sem_addr(kSemStep);
    auto sem_done = [&](uint32_t w) { return sem_addr(kSemDone + w); };
    // Waits until BRISC has started step `step`.
    auto wait_for_step = [&](uint32_t step) { wait_at_least(sem_step, step); };
    // Waits until BRISC has zeroed the bitmap (see sync_zeroed_bitmap in sparse_push_plan.cpp).
    auto sync_zeroed_bitmap = [&]() { wait_at_least(sem_addr(kSemZeroed), 1); };

    // 1. Scatter and map the dirty chunks.
    wait_for_step(kStepScatter);
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
    post(sem_done(worker), kMapped);  // report only: BRISC waits for it

    // 2. Reduce: OR this core's share of the received copies.
    wait_for_step(kStepReduce);
    list_copies<cores>(column, misc[kMiscChunks], lists, copies, worker);
    sync_workers(sem_done, worker, kListed);
    or_copies<cores>(scratch, lists, copies, misc[kMiscChunks], core_idx, bitmap, worker);
    post(sem_done(worker), kOred);  // report only: BRISC waits for it

    // 3. Ranks.
    wait_for_step(kStepRanks);
    visit_ranges<cores, range_blocks>(
        core_idx,
        misc[kMiscRanges],
        misc[kMiscActiveBlocks],
        [&](uint32_t range, uint32_t first_block, uint32_t end_block) {
            range_ranks<num_devices, words_per_block>(
                bitmap, word_ranks, partials + range * owner_stride(num_devices), first_block, end_block, worker);
        });
    post(sem_done(worker), kRanked);  // report only: BRISC waits for it

    // 4. Bases and 5. rewrite.
    wait_for_step(kStepBases);
    bases_and_rewrite<num_devices, max_rows, stripe_chunk, range_blocks, pages_per_packet, banks>(
        idx, index_count, bitmap, word_ranks, partials, misc, worker, sem_done);
}
