// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The three TRISCs of a sparse push plan core, workers 2..4 (see sparse_push_plan.cpp): plain RISC-V loops over L1,
// no Tensix instructions. The math TRISC sets up no circular buffer interfaces, so every TRISC takes its buffer
// and semaphore addresses from the launch message.

#include <cstdint>

#include "api/compute/common.h"
#include "sparse_push_plan_work.hpp"

using namespace sparse_push_plan;

void kernel_main() {
#if defined(TRISC_UNPACK)
    constexpr uint32_t worker = 2;
#elif defined(TRISC_MATH)
    constexpr uint32_t worker = 3;
#else
    constexpr uint32_t worker = 4;
#endif
    constexpr uint32_t num_devices = get_compile_time_arg_val(0);
    constexpr uint32_t stripe_chunk = get_compile_time_arg_val(1);
    constexpr uint32_t bitmap_off = get_compile_time_arg_val(2);
    constexpr uint32_t counts_off = get_compile_time_arg_val(3);
    constexpr uint32_t pages_per_packet = get_compile_time_arg_val(6);
    constexpr uint32_t banks = get_compile_time_arg_val(7);
    constexpr uint32_t cores = get_compile_time_arg_val(8);
    constexpr uint32_t total_chunks = get_compile_time_arg_val(10);
    constexpr uint32_t bucket_cap = get_compile_time_arg_val(15);
    constexpr uint32_t range_blocks = get_compile_time_arg_val(16);
    constexpr uint32_t words_per_block = stripe_chunk / 32;

    const uint32_t core_idx = get_arg_val<uint32_t>(0);
    const uint32_t n_idx = get_arg_val<uint32_t>(1);

    auto* mailboxes = reinterpret_cast<tt_l1_ptr mailboxes_t*>(MEM_MAILBOX_BASE);
    const auto& config = mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config;
    const uint32_t config_base = config.kernel_config_base[ProgrammableCoreType::TENSIX];
    const auto* cb_config = reinterpret_cast<const uint32_t*>(config_base + config.local_cb_offset);
    auto cb_addr = [&](uint32_t cb) { return cb_config[cb * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG]; };
    const uint32_t sem_base = config_base + config.sem_offset[ProgrammableCoreType::TENSIX];
    auto sem = [&](uint32_t id) { return sem_base + id * L1_ALIGNMENT; };

    auto* image = reinterpret_cast<uint32_t*>(cb_addr(kImage));
    uint32_t* bits = image + bitmap_off;
    uint32_t* counts = image + counts_off;
    auto* idx = reinterpret_cast<uint32_t*>(cb_addr(kIdx));
    auto* dirty = reinterpret_cast<uint8_t*>(cb_addr(kDirtyBytes));
    const auto* scratch = reinterpret_cast<const uint32_t*>(cb_addr(kScratch));
    auto* all_dirty = reinterpret_cast<uint32_t*>(cb_addr(kAllDirty));
    auto* partials = reinterpret_cast<uint32_t*>(cb_addr(kRanges));
    auto* rank16 = reinterpret_cast<uint16_t*>(cb_addr(kRank16));
    auto* misc = reinterpret_cast<uint32_t*>(cb_addr(kMisc));
    auto* buckets = reinterpret_cast<uint32_t*>(cb_addr(kBuckets));
    const uint32_t step = sem(kSemStep);
    const uint32_t done = sem(kSemDone + worker);
    auto sem_done = [&](uint32_t w) { return sem(kSemDone + w); };

    wait_at_least(step, kStepScatter);
    misc[kMiscRowEnd + worker] =
        scatter_step(idx, n_idx, buckets, bucket_cap, misc + kMiscBucketed, bits, dirty, worker, sem_done, [&]() {
            wait_at_least(sem(kSemZeroed), 1);
        });
    post(done, kScattered);
    for (uint32_t w = 0; w < kWorkers; ++w) {
        wait_at_least(sem_done(w), kScattered);
    }
    map_dirty<cores, total_chunks>(dirty, all_dirty + kMine * dirty_array_words(cores), worker);
    post(done, kMapped);
    wait_at_least(step, kStepOr);
    auto* lists = reinterpret_cast<uint16_t*>(buckets);
    list_copies<cores>(
        all_dirty + kColumn * dirty_array_words(cores), misc[kMiscChunks], lists, misc + kMiscCopies, worker);
    post(done, kListed);
    for (uint32_t w = 0; w < kWorkers; ++w) {
        wait_at_least(sem_done(w), kListed);
    }
    or_copies<cores>(scratch, lists, misc + kMiscCopies, misc[kMiscChunks], core_idx, bits, worker);
    post(done, kOred);
    wait_at_least(step, kStepRange);
    for (uint32_t r = core_idx, ranges = misc[kMiscRanges], blocks = misc[kMiscActiveBlocks]; r < ranges; r += cores) {
        const uint32_t b0 = r * range_blocks;
        range_ranks<num_devices, words_per_block>(
            bits,
            rank16,
            counts,
            partials + r * num_devices,
            b0,
            b0 + range_blocks < blocks ? b0 + range_blocks : blocks,
            worker);
    }
    post(done, kStepRange);
    wait_at_least(step, kStepBases);
    bases_and_rewrite<num_devices, stripe_chunk, range_blocks, pages_per_packet, banks>(
        idx, n_idx, bits, rank16, partials, misc, worker, sem_done);
}
