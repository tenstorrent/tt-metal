// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The three TRISCs of a sparse_push_hot core, workers 2..4 (see sparse_push_hot.cpp): plain RISC-V loops over L1.
// Buffer and semaphore addresses come from the launch message (the math TRISC sets up no circular buffers).

#include <cstdint>

#include "api/compute/common.h"
#include "sparse_push_hot_work.hpp"

using namespace sparse_push_hot;

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
    constexpr uint32_t pages_per_packet = get_compile_time_arg_val(4);
    constexpr uint32_t banks = get_compile_time_arg_val(5);
    constexpr uint32_t cores = get_compile_time_arg_val(6);
    constexpr uint32_t threshold = get_compile_time_arg_val(7);
    constexpr uint32_t prefix_cap = get_compile_time_arg_val(8);
    constexpr uint32_t current_cap = get_compile_time_arg_val(9);
    constexpr uint32_t range_blocks = get_compile_time_arg_val(10);
    constexpr uint32_t words_per_block = stripe_chunk / 32;
    constexpr uint32_t range_words = range_blocks * words_per_block;

    const uint32_t core_idx = get_arg_val<uint32_t>(0);
    const uint32_t n_idx = get_arg_val<uint32_t>(1);

    auto* mailboxes = reinterpret_cast<tt_l1_ptr mailboxes_t*>(MEM_MAILBOX_BASE);
    const auto& config = mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config;
    const uint32_t config_base = config.kernel_config_base[ProgrammableCoreType::TENSIX];
    const auto* cb_config = reinterpret_cast<const uint32_t*>(config_base + config.local_cb_offset);
    auto cb_addr = [&](uint32_t cb) { return cb_config[cb * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG]; };
    const uint32_t sem_base = config_base + config.sem_offset[ProgrammableCoreType::TENSIX];
    auto sem = [&](uint32_t id) { return sem_base + id * L1_ALIGNMENT; };
    auto sem_done = [&](uint32_t w) { return sem(kSemDone + w); };

    auto* idx = reinterpret_cast<const uint32_t*>(cb_addr(kIdx));
    auto* compact_idx = reinterpret_cast<uint32_t*>(cb_addr(kCompact));
    const auto* scratch = reinterpret_cast<const uint32_t*>(cb_addr(kTables));
    auto* hot = reinterpret_cast<uint32_t*>(cb_addr(kHot));
    auto* rank16 = reinterpret_cast<uint16_t*>(cb_addr(kRank16));
    auto* prefixes = reinterpret_cast<uint32_t*>(cb_addr(kRanges));  // counts, then (below) the prefixes
    auto* misc = reinterpret_cast<uint32_t*>(cb_addr(kMisc));
    auto* cold = reinterpret_cast<uint32_t*>(cb_addr(kCold));
    constexpr uint32_t num_blocks = get_compile_time_arg_val(15);
    constexpr uint32_t rank_cold = cold_rank16(num_blocks * words_per_block);
    constexpr RangesLayout ranges_at = ranges_layout((num_blocks + range_blocks - 1) / range_blocks, num_devices);
    constexpr uint32_t range_stride = ranges_at.range_stride;
    auto* rankc = rank16 + rank_cold;
    auto* counts = prefixes;
    prefixes = counts + ranges_at.prefixes;
    auto* cold_counts = counts + ranges_at.cold;
    const uint32_t step = sem(kSemStep);
    const uint32_t done = sem_done(worker);

    wait_at_least(step, kStepCount);
    const uint32_t ranges = misc[kMiscRanges], blocks = misc[kMiscActiveBlocks];
    for (uint32_t r = core_idx; r < ranges; r += cores) {
        const uint32_t b1 = (r + 1) * range_blocks < blocks ? (r + 1) * range_blocks : blocks;
        const uint32_t n = (b1 - r * range_blocks) * words_per_block;
        count_hot<num_devices>(
            scratch, range_words, r * range_words, share_begin(n, worker), share_begin(n, worker + 1), threshold, hot);
    }
    post(done, kCounted);
    auto wait_all = [&](uint32_t v) {
        for (uint32_t w = 0; w < kWorkers; ++w) {
            wait_at_least(sem_done(w), v);
        }
    };
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
    post(done, kRanked);
    wait_at_least(step, kStepCold);
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
    post(done, kMasked);
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
    post(done, kColdRanked);
    wait_at_least(step, kStepRewrite);
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
        split_block);
    post(done, kStepRewrite);
}
