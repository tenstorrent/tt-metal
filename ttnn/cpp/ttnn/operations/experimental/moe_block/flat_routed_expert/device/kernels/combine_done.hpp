// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Running with the combine (routed_experts_combine): every y writer (down core) reports its progress to the combine's
// collector as its rows land, so the combine takes rows while the flat expert still computes the next ones. Step s is
// local expert s (local expert order); the collector counts per step (cumulative counts) and releases step s once
// every writer has reported it.
//
// A writer reports the longest finished prefix of steps: it counts the blocks it still owes per local expert; the
// caller guarantees the rows have landed (`barrier`: or this waits for its writes).
//
// This core's args (moe_block_args: its per-core block) from flat_options::combine_report_arg: collector noc x,
// collector noc y, the collector's count array address, (unused). Per core, not common args: the collector and its
// counts' address may differ by chip, and one program for the mesh needs identical common args (the count address is
// a fixup: the arena's address added on the device).
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "expert_schedule.hpp"
#include "flat_options.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"

// The progress reports of a writer running with the combine; CombineProgress<false> (not running with it) does
// nothing.
template <bool reports = flat_options::reports_to_combine>
struct CombineProgress {
    uint16_t blocks_left[flat_options::max_experts];  // blocks this core still writes, per local expert (= step)
    uint32_t steps = 0, next_step = 0;

    // `writes`: this core writes y blocks for the schedule's entries (false: it reports every step at once)
    void init(const ExpertSchedule& schedule, uint32_t local_experts, bool writes) {
        steps = local_experts;
        for (uint32_t expert = 0; expert < local_experts; ++expert) {
            blocks_left[expert] = 0;
        }
        for (uint32_t entry = 0; writes && entry < schedule.entries; ++entry) {
            blocks_left[schedule.expert[entry]] += schedule.sub_blocks[entry];
        }
    }
    // One block of local expert `expert` written.
    void wrote(uint32_t expert) { --blocks_left[expert]; }
    bool step_done(uint32_t step) const { return blocks_left[step] == 0; }
    // Report every finished step not reported yet (all earlier ones too). `barrier`: wait for this core's writes to
    // land first (false: the caller knows they have, the row-major y writer's flushed transaction ids).
    void report(bool barrier = true) {
        if (next_step >= steps || !step_done(next_step)) {
            return;
        }
        if (barrier) {
            noc_async_write_barrier();  // the rows have landed
        }
        constexpr uint32_t args = flat_options::combine_report_arg;
        const uint32_t collector_x = moe_block_args::arg(args);
        const uint32_t collector_y = moe_block_args::arg(args + 1);
        const uint32_t counts = moe_block_args::arg(args + 2);
        while (next_step < steps && step_done(next_step)) {
            noc_semaphore_inc(get_noc_addr(collector_x, collector_y, counts + 4 * next_step), 1);
            ++next_step;
        }
        noc_async_atomic_barrier();
    }
};

template <>
struct CombineProgress<false> {
    void init(const ExpertSchedule&, uint32_t, bool) {}
    void wrote(uint32_t) {}
    void report(bool = true) {}
};
