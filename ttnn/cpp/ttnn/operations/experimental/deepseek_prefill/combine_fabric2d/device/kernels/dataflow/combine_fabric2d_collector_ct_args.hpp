// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The collector kernel's compile-time arguments. One field list: the host constructor takes what the program
// factory already has and derives every field from it, the kernel constructor reads the same fields back in
// the same order. The collector has no runtime arguments: it touches no buffer, only L1 the program places.
//
// The waiting cores follow the scalars, (noc_x, noc_y) each.

#include "combine_fabric2d_kernel_interface.hpp"

namespace cmbf2d {

// Scalars packed before the waiting cores, i.e. the index those start at. Asserted against the field list
// below, so it cannot drift out of step with it.
constexpr uint32_t COLLECTOR_SCALAR_CT_ARGS = 11;

struct CollectorCtArgs {
    uint32_t num_writers;  // routed-expert writer cores; each reports once per step
    uint32_t num_steps;    // expert slots walked per pass, times passes
    uint32_t counts_addr;  // the per-step count array the writers bump
    uint32_t ready_sem;
    uint32_t go_addr;
    // The routed expert's cores as one NOC rectangle, which is what lets `go` be one multicast.
    uint32_t re_x0;
    uint32_t re_y0;
    uint32_t re_x1;
    uint32_t re_y1;
    uint32_t re_cores;
    uint32_t num_targets = 0;  // counted from the host target block; read from CT args on device

#ifndef KERNEL_BUILD
    CollectorCtArgs(const op::CombineFabric2dParams& args, const op::CollectorPlan& plan) :
        num_writers(args.routed_expert_writers),
        // The fused pass runs only under a threshold, and then the writers walk every slot twice.
        num_steps((args.hybrid_token_threshold > 0 ? 2u : 1u) * args.experts_per_chip),
        counts_addr(plan.counts_addr),
        ready_sem(plan.ready_sem),
        go_addr(args.routed_expert_go_addr),
        re_cores(static_cast<uint32_t>(args.routed_expert_cores.size())) {
        const auto first = args.device->worker_core_from_logical_core(args.routed_expert_cores.start_coord);
        const auto last = args.device->worker_core_from_logical_core(args.routed_expert_cores.end_coord);
        re_x0 = static_cast<uint32_t>(first.x);
        re_y0 = static_cast<uint32_t>(first.y);
        re_x1 = static_cast<uint32_t>(last.x);
        re_y1 = static_cast<uint32_t>(last.y);
        for (const auto& core : plan.waiting) {
            targets_.push_back(static_cast<uint32_t>(core.x));
            targets_.push_back(static_cast<uint32_t>(core.y));
            num_targets++;
        }
    }

    std::vector<uint32_t> to_ct_word_arr() const {
        std::vector<uint32_t> word_arr{
            num_writers, num_steps, counts_addr, ready_sem, go_addr, re_x0, re_y0, re_x1, re_y1, re_cores, num_targets};
        word_arr.insert(word_arr.end(), targets_.begin(), targets_.end());
        return word_arr;
    }
#else
    constexpr CollectorCtArgs() :
        num_writers(get_compile_time_arg_val(0)),
        num_steps(get_compile_time_arg_val(1)),
        counts_addr(get_compile_time_arg_val(2)),
        ready_sem(get_compile_time_arg_val(3)),
        go_addr(get_compile_time_arg_val(4)),
        re_x0(get_compile_time_arg_val(5)),
        re_y0(get_compile_time_arg_val(6)),
        re_x1(get_compile_time_arg_val(7)),
        re_y1(get_compile_time_arg_val(8)),
        re_cores(get_compile_time_arg_val(9)),
        num_targets(get_compile_time_arg_val(10)) {}

    static constexpr uint32_t target_base = COLLECTOR_SCALAR_CT_ARGS;
#endif

#ifndef KERNEL_BUILD
private:
    std::vector<uint32_t> targets_;  // the waiting cores, appended after the scalars
#endif
};

#ifdef KERNEL_BUILD
// The scalars are read back by index, so the base the waiting cores start at IS the field count.
static_assert(
    sizeof(CollectorCtArgs) == COLLECTOR_SCALAR_CT_ARGS * sizeof(uint32_t),
    "COLLECTOR_SCALAR_CT_ARGS no longer matches the field list; the waiting cores would be misread");
#endif

}  // namespace cmbf2d
