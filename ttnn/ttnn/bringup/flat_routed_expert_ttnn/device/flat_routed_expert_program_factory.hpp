// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "flat_routed_expert_plan.hpp"
#include "flat_routed_expert_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::bringup::flat_routed_expert {

// Buffers whose addresses the runtime args carry (re-applied on every program-cache hit).
enum class AddrSrc : uint8_t {
    X,
    Y,
    Counts,
    Regions,
    Ids,
    GateUp,
    Down,
    ReaderDown,
    Done,
    Arena,
    Words,
    TokenIndex,
    Count
};

struct AddrPatch {
    tt::tt_metal::KernelHandle kernel;
    tt::tt_metal::CoreCoord core;
    uint32_t index;
    AddrSrc src;
    uint32_t offset;  // byte offset added to the buffer address
};

struct FlatRoutedExpertSharedVariables {
    std::vector<AddrPatch> patches;
    std::vector<std::pair<tt::tt_metal::CBHandle, uint32_t>> arena_cbs;  // (CB, arena byte offset)
    // cfg.cmb_rt: the y writer kernels / cores whose combine report args (cmb_rt .. cmb_rt + 3) the overlap fills
    std::vector<std::pair<tt::tt_metal::KernelHandle, tt::tt_metal::CoreCoord>> cmb_writers;
};

// The defines that decide this program's dynamic schedule (se_dyn.hpp: sub-block rows, expert capacity, gate/up ring
// regions, pinning): a kernel of another op built with them rebuilds the same schedule from the counts
// (se_dyn_from_counts), e.g. combine overlapped with this expert walking experts in the order they finish.
std::vector<std::pair<std::string, std::string>> flat_schedule_defines(
    const FlatRoutedExpertConfig& cfg, const FlatRoutedExpertPlan& p);

// Steps of flat_combine_overlap's walk with row-major y: the schedule's entries, at most 2 E (padded with empty ones).
uint32_t flat_combine_overlap_walk_steps(uint32_t experts_per_chip);

struct FlatRoutedExpertProgramFactory {
    using shared_variables_t = FlatRoutedExpertSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const FlatRoutedExpertParams& operation_attributes,
        const FlatRoutedExpertInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const FlatRoutedExpertParams& operation_attributes,
        const FlatRoutedExpertInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::bringup::flat_routed_expert
