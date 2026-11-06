// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "combine_host.hpp"
#include "combine_program.hpp"
#include "ttnn/operations/experimental/moe_block/flat_routed_expert/device/flat_routed_expert_device_operation_types.hpp"
#include "ttnn/operations/experimental/moe_block/flat_routed_expert/device/flat_routed_expert_plan.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

// The combine's op-specific configuration (combine_snap comes from MoEBlockConfig).
struct CombineConfig {
    std::optional<uint32_t> n_acc;      // the accumulator count the fused head planned (pins combine_auto's A)
    std::optional<CombineMode> mode;    // ... and its mode
    uint32_t arena_bytes = 0;           // the combine's L1 budget a core; 0: the reference's sizing at the first call
    uint32_t depth = 8;                 // landing pool slots per source chip (average)
    // the landing pool: rings (one a source chip) or shared (slot grants), in the accumulators' L1 or on the flat
    // cells; nullopt: the planner's (rings, ranked by combine_choice_key)
    std::optional<CombineLanding> landing;
    // credit / grant snapshots: rows a snapshot covers before it goes back (1: at once); nullopt: the planner's (8 with
    // >= 16 sources, else at most half a source's slots, shared at once)
    std::optional<uint32_t> credit_batch;

    bool operator==(const CombineConfig&) const = default;
};

struct RoutedExpertsCombineParams {
    CombineConfig combine;
    uint32_t S = 0;  // home tokens a chip
    uint32_t W = 0;  // the routed width Hr
    flat_routed_expert::FlatRoutedExpertConfig flat_config;  // with the combine overlap (with_combine_overlap)
    std::vector<tt::tt_metal::CoreCoord> cells;              // the cells the flat plan leaves free
    FlatHandoff flat;                                        // its y writers
    // Structural key of the resources (config, dims, mesh, fabric): hashed instead of the config.
    uint64_t plan_key = 0;
    // The plans: functions of the fields above, the mesh and the resources (not hashed).
    std::shared_ptr<const CombinePlan> plan;
    std::shared_ptr<const flat_routed_expert::FlatRoutedExpertPlan> flat_plan;

    static constexpr auto attribute_names = std::forward_as_tuple("S", "W", "plan_key");
    auto attribute_values() const { return std::forward_as_tuple(S, W, plan_key); }
};

struct RoutedExpertsCombineInputs {
    Tensor plan;                // the device plan [1, 1024] uint32 (the fused head's planner)
    Tensor table;               // the hand-off table [rows_cap / 32, 64] uint32 (the fused head's planner)
    Tensor out;                 // y_own [1, 1, S, Hr] bf16 row-major (persistent, the resources')
    Tensor bar;                 // the start barrier / zero gate words (persistent, every worker core)
    Tensor hcache;              // the senders' routed header cache (persistent DRAM)
    std::optional<Tensor> lut;  // per-chip accumulator roles: the senders' accumulator LUT (persistent DRAM)
    // the flat routed expert's: x = the dispatched rows, arena = the call's arena (every worker core: the flat
    // expert's buffers on its cells, the combine's on its own, the y writers' words), output = its (unwritten) y
    flat_routed_expert::FlatRoutedExpertInputs flat;
};

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
