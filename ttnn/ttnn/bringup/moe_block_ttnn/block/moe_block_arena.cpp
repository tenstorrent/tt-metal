// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_block_arena.hpp"

#include <algorithm>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "../common/moe_block_config.hpp"
#include "../common/moe_block_resources.hpp"
#include "../flat_routed_expert/device/flat_routed_expert_plan.hpp"
#include "../moe_head/device/head_sizing.hpp"
#include "../moe_head/moe_head.hpp"
#include "../routed_experts_combine/routed_experts_combine.hpp"
#include "../sparse_dispatch_2d/device/sparse_dispatch_2d_plan.hpp"

namespace ttnn::operations::bringup::moe_block {

namespace block_plan_detail {

// What a group must keep as the smallest group has it. The accumulators' local ring (up to 16 slots) follows the
// arena's bytes: it counts up to the planner's 8-slot keep (GLM-5.3-Flash on the 1x4 ring, group 4 with a 4-slot local
// ring instead of 2 with 8: experts + combine 761.6 -> 924.2 us), not beyond (Kimi-K3 there, group 4 with 9 slots
// instead of 2 with 13: 2598.9 -> 2632.1 us hot, the dispatch 119.1 -> 88.7 us).
struct GroupKey {
    // the flat plan's
    uint32_t row_tiles_per_sub_block = 0, down_h_buffers = 0;
    // the combine's (the local ring counted up to 8 slots)
    uint32_t accumulator_count = 0, landing_slots_per_source = 0, injection_ring_slots = 0, local_ring_slots = 0;
    bool accumulators_on_ports = false;
    bool landing_on_flat_expert_cores = false;
    std::optional<moe_head::HeadLayout> head;  // the head's pick
    bool operator==(const GroupKey&) const = default;
};

}  // namespace block_plan_detail

const MoEBlockPlan& moe_block_plan(
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    std::optional<uint32_t> accumulators,
    uint32_t max_tokens_per_expert,
    const flat_routed_expert::FlatLayoutOptions& layout,
    const routed_experts_combine::CombineConfig& combine,
    std::optional<int64_t> l1_free) {
    using namespace block_plan_detail;
    const auto& plan = resources.plan<MoEBlockPlan>("moe_block/plan", [&] {
        TT_FATAL(
            !accumulators || !combine.accumulator_count || *combine.accumulator_count == *accumulators,
            "moe_block: the combine's accumulators ({}) must be the head's ({})",
            *combine.accumulator_count,
            *accumulators);
        const int64_t l1 = moe_head::moe_head_l1_free(resources, l1_free);
        auto& mesh = resources.mesh_device();
        MoEBlockPlan out;
        std::optional<GroupKey> base;
        for (const uint32_t g : sparse_dispatch_2d::kGroups) {
            auto comb = combine;
            if (accumulators) {
                comb.accumulator_count = *accumulators;
            }
            // (the combine first without the head's ports below its arena: its accumulators plan the head)
            auto c = routed_experts_combine::plan_routed_experts_combine_l1(
                resources, max_tokens_per_expert, layout, comb, g, 0);
            const uint32_t A = c.plan->accumulator_count;
            const auto& fit = moe_head::plan_moe_head(resources, dtypes, A, l1, c.static_bytes);
            if (!fit.ok || fit.planned.empty()) {
                TT_FATAL(base.has_value(), "moe_block: the MoE head does not fit: {}", fit.reason);
                out.note += fmt::format("; group {}: the head does not fit", g);
                break;
            }
            const auto& lay = fit.planned[0].layout;
            // (the head's x-gather CB is an arena region, not a static CB: nothing of the head's under the combine's)
            const uint32_t port = 0;
            comb.accumulator_count = A;  // (pinned: the head is planned for A)
            c = routed_experts_combine::plan_routed_experts_combine_l1(
                resources, max_tokens_per_expert, layout, comb, g, port);
            const auto fp = flat_routed_expert::flat_routed_expert_plan(mesh, c.flat);
            const auto& cp = *c.plan;
            const GroupKey key{
                fp->row_tiles_per_sub_block,
                fp->down_h_buffers,
                cp.accumulator_count,
                cp.landing_slots_per_source,
                cp.injection_ring_slots,
                std::min(cp.local_ring_slots, 8u),
                cp.accumulators_on_ports,
                cp.landing_on_flat_expert_cores,
                lay};
            out.note += fmt::format(
                "{}group {}: static CBs {} KB, combine arena {} KB, flat {} rows x {} h buffers, A {} D {}{}, head {} "
                "(arena {} KB)",
                base ? "; " : "",
                g,
                c.static_bytes >> 10,
                c.arena_bytes >> 10,
                fp->row_tiles_per_sub_block * 32,
                fp->down_h_buffers,
                cp.accumulator_count,
                cp.landing_slots_per_source,
                cp.landing_on_flat_expert_cores ? " (rings on the flat expert cores)" : "",
                moe_head::describe_layout(lay),
                fit.planned[0].arena >> 10);
            if (base && !(key == *base)) {
                break;
            }
            base = base.value_or(key);
            out.accumulators = A;
            out.l1 = BlockL1Plan{.dispatch_group = g, .head_cb_floor = c.static_bytes, .head_port_cb = port};
        }
        log_info(
            tt::LogOp,
            "moe_block: dispatch group {}, accumulators {} ({})",
            out.l1.dispatch_group,
            out.accumulators,
            out.note);
        return out;
    });
    resources.plan<BlockL1Plan>(kBlockL1Plan, [&] { return plan.l1; });
    return plan;
}

tt::tt_metal::TensorSpec moe_block_arena_spec(
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    uint32_t accumulators,
    uint32_t max_tokens_per_expert,
    const flat_routed_expert::FlatLayoutOptions& layout,
    const routed_experts_combine::CombineConfig& combine,
    std::optional<int64_t> l1_free) {
    TT_FATAL(
        !combine.accumulator_count.has_value() || *combine.accumulator_count == accumulators,
        "moe_block: the combine's accumulators ({}) must be the head's ({})",
        *combine.accumulator_count,
        accumulators);
    const auto& bp = moe_block_plan(resources, dtypes, accumulators, max_tokens_per_expert, layout, combine, l1_free);
    TT_FATAL(
        bp.accumulators == accumulators,
        "moe_block: the block was planned for {} accumulators, the arena asked for {}",
        bp.accumulators,
        accumulators);
    const auto& head = moe_head::setup_moe_head(resources, dtypes, accumulators, l1_free);
    const uint32_t head_bytes = moe_head::head_tensor_shapes(*head.model).arena_bytes;
    const uint32_t rows_cap = head.inputs.table.logical_shape()[-2] * 32;
    auto spec = routed_experts_combine::routed_experts_combine_arena_spec(
        resources, rows_cap, max_tokens_per_expert, layout, combine, head_bytes);
    // (moe_head's own checks on the arena: L1, its cores, its bytes a core)
    const auto& shard = *spec.memory_config().shard_spec();
    TT_FATAL(
        shard.grid.contains(moe_head::head_arena_cores(*head.model)),
        "moe_block: the combine's arena cores do not cover the MoE head's");
    TT_FATAL(
        uint64_t(shard.shape[0]) * shard.shape[1] * 2 >= head_bytes,
        "moe_block: the arena holds {} B a core, the MoE head needs {} B",
        uint64_t(shard.shape[0]) * shard.shape[1] * 2,
        head_bytes);
    return spec;
}

}  // namespace ttnn::operations::bringup::moe_block
