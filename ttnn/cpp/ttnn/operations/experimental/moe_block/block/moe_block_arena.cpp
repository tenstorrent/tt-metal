// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_block_arena.hpp"

#include <algorithm>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/experimental/moe_block/common/moe_block_config.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"
#include "ttnn/operations/experimental/moe_block/flat_routed_expert/device/flat_routed_expert_plan.hpp"
#include "ttnn/operations/experimental/moe_block/fused_head/device/head_sizing.hpp"
#include "ttnn/operations/experimental/moe_block/fused_head/fused_head.hpp"
#include "ttnn/operations/experimental/moe_block/routed_experts_combine/routed_experts_combine.hpp"
#include "ttnn/operations/experimental/moe_block/sparse_dispatch_2d/device/sparse_dispatch_2d_plan.hpp"

namespace ttnn::operations::experimental::moe_block {

namespace block_plan_detail {

// What a group must keep as the smallest group has it. The accumulators' local ring (QL slots, up to 16) follows the
// arena's bytes: it counts up to the planner's 8-slot keep (GLM-5.3-Flash on the 1x4 ring, group 4 with QL 4 instead of
// 2 with 8: experts + combine 761.6 -> 924.2 us), not beyond (Kimi-K3 there, group 4 with QL 9 instead of 2 with 13:
// 2598.9 -> 2632.1 us hot, the dispatch 119.1 -> 88.7 us).
struct GroupKey {
    uint32_t mt = 0, hbuf = 0;              // the flat plan
    uint32_t A = 0, D = 0, QP = 0, QL = 0;  // the combine's (QL at most 8)
    bool fused = false;
    std::optional<fused_head::HeadLayout> head;  // the head's pick
    bool operator==(const GroupKey&) const = default;
};

}  // namespace block_plan_detail

const MoEBlockPlan& moe_block_plan(
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    std::optional<uint32_t> accumulators,
    uint32_t max_tokens_per_expert,
    uint32_t activation,
    const flat_routed_expert::FlatLayoutOptions& layout,
    const routed_experts_combine::CombineConfig& combine,
    std::optional<int64_t> l1_free) {
    using namespace block_plan_detail;
    const auto& plan = resources.plan<MoEBlockPlan>("moe_block/plan", [&] {
        TT_FATAL(
            !accumulators || !combine.n_acc || *combine.n_acc == *accumulators,
            "moe_block: the combine's accumulators ({}) must be the head's ({})",
            *combine.n_acc,
            *accumulators);
        const int64_t l1 = fused_head::fused_head_l1_free(resources, l1_free);
        auto& mesh = resources.mesh_device();
        MoEBlockPlan out;
        std::optional<GroupKey> base;
        for (const uint32_t g : sparse_dispatch_2d::kGroups) {
            auto comb = combine;
            if (accumulators) {
                comb.n_acc = *accumulators;
            }
            // (the combine first without the head's ports below its arena: its accumulators plan the head)
            auto c = routed_experts_combine::plan_routed_experts_combine_l1(
                resources, max_tokens_per_expert, activation, layout, comb, g, 0);
            const uint32_t A = c.plan->A;
            const auto& fit = fused_head::plan_fused_head(resources, dtypes, A, l1, c.static_bytes);
            if (!fit.ok || fit.planned.empty()) {
                TT_FATAL(base.has_value(), "moe_block: the fused head does not fit: {}", fit.why);
                out.note += fmt::format("; group {}: the head does not fit", g);
                break;
            }
            const auto& lay = fit.planned[0].lay;
            // (the head's x-gather CB is an arena region, not a static CB: nothing of the head's under the combine's)
            const uint32_t port = 0;
            comb.n_acc = A;  // (pinned: the head is planned for A)
            c = routed_experts_combine::plan_routed_experts_combine_l1(
                resources, max_tokens_per_expert, activation, layout, comb, g, port);
            const auto fp = flat_routed_expert::flat_routed_expert_plan(mesh, c.flat);
            const auto& cp = *c.plan;
            const GroupKey key{fp->mt, fp->hbuf, cp.A, cp.D, cp.QP, std::min(cp.QL, 8u), cp.fused, lay};
            out.note += fmt::format(
                "{}group {}: static CBs {} KB, combine arena {} KB, flat {} rows x {} h buffers, A {} D {}, head {} "
                "(arena {} KB)",
                base ? "; " : "",
                g,
                c.static_bytes >> 10,
                c.arena_bytes >> 10,
                fp->mt * 32,
                fp->hbuf,
                cp.A,
                cp.D,
                fused_head::describe_layout(lay),
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
    uint32_t activation,
    const flat_routed_expert::FlatLayoutOptions& layout,
    const routed_experts_combine::CombineConfig& combine,
    std::optional<int64_t> l1_free) {
    TT_FATAL(
        !combine.n_acc.has_value() || *combine.n_acc == accumulators,
        "moe_block: the combine's accumulators ({}) must be the head's ({})",
        *combine.n_acc,
        accumulators);
    const auto& bp =
        moe_block_plan(resources, dtypes, accumulators, max_tokens_per_expert, activation, layout, combine, l1_free);
    TT_FATAL(
        bp.accumulators == accumulators,
        "moe_block: the block was planned for {} accumulators, the arena asked for {}",
        bp.accumulators,
        accumulators);
    const auto& head = fused_head::setup_fused_head(resources, dtypes, accumulators, l1_free);
    const uint32_t head_bytes = fused_head::head_tensor_shapes(*head.model).arena_bytes;
    const uint32_t rows_cap = head.inputs.table.logical_shape()[-2] * 32;
    auto spec = routed_experts_combine::routed_experts_combine_arena_spec(
        resources, rows_cap, max_tokens_per_expert, activation, layout, combine, head_bytes);
    // (fused_head's own checks on the arena: L1, its cores, its bytes a core)
    const auto& shard = *spec.memory_config().shard_spec();
    TT_FATAL(
        shard.grid.contains(fused_head::head_arena_cores(*head.model)),
        "moe_block: the combine's arena cores do not cover the fused head's");
    TT_FATAL(
        uint64_t(shard.shape[0]) * shard.shape[1] * 2 >= head_bytes,
        "moe_block: the arena holds {} B a core, the fused head needs {} B",
        uint64_t(shard.shape[0]) * shard.shape[1] * 2,
        head_bytes);
    return spec;
}

}  // namespace ttnn::operations::experimental::moe_block
