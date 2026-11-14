// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "expert_slot_copy.hpp"

#include <memory>
#include <set>

#include "device/expert_slot_copy_device_operation.hpp"

namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy {

namespace expert_slot_copy_detail {

struct State {
    std::shared_ptr<const SlotCopyPorts> ports;
    Tensor sems;
};

const State& state_of(MoEBlockResources& resources) {
    return resources.plan<State>("expert_slot_copy/ports", [&] {
        auto ports = std::make_shared<const SlotCopyPorts>(
            SlotCopyPorts{.plan = resources.mesh_plan(), .cores = resources.ports()});
        std::set<std::pair<size_t, size_t>> cells;
        std::vector<CoreRange> ranges;
        for (const auto& [key, core] : ports->cores) {
            if (cells.emplace(core.y, core.x).second) {
                ranges.emplace_back(core, core);
            }
        }
        const uint32_t words = (1 + ports->plan.topo.num_chips()) * kSemStride / 4;
        const Tensor& sems = resources.persistent_words("expert_slot_copy/sems", CoreRangeSet(ranges), words);
        return State{.ports = std::move(ports), .sems = sems};
    });
}

}  // namespace expert_slot_copy_detail

uint32_t expert_slot_copy_chunk_bytes(const MoEBlockResources& resources) {
    const auto payload = static_cast<uint32_t>(resources.fabric_max_payload_bytes());
    return std::min<uint32_t>(payload, 16384u) / 64u * 64u;
}

Tensor expert_slot_copy(
    const Tensor& src_gu,
    const Tensor& src_d,
    const Tensor& dst_gu,
    const Tensor& dst_d,
    const Tensor& runs,
    MoEBlockResources& resources,
    const std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>& jobs,
    std::optional<uint32_t> links,
    uint32_t depth,
    uint32_t batch) {
    using namespace expert_slot_copy_detail;
    const State& state = state_of(resources);
    const uint32_t N = state.ports->plan.topo.num_chips();
    SlotCopyParams params{
        .plan_key = resources.plan_key(),
        .chunk_bytes = expert_slot_copy_chunk_bytes(resources),
        .depth = depth,
        .batch = batch,
        .links = std::min<uint32_t>(links.value_or(state.ports->plan.num_links), kMaxLinks),
        .ports = state.ports,
    };
    const uint32_t tables = static_cast<uint32_t>(runs.logical_volume() / runs.logical_shape()[-1]);
    for (const auto& [s, d, tab] : jobs) {
        TT_FATAL(s < N && d < N, "expert_slot_copy: job ({}, {}) outside the {} chips", s, d, N);
        TT_FATAL(tab < tables, "expert_slot_copy: run table {} of {}", tab, tables);
        params.jobs.push_back(SlotCopyJob{.src = s, .dst = d, .table = tab});
    }
    return ttnn::prim::moe_block_main_expert_slot_copy(
        SlotCopyInputs{
            .src_gu = src_gu, .src_d = src_d, .dst_gu = dst_gu, .dst_d = dst_d, .runs = runs, .sems = state.sems},
        params);
}

}  // namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy
