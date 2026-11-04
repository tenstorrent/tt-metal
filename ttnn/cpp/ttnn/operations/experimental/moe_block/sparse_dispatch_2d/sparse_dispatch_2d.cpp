// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_dispatch_2d.hpp"

#include <memory>
#include <set>

#include "device/sparse_dispatch_2d_device_operation.hpp"
#include "device/sparse_dispatch_2d_plan.hpp"

namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d {

namespace sparse_dispatch_2d_detail {

// What every call needs from the resources, computed once: the plan and port cores (shared with the cached program's
// attributes) and the port cores' semaphore words.
struct State {
    std::shared_ptr<const SparseDispatch2dPorts> ports;
    Tensor sems;
};

const State& state_of(MoEBlockResources& resources) {
    return resources.plan<State>("sparse_dispatch_2d/ports", [&] {
        auto ports = std::make_shared<const SparseDispatch2dPorts>(
            SparseDispatch2dPorts{.plan = resources.mesh_plan(), .cores = resources.ports()});
        // (every chip's port cells: the words sit at one address on all of them, so a port addresses its peer's)
        std::set<std::pair<size_t, size_t>> cells;
        std::vector<CoreRange> ranges;
        for (const auto& [key, core] : ports->cores) {
            if (cells.emplace(core.y, core.x).second) {
                ranges.emplace_back(core, core);
            }
        }
        const Tensor& sems =
            resources.persistent_words("sparse_dispatch_2d/ports/sems", CoreRangeSet(ranges), SEM_WORDS);
        return State{.ports = std::move(ports), .sems = sems};
    });
}

}  // namespace sparse_dispatch_2d_detail

Tensor sparse_dispatch_2d(
    const Tensor& xbuf,
    const Tensor& route,
    MoEBlockResources& resources,
    std::optional<uint32_t> group,
    uint32_t manifest_batch,
    std::optional<uint32_t> pack) {
    using namespace sparse_dispatch_2d_detail;
    const State& state = state_of(resources);
    const uint32_t width = xbuf.logical_shape()[-1];
    const uint32_t row_bytes = xbuf.buffer()->aligned_page_size();
    const SparseDispatch2dParams params{
        .plan_key = resources.plan_key(),
        .group = group.value_or(default_group(width)),
        .manifest_batch = manifest_batch,
        .pack = pack.value_or(default_pack(row_bytes, resources.fabric_max_payload_bytes())),
        .ports = state.ports,
    };
    return ttnn::prim::moe_block_sparse_dispatch_2d(xbuf, route, state.sems, params);
}

}  // namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d
