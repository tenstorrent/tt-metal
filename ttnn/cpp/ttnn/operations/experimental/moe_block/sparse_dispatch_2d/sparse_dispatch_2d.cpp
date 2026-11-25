// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_dispatch_2d.hpp"
#include <tracy/Tracy.hpp>

#include <memory>

#include "device/sparse_dispatch_2d_device_operation.hpp"
#include "device/sparse_dispatch_2d_plan.hpp"
#include "ttnn/operations/experimental/moe_block/common/l1_budget.hpp"

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
        const Tensor& sems = resources.persistent_words(
            "sparse_dispatch_2d/ports/sems", CoreRangeSet(port_cell_ranges(ports->cores)), SEM_WORDS);
        return State{.ports = std::move(ports), .sems = sems};
    });
}

}  // namespace sparse_dispatch_2d_detail

uint32_t dispatch_group(const MoEBlockResources& resources) {
    const auto l1 = block_l1_plan(resources);
    return l1 ? l1->dispatch_group : kGroups.back();
}

Tensor sparse_dispatch_2d(
    const Tensor& xbuf,
    const Tensor& route,
    MoEBlockResources& resources,
    const std::optional<Tensor>& arena) {
    ZoneScopedN("sparse_dispatch_2d: entry");
    using namespace sparse_dispatch_2d_detail;
    const State& state = state_of(resources);
    const uint32_t row_bytes = xbuf.buffer()->aligned_page_size();
    const SparseDispatch2dParams params{
        .plan_key = resources.plan_key(),
        .group = dispatch_group(resources),
        .manifest_batch = kManifestBatch,
        .pack = default_pack(row_bytes, resources.fabric_max_payload_bytes()),
        .ports = state.ports,
    };
    return ttnn::prim::moe_block_sparse_dispatch_2d(xbuf, route, state.sems, params, arena);
}

}  // namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d
