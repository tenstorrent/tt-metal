// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_prefill.hpp"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "device/matmul_reduce_scatter_prefill_device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn {

namespace operations::experimental::ccl::matmul_reduce_scatter_prefill {

namespace {

bool config_wraps(tt::tt_fabric::FabricConfig fabric_config, int32_t cluster_axis) {
    using tt::tt_fabric::FabricConfig;
    if (fabric_config == FabricConfig::FABRIC_2D_TORUS_XY || fabric_config == FabricConfig::FABRIC_1D_RING) {
        return true;
    }
    if (fabric_config == FabricConfig::FABRIC_2D_TORUS_X) {
        return cluster_axis == 1;
    }
    if (fabric_config == FabricConfig::FABRIC_2D_TORUS_Y) {
        return cluster_axis == 0;
    }
    return false;
}

using Coord = std::pair<uint32_t, uint32_t>;  // (row, col) of the mesh

// build_groups: the device lines along cluster_axis.
std::vector<std::vector<Coord>> build_groups(uint32_t rows, uint32_t cols, int32_t cluster_axis) {
    std::vector<std::vector<Coord>> groups;
    if (cluster_axis == 0) {
        for (uint32_t c = 0; c < cols; ++c) {
            auto& group = groups.emplace_back();
            for (uint32_t r = 0; r < rows; ++r) {
                group.emplace_back(r, c);
            }
        }
    } else {
        for (uint32_t r = 0; r < rows; ++r) {
            auto& group = groups.emplace_back();
            for (uint32_t c = 0; c < cols; ++c) {
                group.emplace_back(r, c);
            }
        }
    }
    return groups;
}

}  // namespace

}  // namespace operations::experimental::ccl::matmul_reduce_scatter_prefill

Tensor experimental::matmul_reduce_scatter_prefill(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    int32_t scatter_dim,
    ttnn::ccl::Topology topology,
    const std::optional<int32_t>& num_links,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output_tensor) {
    using namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill;
    namespace mrs_detail = ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill::detail;
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    // the supported inputs, scatter_dim canonical
    const auto rank = static_cast<int32_t>(input_tensor.logical_shape().rank());
    const int32_t canonical_dim = scatter_dim >= 0 ? scatter_dim - rank : scatter_dim;
    check_support(input_tensor, weight, cluster_axis, canonical_dim, topology, num_links);

    // operand checks, then the mesh
    check_operands(input_tensor, weight, memory_config);
    const MemoryConfig out_mem = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);

    auto* mesh = input_tensor.device();
    TT_FATAL(mesh != nullptr, "matmul_reduce_scatter_prefill: input_tensor must be on a mesh device");
    const auto& mesh_shape = mesh->shape();
    TT_FATAL(mesh_shape.dims() == 2, "matmul_reduce_scatter_prefill: the mesh must be 2D, got {}", mesh_shape);
    const uint32_t rows = mesh_shape[0];
    const uint32_t cols = mesh_shape[1];
    const uint32_t G = mesh_shape[cluster_axis];
    TT_FATAL(G >= 2, "matmul_reduce_scatter_prefill: cluster_axis {} has {} device(s); need >= 2", cluster_axis, G);
    const auto& a_shape = input_tensor.logical_shape();
    const uint32_t M = a_shape[-2];
    const uint32_t K = a_shape[-1];
    const uint32_t N = weight.logical_shape()[-1];
    const uint32_t extent = canonical_dim == -2 ? M : N;
    const uint32_t extent_tile = canonical_dim == -2 ? TILE_HEIGHT : TILE_WIDTH;
    TT_FATAL(
        extent % (extent_tile * G) == 0,
        "matmul_reduce_scatter_prefill: scattered extent {} does not split into {} tile blocks",
        extent,
        G);

    const bool ring = topology == ttnn::ccl::Topology::Ring && G >= 3;  // a 2-device ring is the line
    const mrs_detail::ScheduleKind kind =
        !ring ? mrs_detail::ScheduleKind::Line
              : (G % 2 == 0 ? mrs_detail::ScheduleKind::RingBalanced : mrs_detail::ScheduleKind::RingUnbalanced);
    TT_FATAL(
        !ring || config_wraps(tt::tt_fabric::GetFabricConfig(), cluster_axis),
        "matmul_reduce_scatter_prefill: Ring along axis {} needs a fabric config that wraps it (active: {})",
        cluster_axis,
        static_cast<uint32_t>(tt::tt_fabric::GetFabricConfig()));
    auto node = [&](const Coord& c) { return mesh->get_fabric_node_id(ttnn::MeshCoordinate(c.first, c.second)); };
    size_t avail = std::numeric_limits<size_t>::max();
    for (const auto& group : build_groups(rows, cols, cluster_axis)) {
        std::vector<std::pair<Coord, Coord>> pairs;
        for (size_t i = 0; i + 1 < group.size(); ++i) {
            pairs.emplace_back(group[i], group[i + 1]);
        }
        if (ring) {
            pairs.emplace_back(group.back(), group.front());
        }
        for (const auto& [s, d] : pairs) {
            avail = std::min(avail, tt::tt_fabric::get_forwarding_link_indices(*mesh, node(s), node(d)).size());
            avail = std::min(avail, tt::tt_fabric::get_forwarding_link_indices(*mesh, node(d), node(s)).size());
        }
    }
    TT_FATAL(avail > 0, "matmul_reduce_scatter_prefill: a hop of the group has no usable fabric link (no wrap link?)");
    uint32_t L = static_cast<uint32_t>(avail);
    if (num_links.has_value()) {
        TT_FATAL(
            *num_links > 0 && static_cast<size_t>(*num_links) <= avail,
            "matmul_reduce_scatter_prefill: num_links={}, but a hop has only {} link(s)",
            *num_links,
            avail);
        L = static_cast<uint32_t>(*num_links);
    }

    // geometry and the plan (STAGING_POLICY exact: no staging floor)
    const uint32_t Mt = M / TILE_HEIGHT;
    const uint32_t Kt = K / TILE_WIDTH;
    const uint32_t Nt = N / TILE_WIDTH;
    const CoreCoord grid = mesh->compute_with_storage_grid_size();
    const mrs_detail::PlanInputs plan_inputs{
        .G = G,
        .kind = kind,
        .L = L,
        .Mt = Mt,
        .Kt = Kt,
        .Nt = Nt,
        .block_m = canonical_dim == -2 ? Mt / G : Mt,
        .block_n = canonical_dim == -2 ? Nt : Nt / G,
        .gx = static_cast<uint32_t>(grid.x),
        .gy = static_cast<uint32_t>(grid.y),
        .tA = static_cast<uint32_t>(input_tensor.buffer()->aligned_page_size()),
        .tW = static_cast<uint32_t>(weight.buffer()->aligned_page_size()),
        .scatter_dim = canonical_dim,
        .budget = mesh->allocator()->get_bank_size(tt::tt_metal::BufferType::L1) - mrs_detail::kL1Headroom,
    };

    MatmulReduceScatterPrefillParams params{
        .cluster_axis = static_cast<uint32_t>(cluster_axis),
        .scatter_dim = canonical_dim,
        .topology = topology,
        .num_links = L,
        .kind = kind,
        .math_fidelity = compute_kernel_config.has_value() ? compute_kernel_config->math_fidelity
                                                           : tt::tt_metal::MathFidelity::HiFi2,
        .math_approx_mode = compute_kernel_config.has_value() && compute_kernel_config->math_approx_mode,
        .output_memory_config = out_mem,
        .plan = mrs_detail::solve_plan(plan_inputs),
    };
    record_last_plan(params);
    ensure_mesh_resources(mesh, mrs_detail::staging_tiles(params.plan));

    return ttnn::prim::matmul_reduce_scatter_prefill(input_tensor, weight, output_tensor, params);
}

}  // namespace ttnn
