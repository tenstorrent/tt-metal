// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt_stl/indestructible.hpp>

#include "device/matmul_reduce_scatter_device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn {

namespace operations::matmul_reduce_scatter {

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

// The fewest forwarding links over every hop of the groups along cluster_axis, both directions, ring wrap included.
size_t count_usable_links(tt::tt_metal::distributed::MeshDevice* mesh, int32_t cluster_axis, bool ring) {
    const auto& mesh_shape = mesh->shape();
    auto node = [&](const Coord& c) { return mesh->get_fabric_node_id(ttnn::MeshCoordinate(c.first, c.second)); };
    size_t avail = std::numeric_limits<size_t>::max();
    for (const auto& group : build_groups(mesh_shape[0], mesh_shape[1], cluster_axis)) {
        std::vector<std::pair<Coord, Coord>> pairs;
        for (size_t i = 0; i + 1 < group.size(); ++i) {
            pairs.emplace_back(group[i], group[i + 1]);
        }
        if (ring) {
            pairs.emplace_back(group.back(), group.front());
        }
        for (const auto& [s, d] : pairs) {
            avail = std::min(avail, tt::tt_fabric::get_forwarding_link_indices(node(s), node(d)).size());
            avail = std::min(avail, tt::tt_fabric::get_forwarding_link_indices(node(d), node(s)).size());
        }
    }
    return avail;
}

// count_usable_links, counted on a mesh's first call per (axis, ring, fabric config) and read on every later one. The
// count walks a control-plane route per link of every hop (64 lookups on an 8x4 ring), the bulk of a hot call's host
// time; the routing tables it reads are fixed while the mesh is open (SetFabricConfig is refused with devices open)
// and mesh ids are never reused, so the count cannot change under its key.
size_t usable_links(tt::tt_metal::distributed::MeshDevice* mesh, int32_t cluster_axis, bool ring) {
    using Key = std::tuple<int, int32_t, bool, tt::tt_fabric::FabricConfig>;
    static ttsl::Indestructible<std::map<Key, size_t>> counts;
    static std::mutex counts_mutex;
    const Key key{mesh->id(), cluster_axis, ring, tt::tt_fabric::GetFabricConfig()};
    const std::lock_guard<std::mutex> lock(counts_mutex);
    auto it = counts.get().find(key);
    if (it == counts.get().end()) {
        it = counts.get().emplace(key, count_usable_links(mesh, cluster_axis, ring)).first;
    }
    return it->second;
}

}  // namespace

}  // namespace operations::matmul_reduce_scatter

Tensor matmul_reduce_scatter(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    int32_t scatter_dim,
    ttnn::ccl::Topology topology,
    const std::optional<int32_t>& num_links,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output_tensor) {
    using namespace operations::matmul_reduce_scatter;
    namespace mrs_detail = operations::matmul_reduce_scatter::detail;
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    // validate(): the support contract, scatter_dim canonical
    const auto rank = static_cast<int32_t>(input_tensor.logical_shape().rank());
    const int32_t canonical_dim = scatter_dim >= 0 ? scatter_dim - rank : scatter_dim;
    check_support(input_tensor, weight, cluster_axis, canonical_dim, topology, num_links);

    // run(): operand checks, then the mesh
    check_operands(input_tensor, weight, memory_config);
    const MemoryConfig out_mem = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);

    auto* mesh = input_tensor.device();
    TT_FATAL(mesh != nullptr, "matmul_reduce_scatter: input_tensor must be on a mesh device");
    const auto& mesh_shape = mesh->shape();
    TT_FATAL(mesh_shape.dims() == 2, "matmul_reduce_scatter: the mesh must be 2D, got {}", mesh_shape);
    const uint32_t G = mesh_shape[cluster_axis];
    if (G < 2) {
        throw std::invalid_argument(
            fmt::format("matmul_reduce_scatter: cluster_axis {} has {} device(s); need >= 2", cluster_axis, G));
    }
    const auto& a_shape = input_tensor.logical_shape();
    const uint32_t M = a_shape[-2];
    const uint32_t K = a_shape[-1];
    const uint32_t N = weight.logical_shape()[-1];
    const uint32_t extent = canonical_dim == -2 ? M : N;
    const uint32_t extent_tile = canonical_dim == -2 ? TILE_HEIGHT : TILE_WIDTH;
    if (extent % (extent_tile * G) != 0) {
        throw std::invalid_argument(
            fmt::format("matmul_reduce_scatter: scattered extent {} does not split into {} tile blocks", extent, G));
    }

    const bool ring = topology == ttnn::ccl::Topology::Ring && G >= 3;  // a 2-device ring is the line
    const mrs_detail::ScheduleKind kind =
        !ring ? mrs_detail::ScheduleKind::Line
              : (G % 2 == 0 ? mrs_detail::ScheduleKind::RingBalanced : mrs_detail::ScheduleKind::RingUnbalanced);
    if (ring && !config_wraps(tt::tt_fabric::GetFabricConfig(), cluster_axis)) {
        throw std::invalid_argument(fmt::format(
            "matmul_reduce_scatter: Ring along axis {} needs a fabric config that wraps it (active: {})",
            cluster_axis,
            static_cast<uint32_t>(tt::tt_fabric::GetFabricConfig())));
    }
    const size_t avail = usable_links(mesh, cluster_axis, ring);
    if (avail == 0) {
        throw std::invalid_argument(
            "matmul_reduce_scatter: a hop of the group has no usable fabric link (no wrap link?)");
    }
    uint32_t L = static_cast<uint32_t>(avail);
    if (num_links.has_value()) {
        if (*num_links <= 0 || static_cast<size_t>(*num_links) > avail) {
            throw std::invalid_argument(
                fmt::format("matmul_reduce_scatter: num_links={}, but a hop has only {} link(s)", *num_links, avail));
        }
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

    MatmulReduceScatterParams params{
        .cluster_axis = static_cast<uint32_t>(cluster_axis),
        .scatter_dim = canonical_dim,
        .topology = topology,
        .num_links = L,
        .kind = kind,
        .math_fidelity = compute_kernel_config.has_value() ? compute_kernel_config->math_fidelity : MathFidelity::HiFi2,
        .math_approx_mode = compute_kernel_config.has_value() && compute_kernel_config->math_approx_mode,
        .output_memory_config = out_mem,
        .plan = mrs_detail::solve_plan(plan_inputs),
    };
    record_last_plan(params);
    ensure_mesh_resources(mesh, mrs_detail::staging_tiles(params.plan));

    return ttnn::prim::matmul_reduce_scatter(input_tensor, weight, output_tensor, params);
}

}  // namespace ttnn
