// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "device/matmul_reduce_scatter_device_operation.hpp"
#include "ttnn/reports.hpp"

namespace ttnn {

namespace {

namespace mrs = operations::matmul_reduce_scatter;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;

// Bytes per core kept free beyond the op's own footprint.
constexpr uint64_t kL1HeadroomBytes = 64 * 1024;

bool is_dram_interleaved(const Tensor& t) {
    const MemoryConfig& mc = t.memory_config();
    return mc.buffer_type() == BufferType::DRAM && mc.memory_layout() == TensorMemoryLayout::INTERLEAVED;
}

void check_dram_interleaved_tile(const Tensor& t, const char* what) {
    if (!is_dram_interleaved(t)) {
        throw std::invalid_argument(fmt::format("matmul_reduce_scatter: {} must be DRAM interleaved", what));
    }
    if (t.layout() != Layout::TILE) {
        throw std::invalid_argument(fmt::format("matmul_reduce_scatter: {} must be TILE layout", what));
    }
}

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

// Fewest usable fabric links over every hop (both directions) of every group along cluster_axis.
uint32_t usable_links(MeshDevice* mesh, int32_t cluster_axis, bool ring) {
    const uint32_t rows = mesh->shape()[0];
    const uint32_t cols = mesh->shape()[1];
    const uint32_t num_groups = cluster_axis == 0 ? cols : rows;
    const uint32_t group_size = cluster_axis == 0 ? rows : cols;
    auto coord_of = [&](uint32_t group, uint32_t p) {
        return cluster_axis == 0 ? MeshCoordinate(p, group) : MeshCoordinate(group, p);
    };
    auto links = [&](const MeshCoordinate& s, const MeshCoordinate& d) {
        return static_cast<uint32_t>(
            tt::tt_fabric::get_forwarding_link_indices(mesh->get_fabric_node_id(s), mesh->get_fabric_node_id(d))
                .size());
    };
    uint32_t avail = std::numeric_limits<uint32_t>::max();
    for (uint32_t g = 0; g < num_groups; ++g) {
        std::vector<std::pair<uint32_t, uint32_t>> pairs;
        for (uint32_t p = 0; p + 1 < group_size; ++p) {
            pairs.emplace_back(p, p + 1);
        }
        if (ring) {
            pairs.emplace_back(group_size - 1, 0);
        }
        for (const auto& [s, d] : pairs) {
            avail = std::min({avail, links(coord_of(g, s), coord_of(g, d)), links(coord_of(g, d), coord_of(g, s))});
        }
    }
    return avail;
}

}  // namespace

Tensor matmul_reduce_scatter(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    int32_t scatter_dim,
    tt::tt_fabric::Topology topology,
    const std::optional<int32_t>& num_links,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output_tensor) {
    mrs::check_support(input_tensor, weight, cluster_axis, scatter_dim, topology, num_links);
    const auto& a_shape = input_tensor.logical_shape();
    const auto& w_shape = weight.logical_shape();
    const auto rank = static_cast<int32_t>(a_shape.rank());
    const int32_t canonical_scatter_dim = scatter_dim >= 0 ? scatter_dim - rank : scatter_dim;

    bool leading_ones = true;
    for (int32_t i = 0; i + 2 < rank; ++i) {
        leading_ones = leading_ones && a_shape[i] == 1;
    }
    if (rank < 2 || rank > 4 || !leading_ones) {
        throw std::invalid_argument(
            fmt::format("matmul_reduce_scatter: A must be rank 2-4 with leading dims 1, got {}", a_shape));
    }
    if (w_shape.rank() != 2) {
        throw std::invalid_argument(fmt::format("matmul_reduce_scatter: W must be (K, N), got {}", w_shape));
    }
    const uint32_t M = a_shape[-2];
    const uint32_t K = a_shape[-1];
    const uint32_t N = w_shape[-1];
    if (K != w_shape[0]) {
        throw std::invalid_argument(fmt::format("matmul_reduce_scatter: K mismatch, A {} vs W {}", a_shape, w_shape));
    }
    check_dram_interleaved_tile(input_tensor, "input_tensor");
    check_dram_interleaved_tile(weight, "weight");
    if (memory_config.has_value() && memory_config->memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        throw mrs::NotImplementedRefusal(
            "matmul_reduce_scatter: sharded output memory_config (no placement axis in TARGET)");
    }

    MeshDevice* mesh = input_tensor.device();
    TT_FATAL(mesh != nullptr, "matmul_reduce_scatter: input_tensor must be an allocated device tensor");
    const uint32_t G = mesh->shape()[cluster_axis];
    if (G < 2) {
        throw std::invalid_argument(
            fmt::format("matmul_reduce_scatter: cluster_axis {} has {} device(s); need >= 2", cluster_axis, G));
    }
    const uint32_t tile = tt::constants::TILE_HEIGHT;
    const uint32_t extent = canonical_scatter_dim == -2 ? M : N;
    if (extent % (tile * G) != 0) {
        throw std::invalid_argument(
            fmt::format("matmul_reduce_scatter: scattered extent {} does not split into {} tile blocks", extent, G));
    }

    const bool ring = topology == tt::tt_fabric::Topology::Ring && G >= 3;  // a 2-device ring is the line
    const mrs::ScheduleKind kind = !ring        ? mrs::ScheduleKind::LINE
                                   : G % 2 == 0 ? mrs::ScheduleKind::RING_BALANCED
                                                : mrs::ScheduleKind::RING_UNBALANCED;
    if (ring) {
        const auto fabric_config = tt::tt_fabric::GetFabricConfig();
        if (!config_wraps(fabric_config, cluster_axis)) {
            throw std::invalid_argument(fmt::format(
                "matmul_reduce_scatter: Ring along axis {} needs a fabric config that wraps it (active: {})",
                cluster_axis,
                static_cast<int>(fabric_config)));
        }
    }
    const uint32_t avail = usable_links(mesh, cluster_axis, ring);
    if (avail == 0) {
        throw std::invalid_argument(
            "matmul_reduce_scatter: a hop of the group has no usable fabric link (no wrap link?)");
    }
    uint32_t L = avail;
    if (num_links.has_value()) {
        if (*num_links <= 0 || static_cast<uint32_t>(*num_links) > avail) {
            throw std::invalid_argument(
                fmt::format("matmul_reduce_scatter: num_links={}, but a hop has only {} link(s)", *num_links, avail));
        }
        L = static_cast<uint32_t>(*num_links);
    }

    const uint32_t Mt = M / tile;
    const uint32_t Kt = K / tile;
    const uint32_t Nt = N / tile;
    const CoreCoord grid = mesh->compute_with_storage_grid_size();
    const mrs::PlanInputs plan_inputs{
        .group_size = G,
        .kind = kind,
        .num_links = L,
        .Mt = Mt,
        .Kt = Kt,
        .Nt = Nt,
        .block_m = canonical_scatter_dim == -2 ? Mt / G : Mt,
        .block_n = canonical_scatter_dim == -2 ? Nt : Nt / G,
        .grid_x = static_cast<uint32_t>(grid.x),
        .grid_y = static_cast<uint32_t>(grid.y),
        .a_tile_bytes = static_cast<uint32_t>(input_tensor.buffer()->aligned_page_size()),
        .w_tile_bytes = static_cast<uint32_t>(weight.buffer()->aligned_page_size()),
        .scatter_dim = canonical_scatter_dim,
        .l1_budget = ttnn::reports::get_device_info(mesh).l1_bank_size - kL1HeadroomBytes,
    };
    const mrs::Plan plan = mrs::solve_plan(plan_inputs);

    const mrs::MatmulReduceScatterParams params{
        .cluster_axis = static_cast<uint32_t>(cluster_axis),
        .scatter_dim = canonical_scatter_dim,
        .kind = kind,
        .num_links = L,
        .math_fidelity = compute_kernel_config.has_value() ? compute_kernel_config->math_fidelity : MathFidelity::HiFi2,
        .math_approx_mode = compute_kernel_config.has_value() && compute_kernel_config->math_approx_mode,
        .output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        .plan = plan,
    };
    const mrs::MeshResources& resources = mrs::acquire_mesh_resources(mesh, mrs::staging_tiles(plan));
    return ttnn::prim::matmul_reduce_scatter(
        input_tensor, weight, *resources.staging, *resources.mailbox, output_tensor, params);
}

}  // namespace ttnn
