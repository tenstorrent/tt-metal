// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter.hpp"

#include <map>
#include <stdexcept>

#include <fmt/format.h>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt_stl/indestructible.hpp>

#include "device/matmul_reduce_scatter_device_operation.hpp"

namespace ttnn {

namespace {

using namespace operations::matmul_reduce_scatter;

void check_dram_interleaved_tile(const Tensor& t, const char* what) {
    const MemoryConfig& mc = t.memory_config();
    if (mc.buffer_type() != BufferType::DRAM || mc.memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        throw std::invalid_argument(fmt::format("matmul_reduce_scatter: {} must be DRAM interleaved", what));
    }
    if (t.layout() != Layout::TILE) {
        throw std::invalid_argument(fmt::format("matmul_reduce_scatter: {} must be TILE layout", what));
    }
}

// The plan is a pure function of its integer inputs, so each distinct input is solved once (the Python cached its
// chunk search in _PLAN_CACHE): a call that hits the program cache pays a lookup, not the grid / regime / chunk search.
const ChunkPlan& solved_plan(const PlanInputs& inputs) {
    static ttsl::Indestructible<std::map<PlanInputs, ChunkPlan>> plans;
    auto& cache = plans.get();
    auto it = cache.find(inputs);
    if (it == cache.end()) {
        it = cache.emplace(inputs, solve_plan(inputs)).first;
    }
    return it->second;
}

}  // namespace

Tensor matmul_reduce_scatter(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    int32_t scatter_dim,
    tt::tt_fabric::Topology topology,
    std::optional<int32_t> num_links,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output_tensor) {
    const auto& a_shape = input_tensor.logical_shape();
    const auto& w_shape = weight.logical_shape();
    const auto a_rank = static_cast<int32_t>(a_shape.rank());
    const int32_t canonical_dim = scatter_dim >= 0 ? scatter_dim - a_rank : scatter_dim;
    check_support(input_tensor, weight, cluster_axis, canonical_dim, topology, num_links);

    // ---- run(): caller checks, in the Python's order ----
    bool leading_ones = true;
    for (int32_t i = 0; i + 2 < a_rank; ++i) {
        leading_ones = leading_ones && a_shape[i] == 1;
    }
    if (a_rank < 2 || a_rank > 4 || !leading_ones) {
        throw std::invalid_argument(
            fmt::format("matmul_reduce_scatter: A must be rank 2-4 with leading dims 1, got {}", a_shape));
    }
    if (w_shape.rank() != 2) {
        throw std::invalid_argument(fmt::format("matmul_reduce_scatter: W must be (K, N), got {}", w_shape));
    }
    const uint32_t m = a_shape[-2];
    const uint32_t k = a_shape[-1];
    const uint32_t n = w_shape[-1];
    if (k != w_shape[0]) {
        throw std::invalid_argument(fmt::format("matmul_reduce_scatter: K mismatch, A {} vs W {}", a_shape, w_shape));
    }
    check_dram_interleaved_tile(input_tensor, "input_tensor");
    check_dram_interleaved_tile(weight, "weight");
    if (memory_config.has_value() && memory_config->memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        throw NotImplementedRefusal(
            "matmul_reduce_scatter: sharded output memory_config (no placement axis in TARGET)");
    }
    const MemoryConfig out_mem = memory_config.value_or(DRAM_MEMORY_CONFIG);

    auto* mesh = input_tensor.device();
    TT_FATAL(mesh != nullptr, "matmul_reduce_scatter: input_tensor must be on a device");
    const auto axis = static_cast<uint32_t>(cluster_axis);
    const auto& mesh_shape = mesh->shape();
    const uint32_t group_size = mesh_shape[axis];
    if (group_size < 2) {
        throw std::invalid_argument(fmt::format(
            "matmul_reduce_scatter: cluster_axis {} has {} device(s); need >= 2", cluster_axis, group_size));
    }
    const uint32_t extent = canonical_dim == -2 ? m : n;
    if (extent % (tt::constants::TILE_WIDTH * group_size) != 0) {
        throw std::invalid_argument(fmt::format(
            "matmul_reduce_scatter: scattered extent {} does not split into {} tile blocks", extent, group_size));
    }

    const bool ring = topology == tt::tt_fabric::Topology::Ring && group_size >= 3;  // a 2-device ring is the line
    ScheduleKind kind = ScheduleKind::Line;
    if (ring) {
        kind = group_size % 2 == 0 ? ScheduleKind::RingBalanced : ScheduleKind::RingUnbalanced;
    }
    const auto groups = build_groups(mesh_shape, axis);
    if (ring) {
        const auto fabric_config = tt::tt_fabric::GetFabricConfig();
        if (!fabric_config_wraps(fabric_config, axis)) {
            throw std::invalid_argument(fmt::format(
                "matmul_reduce_scatter: Ring along axis {} needs a fabric config that wraps it (active: {})",
                cluster_axis,
                static_cast<uint32_t>(fabric_config)));
        }
    }
    const uint32_t avail = available_links(*mesh, groups, ring);
    if (avail == 0) {
        throw std::invalid_argument(
            "matmul_reduce_scatter: a hop of the group has no usable fabric link (no wrap link?)");
    }
    uint32_t links = avail;
    if (num_links.has_value()) {
        if (*num_links <= 0 || static_cast<uint32_t>(*num_links) > avail) {
            throw std::invalid_argument(
                fmt::format("matmul_reduce_scatter: num_links={}, but a hop has only {} link(s)", *num_links, avail));
        }
        links = static_cast<uint32_t>(*num_links);
    }

    // ---- plan ----
    const uint32_t mt = m / tt::constants::TILE_HEIGHT;
    const uint32_t kt = k / tt::constants::TILE_WIDTH;
    const uint32_t nt = n / tt::constants::TILE_WIDTH;
    const auto grid = mesh->compute_with_storage_grid_size();
    const PlanInputs plan_inputs{
        .group_size = group_size,
        .kind = kind,
        .num_links = links,
        .mt = mt,
        .kt = kt,
        .nt = nt,
        .block_m = canonical_dim == -2 ? mt / group_size : mt,
        .block_n = canonical_dim == -2 ? nt : nt / group_size,
        .grid_x = static_cast<uint32_t>(grid.x),
        .grid_y = static_cast<uint32_t>(grid.y),
        .a_tile_bytes = static_cast<uint32_t>(input_tensor.buffer()->aligned_page_size()),
        .w_tile_bytes = static_cast<uint32_t>(weight.buffer()->aligned_page_size()),
        .hold_tile_bytes = tt::tile_size(tt::DataFormat::Float32),
        .stage_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b),
        .scatter_dim = canonical_dim,
        .budget = mesh->allocator()->get_bank_size(tt::tt_metal::BufferType::L1) - kL1HeadroomBytes,
    };
    const ChunkPlan& plan = solved_plan(plan_inputs);
    MeshResources& res = mesh_resources(mesh, staging_tiles(plan));

    const tt::tt_metal::ComputeConfigDescriptor config =
        compute_kernel_config.value_or(tt::tt_metal::ComputeConfigDescriptor{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi2, .fp32_dest_acc_en = true});
    const MatmulReduceScatterParams params{
        .cluster_axis = axis,
        .scatter_dim = canonical_dim,
        .topology = topology,
        .group_size = group_size,
        .kind = kind,
        .num_links = links,
        .math_fidelity = config.math_fidelity,
        .math_approx_mode = config.math_approx_mode,
        .output_memory_config = out_mem,
        .plan = plan,
    };
    return ttnn::prim::matmul_reduce_scatter(input_tensor, weight, *res.staging, output_tensor, params);
}

}  // namespace ttnn
