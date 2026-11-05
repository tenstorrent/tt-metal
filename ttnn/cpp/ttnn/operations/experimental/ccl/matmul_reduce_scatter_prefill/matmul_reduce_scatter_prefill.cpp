// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_prefill.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt_stl/indestructible.hpp>

#include "device/matmul_reduce_scatter_prefill_device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/core/program_cache_l1.hpp"

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

constexpr size_t kMaxLinks = 2;  // links per hop the op supports (check_support refuses a larger num_links)

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
// Also refuses a mesh with a device on another host: every chip's program places its neighbours' ports itself (the
// port cores its packets target), which needs each neighbour's Ethernet cores, so a remote neighbour would be placed
// differently from its own program and its port would never receive the ready word.
size_t count_usable_links(tt::tt_metal::distributed::MeshDevice* mesh, int32_t cluster_axis, bool ring) {
    const auto& mesh_shape = mesh->shape();
    auto node = [&](const Coord& c) { return mesh->get_fabric_node_id(ttnn::MeshCoordinate(c.first, c.second)); };
    size_t avail = std::numeric_limits<size_t>::max();
    for (const auto& group : build_groups(mesh_shape[0], mesh_shape[1], cluster_axis)) {
        for (const Coord& c : group) {
            TT_FATAL(
                mesh->is_local(ttnn::MeshCoordinate(c.first, c.second)),
                "matmul_reduce_scatter_prefill: device ({}, {}) of the mesh is on another host; a multi-host mesh is "
                "not supported",
                c.first,
                c.second);
        }
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

// Bytes per L1 bank of an interleaved output the op allocates after planning (0 outside L1): it lands in the same free
// block below the live tensors as the circular buffers, so the live budget leaves room for it.
uint64_t l1_output_bytes_per_bank(
    tt::tt_metal::distributed::MeshDevice* mesh, const MemoryConfig& out_mem, uint32_t Mt, uint32_t Nt, uint32_t G) {
    if (out_mem.buffer_type() != tt::tt_metal::BufferType::L1) {
        return 0;
    }
    const auto& allocator = *mesh->allocator();
    const uint64_t banks = allocator.get_num_banks(tt::tt_metal::BufferType::L1);
    const uint64_t tiles = static_cast<uint64_t>(Mt) * Nt / G;
    const uint64_t bytes = tt::div_up(tiles, banks) * tt::tile_size(tt::DataFormat::Float16_b);
    // the L1 free lists allocate in DRAM-alignment granules (see BankManager::init_allocators)
    return tt::round_up(bytes, uint64_t{allocator.get_alignment(tt::tt_metal::BufferType::DRAM)});
}

// Lowers the budget to the solved plan's own footprint when that resolves the same plan. The solver takes the first
// K block, super-block and chunk split that fit, so a budget at the footprint normally does; the program hash then
// covers the plan, not how much L1 happened to be live, and calls with different live L1 share one program.
void canonicalize_budget(detail::PlanInputs& in) {
    const detail::Plan& plan = detail::solve_plan(in);
    detail::PlanInputs at_footprint = in;
    at_footprint.budget = plan.footprint;
    const auto knobs = [](const detail::Plan& p) {
        return std::make_tuple(p.n_m, p.n_n, p.resident, p.kb, p.depths, p.super_block, p.zone_sets);
    };
    if (knobs(detail::solve_plan(at_footprint)) == knobs(plan)) {
        in = at_footprint;
    }
}

// Everything a call resolves before launch: the checks, the links, the plan's inputs.
MatmulReduceScatterPrefillParams resolve_params(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    int32_t scatter_dim,
    ttnn::ccl::Topology topology,
    const std::optional<int32_t>& num_links,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<uint32_t>& l1_headroom,
    bool output_preallocated) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    // the supported inputs, scatter_dim canonical
    const auto rank = static_cast<int32_t>(input_tensor.logical_shape().rank());
    const int32_t canonical_dim = scatter_dim >= 0 ? scatter_dim - rank : scatter_dim;
    check_support(input_tensor, weight, cluster_axis, canonical_dim, topology, num_links);

    // operand checks, then the mesh
    check_operands(input_tensor, weight, memory_config);
    const MemoryConfig out_mem = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);

    const uint32_t G = check_group(input_tensor, weight, cluster_axis, canonical_dim);
    auto* mesh = input_tensor.device();
    const auto& a_shape = input_tensor.logical_shape();
    const uint32_t M = a_shape[-2];
    const uint32_t K = a_shape[-1];
    const uint32_t N = weight.logical_shape()[-1];

    const bool ring = topology == ttnn::ccl::Topology::Ring && G >= 3;  // a 2-device ring is the line
    const detail::ScheduleKind kind =
        !ring ? detail::ScheduleKind::Line
              : (G % 2 == 0 ? detail::ScheduleKind::RingBalanced : detail::ScheduleKind::RingUnbalanced);
    TT_FATAL(
        !ring || config_wraps(tt::tt_fabric::GetFabricConfig(), cluster_axis),
        "matmul_reduce_scatter_prefill: Ring along axis {} needs a fabric config that wraps it (active: {})",
        cluster_axis,
        static_cast<uint32_t>(tt::tt_fabric::GetFabricConfig()));
    const size_t avail = usable_links(mesh, cluster_axis, ring);
    TT_FATAL(avail > 0, "matmul_reduce_scatter_prefill: a hop of the group has no usable fabric link (no wrap link?)");
    uint32_t L = static_cast<uint32_t>(std::min<size_t>(avail, kMaxLinks));  // the default: every link the op supports
    if (num_links.has_value()) {
        TT_FATAL(
            *num_links > 0 && static_cast<size_t>(*num_links) <= avail,
            "matmul_reduce_scatter_prefill: num_links={}, but a hop has only {} link(s)",
            *num_links,
            avail);
        L = static_cast<uint32_t>(*num_links);
    }

    // geometry and the plan
    const uint32_t Mt = M / TILE_HEIGHT;
    const uint32_t Kt = K / TILE_WIDTH;
    const uint32_t Nt = N / TILE_WIDTH;
    const CoreCoord grid = mesh->compute_with_storage_grid_size();
    const uint64_t l1_bank = mesh->allocator()->get_bank_size(tt::tt_metal::BufferType::L1);
    const uint64_t headroom = l1_headroom.value_or(detail::kL1Headroom);
    TT_FATAL(
        headroom < l1_bank,
        "matmul_reduce_scatter_prefill: l1_headroom={} must be smaller than the L1 bank size ({} bytes)",
        headroom,
        l1_bank);
    detail::PlanInputs plan_inputs{
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
        .budget = l1_bank - headroom,
        .w_dram_sharded = weight.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM &&
                          weight.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
    };
    // L1 tensors fill from the top; the CBs (staging included) and an L1 output share the free L1 below them
    const uint64_t output = output_preallocated ? 0 : l1_output_bytes_per_bank(mesh, out_mem, Mt, Nt, G);
    // the mesh's semaphores and mailbox first, so the measurement sees them; rounded so the plan memo stays small
    ensure_mesh_resources(mesh);
    constexpr uint64_t kLiveL1Granule = 16 * 1024;
    const uint64_t free_below_live =
        ttnn::operations::core::available_program_l1_capacity(mesh) / kLiveL1Granule * kLiveL1Granule;
    plan_inputs.budget = std::min(plan_inputs.budget, free_below_live > output ? free_below_live - output : 0);
    canonicalize_budget(plan_inputs);

    return MatmulReduceScatterPrefillParams{
        .cluster_axis = static_cast<uint32_t>(cluster_axis),
        .scatter_dim = canonical_dim,
        .topology = topology,
        .num_links = L,
        .kind = kind,
        .math_fidelity = compute_kernel_config.has_value() ? compute_kernel_config->math_fidelity
                                                           : tt::tt_metal::MathFidelity::HiFi2,
        .math_approx_mode = compute_kernel_config.has_value() && compute_kernel_config->math_approx_mode,
        .output_memory_config = out_mem,
        .plan_inputs = plan_inputs,
    };
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
    const std::optional<Tensor>& output_tensor,
    const std::optional<uint32_t>& l1_headroom) {
    using namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill;
    const MatmulReduceScatterPrefillParams params = resolve_params(
        input_tensor,
        weight,
        cluster_axis,
        scatter_dim,
        topology,
        num_links,
        compute_kernel_config,
        memory_config,
        l1_headroom,
        output_tensor.has_value());
    record_last_plan(params);
    return ttnn::prim::matmul_reduce_scatter_prefill(input_tensor, weight, output_tensor, params);
}

}  // namespace ttnn
