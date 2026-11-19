// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_kv_push_common.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <utility>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "kernels/sparse_push_constants.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

void validate_stripe_chunk(uint32_t stripe_chunk, uint32_t max_rows) {
    TT_FATAL(
        stripe_chunk > 0 && stripe_chunk % 32 == 0 && stripe_chunk <= ::sparse_push::kMaxStripeChunk,
        "sparse KV push stripe_chunk {} must be a positive multiple of 32, at most {}",
        stripe_chunk,
        ::sparse_push::kMaxStripeChunk);
    TT_FATAL(
        max_rows > 0 && max_rows % stripe_chunk == 0,
        "sparse KV push max_rows {} must be a positive multiple of stripe_chunk {}",
        max_rows,
        stripe_chunk);
}

PacketShape packet_shape(tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t page_bytes) {
    const auto max_payload = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
    TT_FATAL(
        page_bytes > 0 && page_bytes <= max_payload,
        "the sparse KV push needs a row page of 1 to {} bytes (the fabric's max payload), got {}",
        max_payload,
        page_bytes);
    return PacketShape{
        .pages_per_packet = max_payload / page_bytes, .banks = static_cast<uint32_t>(mesh_device->num_dram_channels())};
}

bool fabric_supported(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    const auto& mesh_shape = mesh_device->shape();
    return tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig()) && mesh_shape.dims() == 2 &&
           mesh_shape.mesh_size() >= 2;
}

namespace {

using tt::tt_metal::distributed::MeshCoordinate;
using AxisTopologies = std::array<std::optional<tt::tt_fabric::Topology>, 2>;

// Each axis' topology, the same for every chip on it: Ring when the fabric wraps the axis and the closing link is wired
// (never for 2 chips), else Linear; none for an axis of one chip.
AxisTopologies axis_topologies(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    AxisTopologies topologies;
    const auto fabric_config = tt::tt_fabric::GetFabricConfig();
    for (uint32_t axis = 0; axis < topologies.size(); ++axis) {
        if (mesh_device->shape()[axis] >= 2) {
            topologies[axis] = ttnn::ccl::get_axis_topology(*mesh_device, fabric_config, axis);
        }
    }
    return topologies;
}

// The chip's directions into `directions`, or why the push cannot route from it (see fabric_routes).
std::string chip_directions(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const AxisTopologies& topologies,
    const MeshCoordinate& coord,
    uint32_t num_links,
    std::vector<FabricDirection>& directions) {
    // get_neighbors' direction order: E, W along axis 1, then N, S along axis 0.
    enum : uint32_t { kEast = 0, kWest = 1, kNorth = 2, kSouth = 3 };
    constexpr std::array<std::pair<uint32_t, std::array<uint32_t, 2>>, 2> kAxes = {
        {{1, {kEast, kWest}}, {0, {kNorth, kSouth}}}};
    const auto& mesh_view = mesh_device->get_view();
    const auto my_node = mesh_device->get_fabric_node_id(coord);
    for (const auto& [axis, axis_directions] : kAxes) {
        if (!topologies[axis].has_value()) {
            continue;
        }
        const auto [neighbors, present] =
            ttnn::operations::ccl::common::get_neighbors(mesh_view, coord, *topologies[axis], axis);
        auto neighbor = neighbors.begin();
        for (const uint32_t direction : axis_directions) {
            if (!present[direction]) {
                continue;
            }
            const auto neighbor_node = mesh_device->get_fabric_node_id(*neighbor);
            // Resolved in the mesh device's fabric context, like the link indices; the kernels pick a channel by the
            // fabric's next hop, and towards a neighbour that is the direction they are wired in.
            const auto eth_directions =
                tt::tt_fabric::get_neighbor_eth_directions(*mesh_device, my_node, neighbor_node);
            if (eth_directions.empty()) {
                return fmt::format("chip {} has no fabric link to its neighbour {}", coord, *neighbor);
            }
            FabricDirection found{
                .eth_direction = static_cast<uint32_t>(eth_directions.front()),
                .neighbor = *neighbor,
                .links = tt::tt_fabric::get_forwarding_link_indices(*mesh_device, my_node, neighbor_node)};
            if (found.links.size() < num_links) {
                return fmt::format(
                    "chip {} forwards to its neighbour {} on {} links, fewer than {}: {}",
                    coord,
                    *neighbor,
                    found.links.size(),
                    num_links,
                    found.links);
            }
            for (const auto& other : directions) {
                if (other.eth_direction == found.eth_direction) {
                    return fmt::format(
                        "chip {} reaches neighbours {} and {} in the same fabric direction",
                        coord,
                        other.neighbor,
                        found.neighbor);
                }
            }
            directions.push_back(std::move(found));
            ++neighbor;
        }
    }
    for (const auto& other : tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (other == coord) {
            continue;
        }
        // No mesh-context lookup of a multi-hop route exists yet, so this reads the default control plane.
        const auto first_hop =
            tt::tt_fabric::get_eth_forwarding_direction(my_node, mesh_device->get_fabric_node_id(other));
        const bool covered =
            first_hop.has_value() && std::any_of(directions.begin(), directions.end(), [&](const FabricDirection& d) {
                return d.eth_direction == static_cast<uint32_t>(*first_hop);
            });
        if (!covered) {
            return fmt::format("chip {} has no channel on the fabric's route to chip {}", coord, other);
        }
    }
    return {};
}

}  // namespace

uint32_t FabricRoutes::max_directions() const {
    uint32_t most = 0;
    for (const auto& directions : chips) {
        most = std::max(most, static_cast<uint32_t>(directions.size()));
    }
    return most;
}

FabricRoutes fabric_routes(tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t num_links) {
    FabricRoutes routes;
    if (!fabric_supported(mesh_device)) {
        routes.problem = fmt::format(
            "needs a 2D fabric (FABRIC_2D or a 2D torus) on a 2D mesh of at least 2 chips, got {} on {}",
            tt::tt_fabric::GetFabricConfig(),
            mesh_device->shape());
        return routes;
    }
    const auto topologies = axis_topologies(mesh_device);
    for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape())) {
        auto& directions = routes.chips.emplace_back();
        routes.problem = chip_directions(mesh_device, topologies, coord, num_links, directions);
        if (!routes.ok()) {
            routes.chips.clear();
            return routes;
        }
    }
    return routes;
}

std::vector<FabricDirection> fabric_directions(
    tt::tt_metal::distributed::MeshDevice* mesh_device, const MeshCoordinate& coord, uint32_t num_links) {
    std::vector<FabricDirection> directions;
    const auto problem = chip_directions(mesh_device, axis_topologies(mesh_device), coord, num_links, directions);
    TT_FATAL(problem.empty(), "the sparse KV push cannot route: {}", problem);
    return directions;
}

uint32_t plan_cores(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    const auto grid = mesh_device->compute_with_storage_grid_size();
    return grid.x * grid.y;
}

bool plan_fits(tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t max_rows) {
    return max_rows % ::sparse_push_plan::kChunkRows == 0 &&
           tt::div_up(max_rows / ::sparse_push_plan::kChunkRows, plan_cores(mesh_device)) <=
               ::sparse_push_plan::kMaxChunksPerCore;
}

tt::tt_metal::TensorTopology rows_over_every_chip(const tt::tt_metal::distributed::MeshShape& mesh_shape) {
    using tt::tt_metal::distributed::MeshCoordinate;
    using tt::tt_metal::distributed::MeshMapperConfig;
    const tt::tt_metal::distributed::MeshCoordinateRange chips(mesh_shape);
    ttsl::SmallVector<MeshMapperConfig::Placement> placements(mesh_shape.dims(), MeshMapperConfig::Shard{2});
    return tt::tt_metal::TensorTopology(
        mesh_shape, std::move(placements), std::vector<MeshCoordinate>(chips.begin(), chips.end()));
}

uint32_t output_rows(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    uint32_t distinct_rows,
    uint32_t max_rows,
    uint32_t stripe_chunk,
    uint32_t page_bytes) {
    validate_stripe_chunk(stripe_chunk, max_rows);
    const uint32_t num_devices = mesh_device->shape().mesh_size();
    const uint32_t unit = packet_shape(mesh_device, page_bytes).unit();  // each owner's run is padded to whole units
    const uint32_t padded_distinct = std::min(distinct_rows, max_rows) + num_devices * unit;
    const uint32_t owner_rows = tt::div_up(max_rows / stripe_chunk, num_devices) * stripe_chunk;
    const uint32_t all_full = num_devices * tt::round_up(owner_rows, unit);
    return std::min(padded_distinct, all_full);
}

}  // namespace ttnn::operations::experimental::sparse_kv_push
