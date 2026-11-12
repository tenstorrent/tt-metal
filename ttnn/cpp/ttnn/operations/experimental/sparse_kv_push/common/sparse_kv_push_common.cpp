// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_kv_push_common.hpp"

#include <algorithm>
#include <array>
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
    // The configs that say which axes wrap (fabric_directions reads it from them); not MGD_DEFAULT, which leaves it to
    // the mesh graph descriptor.
    using tt::tt_fabric::FabricConfig;
    const auto config = tt::tt_fabric::GetFabricConfig();
    const bool fabric_2d = config == FabricConfig::FABRIC_2D || config == FabricConfig::FABRIC_2D_TORUS_X ||
                           config == FabricConfig::FABRIC_2D_TORUS_Y || config == FabricConfig::FABRIC_2D_TORUS_XY;
    return fabric_2d && mesh_shape.dims() == 2 && mesh_shape.mesh_size() >= 2;
}

std::vector<FabricDirection> fabric_directions(
    const Tensor& tensor, const tt::tt_metal::distributed::MeshCoordinate& coord) {
    // get_neighbors' direction order: E, W along axis 1, then N, S along axis 0.
    enum : uint32_t { kEast = 0, kWest = 1, kNorth = 2, kSouth = 3 };
    constexpr std::array<std::pair<uint32_t, std::array<uint32_t, 2>>, 2> kAxes = {
        {{1, {kEast, kWest}}, {0, {kNorth, kSouth}}}};
    auto* mesh_device = tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    const auto fabric_config = tt::tt_fabric::GetFabricConfig();
    const auto my_node = mesh_device->get_fabric_node_id(coord);
    std::vector<FabricDirection> directions;
    for (const auto& [axis, axis_directions] : kAxes) {
        if (mesh_view.shape()[axis] < 2) {
            continue;
        }
        // Ring when the fabric wraps this axis and the closing link is wired (never for 2 chips), else a line.
        const auto topology = ttnn::ccl::get_axis_topology(tensor, fabric_config, axis);
        const auto [neighbors, present] =
            ttnn::operations::ccl::common::get_neighbors(mesh_view, coord, topology, axis);
        auto neighbor = neighbors.begin();
        for (const uint32_t direction : axis_directions) {
            if (!present[direction]) {
                continue;
            }
            // The kernels pick a channel by the fabric's next hop, so name the direction as the fabric does.
            const auto eth_direction =
                tt::tt_fabric::get_eth_forwarding_direction(my_node, mesh_device->get_fabric_node_id(*neighbor));
            TT_FATAL(
                eth_direction.has_value(),
                "the sparse KV push found no fabric route from chip {} to its neighbour {}",
                coord,
                *neighbor);
            directions.push_back(FabricDirection{
                .eth_direction = static_cast<uint32_t>(*eth_direction),
                .neighbor = *neighbor,
                .links = tt::tt_fabric::get_forwarding_link_indices(
                    *mesh_device, my_node, mesh_device->get_fabric_node_id(*neighbor))});
            ++neighbor;
        }
    }
    return directions;
}

uint32_t max_fabric_directions(const Tensor& tensor) {
    uint32_t most = 0;
    for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(tensor.device()->shape())) {
        most = std::max(most, static_cast<uint32_t>(fabric_directions(tensor, coord).size()));
    }
    return most;
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
