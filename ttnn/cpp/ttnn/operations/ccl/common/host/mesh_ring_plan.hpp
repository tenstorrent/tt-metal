// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/shared_with_host/snake_ring.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::ccl::common {

// Mesh-wide structural description. Coordinate-specific ranks and neighbors
// are intentionally derived separately so this remains safe to hash as a
// device-operation attribute.
struct MeshRingPlan {
    std::optional<uint32_t> cluster_axis;
    bool full_mesh = false;
    ttnn::ccl::snake_ring::Orientation orientation = ttnn::ccl::snake_ring::Orientation::Row;
    uint32_t mesh_rows = 0;
    uint32_t mesh_cols = 0;
    uint32_t ring_size = 0;
    uint32_t num_links = 0;
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear;
    tt::tt_fabric::FabricConfig fabric_config = tt::tt_fabric::FabricConfig::DISABLED;
    std::array<tt::tt_fabric::Topology, 2> axis_topology{
        tt::tt_fabric::Topology::Linear, tt::tt_fabric::Topology::Linear};
    std::optional<uint64_t> route_plan_hash;
};

struct MeshRingPosition {
    uint32_t transport_rank = 0;
    uint32_t tensor_rank = 0;
    std::optional<ttnn::MeshCoordinate> forward_coord;
    std::optional<ttnn::MeshCoordinate> backward_coord;
};

ttnn::MeshCoordinate snake_ring_coordinate(
    uint32_t transport_rank,
    const tt::tt_metal::distributed::MeshShape& shape,
    ttnn::ccl::snake_ring::Orientation orientation);

bool has_row_major_mesh_coordinates(const ttnn::Tensor& tensor);

bool placement_shards_tensor_dim(
    const tt::tt_metal::distributed::MeshMapperConfig::Placement& placement, uint32_t tensor_dim, uint32_t tensor_dims);

uint32_t tensor_dim_shard_factor(const ttnn::Tensor& tensor, uint32_t tensor_dim);

// Prove that every directed edge in an axis line/ring or a full-mesh snake is
// a direct physical Fabric neighbor and exposes each requested link. The hash
// covers the complete resolved route, so callers can safely include it in a
// program-cache key.
std::optional<uint64_t> resolve_direct_neighbor_route_hash(
    const ttnn::Tensor& tensor,
    std::optional<uint32_t> cluster_axis,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    ttnn::ccl::snake_ring::Orientation orientation = ttnn::ccl::snake_ring::Orientation::Row,
    bool log_rejection = true,
    std::string_view operation_name = "mesh ring");

// Resolve an axis line/ring or select the first legal full-mesh snake. For a
// full mesh, row orientation is preferred and column orientation is the
// fallback; only orientations with an even lane count are candidates.
//
// A snake CYCLE closes across a whole axis, so it is a direct hop only on a
// torus (or an extent-2 axis). When no cycle closes and `allow_open_path` is
// set, the same boustrophedon is returned as an open Hamiltonian PATH with
// Topology::Linear: no wrap edge to prove, no parity precondition, so it
// resolves on any wired mesh of two or more devices, 1xN included.
//
// The path costs N-1 hops of latency against a ring's N/2, and it leaves rank 0
// without a backward neighbor and rank N-1 without a forward one. Only callers
// whose schedule handles dead endpoints may opt in -- an op that fuses a closed
// ring must leave this false and keep getting nullopt where no cycle exists.
std::optional<MeshRingPlan> resolve_mesh_ring_plan(
    const ttnn::Tensor& tensor,
    std::optional<uint32_t> cluster_axis,
    uint32_t num_links,
    const std::array<tt::tt_fabric::Topology, 2>& axis_topology,
    bool log_rejection = true,
    std::string_view operation_name = "mesh ring",
    bool allow_open_path = false);

MeshRingPosition get_mesh_ring_position(
    const ttnn::Tensor& tensor, const ttnn::MeshCoordinate& coordinate, const MeshRingPlan& plan);

}  // namespace ttnn::operations::ccl::common
