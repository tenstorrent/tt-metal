// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/common/host/mesh_ring_plan.hpp"

#include <algorithm>

#include <tt-metalium/experimental/fabric/pipeline_builder.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::ccl::common {
namespace {

std::optional<uint32_t> normalize_tensor_dim(int dim, uint32_t rank) {
    const int normalized_dim = dim < 0 ? static_cast<int>(rank) + dim : dim;
    if (normalized_dim < 0 || normalized_dim >= static_cast<int>(rank)) {
        return std::nullopt;
    }
    return static_cast<uint32_t>(normalized_dim);
}

}  // namespace

ttnn::MeshCoordinate snake_ring_coordinate(
    uint32_t transport_rank,
    const tt::tt_metal::distributed::MeshShape& shape,
    ttnn::ccl::snake_ring::Orientation orientation) {
    TT_FATAL(
        shape.dims() == 2 && shape[0] > 0 && shape[1] > 0,
        "snake ring coordinates require a non-empty 2D mesh shape; got {}",
        shape);
    return ttnn::MeshCoordinate(
        ttnn::ccl::snake_ring::coordinate_row(transport_rank, shape[0], shape[1], orientation),
        ttnn::ccl::snake_ring::coordinate_col(transport_rank, shape[0], shape[1], orientation));
}

bool has_row_major_mesh_coordinates(const ttnn::Tensor& tensor) {
    if (tensor.device() == nullptr) {
        return false;
    }
    const auto& mesh_coords = tensor.tensor_topology().mesh_coords();
    const auto shape = tensor.device()->shape();
    if (mesh_coords.size() != shape.mesh_size()) {
        return false;
    }
    uint32_t index = 0;
    for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(shape)) {
        if (mesh_coords[index++] != coord) {
            return false;
        }
    }
    return true;
}

bool placement_shards_tensor_dim(
    const tt::tt_metal::distributed::MeshMapperConfig::Placement& placement,
    uint32_t tensor_dim,
    uint32_t tensor_dims) {
    const auto* shard = std::get_if<tt::tt_metal::distributed::MeshMapperConfig::Shard>(&placement);
    return shard != nullptr && normalize_tensor_dim(shard->dim, tensor_dims) == tensor_dim;
}

uint32_t tensor_dim_shard_factor(const ttnn::Tensor& tensor, uint32_t tensor_dim) {
    const auto& topology = tensor.tensor_topology();
    const auto& distribution_shape = topology.distribution_shape();
    const auto& placements = topology.placements();
    if (placements.size() != distribution_shape.dims()) {
        return 0;
    }
    uint32_t factor = 1;
    const uint32_t rank = tensor.logical_shape().rank();
    for (uint32_t axis = 0; axis < distribution_shape.dims(); ++axis) {
        if (placement_shards_tensor_dim(placements[axis], tensor_dim, rank)) {
            factor *= distribution_shape[axis];
        }
    }
    return factor;
}

std::optional<uint64_t> resolve_direct_neighbor_route_hash(
    const ttnn::Tensor& tensor,
    std::optional<uint32_t> cluster_axis,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    ttnn::ccl::snake_ring::Orientation orientation,
    bool log_rejection,
    std::string_view operation_name) {
    auto* mesh_device = tensor.device();
    if (mesh_device == nullptr) {
        if (log_rejection) {
            log_warning(tt::LogOp, "{} requires a mesh-device tensor", operation_name);
        }
        return std::nullopt;
    }
    const auto shape = mesh_device->shape();
    if (shape.dims() != 2 || (cluster_axis.has_value() && *cluster_axis >= 2)) {
        if (log_rejection) {
            log_warning(
                tt::LogOp,
                "{} direct-neighbor routing requires a 2D mesh and cluster_axis 0 or 1; shape={}, cluster_axis={}",
                operation_name,
                shape,
                cluster_axis);
        }
        return std::nullopt;
    }
    const bool full_mesh = !cluster_axis.has_value();
    if ((cluster_axis.has_value() && shape[*cluster_axis] < 2) || num_links == 0) {
        return std::nullopt;
    }

    const bool is_ring = full_mesh || topology == tt::tt_fabric::Topology::Ring;
    auto hash =
        ttsl::hash::hash_objects_with_default_seed(cluster_axis, full_mesh, orientation, num_links, shape, topology);
    const uint32_t group_count = full_mesh ? 1 : shape[1 - *cluster_axis];
    const uint32_t rank_count = full_mesh ? shape.mesh_size() : shape[*cluster_axis];
    for (uint32_t group = 0; group < group_count; ++group) {
        for (uint32_t rank = 0; rank < rank_count; ++rank) {
            const auto source_coord = full_mesh ? snake_ring_coordinate(rank, shape, orientation)
                                                : (*cluster_axis == 0 ? ttnn::MeshCoordinate(rank, group)
                                                                      : ttnn::MeshCoordinate(group, rank));
            const auto source = mesh_device->get_fabric_node_id(source_coord);
            for (const int32_t offset : {-1, 1}) {
                const int32_t destination_rank_signed = static_cast<int32_t>(rank) + offset;
                if (!is_ring && (destination_rank_signed < 0 || destination_rank_signed >= rank_count)) {
                    continue;
                }
                const auto destination_rank = static_cast<uint32_t>(
                    (destination_rank_signed + static_cast<int32_t>(rank_count)) % static_cast<int32_t>(rank_count));
                const auto destination_coord =
                    full_mesh ? snake_ring_coordinate(destination_rank, shape, orientation)
                              : (*cluster_axis == 0 ? ttnn::MeshCoordinate(destination_rank, group)
                                                    : ttnn::MeshCoordinate(group, destination_rank));
                const auto destination = mesh_device->get_fabric_node_id(destination_coord);
                const auto direction = tt::tt_fabric::pipeline_get_forwarding_direction(source, destination);
                if (!direction.has_value()) {
                    if (log_rejection) {
                        log_warning(
                            tt::LogOp,
                            "{} rejected non-routable neighbor edge {} -> {}",
                            operation_name,
                            source_coord,
                            destination_coord);
                    }
                    return std::nullopt;
                }
                const auto neighbors = tt::tt_fabric::pipeline_get_chip_neighbors(source, *direction);
                const auto mesh_it = neighbors.find(*destination.mesh_id);
                if (mesh_it == neighbors.end() ||
                    std::find(mesh_it->second.begin(), mesh_it->second.end(), destination.chip_id) ==
                        mesh_it->second.end()) {
                    if (log_rejection) {
                        log_warning(
                            tt::LogOp,
                            "{} rejected non-direct neighbor edge {} -> {}",
                            operation_name,
                            source_coord,
                            destination_coord);
                    }
                    return std::nullopt;
                }
                const auto link_indices = tt::tt_fabric::get_forwarding_link_indices(source, destination);
                for (uint32_t link = 0; link < num_links; ++link) {
                    if (std::find(link_indices.begin(), link_indices.end(), link) == link_indices.end()) {
                        if (log_rejection) {
                            log_warning(
                                tt::LogOp,
                                "{} neighbor edge {} -> {} does not provide requested link {}; usable links={}",
                                operation_name,
                                source_coord,
                                destination_coord,
                                link,
                                link_indices);
                        }
                        return std::nullopt;
                    }
                }
                hash = ttsl::hash::hash_objects(
                    hash,
                    source.mesh_id,
                    source.chip_id,
                    destination.mesh_id,
                    destination.chip_id,
                    group,
                    rank,
                    destination_rank,
                    link_indices);
            }
        }
    }
    return hash;
}

std::optional<MeshRingPlan> resolve_mesh_ring_plan(
    const ttnn::Tensor& tensor,
    std::optional<uint32_t> cluster_axis,
    uint32_t num_links,
    const std::array<tt::tt_fabric::Topology, 2>& axis_topology,
    bool log_rejection,
    std::string_view operation_name) {
    auto* mesh_device = tensor.device();
    if (mesh_device == nullptr || num_links == 0) {
        return std::nullopt;
    }
    const auto shape = mesh_device->shape();
    if (shape.dims() != 2) {
        return std::nullopt;
    }
    const auto fabric_config = tt::tt_fabric::GetFabricConfig();
    if (!cluster_axis.has_value() && !tt::tt_fabric::is_2d_fabric_config(fabric_config)) {
        if (log_rejection) {
            log_warning(
                tt::LogOp,
                "{} full-mesh ring requires Fabric2D; active fabric config={}",
                operation_name,
                fabric_config);
        }
        return std::nullopt;
    }

    if (cluster_axis.has_value()) {
        if (*cluster_axis >= 2 || shape[*cluster_axis] < 2) {
            return std::nullopt;
        }
        const auto topology = axis_topology[*cluster_axis];
        const auto route_hash = resolve_direct_neighbor_route_hash(
            tensor,
            cluster_axis,
            num_links,
            topology,
            ttnn::ccl::snake_ring::Orientation::Row,
            log_rejection,
            operation_name);
        if (!route_hash.has_value()) {
            return std::nullopt;
        }
        return MeshRingPlan{
            .cluster_axis = cluster_axis,
            .full_mesh = false,
            .orientation = ttnn::ccl::snake_ring::Orientation::Row,
            .mesh_rows = shape[0],
            .mesh_cols = shape[1],
            .ring_size = shape[*cluster_axis],
            .num_links = num_links,
            .topology = topology,
            .fabric_config = fabric_config,
            .axis_topology = axis_topology,
            .route_plan_hash = ttsl::hash::hash_objects(*route_hash, fabric_config, axis_topology)};
    }

    if (shape[0] < 2 || shape[1] < 2 || (shape[0] % 2 != 0 && shape[1] % 2 != 0)) {
        if (log_rejection) {
            log_warning(
                tt::LogOp,
                "{} full-mesh ring requires both mesh dimensions greater than one and at least one even dimension; "
                "shape={}",
                operation_name,
                shape);
        }
        return std::nullopt;
    }

    // The transport-to-tensor mapping is only valid for row-major mesh coordinates.
    if (!has_row_major_mesh_coordinates(tensor)) {
        if (log_rejection) {
            log_warning(
                tt::LogOp,
                "{} full-mesh ring requires row-major mesh coordinates on the participating tensor",
                operation_name);
        }
        return std::nullopt;
    }
    for (const auto orientation :
         {ttnn::ccl::snake_ring::Orientation::Row, ttnn::ccl::snake_ring::Orientation::Column}) {
        const uint32_t lane_count = orientation == ttnn::ccl::snake_ring::Orientation::Row ? shape[0] : shape[1];
        if (lane_count % 2 != 0) {
            continue;
        }
        const uint32_t closing_axis = orientation == ttnn::ccl::snake_ring::Orientation::Row ? 0 : 1;
        const auto route_hash = resolve_direct_neighbor_route_hash(
            tensor, std::nullopt, num_links, axis_topology[closing_axis], orientation, false, operation_name);
        if (route_hash.has_value()) {
            return MeshRingPlan{
                .cluster_axis = std::nullopt,
                .full_mesh = true,
                .orientation = orientation,
                .mesh_rows = shape[0],
                .mesh_cols = shape[1],
                .ring_size = static_cast<uint32_t>(shape.mesh_size()),
                .num_links = num_links,
                .topology = tt::tt_fabric::Topology::Ring,
                .fabric_config = fabric_config,
                .axis_topology = axis_topology,
                .route_plan_hash = ttsl::hash::hash_objects(*route_hash, fabric_config, axis_topology)};
        }
    }

    if (log_rejection) {
        // Repeat the deterministic fallback orientation with logging enabled
        // so the caller gets the exact edge/link that made the mesh ineligible.
        const auto fallback =
            shape[0] % 2 == 0 ? ttnn::ccl::snake_ring::Orientation::Row : ttnn::ccl::snake_ring::Orientation::Column;
        const uint32_t closing_axis = fallback == ttnn::ccl::snake_ring::Orientation::Row ? 0 : 1;
        (void)resolve_direct_neighbor_route_hash(
            tensor, std::nullopt, num_links, axis_topology[closing_axis], fallback, true, operation_name);
    }
    return std::nullopt;
}

MeshRingPosition get_mesh_ring_position(
    const ttnn::Tensor& tensor, const ttnn::MeshCoordinate& coordinate, const MeshRingPlan& plan) {
    if (plan.full_mesh) {
        TT_FATAL(
            tensor.device() != nullptr && tensor.device()->shape().dims() == 2 && coordinate.dims() == 2,
            "full mesh-ring position requires a 2D mesh-device tensor and 2D coordinate");
        TT_FATAL(
            plan.mesh_rows > 0 && plan.mesh_cols > 0 && plan.ring_size == plan.mesh_rows * plan.mesh_cols,
            "invalid full mesh-ring geometry: rows={}, cols={}, ring_size={}",
            plan.mesh_rows,
            plan.mesh_cols,
            plan.ring_size);
        TT_FATAL(
            coordinate[0] < plan.mesh_rows && coordinate[1] < plan.mesh_cols,
            "mesh coordinate {} is outside resolved ring shape {}x{}",
            coordinate,
            plan.mesh_rows,
            plan.mesh_cols);
        const tt::tt_metal::distributed::MeshShape plan_shape(plan.mesh_rows, plan.mesh_cols);
        const uint32_t transport_rank = ttnn::ccl::snake_ring::index_from_coordinate(
            coordinate[0], coordinate[1], plan.mesh_rows, plan.mesh_cols, plan.orientation);
        return MeshRingPosition{
            .transport_rank = transport_rank,
            .tensor_rank = ttnn::ccl::snake_ring::row_major_index(
                transport_rank, plan.mesh_rows, plan.mesh_cols, plan.orientation),
            .forward_coord = snake_ring_coordinate((transport_rank + 1) % plan.ring_size, plan_shape, plan.orientation),
            .backward_coord = snake_ring_coordinate(
                (transport_rank + plan.ring_size - 1) % plan.ring_size, plan_shape, plan.orientation)};
    }

    TT_FATAL(plan.cluster_axis.has_value(), "axis mesh-ring plan is missing cluster_axis");
    const uint32_t transport_rank =
        ttnn::ccl::get_linearized_index_from_physical_coord(tensor, coordinate, plan.cluster_axis);
    return MeshRingPosition{
        .transport_rank = transport_rank,
        .tensor_rank = transport_rank,
        .forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor, coordinate, 1, plan.topology, plan.cluster_axis),
        .backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor, coordinate, -1, plan.topology, plan.cluster_axis)};
}

}  // namespace ttnn::operations::ccl::common
