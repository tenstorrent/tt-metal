// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/fabric/pipeline_builder.hpp>

#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn::operations::ccl::common {

namespace detail {

bool has_wrap_around(tt::tt_fabric::Topology topology) {
    return topology == tt::tt_fabric::Topology::Ring || topology == tt::tt_fabric::Topology::Torus;
}

tt::tt_metal::distributed::MeshCoordinate::BoundaryMode get_boundary_mode(tt::tt_fabric::Topology topology) {
    return has_wrap_around(topology) ? tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::WRAP
                                     : tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::NONE;
}

uint32_t device_index(const std::vector<tt::tt_metal::IDevice*>& devices, const tt::tt_metal::IDevice* device) {
    for (uint32_t i = 0; i < devices.size(); i++) {
        if (devices[i] == device) {
            return i;
        }
    }
    TT_THROW("Device not found in device_index");
    return std::numeric_limits<uint32_t>::max();
}
}  // namespace detail

std::pair<std::vector<ttnn::MeshCoordinate>, std::array<bool, 4>> get_neighbors(
    const MeshDeviceView& mesh_view,
    const MeshCoordinate& mesh_coordinate,
    const tt::tt_fabric::Topology topology,
    const std::optional<uint32_t> axis) {
    // For readability use symbolic indices instead of raw numbers when accessing the
    // `directions` array `{East, West, North, South}`.
    enum Direction : std::size_t { East = 0, West = 1, North = 2, South = 3 };
    auto boundary_mode = detail::get_boundary_mode(topology);

    std::vector<ttnn::MeshCoordinate> neighbors;
    // directions: {East, West, North, South}
    std::array<bool, 4> directions = {false, false, false, false};

    const bool wrap_around_connection = detail::has_wrap_around(topology);

    // Helper that appends neighbours for a single axis
    auto process_axis = [&](int32_t axis_val) {
        int32_t next_neighbor_offset = 1;
        int32_t prev_neighbor_offset = -1;

        auto add_neighbor = [&](Direction dir, int32_t neighbor_offset) {
            auto neighbor = mesh_coordinate.get_neighbor(mesh_view.shape(), neighbor_offset, axis_val, boundary_mode);
            if (neighbor.has_value()) {
                neighbors.push_back(neighbor.value());
                directions[dir] = true;
            } else {
                directions[dir] = false;
            }
        };

        if (axis_val == 1) {
            // For horizontal axis (rows): process East then West
            // Positive direction (East)
            add_neighbor(Direction::East, next_neighbor_offset);
            // Negative direction (West)
            add_neighbor(Direction::West, prev_neighbor_offset);
        } else {
            // For vertical axis (columns): process North then South to maintain correct order
            // Negative direction (North)
            add_neighbor(Direction::North, prev_neighbor_offset);
            // Positive direction (South)
            add_neighbor(Direction::South, next_neighbor_offset);
        }
    };

    if (axis.has_value()) {
        process_axis(axis.value());
    } else {
        // When no axis is specified, gather neighbours on both axes
        process_axis(1);  // horizontal (row)
        process_axis(0);  // vertical (column)
    }

    TT_FATAL(!neighbors.empty(), "No neighbors found");
    TT_FATAL(!(axis.has_value() && neighbors.size() > 2), "Along a single axis, there can only be 2 neighbors");

    if (!axis.has_value()) {
        TT_FATAL(!(wrap_around_connection && neighbors.size() != 4), "Ring/Torus topology must have 4 neighbors");
    }

    return {neighbors, directions};
}

uint32_t get_linearized_index(const ttnn::MeshCoordinate& mesh_coordinate, const ttnn::MeshDeviceView& mesh_view) {
    return (mesh_coordinate[0] * mesh_view.num_cols()) + mesh_coordinate[1];
}

namespace detail {

// Direction of the single fabric hop src->dst, or nullopt if the two are not directly wired.
// A route always exists between any two nodes, so routability alone does not prove adjacency.
// TODO: belongs in tt::tt_fabric beside the pipeline_get_* wrappers it composes. The same sequence is
// open-coded in multiple CCL ops and fabric's own pipeline_builder.cpp.
std::optional<tt::tt_fabric::RoutingDirection> get_wired_hop_direction(
    tt::tt_fabric::FabricNodeId src, tt::tt_fabric::FabricNodeId dst) {
    const auto direction = tt::tt_fabric::pipeline_get_forwarding_direction(src, dst);
    if (!direction.has_value()) {
        return std::nullopt;
    }
    const auto neighbors = tt::tt_fabric::pipeline_get_chip_neighbors(src, *direction);
    const auto mesh_it = neighbors.find(*dst.mesh_id);
    if (mesh_it == neighbors.end() ||
        std::find(mesh_it->second.begin(), mesh_it->second.end(), dst.chip_id) == mesh_it->second.end()) {
        return std::nullopt;  // routable, but more than one hop away
    }
    return direction;
}

struct AxisEdge {
    tt::tt_fabric::FabricNodeId src;
    tt::tt_fabric::FabricNodeId dst;
    std::optional<tt::tt_fabric::RoutingDirection> direction;  // set iff the pair is wired
    bool is_wrap = false;
};

// Every consecutive coordinate pair along `axis`, for every line on that axis, plus the closing
// pair. Unwired pairs are returned with no direction so callers can tell a missing wrap edge
// (just a line) from a missing interior edge (not a chain at all).
std::vector<AxisEdge> axis_edges(const tt::tt_metal::distributed::MeshDevice& mesh_device, uint32_t axis) {
    const auto mesh_shape = mesh_device.get_view().shape();
    TT_FATAL(mesh_shape.dims() == 2, "Axis geometry requires a 2D mesh shape, got {}", mesh_shape);
    TT_FATAL(axis < 2, "Axis must be 0 or 1, got {}", axis);

    const uint32_t extent = mesh_shape[axis];
    // A 2-device axis is spanned by a single link; closing it would use that link twice, so it
    // is a line and never a ring. Matches the fabric's own is_genuine_torus_dim(dim > 2).
    const bool include_wrap = extent > 2;
    const uint32_t edges_per_line = extent < 2 ? 0 : (include_wrap ? extent : extent - 1);

    std::vector<AxisEdge> edges;
    for (uint32_t group = 0; group < mesh_shape[1 - axis]; group++) {
        for (uint32_t rank = 0; rank < edges_per_line; rank++) {
            const uint32_t next = (rank + 1) % extent;
            const auto src_coord = axis == 0 ? MeshCoordinate(rank, group) : MeshCoordinate(group, rank);
            const auto dst_coord = axis == 0 ? MeshCoordinate(next, group) : MeshCoordinate(group, next);
            const auto src = mesh_device.get_fabric_node_id(src_coord);
            const auto dst = mesh_device.get_fabric_node_id(dst_coord);
            edges.push_back(AxisEdge{src, dst, get_wired_hop_direction(src, dst), next == 0});
        }
    }
    return edges;
}

}  // namespace detail

AxisGeometry get_axis_geometry(const tt::tt_metal::distributed::MeshDevice& mesh_device, uint32_t axis) {
    const auto edges = detail::axis_edges(mesh_device, axis);

    AxisGeometry geometry;
    std::optional<tt::tt_fabric::RoutingDirection> axis_direction;
    size_t wrap_edges = 0;
    size_t wired_wrap_edges = 0;

    for (const auto& edge : edges) {
        wrap_edges += edge.is_wrap;
        if (!edge.direction.has_value()) {
            // A missing wrap edge just means this axis is a line. A missing interior edge means
            // consecutive coords are not neighbours at all, so no hop count describes the axis.
            if (!edge.is_wrap) {
                geometry.is_straight = false;
            }
            continue;
        }
        wired_wrap_edges += edge.is_wrap;
        if (!axis_direction.has_value()) {
            axis_direction = edge.direction;
        } else if (*axis_direction != *edge.direction) {
            geometry.is_straight = false;
        }
    }
    geometry.wrap_edge_exists = wrap_edges > 0 && wired_wrap_edges == wrap_edges;

    log_debug(
        tt::LogOp,
        "axis {}: is_straight {}, wrap_edge_exists {}",
        axis,
        geometry.is_straight,
        geometry.wrap_edge_exists);
    return geometry;
}

size_t get_num_links(const tt::tt_metal::distributed::MeshDevice& mesh_device, std::optional<size_t> cluster_axis) {
    ttsl::SmallVector<uint32_t> cluster_axes;
    if (cluster_axis.has_value()) {
        cluster_axes = {static_cast<uint32_t>(cluster_axis.value())};
    } else {
        cluster_axes = {0, 1};
    }

    size_t num_edges = 0;
    size_t num_available_routing_planes = std::numeric_limits<size_t>::max();
    for (const auto axis : cluster_axes) {
        for (const auto& edge : detail::axis_edges(mesh_device, axis)) {
            if (!edge.direction.has_value()) {
                continue;
            }
            num_edges++;
            // Both ends describe the same link, so take the lower count: reporting more links than
            // either side offers makes the op open a link index the fabric rejects.
            num_available_routing_planes = std::min(
                num_available_routing_planes, tt::tt_fabric::get_num_usable_routing_planes(edge.src, *edge.direction));
            if (const auto reverse = detail::get_wired_hop_direction(edge.dst, edge.src); reverse.has_value()) {
                num_available_routing_planes = std::min(
                    num_available_routing_planes, tt::tt_fabric::get_num_usable_routing_planes(edge.dst, *reverse));
            }
        }
    }

    if (num_edges == 0) {
        return 1;  // nothing to measure (single device, or an axis of extent 1)
    }
    log_debug(tt::LogOp, "num_available_routing_planes: {}", num_available_routing_planes);
    if (num_available_routing_planes == 0 || num_available_routing_planes == std::numeric_limits<size_t>::max()) {
        log_warning(tt::LogOp, "Failed to discover available ethernet links; falling back to 1 link");
        num_available_routing_planes = 1;
    }
    return num_available_routing_planes;
}

}  // namespace ttnn::operations::ccl::common
