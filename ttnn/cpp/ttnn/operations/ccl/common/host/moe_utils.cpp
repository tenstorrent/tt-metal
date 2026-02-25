// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <limits>
#include <utility>
#include <vector>

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

// TODO: once #27196 is fixed we can remove the is_mesh_mmio_capable check
size_t get_num_links(const tt::tt_metal::distributed::MeshDevice& mesh_device, std::optional<size_t> cluster_axis) {
    auto mesh_range = tt::tt_metal::distributed::MeshCoordinateRange(mesh_device.shape());
    auto mesh_range_set = tt::tt_metal::distributed::MeshCoordinateRangeSet(mesh_range);
    const auto& mesh_view = mesh_device.get_view();
    auto mesh_shape = mesh_view.shape();
    auto topology = tt::tt_fabric::get_fabric_topology();

    constexpr std::array<std::array<tt::tt_fabric::RoutingDirection, 2>, 2> directions = {
        {{tt::tt_fabric::RoutingDirection::N, tt::tt_fabric::RoutingDirection::S},
         {tt::tt_fabric::RoutingDirection::W, tt::tt_fabric::RoutingDirection::E}}};

    ttnn::SmallVector<size_t> cluster_axes;
    if (cluster_axis.has_value()) {
        cluster_axes = {cluster_axis.value()};
    } else {
        cluster_axes = {0, 1};
    }

    auto positive_direction = [&](tt::tt_fabric::RoutingDirection direction) {
        return direction == tt::tt_fabric::RoutingDirection::E || direction == tt::tt_fabric::RoutingDirection::S;
    };
    [[maybe_unused]] auto negative_direction = [&](tt::tt_fabric::RoutingDirection direction) {
        return direction == tt::tt_fabric::RoutingDirection::W || direction == tt::tt_fabric::RoutingDirection::N;
    };

    auto applicable_to_coord = [&](const MeshCoordinate& coord,
                                   size_t cluster_axis,
                                   size_t /*axis_size*/,
                                   tt::tt_fabric::RoutingDirection direction) -> bool {
        auto boundary_mode = detail::get_boundary_mode(topology);
        int offset = positive_direction(direction) ? 1 : -1;
        auto neighbor = coord.get_neighbor(mesh_shape, offset, cluster_axis, boundary_mode);
        return neighbor.has_value();
    };

    size_t num_available_routing_planes = std::numeric_limits<size_t>::max();
    bool is_mesh_mmio_capable = true;
    for (const auto& coord : mesh_range_set.coords()) {
        // TODO: remove usage of get_device, need api to return correct routing planes accounting for fast dispatch
        // usage should only be active for T3K
        if (mesh_device.is_local(coord)) {
            auto* device = mesh_device.get_device(coord);
            bool is_mmio_capable = device->is_mmio_capable();
            is_mesh_mmio_capable &= is_mmio_capable;
            log_debug(tt::LogOp, "mesh_coordinate: {}, is_mmio_capable: {}", coord, is_mmio_capable);
        }
        const auto fabric_node_id = mesh_device.get_fabric_node_id(coord);

        for (const auto axis : cluster_axes) {
            for (const auto direction : directions[axis]) {
                if (applicable_to_coord(coord, axis, mesh_shape[axis], direction)) {
                    auto planes_in_direction =
                        tt::tt_fabric::get_num_available_routing_planes_in_direction(fabric_node_id, direction);
                    // if the device is not mmio capable then one link on some axis will be unavailable
                    // ideally we only subtract if we're targetting that cluster axis, but we don't have access to that
                    // information here to be safe, we subtract 1 regardless of the axis when the axis is not available
                    log_debug(
                        tt::LogOp,
                        "fabric_node_id: {}, direction: {}, planes_in_direction: {}",
                        fabric_node_id,
                        direction,
                        planes_in_direction);
                    num_available_routing_planes = std::min(num_available_routing_planes, planes_in_direction);
                }
            }
        }
    }
    if (!is_mesh_mmio_capable && num_available_routing_planes > 1) {
        num_available_routing_planes -= 1;
    }
    log_debug(tt::LogOp, "num_available_routing_planes without max logic: {}", num_available_routing_planes);
    return std::max(num_available_routing_planes, 1ul);
}

}  // namespace ttnn::operations::ccl::common
