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

size_t get_num_links(const tt::tt_metal::distributed::MeshDevice& mesh_device, std::optional<size_t> cluster_axis) {
    const auto mesh_shape = mesh_device.get_view().shape();

    ttsl::SmallVector<size_t> cluster_axes;
    if (cluster_axis.has_value()) {
        cluster_axes = {cluster_axis.value()};
    } else {
        cluster_axes = {0, 1};
    }

    std::optional<size_t> num_links;
    for (const auto axis : cluster_axes) {
        // The op runs on every row (or column) along the axis, and they can be wired differently.
        // All of them must be able to open the link, so take the lowest count. Hops owned by
        // another host report 0 and are skipped, so this counts what this host can see.
        for (uint32_t row_or_col = 0; row_or_col < mesh_shape[1 - axis]; row_or_col++) {
            const size_t planes =
                tt::tt_fabric::experimental::get_number_of_available_routing_planes(mesh_device, axis, row_or_col);
            if (planes > 0) {
                num_links = num_links.has_value() ? std::min(*num_links, planes) : planes;
            }
        }
    }

    if (!num_links.has_value()) {
        log_warning(tt::LogOp, "Failed to discover available ethernet links; falling back to 1 link");
        return 1;
    }
    log_debug(tt::LogOp, "num_links: {}", *num_links);
    return *num_links;
}

}  // namespace ttnn::operations::ccl::common
