// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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
    auto src_device = mesh_view.get_device(mesh_coordinate);

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

    TT_FATAL(neighbors.size() > 0, "No neighbors found");
    TT_FATAL(!(axis.has_value() && neighbors.size() > 2), "Along a single axis, there can only be 2 neighbors");

    if (!axis.has_value()) {
        TT_FATAL(!(wrap_around_connection && neighbors.size() != 4), "Ring/Torus topology must have 4 neighbors");
    }

    return {neighbors, directions};
}

uint32_t get_linearized_index(const ttnn::MeshCoordinate& mesh_coordinate, const ttnn::MeshDeviceView& mesh_view) {
    return mesh_coordinate[0] * mesh_view.num_cols() + mesh_coordinate[1];
}

}  // namespace ttnn::operations::ccl::common
