// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal {
struct TensorTopology {
    tt::tt_metal::distributed::MeshShape mesh_shape = tt::tt_metal::distributed::MeshShape{1};
    std::vector<tt::tt_metal::distributed::MeshCoordinate> mesh_coords = {tt::tt_metal::distributed::MeshCoordinate{0}};

    tt::tt_metal::distributed::MeshCoordinate get_neighbor(
        const tt::tt_metal::distributed::MeshCoordinate& coord, int32_t offset, int32_t dim) const;

    tt::tt_metal::distributed::MeshCoordinate get_next_neighbor(
        const tt::tt_metal::distributed::MeshCoordinate& coord, int32_t dim) const;

    tt::tt_metal::distributed::MeshCoordinate get_prev_neighbor(
        const tt::tt_metal::distributed::MeshCoordinate& coord, int32_t dim) const;

    tt::tt_metal::distributed::MeshCoordinate get_device_coord(
        const tt::tt_metal::distributed::MeshCoordinate& coord) const;
};

}  // namespace tt::tt_metal
