// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_coord.hpp>

#include "ttnn/distributed/distributed_configs.hpp"

namespace tt::tt_metal {
class TensorTopology {
public:
    TensorTopology() :
        mesh_shape_(tt::tt_metal::distributed::MeshShape{1}),
        placements_({tt::tt_metal::distributed::MeshMapperConfig::Replicate{}}),
        mesh_coords_({tt::tt_metal::distributed::MeshCoordinate{0}}) {}

    TensorTopology(
        tt::tt_metal::distributed::MeshShape mesh_shape,
        tt::stl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements,
        std::vector<tt::tt_metal::distributed::MeshCoordinate> mesh_coords) :
        mesh_shape_(std::move(mesh_shape)), placements_(std::move(placements)), mesh_coords_(std::move(mesh_coords)) {}

    const tt::tt_metal::distributed::MeshShape& mesh_shape() const { return mesh_shape_; }
    const tt::stl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>& placements() const {
        return placements_;
    }
    const std::vector<tt::tt_metal::distributed::MeshCoordinate>& mesh_coords() const { return mesh_coords_; }

    tt::tt_metal::distributed::MeshCoordinate get_neighbor(
        const tt::tt_metal::distributed::MeshCoordinate& coord, int32_t offset, int32_t dim) const;

    tt::tt_metal::distributed::MeshCoordinate get_next_neighbor(
        const tt::tt_metal::distributed::MeshCoordinate& coord, int32_t dim) const;

    tt::tt_metal::distributed::MeshCoordinate get_prev_neighbor(
        const tt::tt_metal::distributed::MeshCoordinate& coord, int32_t dim) const;

    tt::tt_metal::distributed::MeshCoordinate get_device_coord(
        const tt::tt_metal::distributed::MeshCoordinate& coord) const;

private:
    tt::tt_metal::distributed::MeshShape mesh_shape_;
    tt::stl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements_;
    std::vector<tt::tt_metal::distributed::MeshCoordinate> mesh_coords_;
};

}  // namespace tt::tt_metal
