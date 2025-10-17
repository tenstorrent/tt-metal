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
        distribution_shape_(tt::tt_metal::distributed::MeshShape{1}),
        placements_({tt::tt_metal::distributed::MeshMapperConfig::Replicate{}}),
        mesh_coords_({tt::tt_metal::distributed::MeshCoordinate{0, 0}}) {}

    TensorTopology(
        tt::tt_metal::distributed::MeshShape distribution_shape,
        tt::stl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements,
        std::vector<tt::tt_metal::distributed::MeshCoordinate> mesh_coords) :
        distribution_shape_(std::move(distribution_shape)),
        placements_(std::move(placements)),
        mesh_coords_(std::move(mesh_coords)) {}

    // Creates a tensor topology for a fully replicated tensor with 1D distribution shape.
    static TensorTopology create_fully_replicated_tensor_topology(
        const tt::tt_metal::distributed::MeshShape& mesh_shape);

    // Creates a tensor topology for a sharded tensor with 1D distribution shape.
    // `shard_dim` is the dimension to shard over (assumes sharded along dim 0 if not provided).
    static TensorTopology create_sharded_tensor_topology(
        const tt::tt_metal::distributed::MeshShape& mesh_shape, int shard_dim = 0);

    // Returns the shape that the original tensor was sharded over.
    const tt::tt_metal::distributed::MeshShape& distribution_shape() const { return distribution_shape_; }
    const tt::stl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>& placements() const {
        return placements_;
    }
    const std::vector<tt::tt_metal::distributed::MeshCoordinate>& mesh_coords() const { return mesh_coords_; }

    tt::tt_metal::distributed::MeshCoordinate get_neighbor(
        const tt::tt_metal::distributed::MeshCoordinate& tensor_coord, int32_t offset, int32_t dim) const;

    tt::tt_metal::distributed::MeshCoordinate get_next_neighbor(
        const tt::tt_metal::distributed::MeshCoordinate& tensor_coord, int32_t dim) const;

    tt::tt_metal::distributed::MeshCoordinate get_prev_neighbor(
        const tt::tt_metal::distributed::MeshCoordinate& tensor_coord, int32_t dim) const;

    // Returns the physical device coordinate for the given tensor coordinate
    tt::tt_metal::distributed::MeshCoordinate get_device_coord(
        const tt::tt_metal::distributed::MeshCoordinate& tensor_coord) const;

    // Returns the tensor coordinate for the given physical device coordinate
    // If no tensor coordinate corresponds to the given physical device coordinate, returns std::nullopt
    std::optional<tt::tt_metal::distributed::MeshCoordinate> get_tensor_coord(
        const tt::tt_metal::distributed::MeshCoordinate& device_coord) const;

private:
    tt::tt_metal::distributed::MeshShape distribution_shape_;
    tt::stl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements_;

    // Physical device coordinates
    std::vector<tt::tt_metal::distributed::MeshCoordinate> mesh_coords_;
};

bool operator==(const TensorTopology& lhs, const TensorTopology& rhs);
bool operator!=(const TensorTopology& lhs, const TensorTopology& rhs);

std::ostream& operator<<(std::ostream& os, const TensorTopology& tensor_topology);

}  // namespace tt::tt_metal
