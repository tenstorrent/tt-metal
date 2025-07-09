// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal {
struct TopologyConfig {
    // TODO: Merge with MeshMapperConfig::Placement
    // Specifies the tensor should be replicated across devices.
    struct Replicate {};

    // Specifies the tensor should be sharded along the specified dimension.
    struct Shard {
        int dim = 0;
    };
    using Placement = std::variant<Replicate, Shard>;

    tt::tt_metal::distributed::MeshShape mesh_shape;
    std::vector<tt::tt_metal::distributed::MeshCoordinate> mesh_coords;
    tt::stl::SmallVector<Placement> placements;

    tt::tt_metal::distributed::MeshCoordinate get_neighbor(
        const tt::tt_metal::distributed::MeshCoordinate& coord) const;
};

}  // namespace tt::tt_metal
