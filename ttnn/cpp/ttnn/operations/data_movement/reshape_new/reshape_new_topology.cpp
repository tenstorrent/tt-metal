// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_new/reshape_new_topology.hpp"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

#include <tt_stl/small_vector.hpp>

namespace ttnn::operations::data_movement::reshape_new {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using tt::tt_metal::distributed::MeshMapperConfig;

std::optional<int> remap_shard_dim(
    int dim, const tt::tt_metal::Shape& old_shape, const tt::tt_metal::Shape& new_shape) {
    const int old_rank = static_cast<int>(old_shape.rank());
    const int new_rank = static_cast<int>(new_shape.rank());
    const int normalized = dim < 0 ? dim + old_rank : dim;
    if (normalized < 0 || normalized >= old_rank) {
        return std::nullopt;
    }

    uint64_t prefix = 1;
    for (int i = 0; i < normalized; ++i) {
        prefix *= old_shape[i];
    }
    uint64_t running = 1;
    for (int i = 0; i < new_rank && running <= prefix; ++i) {
        if (running == prefix && new_shape[i] > 1) {
            return dim < 0 ? i - new_rank : i;
        }
        running *= new_shape[i];
    }
    return std::nullopt;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

tt::tt_metal::TensorTopology remap_topology(
    const tt::tt_metal::TensorTopology& topology,
    const tt::tt_metal::Shape& old_shape,
    const tt::tt_metal::Shape& new_shape) {
    using CMAKE_UNIQUE_NAMESPACE::MeshMapperConfig;
    if (old_shape == new_shape || old_shape.volume() == 0) {
        return topology;
    }
    ttsl::SmallVector<MeshMapperConfig::Placement> placements;
    placements.reserve(topology.placements().size());
    for (const auto& placement : topology.placements()) {
        const auto* shard = std::get_if<MeshMapperConfig::Shard>(&placement);
        if (shard == nullptr) {
            placements.push_back(placement);
            continue;
        }
        const auto remapped = CMAKE_UNIQUE_NAMESPACE::remap_shard_dim(shard->dim, old_shape, new_shape);
        placements.push_back(
            remapped.has_value() ? MeshMapperConfig::Placement{MeshMapperConfig::Shard{*remapped}} : placement);
    }
    return tt::tt_metal::TensorTopology(topology.distribution_shape(), std::move(placements), topology.mesh_coords());
}

}  // namespace ttnn::operations::data_movement::reshape_new
