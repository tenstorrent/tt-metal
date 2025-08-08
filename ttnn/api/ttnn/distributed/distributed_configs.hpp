// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <ostream>
#include <variant>
#include <tt_stl/small_vector.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::distributed {

struct MeshMapperConfig {
    // Specifies the tensor should be replicated across devices.
    struct Replicate {
        bool operator==(const Replicate&) const = default;
    };

    // Specifies the tensor should be sharded along the specified dimension.
    struct Shard {
        int dim = 0;
        bool operator==(const Shard&) const = default;
    };

    // Specifies placements for each dimension of the shape.
    // The size of `placements` must match the dimensions of the shape.
    //
    // For example, sharding a 2x8 tensor over 2x2 mesh with {Replicate(), Shard{1}} will yield the following result:
    //
    //    Input Tensor [2, 8]:
    // +----+----+----+----+----+----+---+-----+
    // |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |
    // |----+----+----+----+----+----+---+-----+
    // |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
    // +----+----+----+----+----+----+---+-----+
    //
    //    Shape [2, 2]:
    // +-------+-------+
    // | (0,0) | (0,1) |
    // +-------+-------+
    // | (1,0) | (1,1) |
    // +-------+-------+
    //
    // Distributed Tensor on Mesh (placements = {Replicate{}, Shard{1}}):
    //
    // +-----------------------------+-----------------------------+
    // |     (0,0)                   |     (0,1)                   |
    // |    +----+----+----+----+    |    +----+----+----+----+    |
    // |    |  0 |  1 |  2 |  3 |    |    |  4 |  5 |  6 |  7 |    |
    // |    +----+----+----+----+    |    +----+----+----+----+    |
    // |    |  8 |  9 | 10 | 11 |    |    | 12 | 13 | 14 | 15 |    |
    // |    +----+----+----+----+    |    +----+----+----+----+    |
    // +-----------------------------+-----------------------------+
    // |     (1,0)                   |     (1,1)                   |
    // |    +----+----+----+----+    |    +----+----+----+----+    |
    // |    |  0 |  1 |  2 |  3 |    |    |  4 |  5 |  6 |  7 |    |
    // |    +----+----+----+----+    |    +----+----+----+----+    |
    // |    |  8 |  9 | 10 | 11 |    |    | 12 | 13 | 14 | 15 |    |
    // |    +----+----+----+----+    |    +----+----+----+----+    |
    // +-----------------------------+-----------------------------+
    //

    using Placement = std::variant<Replicate, Shard>;
    tt::stl::SmallVector<Placement> placements;

    // If provided, the sharding will be performed according to this shape, but re-mapped to the mesh device shape in
    // either row-major order, or preserving the original coordinates (if the shape fits within the mesh device
    // entirely).
    std::optional<tt::tt_metal::distributed::MeshShape> mesh_shape_override = std::nullopt;
};

bool operator==(const MeshMapperConfig::Placement& lhs, const MeshMapperConfig::Placement& rhs);
std::ostream& operator<<(std::ostream& os, const MeshMapperConfig::Placement& placement);
std::ostream& operator<<(std::ostream& os, const MeshMapperConfig& config);

struct MeshComposerConfig {
    // Specifies dimension of the tensor to concatenate.
    tt::stl::SmallVector<int> dims;

    // If provided, the concatenation will be performed according to this shape, but re-mapped to the mesh device shape
    // in either row-major order, or preserving the original coordinates (if the shape fits within the mesh device
    // entirely).
    std::optional<tt::tt_metal::distributed::MeshShape> mesh_shape_override = std::nullopt;
};

std::ostream& operator<<(std::ostream& os, const MeshComposerConfig& config);

}  // namespace tt::tt_metal::distributed
