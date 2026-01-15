// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/distributed_configs.hpp"
#include <tt_stl/overloaded.hpp>

namespace tt::tt_metal::distributed {

std::ostream& operator<<(std::ostream& os, const MeshMapperConfig::Placement& placement) {
    std::visit(
        tt::stl::overloaded{
            [&](const MeshMapperConfig::Replicate& /*replicate*/) { os << "PlacementReplicate()"; },
            [&](const MeshMapperConfig::Shard& shard) { os << "PlacementShard(" << shard.dim << ")"; },
        },
        placement);
    return os;
}

std::ostream& operator<<(std::ostream& os, const MeshMapperConfig& config) {
    os << "MeshMapperConfig(";
    os << "placements: [";
    for (int i = 0; i < config.placements.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << config.placements[i];
    }
    os << "]";
    if (config.mesh_shape_override.has_value()) {
        os << ", mesh_shape_override=" << *config.mesh_shape_override;
    }
    os << ")";
    return os;
}

bool operator==(const MeshMapperConfig::Placement& lhs, const MeshMapperConfig::Placement& rhs) {
    return std::visit(
        tt::stl::overloaded{
            [&](const MeshMapperConfig::Replicate& l, const MeshMapperConfig::Replicate& r) { return l == r; },
            [&](const MeshMapperConfig::Shard& l, const MeshMapperConfig::Shard& r) { return l == r; },
            [&](const auto&, const auto&) { return false; },  // Different types are never equal
        },
        lhs,
        rhs);
}

std::ostream& operator<<(std::ostream& os, const MeshComposerConfig& config) {
    os << "MeshComposerConfig(";
    os << "dims: [";
    for (int i = 0; i < config.dims.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << config.dims[i];
    }
    os << "]";
    if (config.mesh_shape_override.has_value()) {
        os << ", mesh_shape_override=" << *config.mesh_shape_override;
    }
    os << ")";
    return os;
}

}  // namespace tt::tt_metal::distributed
