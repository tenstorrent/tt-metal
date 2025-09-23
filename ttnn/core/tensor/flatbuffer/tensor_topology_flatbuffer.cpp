// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_topology_flatbuffer.hpp"

#include <flatbuffers/flatbuffers.h>

#include <cstdint>
#include <vector>
#include <variant>

#include "mesh_shape_generated.h"
#include <tt-metalium/serialized_descriptors/mesh_coordinate_generated.h>

namespace ttnn {

// Local helpers reusing conversions in tensor_flatbuffer.cpp would create a circular dep.
// Re-implement the MeshShape and MeshCoordinate conversions here.
// Do we want to do it this way? Or move the TensorTopology conversions to tensor_flatbuffer.cpp so we can reuse the
// below conversions?
namespace {

inline flatbuffers::Offset<flatbuffer::MeshShape> to_flatbuffer_mesh_shape(
    const tt::tt_metal::distributed::MeshShape& shape, flatbuffers::FlatBufferBuilder& builder) {
    auto dimensions_vector = builder.CreateVector(std::vector<uint32_t>(shape.cbegin(), shape.cend()));
    return flatbuffer::CreateMeshShape(builder, dimensions_vector);
}

inline tt::tt_metal::distributed::MeshShape from_flatbuffer_mesh_shape(const flatbuffer::MeshShape* shape) {
    return tt::tt_metal::distributed::MeshShape(
        std::vector<uint32_t>(shape->dimensions()->begin(), shape->dimensions()->end()));
}

inline flatbuffers::Offset<tt::tt_metal::distributed::flatbuffer::MeshCoordinate> to_flatbuffer_mesh_coord(
    const tt::tt_metal::distributed::MeshCoordinate& coord, flatbuffers::FlatBufferBuilder& builder) {
    auto values_vector = builder.CreateVector(std::vector<uint32_t>(coord.coords().begin(), coord.coords().end()));
    return tt::tt_metal::distributed::flatbuffer::CreateMeshCoordinate(builder, values_vector);
}

inline tt::tt_metal::distributed::MeshCoordinate from_flatbuffer_mesh_coord(
    const tt::tt_metal::distributed::flatbuffer::MeshCoordinate* coord) {
    return tt::tt_metal::distributed::MeshCoordinate(
        std::vector<uint32_t>(coord->values()->begin(), coord->values()->end()));
}

}  // namespace

flatbuffers::Offset<ttnn::flatbuffer::TensorTopology> to_flatbuffer(
    const tt::tt_metal::TensorTopology& topology, flatbuffers::FlatBufferBuilder& builder) {
    auto dist_shape_offset = to_flatbuffer_mesh_shape(topology.distribution_shape(), builder);

    std::vector<flatbuffers::Offset<ttnn::flatbuffer::MeshMapperPlacement>> placement_offsets;
    placement_offsets.reserve(topology.placements().size());
    for (const auto& placement_variant : topology.placements()) {
        ttnn::flatbuffer::MeshMapperPlacementType type = ttnn::flatbuffer::MeshMapperPlacementType::Replicate;
        int32_t tensor_dim = -1;

        if (std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Replicate>(placement_variant)) {
            type = ttnn::flatbuffer::MeshMapperPlacementType::Replicate;
        } else {
            const auto& shard = std::get<tt::tt_metal::distributed::MeshMapperConfig::Shard>(placement_variant);
            type = ttnn::flatbuffer::MeshMapperPlacementType::Shard;
            tensor_dim = shard.dim;
        }

        placement_offsets.push_back(ttnn::flatbuffer::CreateMeshMapperPlacement(builder, type, tensor_dim));
    }
    auto placements_vec = builder.CreateVector(placement_offsets);

    std::vector<flatbuffers::Offset<tt::tt_metal::distributed::flatbuffer::MeshCoordinate>> coord_offsets;
    coord_offsets.reserve(topology.mesh_coords().size());
    for (const auto& coord : topology.mesh_coords()) {
        coord_offsets.push_back(to_flatbuffer_mesh_coord(coord, builder));
    }
    auto mesh_coords_vec = builder.CreateVector(coord_offsets);

    return ttnn::flatbuffer::CreateTensorTopology(builder, dist_shape_offset, placements_vec, mesh_coords_vec);
}

tt::tt_metal::TensorTopology from_flatbuffer(const ttnn::flatbuffer::TensorTopology* fb_topology) {
    TT_FATAL(fb_topology != nullptr, "TensorTopology flatbuffer pointer must not be null");

    const auto* fb_dist_shape = fb_topology->distribution_shape();
    TT_FATAL(fb_dist_shape != nullptr, "distribution_shape is required in TensorTopology");
    auto dist_shape = from_flatbuffer_mesh_shape(fb_dist_shape);

    tt::stl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements;
    if (const auto* fb_placements = fb_topology->placements()) {
        placements.reserve(fb_placements->size());
        for (const auto* p : *fb_placements) {
            TT_FATAL(p != nullptr, "MeshMapperPlacement element must not be null");
            if (p->type() == ttnn::flatbuffer::MeshMapperPlacementType::Replicate) {
                placements.emplace_back(tt::tt_metal::distributed::MeshMapperConfig::Replicate{});
            } else if (p->type() == ttnn::flatbuffer::MeshMapperPlacementType::Shard) {
                placements.emplace_back(tt::tt_metal::distributed::MeshMapperConfig::Shard{.dim = p->tensor_dim()});
            } else {
                TT_THROW("Unknown MeshMapperPlacementType");
            }
        }
    }

    std::vector<tt::tt_metal::distributed::MeshCoordinate> mesh_coords;
    if (const auto* fb_coords = fb_topology->mesh_coords()) {
        mesh_coords.reserve(fb_coords->size());
        for (const auto* c : *fb_coords) {
            TT_FATAL(c != nullptr, "MeshCoordinate element must not be null");
            mesh_coords.push_back(from_flatbuffer_mesh_coord(c));
        }
    }

    return tt::tt_metal::TensorTopology(std::move(dist_shape), std::move(placements), std::move(mesh_coords));
}

}  // namespace ttnn
