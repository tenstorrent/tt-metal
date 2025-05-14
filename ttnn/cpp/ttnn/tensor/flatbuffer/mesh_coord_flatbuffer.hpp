#pragma once

#include <flatbuffers/flatbuffers.h>
#include <tt-metalium/mesh_coord.hpp>

#include "mesh_shape_generated.h"
#include "mesh_coordinate_generated.h"

namespace ttnn {

flatbuffers::Offset<flatbuffer::MeshCoordinate> to_flatbuffer(
    const tt::tt_metal::distributed::MeshCoordinate& coord, flatbuffers::FlatBufferBuilder& builder);
flatbuffers::Offset<flatbuffer::MeshShape> to_flatbuffer(
    const tt::tt_metal::distributed::MeshShape& shape, flatbuffers::FlatBufferBuilder& builder);

tt::tt_metal::distributed::MeshCoordinate from_flatbuffer(const flatbuffer::MeshCoordinate* coord);
tt::tt_metal::distributed::MeshShape from_flatbuffer(const flatbuffer::MeshShape* shape);

}  // namespace ttnn
