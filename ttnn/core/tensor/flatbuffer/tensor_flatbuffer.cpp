// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/flatbuffer/tensor_flatbuffer.hpp"
#include "tensor/flatbuffer/tensor_spec_flatbuffer.hpp"

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <flatbuffers/flatbuffers.h>

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/distributed/tensor_topology.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include "mesh_shape_generated.h"
#include <tt-metalium/serialized_descriptors/mesh_coordinate_generated.h>
#include "tensor_generated.h"

#include <vector>
#include <cstdint>
#include <unordered_map>
#include <algorithm>

namespace ttnn {
namespace {

flatbuffers::Offset<tt::tt_metal::distributed::flatbuffer::MeshCoordinate> to_flatbuffer(
    const tt::tt_metal::distributed::MeshCoordinate& coord, flatbuffers::FlatBufferBuilder& builder) {
    auto values_vector = builder.CreateVector(std::vector<uint32_t>(coord.coords().begin(), coord.coords().end()));
    return tt::tt_metal::distributed::flatbuffer::CreateMeshCoordinate(builder, values_vector);
}

tt::tt_metal::distributed::MeshCoordinate from_flatbuffer(
    const tt::tt_metal::distributed::flatbuffer::MeshCoordinate* coord) {
    return tt::tt_metal::distributed::MeshCoordinate(
        std::vector<uint32_t>(coord->values()->begin(), coord->values()->end()));
}

flatbuffers::Offset<flatbuffer::MeshShape> to_flatbuffer(
    const tt::tt_metal::distributed::MeshShape& shape, flatbuffers::FlatBufferBuilder& builder) {
    auto dimensions_vector = builder.CreateVector(std::vector<uint32_t>(shape.cbegin(), shape.cend()));
    return flatbuffer::CreateMeshShape(builder, dimensions_vector);
}

tt::tt_metal::distributed::MeshShape from_flatbuffer(const flatbuffer::MeshShape* shape) {
    return tt::tt_metal::distributed::MeshShape(
        std::vector<uint32_t>(shape->dimensions()->begin(), shape->dimensions()->end()));
}

tt::tt_metal::HostBuffer create_host_buffer_from_bytes(
    uint64_t size_bytes,
    const TensorSpec& spec,
    tt::stl::Span<std::byte> data,
    const tt::tt_metal::MemoryPin& memory_pin) {
    switch (spec.data_type()) {
        case tt::tt_metal::DataType::UINT32:
        case tt::tt_metal::DataType::BFLOAT8_B:
        case tt::tt_metal::DataType::BFLOAT4_B: {
            tt::stl::Span<uint32_t> typed_span(reinterpret_cast<uint32_t*>(data.data()), size_bytes / sizeof(uint32_t));
            return tt::tt_metal::HostBuffer(typed_span, memory_pin);
        }
        case tt::tt_metal::DataType::INT32: {
            tt::stl::Span<int32_t> typed_span(reinterpret_cast<int32_t*>(data.data()), size_bytes / sizeof(int32_t));
            return tt::tt_metal::HostBuffer(typed_span, memory_pin);
        }
        case tt::tt_metal::DataType::UINT8: {
            tt::stl::Span<uint8_t> typed_span(reinterpret_cast<uint8_t*>(data.data()), size_bytes / sizeof(uint8_t));
            return tt::tt_metal::HostBuffer(typed_span, memory_pin);
        }
        case tt::tt_metal::DataType::UINT16: {
            tt::stl::Span<uint16_t> typed_span(reinterpret_cast<uint16_t*>(data.data()), size_bytes / sizeof(uint16_t));
            return tt::tt_metal::HostBuffer(typed_span, memory_pin);
        }
        case tt::tt_metal::DataType::FLOAT32: {
            tt::stl::Span<float> typed_span(reinterpret_cast<float*>(data.data()), size_bytes / sizeof(float));
            return tt::tt_metal::HostBuffer(typed_span, memory_pin);
        }
        case tt::tt_metal::DataType::BFLOAT16: {
            tt::stl::Span<bfloat16> typed_span(reinterpret_cast<bfloat16*>(data.data()), size_bytes / sizeof(bfloat16));
            return tt::tt_metal::HostBuffer(typed_span, memory_pin);
        }
        case tt::tt_metal::DataType::INVALID: TT_THROW("Unsupported DataType");
    }
    TT_THROW("Unreachable");
}

flatbuffers::Offset<ttnn::flatbuffer::TensorTopology> to_flatbuffer(
    const tt::tt_metal::TensorTopology& topology, flatbuffers::FlatBufferBuilder& builder) {
    auto dist_shape_offset = to_flatbuffer(topology.distribution_shape(), builder);

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
        coord_offsets.push_back(to_flatbuffer(coord, builder));
    }
    auto mesh_coords_vec = builder.CreateVector(coord_offsets);

    return ttnn::flatbuffer::CreateTensorTopology(builder, dist_shape_offset, placements_vec, mesh_coords_vec);
}

tt::tt_metal::TensorTopology from_flatbuffer(const ttnn::flatbuffer::TensorTopology* fb_topology) {
    TT_FATAL(fb_topology != nullptr, "TensorTopology flatbuffer pointer must not be null");

    const auto* fb_dist_shape = fb_topology->distribution_shape();
    TT_FATAL(fb_dist_shape != nullptr, "distribution_shape is required in TensorTopology");
    auto dist_shape = from_flatbuffer(fb_dist_shape);

    ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements;
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
            mesh_coords.push_back(from_flatbuffer(c));
        }
    }

    return tt::tt_metal::TensorTopology(dist_shape, placements, mesh_coords);
}

}  // namespace

flatbuffers::Offset<ttnn::flatbuffer::Tensor> to_flatbuffer(
    const Tensor& tensor, flatbuffers::FlatBufferBuilder& builder, std::vector<tt::tt_metal::HostBuffer>& buffers) {
    TT_FATAL(buffers.empty(), "Buffers vector must be empty");
    TT_FATAL(!is_device_tensor(tensor), "Device tensors are not supported in flatbuffer serialization");

    auto tensor_spec_offset = ttnn::to_flatbuffer(tensor.tensor_spec(), builder);

    const auto& host_storage = tensor.host_storage();
    const auto& topology = tensor.tensor_topology();
    const auto& placements = topology.placements();
    const auto& dist_shape = topology.distribution_shape();

    // Determine if we should collapse replicate axes in the stored representation.
    // This makes the serialized file topology-portable: a tensor with [R, S(-1)] on a 2x4 mesh
    // is stored as [1,4] with 4 shards, loadable on any mesh with 4+ columns.
    // Condition: buffer dimensionality must match placements (excludes 1D overrides on ND meshes).
    const bool collapse_replicates =
        host_storage.buffer().shape().dims() == placements.size() &&
        std::any_of(placements.begin(), placements.end(), [](const auto& p) {
            return std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Replicate>(p);
        });

    // Build shard entries: pairs of (physical_coord_for_data_lookup, coord_to_store).
    struct ShardEntry {
        tt::tt_metal::distributed::MeshCoordinate phys_coord;
        tt::tt_metal::distributed::MeshCoordinate stored_coord;
    };
    std::vector<ShardEntry> shard_entries;

    tt::tt_metal::distributed::MeshShape stored_mesh_shape = host_storage.buffer().shape();

    if (collapse_replicates) {
        // Compute reduced mesh shape: replicate axes collapse to 1, shard axes keep their size.
        std::vector<uint32_t> reduced_dims;
        for (size_t i = 0; i < placements.size(); i++) {
            reduced_dims.push_back(
                std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Replicate>(placements[i])
                    ? 1
                    : dist_shape[i]);
        }
        stored_mesh_shape = tt::tt_metal::distributed::MeshShape(reduced_dims);

        // Walk distribution coords in lockstep with topology mesh_coords.
        // Keep only the canonical representative for each projected coord
        // (the one where all replicate-axis components are 0).
        const auto& mesh_coords = topology.mesh_coords();
        size_t coord_idx = 0;
        for (const auto& dist_coord : tt::tt_metal::distributed::MeshCoordinateRange(dist_shape)) {
            if (coord_idx >= mesh_coords.size()) {
                break;
            }
            const auto& phys_coord = mesh_coords[coord_idx++];

            // Skip non-canonical entries: any replicate-axis component > 0 means this is a replica.
            bool is_canonical = true;
            for (size_t i = 0; i < dist_coord.dims(); i++) {
                if (std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Replicate>(placements[i]) &&
                    dist_coord[i] != 0) {
                    is_canonical = false;
                    break;
                }
            }
            if (!is_canonical) {
                continue;
            }

            // For canonical entries, the distribution coord already has replicate axes at 0,
            // so it serves directly as the stored coord in the reduced space.
            std::vector<uint32_t> projected;
            for (size_t i = 0; i < dist_coord.dims(); i++) {
                projected.push_back(dist_coord[i]);
            }
            shard_entries.push_back({phys_coord, tt::tt_metal::distributed::MeshCoordinate(projected)});
        }
    } else {
        // No reduction: store all populated shard coords as-is.
        for (const auto& coord : host_storage.buffer().shard_coords()) {
            if (host_storage.buffer().get_shard(coord).has_value()) {
                shard_entries.push_back({coord, coord});
            }
        }
    }

    // Create shard flatbuffer entries.
    std::vector<flatbuffers::Offset<ttnn::flatbuffer::TensorShard>> shards_vector;
    std::unordered_map<const std::byte*, uint64_t> buffer_to_offset;
    uint64_t next_buffer_offset = 0;
    std::vector<tt::tt_metal::distributed::MeshCoordinate> stored_coords;

    for (const auto& [phys_coord, stored_coord] : shard_entries) {
        if (const auto& buffer = host_storage.buffer().get_shard(phys_coord); buffer.has_value()) {
            const auto* buffer_address = buffer->view_bytes().data();
            const std::size_t buffer_size = buffer->view_bytes().size();

            uint64_t shard_buffer_offset = next_buffer_offset;
            if (auto [it, inserted] = buffer_to_offset.try_emplace(buffer_address, shard_buffer_offset); inserted) {
                next_buffer_offset += buffer_size;
                buffers.push_back(*buffer);
            } else {
                shard_buffer_offset = it->second;
            }

            auto inline_storage = ttnn::flatbuffer::InlineFileStorage(shard_buffer_offset, buffer_size);
            auto mesh_coord_offset = to_flatbuffer(stored_coord, builder);

            auto shard_offset = ttnn::flatbuffer::CreateTensorShard(
                builder,
                ttnn::flatbuffer::TensorBuffer::InlineFileStorage,
                builder.CreateStruct(inline_storage).Union(),
                mesh_coord_offset);

            shards_vector.push_back(shard_offset);
            stored_coords.push_back(stored_coord);
        }
    }
    auto shards = builder.CreateVector(shards_vector);

    auto mesh_shape_offset = to_flatbuffer(stored_mesh_shape, builder);

    // Store reduced topology if replicates were collapsed, otherwise store the original.
    auto stored_topology =
        collapse_replicates ? tt::tt_metal::TensorTopology(stored_mesh_shape, placements, stored_coords) : topology;
    auto topology_offset = to_flatbuffer(stored_topology, builder);

    auto tensor_offset =
        ttnn::flatbuffer::CreateTensor(builder, tensor_spec_offset, mesh_shape_offset, shards, topology_offset);

    return tensor_offset;
}

Tensor from_flatbuffer(
    const ttnn::flatbuffer::Tensor* fb_tensor,
    tt::stl::Span<std::byte> tensor_data,
    const tt::tt_metal::MemoryPin& memory_pin) {
    auto spec = ttnn::from_flatbuffer(fb_tensor->tensor_spec());

    const auto* mesh_shape = fb_tensor->mesh_shape();
    TT_FATAL(mesh_shape != nullptr, "Mesh shape is required for tensor");
    const tt::tt_metal::distributed::MeshShape ttnn_mesh_shape = from_flatbuffer(mesh_shape);

    auto distributed_buffer = tt::tt_metal::DistributedHostBuffer::create(ttnn_mesh_shape);
    for (size_t i = 0; i < fb_tensor->shards()->size(); ++i) {
        const auto* shard = fb_tensor->shards()->Get(i);

        const auto* inline_storage = shard->buffer_as<ttnn::flatbuffer::InlineFileStorage>();
        TT_FATAL(inline_storage != nullptr, "Only InlineFileStorage is supported in flatbuffer deserialization");

        const uint64_t offset = inline_storage->offset();
        const uint64_t size = inline_storage->size();

        tt::tt_metal::HostBuffer host_buffer = create_host_buffer_from_bytes(
            size, spec, tt::stl::Span<std::byte>(tensor_data.data() + offset, size), memory_pin);

        TT_FATAL(shard->mesh_coordinate() != nullptr, "Mesh coordinate is required for each shard");
        const auto coord = from_flatbuffer(shard->mesh_coordinate());
        distributed_buffer.emplace_shard(
            coord, [host_buffer = std::move(host_buffer)]() mutable { return std::move(host_buffer); });
    }

    // NOTE: Existing tensor cache files may not have a tensor topology.
    // Create tensor topology from flatbuffer if it exists, otherwise create a fully replicated topology.
    tt::tt_metal::HostStorage host_storage{std::move(distributed_buffer)};
    const auto* fb_topology = fb_tensor->tensor_topology();
    tt::tt_metal::TensorTopology topology =
        fb_topology != nullptr ? from_flatbuffer(fb_topology)
                               : tt::tt_metal::TensorTopology::create_fully_replicated_tensor_topology(ttnn_mesh_shape);

    return Tensor(std::move(host_storage), spec, std::move(topology));
}

}  // namespace ttnn
