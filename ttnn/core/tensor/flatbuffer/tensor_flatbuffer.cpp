// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/flatbuffer/tensor_flatbuffer.hpp"
#include "tensor/flatbuffer/tensor_spec_flatbuffer.hpp"

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <flatbuffers/flatbuffers.h>

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/distributed/tensor_topology.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/config.hpp"

#include "mesh_shape_generated.h"
#include <tt-metalium/serialized_descriptors/mesh_coordinate_generated.h>
#include "tensor_generated.h"

#include <vector>
#include <cstdint>
#include <cstring>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

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
    const tt::tt_metal::TensorSpec& spec,
    ttsl::Span<std::byte> data,
    const tt::tt_metal::MemoryPin& memory_pin) {
    switch (spec.data_type()) {
        case tt::tt_metal::DataType::UINT32:
        case tt::tt_metal::DataType::BFLOAT8_B:
        case tt::tt_metal::DataType::BFLOAT4_B: {
            ttsl::Span<uint32_t> typed_span(reinterpret_cast<uint32_t*>(data.data()), size_bytes / sizeof(uint32_t));
            return tt::tt_metal::HostBuffer(typed_span, memory_pin);
        }
        case tt::tt_metal::DataType::INT32: {
            ttsl::Span<int32_t> typed_span(reinterpret_cast<int32_t*>(data.data()), size_bytes / sizeof(int32_t));
            return tt::tt_metal::HostBuffer(typed_span, memory_pin);
        }
        case tt::tt_metal::DataType::INT8: {
            ttsl::Span<int8_t> typed_span(reinterpret_cast<int8_t*>(data.data()), size_bytes / sizeof(int8_t));
            return tt::tt_metal::HostBuffer(typed_span, memory_pin);
        }
        case tt::tt_metal::DataType::FP8_E4M3:
            TT_THROW("Flatbuffer load for DataType::FP8_E4M3 is not supported during tensor deserialization.");
        case tt::tt_metal::DataType::UINT8: {
            ttsl::Span<uint8_t> typed_span(reinterpret_cast<uint8_t*>(data.data()), size_bytes / sizeof(uint8_t));
            return tt::tt_metal::HostBuffer(typed_span, memory_pin);
        }
        case tt::tt_metal::DataType::UINT16: {
            ttsl::Span<uint16_t> typed_span(reinterpret_cast<uint16_t*>(data.data()), size_bytes / sizeof(uint16_t));
            return tt::tt_metal::HostBuffer(typed_span, memory_pin);
        }
        case tt::tt_metal::DataType::FLOAT32: {
            ttsl::Span<float> typed_span(reinterpret_cast<float*>(data.data()), size_bytes / sizeof(float));
            return tt::tt_metal::HostBuffer(typed_span, memory_pin);
        }
        case tt::tt_metal::DataType::BFLOAT16: {
            ttsl::Span<bfloat16> typed_span(reinterpret_cast<bfloat16*>(data.data()), size_bytes / sizeof(bfloat16));
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
    TT_FATAL(fb_topology != nullptr, "tt::tt_metal::TensorTopology flatbuffer pointer must not be null");

    const auto* fb_dist_shape = fb_topology->distribution_shape();
    TT_FATAL(fb_dist_shape != nullptr, "distribution_shape is required in tt::tt_metal::TensorTopology");
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

// Renders the topology for a diagnostic. Only evaluated on the failure path.
std::string describe(const tt::tt_metal::TensorTopology& topology) {
    std::ostringstream os;
    os << topology;
    return os.str();
}

}  // namespace

flatbuffers::Offset<ttnn::flatbuffer::Tensor> to_flatbuffer(
    const Tensor& tensor, flatbuffers::FlatBufferBuilder& builder, std::vector<SerializedTensorBuffer>& buffers) {
    TT_FATAL(buffers.empty(), "Buffers vector must be empty");
    TT_FATAL(!is_device_tensor(tensor), "Device tensors are not supported in flatbuffer serialization");

    auto tensor_spec_offset = ttnn::to_flatbuffer(tensor.tensor_spec(), builder);

    const auto& host_storage = tensor.host_storage();
    const auto& distributed_buffer = host_storage.buffer();
    const auto& topology = tensor.tensor_topology();

    // Deduplicate replicated shards: two shards are duplicates if their coordinates differ only
    // along Replicate dimensions. The deduplication key is built from coordinates at sharded
    // dimensions only.
    const auto& placements = topology.placements();
    const auto& mesh_shape = topology.distribution_shape();
    size_t unique_keys = 1;
    for (size_t dim = 0; dim < placements.size(); ++dim) {
        if (std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(placements[dim])) {
            unique_keys *= mesh_shape[dim];
        }
    }

    // The topology is a label, and the file trusts it: shards it calls replicas of each other are written once and
    // every record in the group points at that one copy. Device ops relabel their outputs, so the label can be wrong,
    // and a wrong Replicate would drop a shard silently. Each group therefore remembers the buffer that stands for
    // it, and every later member is compared against that buffer before it is folded in.
    struct DedupGroup {
        size_t buffer_index = 0;  // Index into `buffers` of the copy written for this group.
        tt::tt_metal::distributed::MeshCoordinate first_coord;  // Where that copy came from, for diagnostics.
    };
    std::vector<std::optional<DedupGroup>> dedup_groups(unique_keys);
    const bool verify_replicas = ttnn::CONFIG.get<"verify_replicated_shards_on_dump">();

    std::vector<flatbuffers::Offset<ttnn::flatbuffer::TensorShard>> shards_vector;
    shards_vector.reserve(mesh_shape.mesh_size());
    // Two shards backed by the same HostBuffer object hold the same bytes by construction (the fully replicated
    // mapper path aliases one buffer), so they share one copy without a compare, whatever the label says.
    std::unordered_map<const std::byte*, size_t> buffer_to_index;

    const auto& topology_mesh_coords = topology.mesh_coords();
    TT_FATAL(
        topology_mesh_coords.size() == mesh_shape.mesh_size(),
        "Topology mesh coords size {} should match distribution shape size {}",
        topology_mesh_coords.size(),
        mesh_shape.mesh_size());

    // Every populated local shard has to be reachable through the label, or it is left out of the file. Remote
    // coordinates (multi-host LOCAL dumps) hold no data on this host and are exempt.
    const std::set<tt::tt_metal::distributed::MeshCoordinate> labelled_coords(
        topology_mesh_coords.begin(), topology_mesh_coords.end());
    size_t num_populated_local_shards = 0;
    for (const auto& shard_coord : distributed_buffer.shard_coords()) {
        if (!distributed_buffer.is_local(shard_coord) || !distributed_buffer.get_shard(shard_coord).has_value()) {
            continue;
        }
        ++num_populated_local_shards;
        TT_FATAL(
            labelled_coords.contains(shard_coord),
            "Host storage holds a shard at mesh coordinate {} that the tensor topology does not cover ({}; host "
            "storage shape {}). The topology label does not describe the data, so this shard would be left out of "
            "the file. Relabel the tensor with update_tensor_topology() so the label matches how the shards were "
            "produced before dumping it.",
            shard_coord,
            describe(topology),
            distributed_buffer.shape());
    }

    // Iterate over distribution coordinates and map to physical coordinates via the topology.
    uint64_t next_buffer_offset = 0;
    size_t dist_idx = 0;
    for (const auto& dist_coord : tt::tt_metal::distributed::MeshCoordinateRange(mesh_shape)) {
        const auto& coord = topology_mesh_coords[dist_idx++];

        const auto buffer = distributed_buffer.get_shard(coord);
        if (!buffer.has_value()) {
            // A labelled coordinate without a shard is only legitimate when the shard lives on another host.
            TT_FATAL(
                !distributed_buffer.is_local(coord),
                "Tensor topology lists mesh coordinate {} but the host storage has no shard there ({}; host storage "
                "shape {} with {} populated shard(s)). The topology label describes data that does not exist, "
                "typically because the tensor is a single shard taken out of a distributed tensor "
                "(get_device_tensors(...)[i].cpu()), or this is a LOCAL-mode partial file re-dumped on a single host "
                "(re-dump from the multi-host job instead). Relabel the tensor with update_tensor_topology() so the "
                "label matches how the shards were produced before dumping it.",
                coord,
                describe(topology),
                distributed_buffer.shape(),
                num_populated_local_shards);
            continue;
        }

        const auto* buffer_address = buffer->view_bytes().data();
        const std::size_t buffer_size = buffer->view_bytes().size();

        size_t key = 0;
        for (size_t dim = 0; dim < placements.size(); ++dim) {
            if (std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(placements[dim])) {
                key = key * mesh_shape[dim] + dist_coord[dim];
            }
        }

        std::optional<size_t> buffer_index;
        if (auto it = buffer_to_index.find(buffer_address); it != buffer_to_index.end()) {
            buffer_index = it->second;
        }
        if (auto& group = dedup_groups[key]; group.has_value()) {
            if (buffer_index != group->buffer_index) {
                // A distinct buffer that the label calls a replica of the group's copy. Both records will point at
                // that copy, so the bytes have to match: the size always, the contents unless opted out.
                const auto group_bytes = buffers[group->buffer_index].buffer.view_bytes();
                TT_FATAL(
                    group_bytes.size() == buffer_size,
                    "Tensor topology labels mesh coordinates {} and {} as replicas of each other ({}; host storage "
                    "shape {}), but their shard sizes differ ({} vs {} bytes). The topology label does not describe "
                    "the data. Relabel the tensor with update_tensor_topology() so the label matches how the shards "
                    "were produced.",
                    group->first_coord,
                    coord,
                    describe(topology),
                    distributed_buffer.shape(),
                    group_bytes.size(),
                    buffer_size);
                if (verify_replicas) {
                    TT_FATAL(
                        std::memcmp(group_bytes.data(), buffer_address, buffer_size) == 0,
                        "Tensor topology labels mesh coordinates {} and {} as replicas of each other ({}; host "
                        "storage shape {}), but their shard contents differ. The topology label does not describe "
                        "the data, and writing one copy for both would lose the shard at {}. Relabel the tensor with "
                        "update_tensor_topology() so the label matches how the shards were produced; if the replicas "
                        "legitimately differ, set verify_replicated_shards_on_dump=false via "
                        "TTNN_CONFIG_OVERRIDES='{{\"verify_replicated_shards_on_dump\": false}}' to write only the "
                        "first replica.",
                        group->first_coord,
                        coord,
                        describe(topology),
                        distributed_buffer.shape(),
                        coord);
                }
            }
            buffer_index = group->buffer_index;
        } else {
            if (!buffer_index.has_value()) {
                // Start every distinct buffer on `kTensorDataAlignment` so a reader can DMA out of the mapped
                // file directly. The padded position is what gets recorded, so readers never see the gap.
                const uint64_t aligned_offset = tt::align(next_buffer_offset, kTensorDataAlignment);
                next_buffer_offset = aligned_offset + buffer_size;
                buffers.push_back(SerializedTensorBuffer{.buffer = *buffer, .offset = aligned_offset});
                buffer_index = buffers.size() - 1;
            }
            group = DedupGroup{.buffer_index = *buffer_index, .first_coord = coord};
        }
        buffer_to_index.emplace(buffer_address, *buffer_index);
        const uint64_t shard_buffer_offset = buffers[*buffer_index].offset;

        auto inline_storage = ttnn::flatbuffer::InlineFileStorage(shard_buffer_offset, buffer_size);
        auto mesh_coord_offset = to_flatbuffer(coord, builder);

        auto shard_offset = ttnn::flatbuffer::CreateTensorShard(
            builder,
            ttnn::flatbuffer::TensorBuffer::InlineFileStorage,
            builder.CreateStruct(inline_storage).Union(),
            mesh_coord_offset);

        shards_vector.push_back(shard_offset);
    }
    auto shards = builder.CreateVector(shards_vector);

    auto mesh_shape_offset = to_flatbuffer(distributed_buffer.shape(), builder);

    auto topology_offset = to_flatbuffer(topology, builder);

    auto tensor_offset =
        ttnn::flatbuffer::CreateTensor(builder, tensor_spec_offset, mesh_shape_offset, shards, topology_offset);

    return tensor_offset;
}

Tensor from_flatbuffer(
    const ttnn::flatbuffer::Tensor* fb_tensor,
    ttsl::Span<std::byte> tensor_data,
    const tt::tt_metal::MemoryPin& memory_pin) {
    auto spec = ttnn::from_flatbuffer(fb_tensor->tensor_spec());

    const auto* mesh_shape = fb_tensor->mesh_shape();
    TT_FATAL(mesh_shape != nullptr, "Mesh shape is required for tensor");
    const tt::tt_metal::distributed::MeshShape ttnn_mesh_shape = from_flatbuffer(mesh_shape);

    // File shards are host-local. Loading them must not initialize MetalContext or acquire device locks.
    auto distributed_buffer = tt::tt_metal::DistributedHostBuffer::create(
        ttnn_mesh_shape,
        ttnn_mesh_shape,
        tt::tt_metal::distributed::MeshCoordinate::zero_coordinate(ttnn_mesh_shape.dims()),
        /*context=*/nullptr);
    for (size_t i = 0; i < fb_tensor->shards()->size(); ++i) {
        const auto* shard = fb_tensor->shards()->Get(i);

        const auto* inline_storage = shard->buffer_as<ttnn::flatbuffer::InlineFileStorage>();
        TT_FATAL(inline_storage != nullptr, "Only InlineFileStorage is supported in flatbuffer deserialization");

        const uint64_t offset = inline_storage->offset();
        const uint64_t size = inline_storage->size();

        tt::tt_metal::HostBuffer host_buffer = create_host_buffer_from_bytes(
            size, spec, ttsl::Span<std::byte>(tensor_data.data() + offset, size), memory_pin);

        TT_FATAL(shard->mesh_coordinate() != nullptr, "Mesh coordinate is required for each shard");
        const auto coord = from_flatbuffer(shard->mesh_coordinate());
        distributed_buffer.emplace_shard(
            coord, [host_buffer = std::move(host_buffer)]() mutable { return std::move(host_buffer); });
    }

    // NOTE: Existing tensor cache files may not have a tensor topology.
    // Create tensor topology from flatbuffer if it exists, otherwise create a fully replicated topology.
    const auto* fb_topology = fb_tensor->tensor_topology();
    tt::tt_metal::TensorTopology topology =
        fb_topology != nullptr ? from_flatbuffer(fb_topology)
                               : tt::tt_metal::TensorTopology::create_fully_replicated_tensor_topology(ttnn_mesh_shape);

    return Tensor(
        tt::tt_metal::host_tensor_from_buffer_with_topology(std::move(distributed_buffer), spec, std::move(topology)));
}

}  // namespace ttnn
