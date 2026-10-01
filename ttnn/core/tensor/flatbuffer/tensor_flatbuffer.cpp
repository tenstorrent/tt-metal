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
#include <tt-logger/tt-logger.hpp>
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
#include <map>
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

// Shards whose distribution coordinates differ only along Replicate axes are replicas of each other and form one
// group; the writer stores a single copy per group. The group key is the row-major index of the distribution
// coordinate restricted to the Shard axes, so keys run over [0, num_replica_groups).
size_t num_replica_groups(const tt::tt_metal::TensorTopology& topology) {
    const auto& placements = topology.placements();
    const auto& dist_shape = topology.distribution_shape();
    size_t num_groups = 1;
    for (size_t dim = 0; dim < placements.size(); ++dim) {
        if (std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(placements[dim])) {
            num_groups *= dist_shape[dim];
        }
    }
    return num_groups;
}

size_t replica_group_key(
    const tt::tt_metal::TensorTopology& topology, const tt::tt_metal::distributed::MeshCoordinate& dist_coord) {
    const auto& placements = topology.placements();
    const auto& dist_shape = topology.distribution_shape();
    size_t key = 0;
    for (size_t dim = 0; dim < placements.size(); ++dim) {
        if (std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(placements[dim])) {
            key = key * dist_shape[dim] + dist_coord[dim];
        }
    }
    return key;
}

// One `TensorShard` record of a file: where the shard sits in the mesh and where its bytes sit in the data region.
struct ShardRecord {
    tt::tt_metal::distributed::MeshCoordinate coord;
    uint64_t offset = 0;
    uint64_t size = 0;
};

// Checks a topology read from a file against the shard records it came with, before any shard is placed. The
// writer now enforces the same rules (the placement count is new alongside this check; the others came with the
// replica dedup guards), so a failure means the header does not describe the data: corruption, a file assembled by
// something other than `to_flatbuffer`, or a label with the wrong placement count dumped before the writer checked it.
void validate_loaded_topology(
    const tt::tt_metal::TensorTopology& topology,
    const std::vector<ShardRecord>& records,
    const tt::tt_metal::distributed::MeshShape& storage_shape) {
    const auto& dist_shape = topology.distribution_shape();
    const auto& placements = topology.placements();
    const auto& mesh_coords = topology.mesh_coords();

    // `TensorTopology` itself checks nothing, and everything below indexes by these sizes.
    TT_FATAL(
        placements.size() == dist_shape.dims(),
        "Tensor file topology has {} placement(s) for the {}-dimensional distribution shape {} (host storage shape "
        "{}); the header is corrupt. Re-dump the tensor.",
        placements.size(),
        dist_shape.dims(),
        dist_shape,
        storage_shape);
    TT_FATAL(
        mesh_coords.size() == dist_shape.mesh_size(),
        "Tensor file topology lists {} mesh coordinate(s) for distribution shape {}, which has {} position(s) (host "
        "storage shape {}); the header is corrupt. Re-dump the tensor.",
        mesh_coords.size(),
        dist_shape,
        dist_shape.mesh_size(),
        storage_shape);

    // Every record must be reachable through the label, or the loaded tensor's topology does not describe its data.
    const std::set<tt::tt_metal::distributed::MeshCoordinate> labelled_coords(mesh_coords.begin(), mesh_coords.end());
    std::map<tt::tt_metal::distributed::MeshCoordinate, uint64_t> offset_at;
    for (const auto& record : records) {
        TT_FATAL(
            labelled_coords.contains(record.coord),
            "Tensor file holds a shard at mesh coordinate {} that its topology does not cover ({}; host storage shape "
            "{}). The label does not describe the data; the header is corrupt. Re-dump the tensor.",
            record.coord,
            describe(topology),
            storage_shape);
        offset_at[record.coord] = record.offset;
    }

    // Fewer records than labelled coordinates is legitimate: a LOCAL-mode dump from one host of a multi-host job
    // records only that host's shards. The file does not say which mode wrote it, so this can only be a warning.
    if (offset_at.size() < mesh_coords.size()) {
        log_warning(
            tt::LogAlways,
            "Tensor file holds {} of the {} shards its topology lists ({}; host storage shape {}). A LOCAL-mode dump "
            "from one host of a multi-host job records only that host's shards; the other coordinates load "
            "unpopulated.",
            offset_at.size(),
            mesh_coords.size(),
            describe(topology),
            storage_shape);
    }

    // Records of one replica group must point at the same copy, since that is how the writer stores them. Different
    // groups pointing at one copy is fine: shards backed by the same HostBuffer are written once whatever the label.
    struct GroupRecord {
        uint64_t offset = 0;
        tt::tt_metal::distributed::MeshCoordinate coord;
    };
    std::vector<std::optional<GroupRecord>> groups(num_replica_groups(topology));
    size_t dist_idx = 0;
    for (const auto& dist_coord : tt::tt_metal::distributed::MeshCoordinateRange(dist_shape)) {
        const auto& coord = mesh_coords[dist_idx++];
        const auto it = offset_at.find(coord);
        if (it == offset_at.end()) {
            continue;
        }
        auto& group = groups[replica_group_key(topology, dist_coord)];
        if (!group.has_value()) {
            group = GroupRecord{.offset = it->second, .coord = coord};
            continue;
        }
        TT_FATAL(
            group->offset == it->second,
            "Tensor file topology labels mesh coordinates {} and {} as replicas of each other ({}; host storage shape "
            "{}), but their records point at different data (offsets {} and {} in the data region). The writer "
            "stores replicas once, so this is not the label the shards were written under; the header is corrupt. "
            "Re-dump the tensor.",
            group->coord,
            coord,
            describe(topology),
            storage_shape,
            group->offset,
            it->second);
    }
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

    // Deduplicate replicated shards: two shards are duplicates if their coordinates differ only along Replicate
    // dimensions, see `replica_group_key`.
    const auto& mesh_shape = topology.distribution_shape();

    // Structural checks first: `TensorTopology` itself accepts any sizes, and the group-key computation below indexes
    // the distribution shape by placement index. A label with too few placements would otherwise dump (the Shard axes
    // under-indexed silently) and then be refused on load as corrupt.
    TT_FATAL(
        topology.placements().size() == mesh_shape.dims(),
        "Tensor topology has {} placement(s) for the {}-dimensional distribution shape {} ({}; host storage shape {}). "
        "The label does not describe the data. Relabel the tensor with update_tensor_topology() using one placement "
        "per distribution dimension before dumping it.",
        topology.placements().size(),
        mesh_shape.dims(),
        mesh_shape,
        describe(topology),
        distributed_buffer.shape());
    const auto& topology_mesh_coords = topology.mesh_coords();
    TT_FATAL(
        topology_mesh_coords.size() == mesh_shape.mesh_size(),
        "Topology mesh coords size {} should match distribution shape size {}",
        topology_mesh_coords.size(),
        mesh_shape.mesh_size());

    // The topology is a label, and the file trusts it: shards it calls replicas of each other are written once and
    // every record in the group points at that one copy. Device ops relabel their outputs, so the label can be wrong,
    // and a wrong Replicate would drop a shard silently. Each group therefore remembers the buffer that stands for
    // it, and every later member is compared against that buffer before it is folded in.
    struct DedupGroup {
        size_t buffer_index = 0;  // Index into `buffers` of the copy written for this group.
        tt::tt_metal::distributed::MeshCoordinate first_coord;  // Where that copy came from, for diagnostics.
    };
    std::vector<std::optional<DedupGroup>> dedup_groups(num_replica_groups(topology));
    const bool verify_replicas = ttnn::CONFIG.get<"verify_replicated_shards_on_dump">();

    std::vector<flatbuffers::Offset<ttnn::flatbuffer::TensorShard>> shards_vector;
    shards_vector.reserve(mesh_shape.mesh_size());
    // Two shards backed by the same HostBuffer object hold the same bytes by construction (the fully replicated
    // mapper path aliases one buffer), so they share one copy without a compare, whatever the label says.
    std::unordered_map<const std::byte*, size_t> buffer_to_index;

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

        std::optional<size_t> buffer_index;
        if (auto it = buffer_to_index.find(buffer_address); it != buffer_to_index.end()) {
            buffer_index = it->second;
        }
        if (auto& group = dedup_groups[replica_group_key(topology, dist_coord)]; group.has_value()) {
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

    auto tensor_offset = ttnn::flatbuffer::CreateTensor(
        builder, tensor_spec_offset, mesh_shape_offset, shards, topology_offset, kTensorFileSchemaVersion);

    return tensor_offset;
}

Tensor from_flatbuffer(
    const ttnn::flatbuffer::Tensor* fb_tensor,
    ttsl::Span<std::byte> tensor_data,
    const tt::tt_metal::MemoryPin& memory_pin) {
    // Absent (0) in files written before the field existed. A reader understands every revision up to its own.
    const uint32_t schema_version = fb_tensor->schema_version();
    TT_FATAL(
        schema_version <= kTensorFileSchemaVersion,
        "Tensor file records schema version {}, but this build reads versions up to {}. The file was written by a "
        "newer tt-metal; load it with that version or re-dump the tensor with this one.",
        schema_version,
        kTensorFileSchemaVersion);

    auto spec = ttnn::from_flatbuffer(fb_tensor->tensor_spec());

    const auto* mesh_shape = fb_tensor->mesh_shape();
    TT_FATAL(mesh_shape != nullptr, "Mesh shape is required for tensor");
    const tt::tt_metal::distributed::MeshShape ttnn_mesh_shape = from_flatbuffer(mesh_shape);

    // Read every shard record first: the topology is checked against them before any shard is placed.
    TT_FATAL(fb_tensor->shards() != nullptr, "Shards are required for tensor");
    std::vector<ShardRecord> records;
    records.reserve(fb_tensor->shards()->size());
    for (const auto* shard : *fb_tensor->shards()) {
        const auto* inline_storage = shard->buffer_as<ttnn::flatbuffer::InlineFileStorage>();
        TT_FATAL(inline_storage != nullptr, "Only InlineFileStorage is supported in flatbuffer deserialization");
        TT_FATAL(shard->mesh_coordinate() != nullptr, "Mesh coordinate is required for each shard");
        records.push_back(ShardRecord{
            .coord = from_flatbuffer(shard->mesh_coordinate()),
            .offset = inline_storage->offset(),
            .size = inline_storage->size()});
    }

    tt::tt_metal::TensorTopology topology = [&]() {
        if (const auto* fb_topology = fb_tensor->tensor_topology(); fb_topology != nullptr) {
            auto loaded = from_flatbuffer(fb_topology);
            validate_loaded_topology(loaded, records, ttnn_mesh_shape);
            return loaded;
        }
        // A versioned writer always records the topology, so a versioned file without one is corrupt.
        TT_FATAL(
            schema_version == 0,
            "Tensor file records schema version {} but no tensor topology; a writer of that version always records "
            "one, so the header is corrupt. Re-dump the tensor.",
            schema_version);
        // Files written before the topology field existed (#29158) are labelled fully replicated, as they always
        // were. That is right for a single buffer; for several, the placement cannot be known from the file. The
        // shards still load intact, so this is a warning rather than a refusal.
        std::set<uint64_t> distinct_offsets;
        for (const auto& record : records) {
            distinct_offsets.insert(record.offset);
        }
        if (distinct_offsets.size() > 1) {
            log_warning(
                tt::LogAlways,
                "Tensor file records no tensor topology (it predates the field) and holds {} distinct shard buffers "
                "for {} mesh coordinates (host storage shape {}). The placement label is unknown, so the tensor is "
                "labelled fully replicated as before: the shards load intact, but anything that reads the label sees "
                "copies of one shard, and re-dumping under it fails the replica check. Set the right label with "
                "update_tensor_topology() and re-dump to record the topology.",
                distinct_offsets.size(),
                records.size(),
                ttnn_mesh_shape);
        }
        return tt::tt_metal::TensorTopology::create_fully_replicated_tensor_topology(ttnn_mesh_shape);
    }();

    // File shards are host-local. Loading them must not initialize MetalContext or acquire device locks.
    auto distributed_buffer = tt::tt_metal::DistributedHostBuffer::create(
        ttnn_mesh_shape,
        ttnn_mesh_shape,
        tt::tt_metal::distributed::MeshCoordinate::zero_coordinate(ttnn_mesh_shape.dims()),
        /*context=*/nullptr);
    for (const auto& record : records) {
        tt::tt_metal::HostBuffer host_buffer = create_host_buffer_from_bytes(
            record.size, spec, ttsl::Span<std::byte>(tensor_data.data() + record.offset, record.size), memory_pin);
        distributed_buffer.emplace_shard(
            record.coord, [host_buffer = std::move(host_buffer)]() mutable { return std::move(host_buffer); });
    }

    return Tensor(
        tt::tt_metal::host_tensor_from_buffer_with_topology(std::move(distributed_buffer), spec, std::move(topology)));
}

}  // namespace ttnn
