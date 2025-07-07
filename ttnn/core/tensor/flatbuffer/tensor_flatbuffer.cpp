// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include "mesh_shape_generated.h"
#include <tt-metalium/serialized_descriptors/mesh_coordinate_generated.h>
#include "tensor_generated.h"

#include <vector>

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
    uint64_t size_bytes, const TensorSpec& spec, tt::stl::Span<std::byte> data, tt::tt_metal::MemoryPin memory_pin) {
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

}  // namespace

flatbuffers::Offset<ttnn::flatbuffer::Tensor> to_flatbuffer(
    const Tensor& tensor, flatbuffers::FlatBufferBuilder& builder, std::vector<tt::tt_metal::HostBuffer>& buffers) {
    const auto& storage = tensor.storage();

    TT_FATAL(buffers.empty(), "Buffers vector must be empty");
    TT_FATAL(!is_device_tensor(tensor), "Device tensors are not supported in flatbuffer serialization");

    auto tensor_spec_offset = ttnn::to_flatbuffer(tensor.tensor_spec(), builder);

    const auto& host_storage = tensor.host_storage();

    std::vector<flatbuffers::Offset<ttnn::flatbuffer::TensorShard>> shards_vector;
    // Used to deduplicate buffer addresses for replicated tensor data.
    std::unordered_map<const std::byte*, uint64_t> buffer_to_offset;
    uint64_t next_buffer_offset = 0;
    for (const auto& coord : host_storage.buffer().shard_coords()) {
        // Iterate over local populated shards.
        if (const auto& buffer = host_storage.buffer().get_shard(coord); buffer.has_value()) {
            const auto* buffer_address = buffer->view_bytes().data();
            const std::size_t buffer_size = buffer->view_bytes().size();

            uint64_t shard_buffer_offset = next_buffer_offset;
            if (auto [it, inserted] = buffer_to_offset.try_emplace(buffer_address, shard_buffer_offset); inserted) {
                // Encountered a new buffer, add it to the buffers vector.
                next_buffer_offset += buffer_size;
                buffers.push_back(*buffer);
            } else {
                // Point to the existing buffer.
                shard_buffer_offset = it->second;
            }

            auto inline_storage = ttnn::flatbuffer::InlineFileStorage(shard_buffer_offset, buffer_size);
            auto mesh_coord_offset = to_flatbuffer(coord, builder);

            auto shard_offset = ttnn::flatbuffer::CreateTensorShard(
                builder,
                ttnn::flatbuffer::TensorBuffer::InlineFileStorage,
                builder.CreateStruct(inline_storage).Union(),
                mesh_coord_offset);

            shards_vector.push_back(shard_offset);
        }
    }
    auto shards = builder.CreateVector(shards_vector);

    auto mesh_shape_offset = to_flatbuffer(host_storage.buffer().shape(), builder);

    auto tensor_offset = ttnn::flatbuffer::CreateTensor(builder, tensor_spec_offset, mesh_shape_offset, shards);

    return tensor_offset;
}

Tensor from_flatbuffer(
    const ttnn::flatbuffer::Tensor* fb_tensor,
    tt::stl::Span<std::byte> tensor_data,
    tt::tt_metal::MemoryPin memory_pin) {
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

    // TODO: #24115 - `DistributedTensorConfig` will be replaced by distributed host buffer, which can be used
    // directly in Tensor storage.
    const auto strategy = [&]() -> tt::tt_metal::DistributedTensorConfig {
        std::unordered_set<const std::byte*> buffer_addresses;
        distributed_buffer.apply([&buffer_addresses](const tt::tt_metal::HostBuffer& shard) {
            buffer_addresses.insert(shard.view_bytes().data());
        });
        if (buffer_addresses.size() == 1) {
            return tt::tt_metal::ReplicateTensor();
        } else if (ttnn_mesh_shape.dims() == 2) {
            return tt::tt_metal::ShardTensor2D{
                tt::tt_metal::ShardMesh{.y = ttnn_mesh_shape[0], .x = ttnn_mesh_shape[1]}};
        } else {
            return tt::tt_metal::AllGatherTensor{};
        }
    }();

    tt::tt_metal::HostStorage host_storage{std::move(distributed_buffer)};

    // TODO (#25340): Add TensorTopology to flatbuffer serialization and properly handle it in deserialization.
    return Tensor(std::move(host_storage), spec, strategy, tt::tt_metal::TensorTopology{});
}

}  // namespace ttnn
