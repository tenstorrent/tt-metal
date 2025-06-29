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

    if (const auto* host_storage = std::get_if<tt::tt_metal::HostStorage>(&storage); host_storage != nullptr) {
        buffers.push_back(host_storage->buffer);

        auto inline_storage =
            ttnn::flatbuffer::InlineFileStorage(/*offset=*/0, host_storage->buffer.view_bytes().size());

        auto replicated_tensor = ttnn::flatbuffer::CreateReplicatedTensor(
            builder, ttnn::flatbuffer::TensorBuffer::InlineFileStorage, builder.CreateStruct(inline_storage).Union());

        auto tensor_offset = ttnn::flatbuffer::CreateTensor(
            builder, tensor_spec_offset, ttnn::flatbuffer::TensorType::ReplicatedTensor, replicated_tensor.Union());

        return tensor_offset;
    } else {
        const auto* multi_device_storage = std::get_if<tt::tt_metal::MultiDeviceHostStorage>(&storage);
        TT_FATAL(multi_device_storage != nullptr, "Sharded tensor requires MultiDeviceHostStorage");

        std::vector<flatbuffers::Offset<ttnn::flatbuffer::TensorShard>> shards_vector;
        // Used to deduplicate buffer addresses for replicated tensor data.
        std::unordered_map<const std::byte*, uint64_t> buffer_to_offset;
        uint64_t next_buffer_offset = 0;
        for (const auto& coord : multi_device_storage->distributed_buffer().shard_coords()) {
            // Iterate over local populated shards.
            if (const auto& buffer = multi_device_storage->distributed_buffer().get_shard(coord); buffer.has_value()) {
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

        auto mesh_shape_offset = to_flatbuffer(multi_device_storage->distributed_buffer().shape(), builder);

        auto sharded_tensor = ttnn::flatbuffer::CreateShardedTensor(builder, mesh_shape_offset, shards);
        auto tensor_offset = ttnn::flatbuffer::CreateTensor(
            builder, tensor_spec_offset, ttnn::flatbuffer::TensorType::ShardedTensor, sharded_tensor.Union());

        return tensor_offset;
    }
}

Tensor from_flatbuffer(
    const ttnn::flatbuffer::Tensor* fb_tensor,
    tt::stl::Span<std::byte> tensor_data,
    tt::tt_metal::MemoryPin memory_pin) {
    auto spec = ttnn::from_flatbuffer(fb_tensor->tensor_spec());

    switch (fb_tensor->tensor_type_type()) {
        case ttnn::flatbuffer::TensorType::NONE: TT_THROW("Invalid TensorType");
        case ttnn::flatbuffer::TensorType::ReplicatedTensor: {
            auto replicated = fb_tensor->tensor_type_as_ReplicatedTensor();

            auto* inline_storage = replicated->buffer_as<ttnn::flatbuffer::InlineFileStorage>();
            TT_FATAL(inline_storage != nullptr, "Only InlineFileStorage is supported in flatbuffer deserialization");

            const uint64_t offset = inline_storage->offset();
            const uint64_t size = inline_storage->size();

            tt::tt_metal::HostBuffer host_buffer = create_host_buffer_from_bytes(
                size, spec, tt::stl::Span<std::byte>(tensor_data.data() + offset, size), memory_pin);
            return Tensor(std::move(host_buffer), spec);
        }
        case ttnn::flatbuffer::TensorType::ShardedTensor: {
            const auto* sharded = fb_tensor->tensor_type_as_ShardedTensor();

            const auto* mesh_shape = sharded->mesh_shape();
            TT_FATAL(mesh_shape != nullptr, "Mesh shape is required for sharded tensor");
            const tt::tt_metal::distributed::MeshShape ttnn_mesh_shape = from_flatbuffer(mesh_shape);

            auto distributed_buffer = tt::tt_metal::DistributedHostBuffer::create(ttnn_mesh_shape);
            for (size_t i = 0; i < sharded->shards()->size(); ++i) {
                const auto* shard = sharded->shards()->Get(i);

                const auto* inline_storage = shard->buffer_as<ttnn::flatbuffer::InlineFileStorage>();
                TT_FATAL(
                    inline_storage != nullptr, "Only InlineFileStorage is supported in flatbuffer deserialization");

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

            tt::tt_metal::MultiDeviceHostStorage multi_device_storage{std::move(distributed_buffer)};

            return Tensor(std::move(multi_device_storage), spec, strategy);
        }
    }
    TT_THROW("Unreachable");
}

}  // namespace ttnn
