// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/flatbuffer/tensor_flatbuffer.hpp"
#include "tensor/flatbuffer/tensor_spec_flatbuffer.hpp"

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <flatbuffers/flatbuffers.h>

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include "mesh_shape_generated.h"
#include "mesh_coordinate_generated.h"
#include "tensor_generated.h"

#include <vector>

namespace ttnn {
namespace {

flatbuffers::Offset<flatbuffer::MeshCoordinate> to_flatbuffer(
    const tt::tt_metal::distributed::MeshCoordinate& coord, flatbuffers::FlatBufferBuilder& builder) {
    auto values_vector = builder.CreateVector(std::vector<uint32_t>(coord.coords().begin(), coord.coords().end()));
    return flatbuffer::CreateMeshCoordinate(builder, values_vector);
}

tt::tt_metal::distributed::MeshCoordinate from_flatbuffer(const flatbuffer::MeshCoordinate* coord) {
    return tt::tt_metal::distributed::MeshCoordinate(
        std::vector<uint32_t>(coord->values()->begin(), coord->values()->end()));
}

flatbuffers::Offset<flatbuffer::MeshShape> to_flatbuffer(
    const tt::tt_metal::distributed::MeshShape& shape, flatbuffers::FlatBufferBuilder& builder) {
    auto dimensions_vector = builder.CreateVector(std::vector<uint32_t>(shape.view().begin(), shape.view().end()));
    return flatbuffer::CreateMeshShape(builder, dimensions_vector);
}

tt::tt_metal::distributed::MeshShape from_flatbuffer(const flatbuffer::MeshShape* shape) {
    return tt::tt_metal::distributed::MeshShape(
        std::vector<uint32_t>(shape->dimensions()->begin(), shape->dimensions()->end()));
}

tt::tt_metal::HostBuffer create_host_buffer_from_bytes(uint64_t size_bytes, const TensorSpec& spec) {
    switch (spec.data_type()) {
        case tt::tt_metal::DataType::UINT32:
        case tt::tt_metal::DataType::BFLOAT8_B:
        case tt::tt_metal::DataType::BFLOAT4_B: {
            std::vector<uint32_t> data(size_bytes / sizeof(uint32_t));
            return tt::tt_metal::HostBuffer(std::move(data));
        }
        case tt::tt_metal::DataType::INT32: {
            std::vector<int32_t> data(size_bytes / sizeof(int32_t));
            return tt::tt_metal::HostBuffer(std::move(data));
        }
        case tt::tt_metal::DataType::UINT8: {
            std::vector<uint8_t> data(size_bytes / sizeof(uint8_t));
            return tt::tt_metal::HostBuffer(std::move(data));
        }
        case tt::tt_metal::DataType::UINT16: {
            std::vector<uint16_t> data(size_bytes / sizeof(uint16_t));
            return tt::tt_metal::HostBuffer(std::move(data));
        }
        case tt::tt_metal::DataType::FLOAT32: {
            std::vector<float> data(size_bytes / sizeof(float));
            return tt::tt_metal::HostBuffer(std::move(data));
        }
        case tt::tt_metal::DataType::BFLOAT16: {
            std::vector<bfloat16> data(size_bytes / sizeof(bfloat16));
            return tt::tt_metal::HostBuffer(std::move(data));
        }
        case tt::tt_metal::DataType::INVALID: TT_THROW("Unsupported DataType");
    }
    TT_THROW("Unreachable");
}

}  // namespace

flatbuffers::Offset<ttnn::flatbuffer::Tensor> to_flatbuffer(
    const Tensor& tensor, flatbuffers::FlatBufferBuilder& builder) {
    const auto& storage = tensor.get_storage();

    TT_FATAL(!is_device_tensor(tensor), "Device tensors are not supported in flatbuffer serialization");

    auto tensor_spec_offset = ttnn::to_flatbuffer(tensor.get_tensor_spec(), builder);

    const auto& strategy = tensor.get_distributed_tensor_config();
    if (std::holds_alternative<tt::tt_metal::ReplicateTensor>(strategy)) {
        std::size_t buffer_size = 0;
        if (std::holds_alternative<tt::tt_metal::HostStorage>(storage)) {
            const auto& host_storage = std::get<tt::tt_metal::HostStorage>(storage);
            buffer_size = host_storage.buffer.view_bytes().size();
        } else {
            const auto buffer = std::get<tt::tt_metal::MultiDeviceHostStorage>(storage).get_buffer(0);
            buffer_size = buffer.view_bytes().size();
        }

        auto inline_storage = ttnn::flatbuffer::InlineFileStorage(/*offset=*/0, buffer_size);

        auto replicated_tensor = ttnn::flatbuffer::CreateReplicatedTensor(
            builder, ttnn::flatbuffer::TensorBuffer::InlineFileStorage, builder.CreateStruct(inline_storage).Union());

        auto tensor_offset = ttnn::flatbuffer::CreateTensor(
            builder, tensor_spec_offset, ttnn::flatbuffer::TensorType::ReplicatedTensor, replicated_tensor.Union());

        return tensor_offset;
    } else {
        TT_FATAL(
            std::holds_alternative<tt::tt_metal::MultiDeviceHostStorage>(storage),
            "Sharded tensor requires MultiDeviceHostStorage");

        const auto& multi_device_storage = std::get<tt::tt_metal::MultiDeviceHostStorage>(storage);

        std::vector<flatbuffers::Offset<ttnn::flatbuffer::TensorShard>> shards_vector;
        uint64_t data_offset = 0;
        for (std::size_t i = 0; i < multi_device_storage.num_buffers(); i++) {
            const auto& buffer = multi_device_storage.get_buffer(i);
            const std::size_t buffer_size = buffer.view_bytes().size();

            auto inline_storage = ttnn::flatbuffer::InlineFileStorage(data_offset, buffer_size);

            auto shard_offset = ttnn::flatbuffer::CreateTensorShard(
                builder,
                ttnn::flatbuffer::TensorBuffer::InlineFileStorage,
                builder.CreateStruct(inline_storage).Union());

            shards_vector.push_back(shard_offset);
            data_offset += buffer_size;
        }
        auto shards = builder.CreateVector(shards_vector);

        flatbuffers::Offset<ttnn::flatbuffer::MeshShape> mesh_shape_offset;
        if (const auto* shard_2d = std::get_if<tt::tt_metal::ShardTensor2D>(&strategy)) {
            ttnn::MeshShape mesh_shape{shard_2d->shard_mesh.x, shard_2d->shard_mesh.y};
            mesh_shape_offset = to_flatbuffer(mesh_shape, builder);
        }
        auto sharded_tensor = ttnn::flatbuffer::CreateShardedTensor(builder, mesh_shape_offset, shards);

        auto tensor_offset = ttnn::flatbuffer::CreateTensor(
            builder, tensor_spec_offset, ttnn::flatbuffer::TensorType::ShardedTensor, sharded_tensor.Union());

        return tensor_offset;
    }
}

Tensor from_flatbuffer(const ttnn::flatbuffer::Tensor* fb_tensor, tt::stl::Span<std::byte> tensor_data) {
    auto spec = ttnn::from_flatbuffer(fb_tensor->tensor_spec());

    switch (fb_tensor->tensor_type_type()) {
        case ttnn::flatbuffer::TensorType::NONE: TT_THROW("Invalid TensorType");
        case ttnn::flatbuffer::TensorType::ReplicatedTensor: {
            auto replicated = fb_tensor->tensor_type_as_ReplicatedTensor();

            auto* inline_storage = replicated->buffer_as<ttnn::flatbuffer::InlineFileStorage>();
            TT_FATAL(inline_storage != nullptr, "Only InlineFileStorage is supported in flatbuffer deserialization");

            const uint64_t offset = inline_storage->offset();
            const uint64_t size = inline_storage->size();

            tt::tt_metal::HostBuffer host_buffer = create_host_buffer_from_bytes(size, spec);
            TT_FATAL(offset + size <= tensor_data.size(), "Tensor data out of bounds");
            std::memcpy(host_buffer.view_bytes().data(), tensor_data.data() + offset, size);

            return Tensor(std::move(host_buffer), spec);
        }
        case ttnn::flatbuffer::TensorType::ShardedTensor: {
            auto sharded = fb_tensor->tensor_type_as_ShardedTensor();

            tt::tt_metal::DistributedTensorConfig strategy;
            auto* mesh_shape = sharded->mesh_shape();
            if (mesh_shape != nullptr) {
                const tt::tt_metal::distributed::MeshShape ttnn_mesh_shape = from_flatbuffer(mesh_shape);
                strategy = tt::tt_metal::ShardTensor2D{
                    tt::tt_metal::ShardMesh{.y = ttnn_mesh_shape[0], .x = ttnn_mesh_shape[1]}};
            }

            const size_t num_shards = sharded->shards()->size();

            std::vector<tt::tt_metal::HostBuffer> buffers;
            std::vector<TensorSpec> specs(num_shards, spec);
            buffers.reserve(num_shards);
            for (size_t i = 0; i < num_shards; i++) {
                auto* shard = sharded->shards()->Get(i);

                auto* inline_storage = shard->buffer_as<ttnn::flatbuffer::InlineFileStorage>();
                TT_FATAL(
                    inline_storage != nullptr, "Only InlineFileStorage is supported in flatbuffer deserialization");

                const uint64_t offset = inline_storage->offset();
                const uint64_t size = inline_storage->size();

                tt::tt_metal::HostBuffer host_buffer = create_host_buffer_from_bytes(size, spec);
                TT_FATAL(offset + size <= tensor_data.size(), "Tensor data out of bounds for shard {}", i);
                std::memcpy(static_cast<void*>(host_buffer.view_bytes().data()), tensor_data.data() + offset, size);
                buffers.push_back(std::move(host_buffer));
            }

            tt::tt_metal::MultiDeviceHostStorage multi_device_storage{std::move(buffers)};

            return Tensor(std::move(multi_device_storage), spec, strategy);
        }
    }
    TT_THROW("Unreachable");
}

}  // namespace ttnn
