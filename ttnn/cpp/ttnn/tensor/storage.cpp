// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt-metalium/mesh_coord.hpp"

#include "ttnn/tensor/storage.hpp"

namespace tt::tt_metal {

DeviceStorage::DeviceStorage(std::shared_ptr<Buffer> buffer_) { buffer = std::move(buffer_); }

MemoryConfig DeviceStorage::memory_config() const {
    auto* buffer_to_use = get_buffer();

    std::optional<ShardSpec> shard_spec = std::nullopt;

    if (is_sharded(buffer_to_use->buffer_layout())) {
        shard_spec = buffer_to_use->shard_spec().tensor_shard_spec;
    }
    return MemoryConfig{
        .memory_layout = buffer_to_use->buffer_layout(),
        .buffer_type = buffer_to_use->buffer_type(),
        .shard_spec = shard_spec,
    };
}

DeviceStorage::DeviceStorage(
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer_, std::vector<distributed::MeshCoordinate> shards_) :
    shards(std::move(shards_)), mesh_buffer(std::move(mesh_buffer_)) {}

Buffer* DeviceStorage::get_buffer() const {
    if (this->mesh_buffer.get() != nullptr) {
        return this->mesh_buffer->get_reference_buffer();
    }
    TT_FATAL(this->buffer != nullptr, "Buffer is not allocated");
    return this->buffer.get();
}

std::shared_ptr<distributed::MeshBuffer> DeviceStorage::get_mesh_buffer() const {
    TT_FATAL(mesh_buffer != nullptr, "Mesh buffer is not allocated");
    return mesh_buffer;
}

bool DeviceStorage::is_allocated() const {
    if (this->mesh_buffer.get() != nullptr) {
        return this->mesh_buffer->is_allocated();
    }
    return this->buffer != nullptr && this->buffer->is_allocated();
}

IDevice* DeviceStorage::get_device() const {
    if (this->mesh_buffer.get() != nullptr) {
        return this->mesh_buffer->device();
    }
    TT_FATAL(this->buffer != nullptr, "Buffer is not allocated");
    return this->buffer->device();
}

bool DeviceStorage::is_uniform_storage() const {
    if (mesh_buffer.get() == nullptr) {
        return true;
    }
    return shards.size() == mesh_buffer->device()->num_devices();
}

}  // namespace tt::tt_metal
