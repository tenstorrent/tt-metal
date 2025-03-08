// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/storage.hpp"
#include "tt-metalium/mesh_coord.hpp"

namespace tt::tt_metal {

DeviceStorage::DeviceStorage(std::shared_ptr<Buffer> buffer_) { buffer = std::move(buffer_); }

MemoryConfig DeviceStorage::memory_config() const {
    if (this->mesh_buffer.get() != nullptr) {
        auto buffer = this->mesh_buffer->get_device_buffer();
        std::optional<ShardSpec> shard_spec = std::nullopt;

        if (is_sharded(buffer->buffer_layout())) {
            shard_spec = buffer->shard_spec().tensor_shard_spec;
        }
        return MemoryConfig{
            .memory_layout = buffer->buffer_layout(), .buffer_type = buffer->buffer_type(), .shard_spec = shard_spec};
    }
    std::optional<ShardSpec> shard_spec = std::nullopt;
    if (is_sharded(this->buffer->buffer_layout())) {
        shard_spec = this->buffer->shard_spec().tensor_shard_spec;
    }
    return MemoryConfig{
        .memory_layout = this->buffer->buffer_layout(),
        .buffer_type = this->buffer->buffer_type(),
        .shard_spec = shard_spec};
}

DeviceStorage::DeviceStorage(
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer_,
    std::map<distributed::MeshCoordinate, TensorSpec> specs_,
    DistributedTensorConfig strategy_) :
    strategy(std::move(strategy_)), mesh_buffer(std::move(mesh_buffer_)), specs(std::move(specs_)) {}

void DeviceStorage::insert_buffer(const std::shared_ptr<Buffer>& buffer_) { this->buffer = buffer_; }

Buffer* DeviceStorage::get_buffer() const {
    if (this->mesh_buffer.get() == nullptr) {
        TT_FATAL(this->buffer != nullptr, "Buffer is not allocated");
        return this->buffer.get();
    }
    return this->mesh_buffer->get_device_buffer();
}

bool DeviceStorage::is_allocated() const {
    if (this->mesh_buffer.get() == nullptr) {
        return this->buffer != nullptr && this->buffer->is_allocated();
    }
    return this->mesh_buffer->is_allocated();
}

}  // namespace tt::tt_metal
