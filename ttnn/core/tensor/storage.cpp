// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <vector>

#include "tt-metalium/mesh_coord.hpp"

#include "ttnn/tensor/storage.hpp"

namespace tt::tt_metal {

HostStorage::HostStorage(HostBuffer buffer) :
    distributed_buffer_(DistributedHostBuffer::create(distributed::MeshShape(1, 1))) {
    distributed_buffer_.emplace_shard(distributed::MeshCoordinate(0, 0), [&buffer]() { return std::move(buffer); });
}
HostStorage::HostStorage(DistributedHostBuffer buffer) : distributed_buffer_(std::move(buffer)) {}

const DistributedHostBuffer& HostStorage::buffer() const { return distributed_buffer_; }

HostStorage HostStorage::transform(const std::function<HostBuffer(const HostBuffer&)>& callable) const {
    return HostStorage(
        distributed_buffer_.transform(callable, DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL));
}

DeviceStorage::DeviceStorage(std::shared_ptr<Buffer> buffer_) { buffer = std::move(buffer_); }

DeviceStorage::DeviceStorage(
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer_, std::vector<distributed::MeshCoordinate> coords_) :
    coords(std::move(coords_)), mesh_buffer(std::move(mesh_buffer_)) {}

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
    return coords.size() == mesh_buffer->device()->num_devices();
}

}  // namespace tt::tt_metal
