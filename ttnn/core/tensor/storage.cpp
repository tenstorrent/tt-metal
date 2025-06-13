// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <vector>

#include "tt-metalium/mesh_coord.hpp"

#include "ttnn/tensor/storage.hpp"

namespace tt::tt_metal {

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

std::optional<HostBuffer> MultiDeviceHostStorage::get_shard_at_origin() const {
    return distributed_buffer_.get_shard(
        distributed::MeshCoordinate::zero_coordinate(distributed_buffer_.shape().dims()));
}

const DistributedHostBuffer& MultiDeviceHostStorage::distributed_buffer() const { return distributed_buffer_; }

MultiDeviceHostStorage::MultiDeviceHostStorage(std::vector<HostBuffer> buffers) :
    distributed_buffer_(DistributedHostBuffer::create(tt::tt_metal::distributed::MeshShape(buffers.size()))) {
    for (size_t i = 0; i < buffers.size(); ++i) {
        distributed_buffer_.emplace_shard(
            tt::tt_metal::distributed::MeshCoordinate(i), [&buffers, i]() { return std::move(buffers[i]); });
    }
}
MultiDeviceHostStorage::MultiDeviceHostStorage(DistributedHostBuffer buffer) : distributed_buffer_(std::move(buffer)) {}

}  // namespace tt::tt_metal
