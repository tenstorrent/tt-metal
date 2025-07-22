// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <vector>

#include "tt-metalium/mesh_coord.hpp"

#include "ttnn/tensor/storage.hpp"

namespace tt::tt_metal {

DeviceStorage::DeviceStorage(
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer_, std::vector<distributed::MeshCoordinate> coords_) :
    coords(std::move(coords_)), mesh_buffer(std::move(mesh_buffer_)) {}

Buffer* DeviceStorage::get_buffer() const {
    if (this->mesh_buffer.get() != nullptr) {
        return this->mesh_buffer->get_reference_buffer();
    }
    TT_THROW("Buffer is not allocated");
}

std::shared_ptr<distributed::MeshBuffer> DeviceStorage::get_mesh_buffer() const {
    TT_FATAL(mesh_buffer != nullptr, "Buffer is not allocated");
    return mesh_buffer;
}

bool DeviceStorage::is_allocated() const {
    return this->mesh_buffer.get() != nullptr && this->mesh_buffer->is_allocated();
}

distributed::MeshDevice* DeviceStorage::get_device() const {
    if (this->mesh_buffer.get() != nullptr) {
        return this->mesh_buffer->device();
    }
    TT_THROW("Buffer is not allocated");
}

bool DeviceStorage::is_uniform_storage() const {
    if (mesh_buffer.get() == nullptr) {
        return true;
    }
    return coords.size() == mesh_buffer->device()->num_devices();
}

const DistributedHostBuffer& MultiDeviceHostStorage::distributed_buffer() const { return distributed_buffer_; }

MultiDeviceHostStorage::MultiDeviceHostStorage(DistributedHostBuffer buffer) : distributed_buffer_(std::move(buffer)) {}

}  // namespace tt::tt_metal
