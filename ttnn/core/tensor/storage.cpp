// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <unordered_set>
#include <vector>

#include <ttnn/tensor/layout/layout.hpp>
#include "tt-metalium/mesh_coord.hpp"

#include "ttnn/tensor/storage.hpp"

namespace tt::tt_metal {

DistributedHostBuffer create_unit_distributed_host_buffer(HostBuffer buffer) {
    auto distributed_buffer = DistributedHostBuffer::create(distributed::MeshShape(1, 1));
    distributed_buffer.emplace_shard(distributed::MeshCoordinate(0, 0), [&buffer]() { return std::move(buffer); });
    return distributed_buffer;
}

HostStorage::HostStorage(HostBuffer buffer) : HostStorage(create_unit_distributed_host_buffer(std::move(buffer))) {}

HostTensor create_dummy_host_tensor(DistributedHostBuffer buffer) {
    TensorSpec spec{Shape{}, TensorLayout{DataType::BFLOAT16, PageConfig{Layout::ROW_MAJOR}, MemoryConfig{}}};
    TensorTopology topology;
    return HostTensor(std::move(buffer), std::move(spec), std::move(topology));
}

HostStorage::HostStorage(DistributedHostBuffer buffer) : tensor(create_dummy_host_tensor(std::move(buffer))) {}

HostStorage::HostStorage(HostTensor tensor) : tensor(std::move(tensor)) {}

HostStorage::HostStorage(const HostStorage& other, TensorSpec spec, TensorTopology topology) :
    tensor(HostTensor(other.buffer(), std::move(spec), std::move(topology))) {}

HostStorage::HostStorage(HostStorage&& other, TensorSpec spec, TensorTopology topology) :
    tensor(HostTensor(std::move(other.tensor), std::move(spec), std::move(topology))) {}

const DistributedHostBuffer& HostStorage::buffer() const { return tensor.buffer(); }

const HostTensor& HostStorage::host_tensor() const { return tensor; }

HostStorage HostStorage::transform(const std::function<HostBuffer(const HostBuffer&)>& callable) const {
    auto transformed_buffer =
        tensor.buffer().transform(callable, DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);
    auto transformed_tensor = HostTensor(std::move(transformed_buffer), tensor.tensor_spec(), tensor.tensor_topology());
    return HostStorage(std::move(transformed_tensor));
}

DeviceStorage::DeviceStorage(
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer_,
    std::vector<distributed::MeshCoordinate> coords_,
    std::shared_ptr<distributed::MeshBuffer> root_buffer_) :
    coords(std::move(coords_)), mesh_buffer(std::move(mesh_buffer_)), root_mesh_buffer(std::move(root_buffer_)) {}

Buffer* DeviceStorage::get_buffer() const {
    if (this->mesh_buffer != nullptr) {
        return this->mesh_buffer->get_reference_buffer();
    }
    TT_THROW("Buffer is not allocated");
}

const distributed::MeshBuffer& DeviceStorage::get_mesh_buffer() const {
    TT_FATAL(mesh_buffer != nullptr, "Buffer is not allocated");
    return *mesh_buffer;
}

bool DeviceStorage::is_sole_owner_of_device_memory() const { return mesh_buffer.use_count() == 1; }

std::shared_ptr<distributed::MeshBuffer> DeviceStorage::get_mesh_buffer_leak_ownership() const {
    TT_FATAL(mesh_buffer != nullptr, "Buffer is not allocated");
    return mesh_buffer;
}

const std::shared_ptr<distributed::MeshBuffer>& DeviceStorage::get_root_mesh_buffer() const {
    return root_mesh_buffer ? root_mesh_buffer : mesh_buffer;
}

void DeviceStorage::deallocate_root_mesh_buffer() {
    if (root_mesh_buffer) {
        root_mesh_buffer->deallocate();
    } else {
        mesh_buffer->deallocate();
    }
}

void DeviceStorage::reset_root_mesh_buffer() {
    if (root_mesh_buffer) {
        root_mesh_buffer.reset();
    } else {
        mesh_buffer.reset();
    }
}

bool DeviceStorage::is_allocated() const { return this->mesh_buffer != nullptr && this->mesh_buffer->is_allocated(); }

distributed::MeshDevice* DeviceStorage::get_device() const {
    if (this->mesh_buffer != nullptr) {
        return this->mesh_buffer->device();
    }
    TT_THROW("Buffer is not allocated");
}

bool DeviceStorage::is_uniform_storage() const {
    if (mesh_buffer == nullptr) {
        return true;
    }
    return coords.size() == mesh_buffer->device()->num_devices();
}

}  // namespace tt::tt_metal
