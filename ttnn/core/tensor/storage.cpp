// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <numeric>
#include <functional>
#include <unordered_set>
#include <vector>

#include <ttnn/tensor/layout/layout.hpp>
#include "tt-metalium/mesh_coord.hpp"

#include "ttnn/tensor/storage.hpp"

namespace tt::tt_metal {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
std::vector<distributed::MeshCoordinate> get_all_mesh_coordinates(const distributed::MeshDevice& device) {
    std::vector<distributed::MeshCoordinate> coordinates;
    coordinates.reserve(device.num_devices());
    for (const auto& coord : distributed::MeshCoordinateRange(device.shape())) {
        coordinates.push_back(coord);
    }
    return coordinates;
}

void validate_mesh_coordinates(
    const std::vector<distributed::MeshCoordinate>& coords, const distributed::MeshDevice& device) {
    const distributed::MeshCoordinateRange valid_range(device.shape());
    for (const auto& coord : coords) {
        TT_FATAL(
            valid_range.contains(coord),
            "DeviceStorage coordinate {} is out of bounds for mesh device shape {}",
            coord,
            device.shape());
    }
}

DistributedHostBuffer create_unit_distributed_host_buffer(HostBuffer buffer) {
    auto distributed_buffer = DistributedHostBuffer::create(distributed::MeshShape(1, 1));
    distributed_buffer.emplace_shard(distributed::MeshCoordinate(0, 0), [&buffer]() { return std::move(buffer); });
    return distributed_buffer;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

HostStorage::HostStorage(HostBuffer buffer) : HostStorage(CMAKE_UNIQUE_NAMESPACE::create_unit_distributed_host_buffer(std::move(buffer))) {}

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
HostTensor& HostStorage::host_tensor() { return tensor; }

HostStorage HostStorage::transform(const std::function<HostBuffer(const HostBuffer&)>& callable) const {
    return HostStorage(tensor.transform(callable));
}

DeviceStorage::DeviceStorage(const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer_) :
    DeviceStorage(mesh_buffer_, CMAKE_UNIQUE_NAMESPACE::get_all_mesh_coordinates(*mesh_buffer_->device())) {}

DeviceStorage::DeviceStorage(
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer_, std::vector<distributed::MeshCoordinate> coords) :
    DeviceStorage(std::move(mesh_buffer_), std::move(coords), nullptr) {}

DeviceStorage::DeviceStorage(const DeviceStorage& other, std::vector<distributed::MeshCoordinate> coords) :
    DeviceStorage(other.mesh_buffer, std::move(coords), other.root_mesh_buffer) {}

DeviceStorage::DeviceStorage(
    const DeviceStorage& owning_storage, std::shared_ptr<distributed::MeshBuffer> surface_buffer) :
    DeviceStorage(std::move(surface_buffer), owning_storage.coords_, owning_storage.get_root_mesh_buffer()) {}

DeviceStorage::DeviceStorage(
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer,
    std::vector<distributed::MeshCoordinate> coords,
    std::shared_ptr<distributed::MeshBuffer> root_mesh_buffer) :
    coords_(std::move(coords)), mesh_buffer(std::move(mesh_buffer)), root_mesh_buffer(std::move(root_mesh_buffer)) {
    if (!is_allocated()) {
        this->mesh_buffer = nullptr;
        this->root_mesh_buffer = nullptr;
        return;
    }
    CMAKE_UNIQUE_NAMESPACE::validate_mesh_coordinates(coords_, *get_device());
}

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

bool DeviceStorage::is_sole_owner_of_device_memory() const {
    return mesh_buffer.use_count() == 1 && get_root_mesh_buffer().use_count() == 1;
}

std::shared_ptr<distributed::MeshBuffer> DeviceStorage::get_mesh_buffer_leak_ownership() const {
    TT_FATAL(mesh_buffer != nullptr, "Buffer is not allocated");
    return mesh_buffer;
}

const std::shared_ptr<distributed::MeshBuffer>& DeviceStorage::get_root_mesh_buffer() const {
    return root_mesh_buffer ? root_mesh_buffer : mesh_buffer;
}

void DeviceStorage::deallocate() {
    if (!is_allocated()) {
        return;
    }

    get_root_mesh_buffer()->deallocate();
    root_mesh_buffer = nullptr;
    mesh_buffer = nullptr;
}

bool DeviceStorage::is_allocated() const { return this->mesh_buffer != nullptr && this->mesh_buffer->is_allocated(); }

distributed::MeshDevice* DeviceStorage::get_device_bypass_deallocate_check() const {
    return this->mesh_buffer ? this->mesh_buffer->device() : nullptr;
}

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
    return coords_.size() == mesh_buffer->device()->num_devices();
}

std::span<const distributed::MeshCoordinate> DeviceStorage::get_coords() const {
    // Conv breaks if we keep the assert here.
    // TT_FATAL(is_allocated(), "Device memory is not allocated");
    return coords_;
}

DeviceStorage DeviceStorage::combine_device_storages(
    const std::vector<std::reference_wrapper<const DeviceStorage>>& storages) {
    TT_FATAL(!storages.empty(), "Cannot aggregate empty vector of DeviceStorages");

    const auto& model_storage = storages.front().get();
    TT_FATAL(
        std::all_of(
            storages.begin(),
            storages.end(),
            [&](const auto& storage) { return storage.get().mesh_buffer == model_storage.mesh_buffer; }),
        "All DeviceStorages must point to the same device memory");

    auto num_coords = std::accumulate(storages.begin(), storages.end(), 0, [](size_t sum, const auto& storage) {
        return sum + storage.get().get_coords().size();
    });

    std::unordered_set<distributed::MeshCoordinate> joint_coords;
    joint_coords.reserve(num_coords);
    for (const auto& storage : storages) {
        auto other_coords = storage.get().get_coords();
        joint_coords.insert(other_coords.begin(), other_coords.end());
    }

    return DeviceStorage(
        model_storage, std::vector<distributed::MeshCoordinate>(joint_coords.begin(), joint_coords.end()));
}

}  // namespace tt::tt_metal
