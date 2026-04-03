// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <numeric>
#include <functional>
#include <unordered_set>
#include <vector>

#include <ttnn/tensor/layout/layout.hpp>
#include <ttnn/distributed/types.hpp>
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

DeviceStorage::DeviceStorage(MeshTensor mesh_tensor) :
    state_(LocallyAllocatedState(std::move(mesh_tensor))),
    coords_(CMAKE_UNIQUE_NAMESPACE::get_all_mesh_coordinates(get_device())) {}

DeviceStorage::DeviceStorage(MeshTensor mesh_tensor_, std::vector<distributed::MeshCoordinate> coords) :
    DeviceStorage(LocallyAllocatedState(std::move(mesh_tensor_)), std::move(coords), nullptr) {}

DeviceStorage::DeviceStorage(const DeviceStorage& other, std::vector<distributed::MeshCoordinate> coords) :
    DeviceStorage(other.state_, std::move(coords), other.root_mesh_tensor_) {}

DeviceStorage::DeviceStorage(const DeviceStorage& owning_storage, MeshTensor reinterpreted_mesh_tensor) :
    DeviceStorage(
        LocallyAllocatedState(std::move(reinterpreted_mesh_tensor)),
        owning_storage.coords_,
        owning_storage.get_root_mesh_tensor()) {}

DeviceStorage::DeviceStorage(
    States state, std::vector<distributed::MeshCoordinate> coords, std::shared_ptr<MeshTensor> root_mesh_tensor) :
    state_(std::move(state)), coords_(std::move(coords)), root_mesh_tensor_(std::move(root_mesh_tensor)) {
    if (is_allocated()) {
        CMAKE_UNIQUE_NAMESPACE::validate_mesh_coordinates(coords_, get_device());
    }
}

Buffer* DeviceStorage::get_buffer() const { return get_mesh_buffer().get_reference_buffer(); }

const std::shared_ptr<MeshTensor>& DeviceStorage::get_mesh_tensor_bypass_deallocate_check() const {
    TT_FATAL(std::holds_alternative<LocallyAllocatedState>(state_), "Device Memory is not allocated");
    return std::get<LocallyAllocatedState>(state_).mesh_tensor_;
}

const distributed::MeshBuffer& DeviceStorage::get_mesh_buffer() const {
    return get_mesh_tensor_bypass_deallocate_check()->mesh_buffer();
}

bool DeviceStorage::is_sole_owner_of_device_memory() const {
    if (!is_allocated()) {
        return false;
    }
    return get_mesh_tensor_bypass_deallocate_check().use_count() == 1 && get_root_mesh_tensor().use_count() == 1;
}

const MeshTensor& DeviceStorage::get_mesh_tensor() const {
    TT_FATAL(is_allocated(), "MeshTensor is not allocated");
    return *get_mesh_tensor_bypass_deallocate_check();
}

MeshTensor& DeviceStorage::get_mesh_tensor() {
    TT_FATAL(is_allocated(), "MeshTensor is not allocated");
    return *get_mesh_tensor_bypass_deallocate_check();
}

std::shared_ptr<distributed::MeshBuffer> DeviceStorage::get_mesh_buffer_leak_ownership() const {
    return get_mesh_tensor_bypass_deallocate_check()->mesh_buffer_invariant_breaking();
}

const std::shared_ptr<MeshTensor>& DeviceStorage::get_root_mesh_tensor() const {
    return root_mesh_tensor_ ? root_mesh_tensor_ : std::get<LocallyAllocatedState>(state_).mesh_tensor_;
}

void DeviceStorage::deallocate() {
    if (!is_allocated()) {
        return;
    }

    get_root_mesh_tensor()->mesh_buffer().deallocate();
    DeallocatedState new_state(*get_mesh_tensor_bypass_deallocate_check());
    state_ = std::move(new_state);
}

bool DeviceStorage::is_allocated() const {
    if (const auto* allocated = std::get_if<LocallyAllocatedState>(&state_)) {
        return allocated->mesh_tensor_->mesh_buffer().is_allocated();
    }
    return false;
}

distributed::MeshDevice* DeviceStorage::get_device_bypass_deallocate_check() const {
    if (const auto* allocated = std::get_if<LocallyAllocatedState>(&state_)) {
        return allocated->mesh_tensor_->mesh_buffer().device();
    }
    return nullptr;
}

distributed::MeshDevice& DeviceStorage::get_device() const { return *get_mesh_buffer().device(); }

bool DeviceStorage::is_uniform_storage() const {
    if (std::holds_alternative<DeallocatedState>(state_)) {
        return true;
    }
    return coords_.size() == get_device_bypass_deallocate_check()->num_devices();
}

std::span<const distributed::MeshCoordinate> DeviceStorage::get_coords() const {
    // Conv breaks if we keep the assert here.
    // TT_FATAL(is_allocated(), "Device memory is not allocated");
    return coords_;
}

DeviceStorage DeviceStorage::combine_device_storages(
    const std::vector<std::reference_wrapper<const DeviceStorage>>& storages, int shard_dim) {
    TT_FATAL(!storages.empty(), "Cannot aggregate empty vector of DeviceStorages");

    const auto& model_storage = storages.front().get();
    TT_FATAL(
        std::all_of(
            storages.begin(),
            storages.end(),
            [&](const auto& storage) { return std::holds_alternative<LocallyAllocatedState>(storage.get().state_); }),
        "All DeviceStorages must be allocated");
    TT_FATAL(
        std::all_of(
            storages.begin(),
            storages.end(),
            [&](const auto& storage) {
                return storage.get().get_mesh_tensor_bypass_deallocate_check() ==
                       model_storage.get_mesh_tensor_bypass_deallocate_check();
            }),
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

    TensorTopology topology =
        TensorTopology::create_sharded_tensor_topology(distributed::MeshShape(joint_coords.size()), shard_dim);

    DeviceStorage res(
        model_storage, std::vector<distributed::MeshCoordinate>(joint_coords.begin(), joint_coords.end()));
    res.update_tensor_topology(topology);
    return res;
}

void DeviceStorage::update_tensor_topology(const TensorTopology& tensor_topology) {
    TT_FATAL(is_allocated(), "Device memory is not allocated");
    get_mesh_tensor().update_tensor_topology(tensor_topology);
}

const TensorSpec& DeviceStorage::get_tensor_spec() const {
    return std::visit([](const auto& state) -> const TensorSpec& { return state.get_tensor_spec(); }, state_);
}

}  // namespace tt::tt_metal
