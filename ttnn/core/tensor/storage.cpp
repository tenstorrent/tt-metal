// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <numeric>
#include <functional>
#include <tt-logger/tt-logger.hpp>
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

// MeshTensor lifetime holder:
// This holder type has two states:
// - Allocated: actively holding a MeshTensor.
// - Deallocated: the MeshTensor was deallocated by any of the DeviceStorage instances.
//
// To ease transition, we keep a tombstone of the MeshTensor's spec, topology, and buffer when the MeshTensor is
// deallocated.
struct DeviceStorage::MeshTensorHolder {
    struct DeallocatedDefaultConstructed {};

    struct Allocated {
        MeshTensor mesh_tensor_;
    };

    struct DeallocatedTombStone {
        TensorSpec tensor_spec_;
        TensorTopology tensor_topology_;
        // Deallocated buffer kept so device() stays valid without dangling MeshDevice.
        // Remove once post-deallocation mesh_device access is no longer needed.
        // See: get_device_bypass_deallocate_check
        std::shared_ptr<distributed::MeshBuffer> mesh_buffer_;
    };

    using States = std::variant<DeallocatedDefaultConstructed, Allocated, DeallocatedTombStone>;
    States state_;

    MeshTensorHolder() : state_(DeallocatedDefaultConstructed{}) {}
    MeshTensorHolder(MeshTensor mesh_tensor) : state_(Allocated{std::move(mesh_tensor)}) {
        TT_FATAL(
            std::get<Allocated>(state_).mesh_tensor_.has_value(),
            "MeshTensor must not be in default constructed state.");
    }

    bool is_allocated() const { return std::holds_alternative<Allocated>(state_); }

    void deallocate() {
        if (auto* allocated = std::get_if<Allocated>(&state_)) {
            // We should favor letting MeshTensor go out of scope instead of explicitly calling the underlying
            // MeshBuffer. Calling deallocate is currently needed as we keep the MeshBuffer object alive in the
            // DeallocatedTombStone state.
            // Calling mesh_buffer_invariant_breaking() as we wish to get a mutable pointer to the MeshBuffer,
            // and this is breaking the invariant of MeshTensor (Device memory is allocated when the MeshTensor object
            // is alive).
            allocated->mesh_tensor_.mesh_buffer_invariant_breaking()->deallocate();
            // MeshTensor goes out of scope at this assignment:
            state_ = DeallocatedTombStone{
                allocated->mesh_tensor_.tensor_spec(),
                allocated->mesh_tensor_.tensor_topology(),
                allocated->mesh_tensor_.mesh_buffer_invariant_breaking()};
        }
    }
};

DeviceStorage::DeviceStorage() : mesh_tensor_holder_(std::make_shared<MeshTensorHolder>()) {}

DeviceStorage::DeviceStorage(MeshTensor mesh_tensor) :
    mesh_tensor_holder_(std::make_shared<MeshTensorHolder>(std::move(mesh_tensor))),
    coords_(CMAKE_UNIQUE_NAMESPACE::get_all_mesh_coordinates(get_mesh_tensor().device())) {}

DeviceStorage::DeviceStorage(MeshTensor mesh_tensor_, std::vector<distributed::MeshCoordinate> coords) :
    DeviceStorage(std::make_shared<MeshTensorHolder>(std::move(mesh_tensor_)), std::move(coords), nullptr) {}

DeviceStorage::DeviceStorage(const DeviceStorage& other, std::vector<distributed::MeshCoordinate> coords) :
    DeviceStorage(other.mesh_tensor_holder_, std::move(coords), other.root_mesh_tensor_holder_) {}

DeviceStorage::DeviceStorage(const DeviceStorage& owning_storage, MeshTensor reinterpreted_mesh_tensor) :
    DeviceStorage(
        std::make_shared<MeshTensorHolder>(std::move(reinterpreted_mesh_tensor)),
        owning_storage.coords_,
        owning_storage.get_root_mesh_tensor()) {}

DeviceStorage::DeviceStorage(
    std::shared_ptr<MeshTensorHolder> mesh_tensor_holder,
    std::vector<distributed::MeshCoordinate> coords,
    std::shared_ptr<MeshTensorHolder> root_mesh_tensor_holder) :
    mesh_tensor_holder_(std::move(mesh_tensor_holder)),
    coords_(std::move(coords)),
    root_mesh_tensor_holder_(std::move(root_mesh_tensor_holder)) {
    if (is_allocated()) {
        CMAKE_UNIQUE_NAMESPACE::validate_mesh_coordinates(coords_, get_mesh_tensor().device());
    }
}

Buffer* DeviceStorage::get_buffer() const { return get_mesh_buffer().get_reference_buffer(); }

const distributed::MeshBuffer& DeviceStorage::get_mesh_buffer() const {
    return std::visit(
        ttsl::overloaded{
            [](const MeshTensorHolder::Allocated& allocated) -> const distributed::MeshBuffer& {
                return allocated.mesh_tensor_.mesh_buffer();
            },
            [](const auto&) -> const distributed::MeshBuffer& { TT_THROW("Tensor is not allocated"); }},
        mesh_tensor_holder_->state_);
}

bool DeviceStorage::is_sole_owner_of_device_memory() const {
    if (!is_allocated()) {
        return false;
    }
    return mesh_tensor_holder_.use_count() == 1 && get_root_mesh_tensor().use_count() == 1;
}

const MeshTensor& DeviceStorage::get_mesh_tensor() const {
    return std::visit(
        ttsl::overloaded{
            [](const MeshTensorHolder::Allocated& allocated) -> const MeshTensor& { return allocated.mesh_tensor_; },
            [](const auto&) -> const MeshTensor& { TT_THROW("Tensor is not allocated"); }},
        mesh_tensor_holder_->state_);
}

MeshTensor& DeviceStorage::get_mesh_tensor() {
    return std::visit(
        ttsl::overloaded{
            [](MeshTensorHolder::Allocated& allocated) -> MeshTensor& { return allocated.mesh_tensor_; },
            [](const auto&) -> MeshTensor& { TT_THROW("Tensor is not allocated"); }},
        mesh_tensor_holder_->state_);
}

std::shared_ptr<distributed::MeshBuffer> DeviceStorage::get_mesh_buffer_leak_ownership() const {
    return std::visit(
        ttsl::overloaded{
            [](const MeshTensorHolder::Allocated& allocated) -> std::shared_ptr<distributed::MeshBuffer> {
                return allocated.mesh_tensor_.mesh_buffer_invariant_breaking();
            },
            [](const MeshTensorHolder::DeallocatedTombStone& tombstone) -> std::shared_ptr<distributed::MeshBuffer> {
                return tombstone.mesh_buffer_;
            },
            [](const auto&) -> std::shared_ptr<distributed::MeshBuffer> { TT_THROW("Tensor is not allocated"); }},
        mesh_tensor_holder_->state_);
}

const std::shared_ptr<DeviceStorage::MeshTensorHolder>& DeviceStorage::get_root_mesh_tensor() const {
    return root_mesh_tensor_holder_ ? root_mesh_tensor_holder_ : mesh_tensor_holder_;
}

void DeviceStorage::deallocate() {
    if (!is_allocated()) {
        return;
    }

    get_root_mesh_tensor()->deallocate();
    mesh_tensor_holder_->deallocate();
}

bool DeviceStorage::is_allocated() const { return mesh_tensor_holder_->is_allocated(); }

distributed::MeshDevice* DeviceStorage::get_device_bypass_deallocate_check() const {
    return std::visit(
        ttsl::overloaded{
            [](const MeshTensorHolder::Allocated& allocated) { return &allocated.mesh_tensor_.device(); },
            [](const MeshTensorHolder::DeallocatedTombStone& tombstone) { return tombstone.mesh_buffer_->device(); },
            [](const auto&) -> distributed::MeshDevice* { TT_THROW("Tensor is not allocated"); }},
        mesh_tensor_holder_->state_);
}

bool DeviceStorage::is_uniform_storage() const {
    if (!is_allocated()) {
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
            storages.begin(), storages.end(), [&](const auto& storage) { return storage.get().is_allocated(); }),
        "All DeviceStorages must be allocated");
    TT_FATAL(
        std::all_of(
            storages.begin(),
            storages.end(),
            [&](const auto& storage) {
                return std::addressof(storage.get().get_mesh_tensor()) ==
                       std::addressof(model_storage.get_mesh_tensor());
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
    get_mesh_tensor().update_tensor_topology(tensor_topology);
}

const TensorSpec& DeviceStorage::get_tensor_spec() const {
    return std::visit(
        ttsl::overloaded{
            [](const MeshTensorHolder::Allocated& allocated) -> const TensorSpec& {
                return allocated.mesh_tensor_.tensor_spec();
            },
            [](const MeshTensorHolder::DeallocatedTombStone& tombstone) -> const TensorSpec& {
                return tombstone.tensor_spec_;
            },
            [](const auto&) -> const TensorSpec& { TT_THROW("Tensor is not allocated"); }},
        mesh_tensor_holder_->state_);
}

const TensorTopology& DeviceStorage::get_tensor_topology() const {
    return std::visit(
        ttsl::overloaded{
            [](const MeshTensorHolder::Allocated& allocated) -> const TensorTopology& {
                return allocated.mesh_tensor_.tensor_topology();
            },
            [](const MeshTensorHolder::DeallocatedTombStone& tombstone) -> const TensorTopology& {
                return tombstone.tensor_topology_;
            },
            [](const auto&) -> const TensorTopology& { TT_THROW("Tensor is not allocated"); }},
        mesh_tensor_holder_->state_);
}

}  // namespace tt::tt_metal
