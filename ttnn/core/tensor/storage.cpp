// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <iterator>
#include <unordered_set>
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

DeviceStorage::DeviceStorage(std::shared_ptr<distributed::MeshBuffer> mesh_buffer_) :
    mesh_buffer(std::move(mesh_buffer_)) {
    auto* device = this->get_device();
    distributed::MeshCoordinateRange coord_range(device->shape());
    std::copy(coord_range.begin(), coord_range.end(), std::back_inserter(coords));
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

namespace {
namespace CMAKE_UNQIUE_NAMESPACE {
// Returns true if all the coordinates are unique.
bool all_unique_coords(std::span<const distributed::MeshCoordinate> storages) {
    std::unordered_set<distributed::MeshCoordinate> coords_set(storages.begin(), storages.end());
    return storages.size() == coords_set.size();
}
}  // namespace CMAKE_UNQIUE_NAMESPACE
}  // namespace

DeviceStorage DeviceStorage::combine_to_multi_device_storage(
    std::span<std::reference_wrapper<const DeviceStorage>> storages) {
    TT_FATAL(!storages.empty(), "At least one storage must be provided");
    // Check that all storages are allocated on the same mesh buffer.
    auto prototype = storages[0].get().mesh_buffer;
    for (const auto& storage : storages) {
        TT_FATAL(storage.get().mesh_buffer != prototype, "Given storages must be allocated on the same mesh buffer.");
    }

    // Collect all coodinates.
    auto total_num_coords =
        std::accumulate(storages.begin(), storages.end(), std::size_t{0}, [](auto acc, const auto& storage) {
            return acc + storage.get().coords.size();
        });
    std::vector<distributed::MeshCoordinate> coords;
    coords.reserve(total_num_coords);
    for (const auto& storage : storages) {
        for (const auto& coord : storage.get().coords) {
            coords.push_back(coord);
        }
    }

    // Validations:

    // No duplicated coordinates
    TT_FATAL(
        CMAKE_UNQIUE_NAMESPACE::all_unique_coords(coords), "There are duplicate coordinates in the given storages.");

    DeviceStorage result(prototype);
    result.coords = std::move(coords);
    return result;
}

}  // namespace tt::tt_metal
