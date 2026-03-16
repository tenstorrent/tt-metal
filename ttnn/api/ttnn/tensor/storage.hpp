// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <span>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tuple>

#include "tt-metalium/distributed_host_buffer.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

class HostStorage {
public:
    // Creates HostStorage distributed over a mesh that matches `buffer` shape.
    explicit HostStorage(DistributedHostBuffer buffer);

    // Creates HostStorage distributed over 1x1 mesh.
    explicit HostStorage(HostBuffer buffer);

    // Returns the distributed host buffer.
    const DistributedHostBuffer& buffer() const;

    // Applies a transformation function to each device buffer in parallel, returning a new HostStorage.
    HostStorage transform(const std::function<HostBuffer(const HostBuffer&)>& callable) const;

    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const { return std::forward_as_tuple(); }

private:
    DistributedHostBuffer distributed_buffer_;
};

// DeviceStorage manages a piece of device memory and that is valid for a vector of MeshCoordinates.
//
// DeviceStorage owns the lifetime of the underlying device memory.
// Copying the DeviceStorage will share the ownership of the underlying device memory.
// DeviceStorage currently allow "leaking" of the underlying device memory ownership via get_mesh_buffer(),
// this will be addressed in #39064.
//
// DeviceStorage has two possible states:
// - Allocated: the underlying device memory is allocated.
//   - Can query device memory state/ coordinates/ device.
//   - The MeshCoordinates obtained from get_coords will be within the boundaries of the underlying device memory.
//   - Can be switched to the deallocated state by calling deallocate.
// - Deallocated: the underlying device memory is released.
//   - Query of device memory state/ coordinates/ device will throw.
//   - is_uniform_storage will return true.
//   - Calls to deallocate will have no effect.
struct DeviceStorage {
    // Construct a DeviceStorage that is deallocated
    DeviceStorage() = default;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructs a DeviceStorage from a device memory

    // Constructs DeviceStorage with coords covering the full mesh device shape.
    explicit DeviceStorage(std::shared_ptr<distributed::MeshBuffer> mesh_buffer_);

    // Constructs DeviceStorage that is a view of the mesh_buffer_ at the given coords_.
    // Throws if the coords_ are out of bounds for the mesh_buffer_ device shape.
    DeviceStorage(
        std::shared_ptr<distributed::MeshBuffer> mesh_buffer_, std::vector<distributed::MeshCoordinate> coords_);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Copys an existing DeviceStorage and share it's underlying device memory.

    // Creates a copy of the DeviceStorage that shares the underlying device memory
    DeviceStorage(const DeviceStorage&) = default;
    DeviceStorage(DeviceStorage&&) noexcept = default;
    DeviceStorage& operator=(const DeviceStorage&) = default;
    DeviceStorage& operator=(DeviceStorage&&) noexcept = default;

    // Creates a copy of the DeviceStorage that shares the underlying device memory,
    // but with a different set of coords.
    // Throws if the coords_ are out of bounds for the mesh_buffer_ device shape.
    DeviceStorage(const DeviceStorage& other, std::vector<distributed::MeshCoordinate> coords);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Device Memory getters

    // Get legacy single device buffer
    // Throws if the DeviceStorage is not allocated.
    Buffer* get_buffer() const;

    // Get mesh buffer that represents the device memory
    // Throws if the DeviceStorage is not allocated.
    // TODO(#39064): the ownership sharing will be removed
    const std::shared_ptr<distributed::MeshBuffer>& get_mesh_buffer() const;

    // Get the device the device memory is allocated on
    // Throws if the DeviceStorage is not allocated.
    distributed::MeshDevice* get_device() const;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DeviceStorage as a view of the undelrying device memory at specific coordinates:

    // Returns true if the tensor spans across all devices in a mesh or if the DeviceStorage is not allocated.
    bool is_uniform_storage() const;

    // Returns the coordinates the tensor spans across.
    // Throws if the DeviceStorage is not allocated.
    std::span<const distributed::MeshCoordinate> get_coords() const;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Deallocation management

    // Deallocate the underlying device memory.
    // The underlying device memory could be shared by multiple instances of the DeviceStorage.
    //
    // If force is set, the underlying device memory will be deallocated irespective of the number of device storage
    // owners. If force is not set, the underlying device memory will be deallocated only if "this" is the sole owner of
    // the underlying device memory.
    void deallocate(bool force);

    // Returns true if the underlying device memory is allocated.
    bool is_allocated() const;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Begin internal functions:
    //
    // These functions allows the use of the get_mesh_buffer as a view.
    // These are considered internal functions and are not part of the public API.
    // They will be replaced with a new initiative as described in: #38093
    DeviceStorage(const DeviceStorage& owning_storage, std::shared_ptr<distributed::MeshBuffer> surface_buffer);
    // End internal functions.

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Serialization

    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const { return std::forward_as_tuple(); }

private:
    // Main internal constructor, performs all validation
    DeviceStorage(
        std::shared_ptr<distributed::MeshBuffer> mesh_buffer_,
        std::vector<distributed::MeshCoordinate> coords_,
        std::shared_ptr<distributed::MeshBuffer> root_mesh_buffer_);

    std::vector<distributed::MeshCoordinate> coords_;
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer;

    // Experimental features for viewing an existing DeviceStorage
    const std::shared_ptr<distributed::MeshBuffer>& get_root_mesh_buffer() const;
    void deallocate_root_mesh_buffer();
    void reset_root_mesh_buffer();

    std::shared_ptr<distributed::MeshBuffer> root_mesh_buffer;
    // End experimental features
};

using Storage = std::variant<HostStorage, DeviceStorage>;

}  // namespace tt::tt_metal
