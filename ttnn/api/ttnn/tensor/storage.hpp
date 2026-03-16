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
#include "tt-metalium/experimental/tensor/host_tensor.hpp"
#include <ttnn/tensor/tensor_spec.hpp>
#include <ttnn/distributed/tensor_topology.hpp>
#include "ttnn/tensor/types.hpp"


namespace tt::tt_metal {

class HostStorage {
public:
    // Creates HostStorage distributed over a mesh that matches `buffer` shape.
    [[deprecated("Use HostStorage(HostTensor tensor) instead")]]
    explicit HostStorage(DistributedHostBuffer buffer);

    // Creates HostStorage distributed over 1x1 mesh.
    [[deprecated("Use HostStorage(HostTensor tensor) instead")]]
    explicit HostStorage(HostBuffer buffer);

    // Creates HostStorage from a HostTensor.
    explicit HostStorage(HostTensor tensor);

    // Transitional constructors: accept a pre-transition HostStorage (constructed
    // without TensorSpec and TensorTopology) and assign them during construction.
    // Overrides any existing spec/topology in the HostStorage.
    //
    // TODO(#40348): Remove these.
    HostStorage(const HostStorage& other, TensorSpec spec, TensorTopology topology);
    HostStorage(HostStorage&& other, TensorSpec spec, TensorTopology topology);

    // Returns the distributed host buffer.
    const DistributedHostBuffer& buffer() const;

    // Returns the host tensor.
    const HostTensor& host_tensor() const;
    HostTensor& host_tensor();

    // Applies a transformation function to each device buffer in parallel, returning a new HostStorage.
    HostStorage transform(const std::function<HostBuffer(const HostBuffer&)>& callable) const;

    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const { return std::forward_as_tuple(); }

private:
    HostTensor tensor;
};

struct DeviceStorage {
    // Construct a DeviceStorage that is deallocated
    DeviceStorage() = default;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructs a DeviceStorage from a device memory

    // Constructs DeviceStorage with coords covering the full mesh device shape.
    explicit DeviceStorage(std::shared_ptr<distributed::MeshBuffer> mesh_buffer_);

    // Constructs DeviceStorage that is a view of the mesh_buffer_ at the given coords_
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
    DeviceStorage(const DeviceStorage& other, std::vector<distributed::MeshCoordinate> coords);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Device Memory getters

    // Get legacy single device buffer
    Buffer* get_buffer() const;

    // Get mesh buffer that represents the device memory
    const distributed::MeshBuffer& get_mesh_buffer() const;

    // Get the device the device memory is allocated on
    distributed::MeshDevice* get_device() const;

    // Returns the MeshDevice pointer if mesh_buffer exists, or nullptr otherwise.
    // Unlike get_device(), this does NOT throw when the buffer is deallocated.
    //
    // Workaround for https://github.com/tenstorrent/tt-metal/issues/40716:
    // When a tensor's DeviceStorage is copied (e.g., by view/reshape) and the original
    // is deallocated, the copy's MeshBuffer is in DeallocatedState but still exists.
    // This function allows retrieving the device even in that state, preventing nullptr
    // device propagation when constructing new tensors from such storage.
    //
    // TODO: Remove this workaround once models properly manage tensor lifetimes and
    // don't operate on deallocated tensors.
    distributed::MeshDevice* get_device_bypass_deallocate_check() const;


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DeviceStorage as a view of the undelrying device memory at specific coordinates:

    // Returns true if the tensor spans across all devices in a mesh.
    bool is_uniform_storage() const;

    // Returns the coordinates the tensor spans across.
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

    // Returns true if no other DeviceStorage or third party has a shared reference to the device memory (MeshBuffer).
    bool is_sole_owner_of_device_memory() const;

    // Returns true if the underlying device memory is allocated.
    bool is_allocated() const;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Begin internal functions:
    std::shared_ptr<distributed::MeshBuffer> get_mesh_buffer_leak_ownership() const;
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
