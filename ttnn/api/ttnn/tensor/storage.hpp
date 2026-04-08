// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <span>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tuple>
#include <vector>

#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
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
    std::shared_ptr<HostTensor> tensor_;
};

// DeviceStorage is a wrapper around the MeshTensor to fit the semantics of ttnn::Tensor.
//
// Different from MeshTensor, DeviceStorage has the following additional features:
// - It can be in a deallocated state.
// - It is copyable, copying a DeviceStorage shares the underlying device memory.
// - It represents a MeshTensor at specific coordinates of the MeshDevice.
//
// Invariant:
// - A default-constructed DeviceStorage acts like a deallocated DeviceStorage. However it is not associated with
// TensorSpec and TensorTopology.
// - An allocated DeviceStorage always holds a non-default constructed MeshTensor. Do not move the MeshTensor out of
//   DeviceStorage.
// - TensorSpec and TensorTopology are always accessible for a DeviceStorage constructed from a MeshTensor. This stays
// true even after deallocate() is called.
// - deallocate() releases the underlying device memory.
// - MeshTensor getters will always throw if the DeviceStorage is deallocated.
struct DeviceStorage {
    // Construct a DeviceStorage that is deallocated
    DeviceStorage();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructs a DeviceStorage from a device memory

    // Constructs DeviceStorage with coords covering the full mesh device shape.
    explicit DeviceStorage(MeshTensor mesh_tensor);

    // Constructs DeviceStorage that is a view of the mesh_buffer_ at the given coords_.
    // Throws if the coords_ are out of bounds for the mesh_buffer_ device shape.
    DeviceStorage(MeshTensor mesh_tensor, std::vector<distributed::MeshCoordinate> coords);

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
    // Throws if the DeviceStorage is deallocated.
    const distributed::MeshBuffer& get_mesh_buffer() const;

    // Get the underlying MeshTensor, throws if the DeviceStorage is deallocated.
    const MeshTensor& get_mesh_tensor() const;

    // Get the underlying MeshTensor, throws if the DeviceStorage is deallocated.
    // Please do not move the MeshTensor out of the DeviceStorage using this function.
    MeshTensor& get_mesh_tensor();

    // Returns the MeshDevice associated with the underlying device memory.
    // Throws if the DeviceStorage is not constructed from a MeshTensor.
    //
    // Workaround for https://github.com/tenstorrent/tt-metal/issues/40716:
    // When DeviceStorage is copied (e.g. view/reshape) and the original is deallocated, the copy's
    // holder becomes DeallocatedTombStone while the MeshBuffer reference is still present.
    // This path preserves a valid device pointer when constructing new tensors from such storage.
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
    // Deallocate will deallocate the device memory of among all shared instances.
    void deallocate();

    // Returns true when DeviceStorage is allocated and is the sole owner of the underlying MeshTensor.
    bool is_sole_owner_of_device_memory() const;

    // Returns true if the underlying device memory is allocated.
    bool is_allocated() const;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Begin internal functions:

    // Returns the MeshBuffer associated with the underlying device memory.
    // This function should be removed in-favor of `get_mesh_tensor()`.
    // The function also leaks the ownership of the underlying device memory out.
    // This is meant to be transitional and is to be removed.
    // Throws if the DeviceStorage is not constructed from a MeshTensor.
    std::shared_ptr<distributed::MeshBuffer> get_mesh_buffer_leak_ownership() const;

    // There are situations where we want to "reinterpret" an existing Tensor without modifying its underlying memory.
    // For example, select slice ops can be done in-place, as can select reshapes. This DeviceStorage constructor
    // addresses such cases.
    //  - owning_storage is the original DeviceStorage object that owns the device memory
    //  - reinterpreted_mesh_tensor is the new interpretation of the device memory
    //
    // Upon construction, ownership of the underlying device memory is SHARED by the new DeviceStorage and the original
    // owning_storage.
    //
    // Note that deallocation will affect BOTH the new DeviceStorage and the original owning_storage.
    //
    // This is currently the recommended method to reinterpret an existing Tensor.
    // This is  internal functionality: it is not part of the public API.
    // TODO: implement a more robust mechanism for Tensor reinterpretation (#38093)
    DeviceStorage(const DeviceStorage& owning_storage, MeshTensor reinterpreted_mesh_tensor);
    // End internal functions.

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Aggregate DeviceStorages
    //
    // Combines a vector of DeviceStorages that shares the same device storage but spread across difference coordinates
    // into a single DeviceStorage.
    //
    // Throws if:
    // - The vector of DeviceStorages is empty.
    // - The vector of DeviceStorages does not share the same device storage.
    // - Any DeviceStorage is deallocated.
    static DeviceStorage combine_device_storages(
        const std::vector<std::reference_wrapper<const DeviceStorage>>& storages, int shard_dim);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Update tensor topology
    // Throws if the device storage is deallocated.
    void update_tensor_topology(const TensorTopology& tensor_topology);

    // Returns the tensor spec associated with the MeshTensor.
    // Throws if the DeviceStorage is not constructed from a MeshTensor.
    const TensorSpec& get_tensor_spec() const;
    // Returns the tensor topology associated with the MeshTensor.
    // Throws if the DeviceStorage is not constructed from a MeshTensor.
    const TensorTopology& get_tensor_topology() const;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Serialization

    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const { return std::forward_as_tuple(); }

private:
    struct MeshTensorHolder;

    // Main internal constructor, performs all validation
    DeviceStorage(
        std::shared_ptr<MeshTensorHolder> mesh_tensor_holder,
        std::vector<distributed::MeshCoordinate> coords,
        std::shared_ptr<MeshTensorHolder> root_mesh_tensor_holder);

    std::shared_ptr<MeshTensorHolder> mesh_tensor_holder_;
    std::vector<distributed::MeshCoordinate> coords_;

    // Experimental features for viewing an existing DeviceStorage
    const std::shared_ptr<MeshTensorHolder>& get_root_mesh_tensor() const;
    std::shared_ptr<MeshTensorHolder> root_mesh_tensor_holder_;
    // End experimental features
};

using Storage = std::variant<HostStorage, DeviceStorage>;

}  // namespace tt::tt_metal
