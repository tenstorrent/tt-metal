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
    HostTensor tensor;
};

// DeviceStorage manages a piece of device memory and that is valid for a vector of MeshCoordinates.
//
// DeviceStorage owns the lifetime of the underlying device memory.
// Copying the DeviceStorage will share the ownership of the underlying device memory.
//
// DeviceStorage has two possible states:
// - Allocated: the underlying device memory is allocated.
//   - Can query device memory state/ coordinates/ device.
//   - The MeshCoordinates obtained from get_coords will be within the boundaries of the underlying device memory.
//   - Can be switched to the deallocated state by calling deallocate.
// - Deallocated: the underlying device memory is released.
//   - Query of device memory state/ device will throw.
//   - get_coords/ is_uniform_storage will be undefined (for transition, ideally they should all throw).
//   - Calls to deallocate will have no effect.
//
// Right now the deallocated state is only restircted to have a nullptr to the MeshTensor,
// However, the MeshTensor could still be in a deallocated state when other owners of DeviceStorage calls deallocate.
// This is problematic in two ways:
// 1. This breaks the invariant that the MeshTensor is always allocated.
// 2. The definition of deallocated state is not conforming to `DeviceStorage::is_allocated()`.
//
// Long term, "deallocated" and `!is_allocated()` should be synonymous. Migration is difficult: some
// operations (e.g. move) still rely on accessing tensors in a deallocated state, and asserts do not
// cover every intentional or accidental use of deallocated tensors.
struct DeviceStorage {
    // Construct a DeviceStorage that is deallocated
    DeviceStorage() = default;

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
    // Throws if the DeviceStorage is not allocated.
    const distributed::MeshBuffer& get_mesh_buffer() const;

    const MeshTensor& get_mesh_tensor() const;
    MeshTensor& get_mesh_tensor();

    // Get the device the device memory is allocated on
    // Throws if the DeviceStorage is not allocated.
    distributed::MeshDevice& get_device() const;

    // Returns the MeshDevice pointer if mesh_tensor exists, or nullptr otherwise.
    // Unlike get_device(), this does NOT throw when the mesh_tensor exist but it's mesh_buffer being in a deallocated
    // state.
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
    // Deallocate will deallocate the device memory of among all shared instances.
    void deallocate();

    // Returns true if no other DeviceStorage shares this storage's MeshTensor handle(s) (surface and optional root).
    bool is_sole_owner_of_device_memory() const;

    // Returns true if the underlying device memory is allocated.
    bool is_allocated() const;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Begin internal functions:
    std::shared_ptr<distributed::MeshBuffer> get_mesh_buffer_leak_ownership() const;
    //
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
    static DeviceStorage combine_device_storages(
        const std::vector<std::reference_wrapper<const DeviceStorage>>& storages, int shard_dim);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Update tensor topology
    void update_tensor_topology(const TensorTopology& tensor_topology);

    const TensorSpec& get_tensor_spec() const;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Serialization

    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const { return std::forward_as_tuple(); }

private:
    struct LocallyAllocatedState {
        // Engaged and actively sharing an allocated MeshTensor.
        LocallyAllocatedState() = default;
        explicit LocallyAllocatedState(MeshTensor mesh_tensor) :
            mesh_tensor_(std::make_shared<MeshTensor>(std::move(mesh_tensor))) {}
        std::shared_ptr<MeshTensor> mesh_tensor_;

        const TensorSpec& get_tensor_spec() const { return mesh_tensor_->tensor_spec(); }
        const TensorTopology& get_tensor_topology() const { return mesh_tensor_->tensor_topology(); }
    };
    struct DeallocatedState {
        // Tombstone of a deallocated MeshTensor.
        // Transitionally preserve the tensor spec for backwards compatibility.
        explicit DeallocatedState(const MeshTensor& mesh_tensor) :
            tensor_spec_(mesh_tensor.tensor_spec()), tensor_topology_(mesh_tensor.tensor_topology()) {}
        TensorSpec tensor_spec_;
        TensorTopology tensor_topology_;

        const TensorSpec& get_tensor_spec() const { return tensor_spec_; }
        const TensorTopology& get_tensor_topology() const { return tensor_topology_; }
    };

    using States = std::variant<LocallyAllocatedState, DeallocatedState>;

    // Main internal constructor, performs all validation
    DeviceStorage(
        States state, std::vector<distributed::MeshCoordinate> coords, std::shared_ptr<MeshTensor> root_mesh_tensor);

    const std::shared_ptr<MeshTensor>& get_mesh_tensor_bypass_deallocate_check() const;

    States state_;
    std::vector<distributed::MeshCoordinate> coords_;

    // Experimental features for viewing an existing DeviceStorage
    const std::shared_ptr<MeshTensor>& get_root_mesh_tensor() const;
    std::shared_ptr<MeshTensor> root_mesh_tensor_;
    // End experimental features
};

using Storage = std::variant<HostStorage, DeviceStorage>;

}  // namespace tt::tt_metal
