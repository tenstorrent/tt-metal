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

struct DeviceStorage {
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

    Buffer* get_buffer() const;
    const std::shared_ptr<distributed::MeshBuffer>& get_mesh_buffer() const;
    void deallocate(bool force);

    // Begin internal functions:
    //
    // These functions allows the use of the get_mesh_buffer as a view.
    // These are considered internal functions and are not part of the public API.
    // They will be replaced with a new initiative as described in: #38093
    DeviceStorage(const DeviceStorage& owning_storage, std::shared_ptr<distributed::MeshBuffer> surface_buffer);
    // End internal functions.

    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const { return std::forward_as_tuple(); }

    bool is_allocated() const;

    distributed::MeshDevice* get_device() const;

    // Returns true if the tensor spans across all devices in a mesh.
    bool is_uniform_storage() const;

    // Returns the coordinates the tensor spans across.
    std::span<const distributed::MeshCoordinate> get_coords() const { return coords_; }

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
