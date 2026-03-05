// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
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
    std::vector<distributed::MeshCoordinate> coords;

private:
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer;
    // Workaround for managing view MeshBuffer; expected to be refactored in #38093
    std::shared_ptr<distributed::MeshBuffer> root_mesh_buffer;

public:
    DeviceStorage() = default;
    DeviceStorage(
        std::shared_ptr<distributed::MeshBuffer> mesh_buffer_,
        std::vector<distributed::MeshCoordinate> coords_,
        std::shared_ptr<distributed::MeshBuffer> root_buffer_ = nullptr);

    Buffer* get_buffer() const;
    const distributed::MeshBuffer& get_mesh_buffer() const;
    std::shared_ptr<distributed::MeshBuffer> get_mesh_buffer_leak_ownership() const;

    // Begin internal functions:
    //
    // These functions allows the use of the get_mesh_buffer as a view.
    // These are considered internal functions and are not part of the public API.
    // They will be replaced with a new initiative as described in: #38093
    const std::shared_ptr<distributed::MeshBuffer>& get_root_mesh_buffer() const;
    void deallocate_root_mesh_buffer();
    void reset_root_mesh_buffer();
    // End internal functions.

    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const { return std::forward_as_tuple(); }

    bool is_allocated() const;

    distributed::MeshDevice* get_device() const;

    // Returns true if the tensor spans across all devices in a mesh.
    bool is_uniform_storage() const;

    // These are internal functions and should be treated as a public API.
    // They are here to support distributed API.
    DeviceStorage reduce_to_single_device_storage(const distributed::MeshCoordinate& coord) const;
    static DeviceStorage combine_to_multi_device_storage(
        std::span<std::reference_wrapper<const DeviceStorage>> storages);

    // Low level function, strickly internal:
    DeviceStorage with_coords(std::vector<distributed::MeshCoordinate> new_coords) const;
};

using Storage = std::variant<HostStorage, DeviceStorage>;

}  // namespace tt::tt_metal
