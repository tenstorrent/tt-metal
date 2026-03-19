// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
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
    explicit HostStorage(DistributedHostBuffer buffer);

    // Creates HostStorage distributed over 1x1 mesh.
    explicit HostStorage(HostBuffer buffer);

    // Creates HostStorage from a HostTensor.
    explicit HostStorage(HostTensor tensor);

    // Copy a HostStorage with a new TensorSpec and TensorTopology, this is meant for transition
    HostStorage(const HostStorage& other, TensorSpec spec, TensorTopology topology);

    // Move a HostStorage with a new TensorSpec and TensorTopology
    HostStorage(HostStorage&& other, TensorSpec spec, TensorTopology topology);

    // Returns the distributed host buffer.
    const DistributedHostBuffer& buffer() const;

    // Returns the host tensor.
    const HostTensor& host_tensor() const;

    // Applies a transformation function to each device buffer in parallel, returning a new HostStorage.
    HostStorage transform(const std::function<HostBuffer(const HostBuffer&)>& callable) const;

    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const { return std::forward_as_tuple(); }

private:
    HostTensor tensor;
};

struct DeviceStorage {
    std::vector<distributed::MeshCoordinate> coords;
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer;
    // Workaround for managing view MeshBuffer; expected to be refactored in #38093
    std::shared_ptr<distributed::MeshBuffer> root_mesh_buffer;

    DeviceStorage() = default;
    DeviceStorage(
        std::shared_ptr<distributed::MeshBuffer> mesh_buffer_,
        std::vector<distributed::MeshCoordinate> coords_,
        std::shared_ptr<distributed::MeshBuffer> root_buffer_ = nullptr);

    Buffer* get_buffer() const;
    std::shared_ptr<distributed::MeshBuffer> get_mesh_buffer() const;

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
};

using Storage = std::variant<HostStorage, DeviceStorage>;

}  // namespace tt::tt_metal
