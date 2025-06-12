// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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

struct HostStorage {
    HostBuffer buffer;
    HostStorage() = default;
    HostStorage(HostBuffer buffer_) : buffer(std::move(buffer_)) {}

    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const { return std::forward_as_tuple(); }
};

struct DeviceStorage {
    std::vector<distributed::MeshCoordinate> coords;
    std::shared_ptr<Buffer> buffer;
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer;

    DeviceStorage() = default;
    DeviceStorage(std::shared_ptr<Buffer> buffer_);
    DeviceStorage(
        std::shared_ptr<distributed::MeshBuffer> mesh_buffer_, std::vector<distributed::MeshCoordinate> coords_);

    Buffer* get_buffer() const;
    std::shared_ptr<distributed::MeshBuffer> get_mesh_buffer() const;

    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const { return std::forward_as_tuple(); }

    bool is_allocated() const;

    IDevice* get_device() const;

    // Returns true if the tensor spans across all devices in a mesh.
    bool is_uniform_storage() const;
};

class MultiDeviceHostStorage {
public:
    // TODO: #15840 - Remove this once `HostStorage` and `MultiDeviceHostStorage` are unified.
    std::optional<HostBuffer> get_shard_at_origin() const;

    // Constructor that creates a linearized distributed host buffer from a vector of host buffers.
    // The buffer is re-shaped upon a write to device, to fit the actual shape of the device.
    // TODO: #22169 - Remove this once there are no more usages of this constructor.
    explicit MultiDeviceHostStorage(std::vector<HostBuffer> buffers);

    explicit MultiDeviceHostStorage(DistributedHostBuffer buffer);

    const DistributedHostBuffer& distributed_buffer() const;

    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const { return std::forward_as_tuple(); }

private:
    DistributedHostBuffer distributed_buffer_;
};

using Storage = std::variant<HostStorage, DeviceStorage, MultiDeviceHostStorage>;

}  // namespace tt::tt_metal
