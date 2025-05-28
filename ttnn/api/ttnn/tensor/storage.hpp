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
    std::vector<std::pair<distributed::MeshCoordinate, TensorSpec>> specs;

    std::shared_ptr<Buffer> buffer;
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer;

    DeviceStorage() = default;
    DeviceStorage(std::shared_ptr<Buffer> buffer_);
    DeviceStorage(
        std::shared_ptr<distributed::MeshBuffer> mesh_buffer_,
        std::vector<std::pair<distributed::MeshCoordinate, TensorSpec>> specs_);

    MemoryConfig memory_config() const;
    Buffer* get_buffer() const;
    std::shared_ptr<distributed::MeshBuffer> get_mesh_buffer() const;

    static constexpr auto attribute_names = std::forward_as_tuple("memory_config");
    auto attribute_values() const { return std::make_tuple(this->memory_config()); }

    bool is_allocated() const;

    IDevice* get_device() const;

    void update_specs(const TensorSpec& new_spec);

    // Returns true if the tensor spans across all devices in a mesh, and all specs are the same.
    bool is_uniform_storage() const;
};

class MultiDeviceHostStorage {
public:
    MultiDeviceHostStorage(std::vector<HostBuffer> buffers, std::vector<TensorSpec> specs);

    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const { return std::forward_as_tuple(); }

    // Returns `HostBuffer` at position `buffer_index`;
    HostBuffer get_buffer(int buffer_index) const;

    // Returns `TensorSpec` at position `spec_index`;
    TensorSpec get_tensor_spec(int spec_index) const;

    // Returns the number of `HostBuffer`s in the storage;
    size_t num_buffers() const;

private:
    std::vector<HostBuffer> buffers_;
    std::vector<TensorSpec> specs_;
};

using Storage = std::variant<HostStorage, DeviceStorage, MultiDeviceHostStorage>;

}  // namespace tt::tt_metal
