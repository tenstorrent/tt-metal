// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/host_buffer.hpp>

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

struct HostStorage {
    HostBuffer buffer;
    HostStorage() = default;
    HostStorage(HostBuffer buffer_) : buffer(std::move(buffer_)) {}

    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const { return std::forward_as_tuple(); }

    bool is_allocated() const { return buffer.is_allocated(); }
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

struct MultiDeviceHostStorage {
    std::vector<HostBuffer> buffers;
    std::vector<TensorSpec> specs;

    friend void swap(MultiDeviceHostStorage& first, MultiDeviceHostStorage& second) noexcept {
        // enable ADL (not necessary, but good practice)
        using std::swap;

        swap(first.buffers, second.buffers);
        swap(first.specs, second.specs);
    }

    MultiDeviceHostStorage() = default;
    MultiDeviceHostStorage(std::vector<HostBuffer> buffers_, std::vector<TensorSpec> specs_) :
        buffers(std::move(buffers_)), specs(std::move(specs_)) {}
    MultiDeviceHostStorage(MultiDeviceHostStorage&& other) noexcept { swap(*this, other); }
    // unfotunately we need to have this code written manually.
    MultiDeviceHostStorage(const MultiDeviceHostStorage& other) {
        buffers = other.buffers;
        specs = other.specs;
    }

    MultiDeviceHostStorage& operator=(const MultiDeviceHostStorage& other) {
        MultiDeviceHostStorage temp(other);
        swap(*this, temp);
        return *this;
    }

    MultiDeviceHostStorage& operator=(MultiDeviceHostStorage&& other) noexcept {
        swap(*this, other);
        return *this;
    }

    bool operator==(const MultiDeviceHostStorage& other) {
        return this->buffers == other.buffers and this->specs == other.specs;
    }

    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const { return std::forward_as_tuple(); }

    HostBuffer get_buffer(int buffer_index) const {
        TT_FATAL(buffer_index < buffers.size(), "Buffer not found for buffer_index {}", buffer_index);
        return buffers[buffer_index];
    }

    HostBuffer& get_buffer(int buffer_index) {
        TT_FATAL(buffer_index < buffers.size(), "Buffer not found for buffer_index {}", buffer_index);
        return buffers[buffer_index];
    }

    TensorSpec get_tensor_spec(int spec_index) const {
        TT_FATAL(spec_index < specs.size(), "Spec for device {} not found in spec list", spec_index);
        return specs[spec_index];
    }

    uint32_t num_buffers() const { return buffers.size(); }

    bool is_allocated() const {
        return std::all_of(buffers.begin(), buffers.end(), [](auto&& buffer) { return buffer.is_allocated(); });
    }
};

using Storage = std::variant<HostStorage, DeviceStorage, MultiDeviceHostStorage>;

}  // namespace tt::tt_metal
