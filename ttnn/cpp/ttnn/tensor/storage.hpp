// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "tt-metalium/mesh_coord.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/host_buffer/host_buffer.hpp"

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
    std::vector<distributed::MeshCoordinate> shards;

    std::shared_ptr<Buffer> buffer;
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer;

    DeviceStorage() = default;
    DeviceStorage(std::shared_ptr<Buffer> buffer_);
    DeviceStorage(
        std::shared_ptr<distributed::MeshBuffer> mesh_buffer_, std::vector<distributed::MeshCoordinate> shards_);

    MemoryConfig memory_config() const;
    Buffer* get_buffer() const;
    std::shared_ptr<distributed::MeshBuffer> get_mesh_buffer() const;

    static constexpr auto attribute_names = std::forward_as_tuple("memory_config");
    auto attribute_values() const { return std::make_tuple(this->memory_config()); }

    bool is_allocated() const;

    IDevice* get_device() const;

    // Returns true if the tensor spans across all devices in a mesh, and all specs are the same.
    bool is_uniform_storage() const;
};

struct MultiDeviceHostStorage {
    std::vector<HostBuffer> buffers;
    mutable std::mutex mtx;

    friend void swap(MultiDeviceHostStorage& first, MultiDeviceHostStorage& second) {
        std::scoped_lock lock(first.mtx, second.mtx);
        // enable ADL (not necessary, but good practice)
        using std::swap;

        swap(first.buffers, second.buffers);
    }

    MultiDeviceHostStorage() = default;
    MultiDeviceHostStorage(std::vector<HostBuffer> buffers_) : buffers(std::move(buffers_)) {}
    MultiDeviceHostStorage(MultiDeviceHostStorage&& other) noexcept { swap(*this, other); }
    // unfotunately we need to have this code written manually.
    MultiDeviceHostStorage(const MultiDeviceHostStorage& other) {
        std::scoped_lock lock(other.mtx);
        buffers = other.buffers;
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

    bool operator==(const MultiDeviceHostStorage& other) { return this->buffers == other.buffers; }

    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const { return std::forward_as_tuple(); }

    HostBuffer get_buffer(int buffer_index) const {
        std::lock_guard<std::mutex> lock(mtx);
        TT_FATAL(buffer_index < buffers.size(), "Buffer not found for buffer_index {}", buffer_index);
        return buffers[buffer_index];
    }

    HostBuffer& get_buffer(int buffer_index) {
        std::lock_guard<std::mutex> lock(mtx);
        TT_FATAL(buffer_index < buffers.size(), "Buffer not found for buffer_index {}", buffer_index);
        return buffers[buffer_index];
    }

    uint32_t num_buffers() const {
        std::lock_guard<std::mutex> lock(mtx);
        return buffers.size();
    }

    bool is_allocated() const {
        // not sure what is better mutex for each buffer 10 times or one here.
        // I think this one is better.
        std::lock_guard<std::mutex> lock(mtx);
        return std::all_of(buffers.begin(), buffers.end(), [](auto&& buffer) { return buffer.is_allocated(); });
    }
};

using Storage = std::variant<HostStorage, DeviceStorage, MultiDeviceHostStorage>;

}  // namespace tt::tt_metal
