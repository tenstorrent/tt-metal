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

    void insert_buffer(const HostBuffer& buffer_) { this->buffer = buffer_; }

    HostBuffer get_buffer() const { return this->buffer; }

    bool is_allocated() const { return buffer.is_allocated(); }
};

struct DeviceStorage {
    // TODO: come up with a better abstraction for this.
    DistributedTensorConfig strategy;
    std::vector<std::pair<distributed::MeshCoordinate, TensorSpec>> specs;

    std::shared_ptr<Buffer> buffer;
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer;

    DeviceStorage() = default;
    DeviceStorage(std::shared_ptr<Buffer> buffer_);
    DeviceStorage(
        std::shared_ptr<distributed::MeshBuffer> mesh_buffer_,
        DistributedTensorConfig strategy_,
        std::vector<std::pair<distributed::MeshCoordinate, TensorSpec>> specs_);

    MemoryConfig memory_config() const;
    void insert_buffer(const std::shared_ptr<Buffer>& buffer_);
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
    DistributedTensorConfig strategy;
    std::vector<HostBuffer> buffers;
    std::vector<TensorSpec> specs;
    mutable std::mutex mtx;

    friend void swap(MultiDeviceHostStorage& first, MultiDeviceHostStorage& second) {
        std::scoped_lock lock(first.mtx, second.mtx);
        // enable ADL (not necessary, but good practice)
        using std::swap;

        swap(first.strategy, second.strategy);
        swap(first.buffers, second.buffers);
        swap(first.specs, second.specs);
    }

    MultiDeviceHostStorage() = default;
    MultiDeviceHostStorage(
        DistributedTensorConfig strategy_, std::vector<HostBuffer> buffers_, std::vector<TensorSpec> specs_) :
        strategy(strategy_), buffers(std::move(buffers_)), specs(std::move(specs_)) {}
    MultiDeviceHostStorage(MultiDeviceHostStorage&& other) { swap(*this, other); }
    // unfotunately we need to have this code written manually.
    MultiDeviceHostStorage(const MultiDeviceHostStorage& other) {
        std::scoped_lock lock(other.mtx);
        strategy = other.strategy;
        buffers = other.buffers;
        specs = other.specs;
    }

    MultiDeviceHostStorage& operator=(const MultiDeviceHostStorage& other) {
        MultiDeviceHostStorage temp(other);
        swap(*this, temp);
        return *this;
    }

    MultiDeviceHostStorage& operator=(MultiDeviceHostStorage&& other) {
        swap(*this, other);
        return *this;
    }

    bool operator==(const MultiDeviceHostStorage& other) {
        return this->strategy == other.strategy and this->buffers == other.buffers and this->specs == other.specs;
    }

    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const { return std::forward_as_tuple(); }

    // Helper Functions - Getters and setters to get/modify storage attributes. These are needed to
    // preinitialize empty tensor handles and use/populate them in the worker threads.
    void insert_buffer_and_spec_for_device(int buffer_index, const HostBuffer& buffer, TensorSpec spec) {
        std::lock_guard<std::mutex> lock(mtx);
        buffers[buffer_index] = buffer;
        specs[buffer_index] = std::move(spec);
    }

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

    TensorSpec get_tensor_spec(int spec_index) const {
        std::lock_guard<std::mutex> lock(mtx);
        TT_FATAL(spec_index < specs.size(), "Spec for device {} not found in spec list", spec_index);
        return specs[spec_index];
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
