// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "tt-metalium/mesh_coord.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

using OwnedBuffer = std::variant<
    owned_buffer::Buffer<uint8_t>,
    owned_buffer::Buffer<uint16_t>,
    owned_buffer::Buffer<int32_t>,
    owned_buffer::Buffer<uint32_t>,
    owned_buffer::Buffer<float>,
    owned_buffer::Buffer<bfloat16>>;

// HostDataType supports all types included in OwnedBuffer as well as void*
static_assert(
    std::variant_size_v<OwnedBuffer> + 1 == std::variant_size_v<tt::tt_metal::HostDataType>,
    "The data types supported in OwnedBuffer must match those in HostDataType.");
struct OwnedStorage {
    OwnedBuffer buffer;
    OwnedStorage() = default;
    OwnedStorage(OwnedBuffer buffer_) : buffer(std::move(buffer_)) {}

    static constexpr auto attribute_names = std::forward_as_tuple();
    const auto attribute_values() const { return std::forward_as_tuple(); }

    inline void insert_buffer(const OwnedBuffer& buffer_) { this->buffer = buffer_; }

    inline OwnedBuffer get_buffer() const { return this->buffer; }

    inline bool is_allocated() const {
        return std::visit([](auto&& buffer) -> bool { return buffer.is_allocated(); }, buffer);
    }
};

// TODO: #17215 - Replace `DeviceStorage` with "mesh storage".
struct DeviceStorage {
    // TODO: come up with a better abstraction for this.
    DistributedTensorConfig strategy;

    std::shared_ptr<Buffer> buffer;
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer;
    std::map<distributed::MeshCoordinate, TensorSpec> specs;

    DeviceStorage() = default;
    DeviceStorage(std::shared_ptr<Buffer> buffer_);
    DeviceStorage(std::shared_ptr<distributed::MeshBuffer> mesh_buffer_);

    MemoryConfig memory_config() const;
    void insert_buffer(const std::shared_ptr<Buffer>& buffer_);
    Buffer* get_buffer() const;

    static constexpr auto attribute_names = std::forward_as_tuple("memory_config");
    const auto attribute_values() const { return std::make_tuple(this->memory_config()); }

    bool is_allocated() const;
    distributed::MeshBuffer* get_mesh_buffer() const {
        TT_FATAL(mesh_buffer != nullptr, "Mesh buffer is not allocated");
        return mesh_buffer.get();
    }
    IDevice* get_device() const {
        if (mesh_buffer != nullptr) {
            return mesh_buffer->device();
        }
        TT_FATAL(buffer != nullptr, "Buffer is not allocated");
        return buffer->device();
    }
};

using BorrowedBuffer = std::variant<
    borrowed_buffer::Buffer<uint8_t>,
    borrowed_buffer::Buffer<uint16_t>,
    borrowed_buffer::Buffer<int32_t>,
    borrowed_buffer::Buffer<uint32_t>,
    borrowed_buffer::Buffer<float>,
    borrowed_buffer::Buffer<bfloat16>>;
struct BorrowedStorage {
    BorrowedBuffer buffer;

    BorrowedStorage() = default;
    std::function<void()> on_creation_callback = [] {};
    std::function<void()> on_destruction_callback = [] {};

    explicit BorrowedStorage(
        const BorrowedBuffer& buffer,
        const std::function<void()>& on_creation_callback,
        const std::function<void()>& on_destruction_callback) :
        buffer(buffer), on_creation_callback(on_creation_callback), on_destruction_callback(on_destruction_callback) {
        this->on_creation_callback();
    }

    BorrowedStorage(const BorrowedStorage& other) :
        buffer(other.buffer),
        on_creation_callback(other.on_creation_callback),
        on_destruction_callback(other.on_destruction_callback) {
        this->on_creation_callback();
    }

    BorrowedStorage operator=(const BorrowedStorage& other) {
        this->buffer = other.buffer;
        this->on_creation_callback = other.on_creation_callback;
        this->on_destruction_callback = other.on_destruction_callback;
        this->on_creation_callback();
        return *this;
    }

    BorrowedStorage(BorrowedStorage&& other) :
        buffer(other.buffer),
        on_creation_callback(other.on_creation_callback),
        on_destruction_callback(other.on_destruction_callback) {
        other.on_creation_callback = [] {};
        other.on_destruction_callback = [] {};
    }

    BorrowedStorage operator=(BorrowedStorage&& other) {
        this->buffer = other.buffer;
        this->on_creation_callback = other.on_creation_callback;
        this->on_destruction_callback = other.on_destruction_callback;
        other.on_creation_callback = [] {};
        other.on_destruction_callback = [] {};
        return *this;
    }

    ~BorrowedStorage() { this->on_destruction_callback(); }

    static constexpr auto attribute_names = std::forward_as_tuple();
    const auto attribute_values() const { return std::forward_as_tuple(); }

    inline bool is_allocated() const { return true; }
};

struct MultiDeviceHostStorage {
    DistributedTensorConfig strategy;
    std::vector<OwnedBuffer> buffers;
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
        DistributedTensorConfig strategy_, std::vector<OwnedBuffer> buffers_, std::vector<TensorSpec> specs_) :
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
    const auto attribute_values() const { return std::forward_as_tuple(); }

    // Helper Functions - Getters and setters to get/modify storage attributes. These are needed to
    // preinitialize empty tensor handles and use/populate them in the worker threads.
    void insert_buffer_and_spec_for_device(int buffer_index, const OwnedBuffer& buffer, TensorSpec spec) {
        std::lock_guard<std::mutex> lock(mtx);
        buffers[buffer_index] = buffer;
        specs[buffer_index] = std::move(spec);
    }

    OwnedBuffer get_buffer(int buffer_index) const {
        std::lock_guard<std::mutex> lock(mtx);
        TT_FATAL(buffer_index < buffers.size(), "Buffer not found for buffer_index {}", buffer_index);
        return buffers[buffer_index];
    }

    OwnedBuffer& get_buffer(int buffer_index) {
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

    inline bool is_allocated() const {
        // not sure what is better mutex for each buffer 10 times or one here.
        // I think this one is better.
        std::lock_guard<std::mutex> lock(mtx);

        return std::all_of(buffers.begin(), buffers.end(), [](auto&& buffer) {
            return std::visit([](auto&& buffer) -> bool { return buffer.is_allocated(); }, buffer);
        });
    }
};

using Storage = std::variant<OwnedStorage, DeviceStorage, BorrowedStorage, MultiDeviceHostStorage>;

template <typename T>
concept OwnedOrBorrowedStorage = std::is_same_v<T, OwnedStorage> || std::is_same_v<T, BorrowedStorage>;

template <typename T>
constexpr void raise_unsupported_storage() {
    static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported Storage");
}

}  // namespace tt::tt_metal
