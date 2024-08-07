// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <random>
#include <tuple>
#include <variant>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/bfloat4.hpp"
#include "common/bfloat8.hpp"
#include "common/test_tiles.hpp"
#include "common/tt_backend_api_types.hpp"
#include "ttnn/tensor/types.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/device/device_mesh.hpp"
#include "tt_metal/tt_stl/reflection.hpp"

namespace tt {

namespace tt_metal {

struct Tensor {

    std::optional<std::size_t> tensor_id = std::nullopt;
    Storage storage;
    ttnn::Shape shape;
    DataType dtype;
    Layout layout;

    // ======================================================================================
    //                                  Hi Level APIs
    // ======================================================================================

    Tensor(const Storage storage, const ttnn::Shape shape, DataType dtype, Layout layout);
    Tensor(const Storage storage, const Shape shape, DataType dtype, Layout layout);

    // TODO: should we remove this constructor and disallow creating an uninitialized tensor?
    explicit Tensor() : shape{std::array<uint32_t, 4>{0xff, 0xff, 0xff, 0xff}}, dtype{DataType::INVALID}, layout{Layout::INVALID} {}

    Tensor(const Tensor &other) = default;
    Tensor &operator=(const Tensor &other) = default;

    Tensor(Tensor &&other) noexcept = default;
    Tensor &operator=(Tensor &&other) = default;

    ~Tensor();

    void deallocate(bool force = false);

    std::vector<Device *> get_workers(bool blocking = false) const;

    Tensor to(
        Device *target_device,
        const MemoryConfig &mem_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) const;

    Tensor to(
        DeviceMesh *device_mesh,
        const MemoryConfig &mem_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) const;

    Tensor to(
        CommandQueue &queue,
        const MemoryConfig &mem_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) const;

    Tensor to(
        const std::vector<Device *> &workers,
        const MemoryConfig &mem_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) const;

    Tensor to(Layout target_layout, Device *worker = nullptr) const;

    Tensor to(Layout target_layout, DeviceMesh *device_mesh) const;

    Tensor pad(const Shape &output_tensor_shape, const Shape &input_tensor_start, float pad_value) const;

    Tensor cpu(CommandQueue &queue, bool blocking = true) const;
    Tensor cpu(bool blocking = true) const;

    Tensor cpu_sharded() const;

    Tensor unpad(const Shape &output_tensor_start, const Shape &output_tensor_end) const;

    Tensor pad_to_tile(float pad_value) const;

    Tensor unpad_from_tile(const Shape &output_tensor_shape) const;

    const std::string write_to_string() const;
    void print() const;

    Tensor extract_shard(const CoreCoord &core) const;
    Tensor extract_shard(const uint32_t &core_id) const;

    // ======================================================================================
    //                                  Low Level APIs
    // ======================================================================================
    Tensor reshape(int N, int C, int H, int W) const;
    Tensor reshape(const Shape &new_shape) const;

    // ======================================================================================
    //                                      Getters
    // ======================================================================================
    const Storage &get_storage() const;
    // [[deprecated("Use get_shape() instead.")]]
    const Shape &get_legacy_shape() const;
    const ttnn::Shape &get_shape() const;
    const DataType &get_dtype() const;
    const Layout &get_layout() const;

    // ======================================================================================
    //                                      Setters
    // ======================================================================================
    inline void set_storage(const Storage &storage) { this->storage = storage; }
    inline void set_shape(const ttnn::Shape &shape) { this->shape = shape; }
    inline void set_dtype(const DataType &dtype) { this->dtype = dtype; }
    inline void set_layout(const Layout &layout) { this->layout = layout; }
    // ======================================================================================
    //                                      Extra Helper Functions
    // ======================================================================================
    StorageType storage_type() const;
    const Shape strides() const;
    uint32_t volume() const;
    uint32_t intended_volume() const;

    bool is_allocated() const;

    bool is_contiguous() const {
        if (this->get_layout() == tt::tt_metal::Layout::ROW_MAJOR) {
            return this->get_legacy_shape() == this->get_legacy_shape().without_padding();
        } else {
            return false;
        }
    }

    // TODO(arakhmati): clean up the methods below
    Buffer *buffer() const { return std::get<DeviceStorage>(this->get_storage()).get_buffer().get(); }
    DeviceBuffer device_buffer() const { return std::get<DeviceStorage>(this->get_storage()).get_buffer(); }

    Device *device() const {
        if (this->storage_type() == tt::tt_metal::StorageType::DEVICE) {
            auto buffer = this->buffer();
            if (buffer == nullptr)
                TT_THROW("Cannot get the device from a tensor without an allocated buffer");
            return buffer->device();
        } else if (this->storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE) {
            auto &storage = std::get<MultiDeviceStorage>(this->get_storage());
            return this->get_workers().at(0);
        } else {
            TT_THROW("Cannot get the device from a tensor with host storage");
        }
    }

    const MemoryConfig memory_config() const {
        return std::visit(
            [](const auto &storage) -> MemoryConfig {
                using T = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<T, DeviceStorage>) {
                    return storage.memory_config();
                } else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                    return storage.memory_config();
                } else {
                    TT_THROW("MemoryConfig can only be obtained for a tensor with DeviceStorage");
                }
            },
            this->get_storage());
    }
    const std::optional<ShardSpec> shard_spec() const { return this->memory_config().shard_spec; }

    const bool is_sharded() const;

    // Size in bytes of a single element held in tensor
    uint32_t element_size() const;

    static constexpr auto attribute_names = std::forward_as_tuple("storage", "shape", "dtype", "layout");
    const auto attribute_values() const {
        return std::forward_as_tuple(this->storage, this->shape, this->dtype, this->layout);
    }

    std::vector<uint32_t> host_page_ordering();
};

Tensor create_device_tensor(
    const Shape &shape,
    DataType dtype,
    Layout layout,
    Device *device,
    const MemoryConfig &memory_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED});

static Tensor create_device_tensor(
    const ttnn::Shape &shape,
    DataType dtype,
    Layout layout,
    Device *device,
    const MemoryConfig &memory_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    return create_device_tensor(shape.value, dtype, layout, device, memory_config);
}

// template<typename Buffer>
// void *get_host_buffer(const Tensor &tensor);
void *get_raw_host_data_ptr(const Tensor &tensor);

void memcpy(
    CommandQueue &queue,
    void *dst,
    const Tensor &src,
    const std::optional<std::size_t> transfer_size = std::nullopt,
    bool blocking = true);
void memcpy(
    CommandQueue &queue, Tensor &dst, const void *src, const std::optional<std::size_t> transfer_size = std::nullopt);
void memcpy(
    CommandQueue &queue, Tensor &dst, const Tensor &src, const std::optional<std::size_t> transfer_size = std::nullopt);

void memcpy(
    void *dst, const Tensor &src, const std::optional<std::size_t> transfer_size = std::nullopt, bool blocking = true);
void memcpy(Tensor &dst, const void *src, const std::optional<std::size_t> transfer_size = std::nullopt);
void memcpy(Tensor &dst, const Tensor &src, const std::optional<std::size_t> transfer_size = std::nullopt);

Tensor allocate_tensor_on_device(
    const ttnn::Shape &shape,
    DataType data_type,
    Layout layout,
    Device *device,
    const MemoryConfig &memory_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED});
Tensor allocate_tensor_on_device(
    const ttnn::Shape &shape,
    DataType data_type,
    Layout layout,
    DeviceMesh *device_mesh,
    const MemoryConfig &memory_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED});
void write_tensor(const Tensor& host_tensor, Tensor& device_tensor, uint8_t cq_id = 0);

// Maps a tensor to the set of devices in the device-mesh that the shards will be distributed across.
std::vector<Device*> distribute_tensor_to_mesh(const Tensor& tensor, DeviceMesh& device_mesh);

}  // namespace tt_metal

}  // namespace tt

namespace ttnn {

using Tensor = tt::tt_metal::Tensor;

}  // namespace ttnn
