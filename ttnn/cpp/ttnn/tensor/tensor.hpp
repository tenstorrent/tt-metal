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
#include "ttnn/any_device.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/tile/tile.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/tt_stl/reflection.hpp"
#include "types.hpp"

namespace tt {

namespace tt_metal {

namespace distributed {
class MeshDevice;
}

struct Tensor {
    struct TensorAttributes : public std::enable_shared_from_this<TensorAttributes> {
        Storage storage;
        TensorSpec tensor_spec;
        uint32_t num_shards_to_be_populated = 0;
        uint32_t main_thread_ref_count = 0;
        std::atomic<uint32_t> num_sibling_workers_sharing_tensor = 0;
        std::atomic<bool> main_thread_tensor = true;
        std::atomic<bool> metadata_populated = false;
        std::atomic<int> num_workers_completed = 0;
        bool deallocated = false;      // Set to true if device side storage was deallocated
        bool dynamic_storage = false;  // Storage type can change, depending on op behaviour
        bool track_ref_count = false;
        TensorAttributes(Storage storage, TensorSpec tensor_spec);
        TensorAttributes();
        ~TensorAttributes() = default;

        // Use these functions to manage the main_thread_ref_count for a tensor attr instance.
        // This variable is used for on device memory deallocation in async mode, where the main
        // thread owns all tensors and enqueues a deallocate command for each shard, when a tensor
        // is implicitly or explicitly dellocated.
        // Call increment when a tensor is default, copy or assignment constructed, since an additional
        // object will own a tensor_attr instance.
        // Call decrement when a tensor is destroyed and the number of owners of the tensor_attr object
        // decreases.
        // Record the main thread ref count before pushing to a worker queue (number of owners in main thread).
        // Update the main thread ref count with the recorded value after the tensor is pushed to the queue(s),
        // since pushing to the queue requires an extra datacopy in the main thread, that gets balanced by the
        // worker, however the worker cannot modify main_thread_ref_count.
        void increment_main_thread_ref_count(Device* worker);

        void decrement_main_thread_ref_count(Device* worker);

        uint32_t record_main_thread_ref_count();

        void update_main_thread_ref_count(Device* worker, uint32_t ref_count);
    };

    std::optional<std::int64_t> tensor_id = std::nullopt;
    // Shared pointer to all attributes associated with this tensor
    // Can be safely passed between threads when the tensor is copied
    std::shared_ptr<TensorAttributes> tensor_attributes = nullptr;

    // Tensor gets worker queue handle through the device
    std::vector<Device*> workers = {};
    bool deallocate_through_destructor = false;

    // ======================================================================================
    //                                  Hi Level APIs
    // ======================================================================================
    explicit Tensor() = default;

    Tensor(
        Storage storage,
        const ttnn::Shape& shape,
        DataType dtype,
        Layout layout,
        const std::optional<Tile>& tile = std::nullopt);
    Tensor(
        Storage storage,
        const ttnn::SimpleShape& shape,
        DataType dtype,
        Layout layout,
        const std::optional<Tile>& tile = std::nullopt);
    Tensor(
        Storage storage,
        const ttnn::SimpleShape& logical_shape,
        const ttnn::SimpleShape& padded_shape,
        DataType dtype,
        Layout layout,
        const std::optional<Tile>& tile = std::nullopt);
    Tensor(Storage storage, TensorSpec tensor_spec);

    // Constructors to initialize unpopulated tensor with workers and storage specified. Use this when creating tensor
    // handles in async mode.
    explicit Tensor(
        uint32_t num_buffers, std::optional<DistributedTensorConfig> distributed_tensor_config = std::nullopt);
    explicit Tensor(const std::vector<Device*>& workers);

    Tensor(const Tensor& other);

    Tensor& operator=(const Tensor& other);

    Tensor(Tensor&& other) noexcept = default;

    Tensor& operator=(Tensor&& other) {
        // Don't self assign
        this->tensor_id = std::move(other.tensor_id);
        if (this->tensor_attributes != other.tensor_attributes) {
            // Update ref count for curr tensor_attr and deallocate if needed
            perform_cleanup_for_async_mode();
            this->workers = std::move(other.workers);
            this->tensor_attributes = std::move(other.tensor_attributes);
            this->deallocate_through_destructor = std::move(other.deallocate_through_destructor);
        }
        return *this;
    }

    ~Tensor();

    void track_ref_count() { this->tensor_attributes->track_ref_count = true; }

    void perform_cleanup_for_async_mode();

    void populate_buffers_and_metadata(const Tensor& other);

    void deallocate(bool force = false);

    std::vector<Device*> get_workers(bool blocking = false) const;

    // Converts a buffer of elements of type `T` to a `Tensor`.
    // Elements in the buffer are assumed to be stored in row-major order. The size of the buffer and the type of the
    // elements have to match `spec`; block float formats such as BFLOAT8_B and BFLOAT4_B require `T` equal `float`.
    //
    // The data in the buffer is copied into a tensor with an owned storage.
    //
    // TODO: add support for returning a tensor with borrowed storage based off the buffer.
    // TODO: handle tilization and padding in face of sharding.
    template <typename T>
    static Tensor from_span(
        tt::stl::Span<const T> buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device = std::nullopt);

    // Same as `from_span`, but takes a vector instead.
    template <typename T>
    static Tensor from_vector(
        const std::vector<T>& buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device = std::nullopt) {
        return from_span(tt::stl::Span<const T>(buffer.data(), buffer.size()), spec, device);
    }

    // Converts a `Tensor` to a `std::vector<T>`.
    // Elements in the vector will be stored in row-major order. The type of the requested vector has to match that of
    // the `Tensor`; block float formats such as BFLOAT8_B and BFLOAT4_B require `T` equal `float`.
    //
    // If the tensor resides on a device, it will be brough back to host.
    //
    // TODO: handle tilization and padding in face of sharding.
    template <typename T>
    std::vector<T> to_vector() const;

    Tensor to(
        Device* target_device,
        const MemoryConfig& mem_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED},
        uint8_t cq_id = ttnn::DefaultQueueId,
        const std::vector<SubDeviceId>& sub_device_ids = {}) const;

    Tensor to(
        distributed::MeshDevice* mesh_device,
        const MemoryConfig& mem_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED},
        uint8_t cq_id = ttnn::DefaultQueueId,
        const std::vector<SubDeviceId>& sub_device_ids = {}) const;

    Tensor to(
        const std::vector<Device*>& workers,
        const MemoryConfig& mem_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED},
        uint8_t cq_id = ttnn::DefaultQueueId,
        const std::vector<SubDeviceId>& sub_device_ids = {}) const;

    Tensor to(Layout target_layout, Device* worker = nullptr) const;

    Tensor to(Layout target_layout, distributed::MeshDevice* mesh_device) const;

    Tensor pad(
        const ttnn::SimpleShape& output_padded_shape,
        const ttnn::SimpleShape& input_tensor_start,
        float pad_value) const;

    Tensor cpu(
        bool blocking = true,
        uint8_t cq_id = ttnn::DefaultQueueId,
        const std::vector<SubDeviceId>& sub_device_ids = {}) const;

    Tensor unpad(const ttnn::SimpleShape& output_tensor_start, const ttnn::SimpleShape& output_tensor_end) const;

    Tensor pad_to_tile(float pad_value) const;

    Tensor unpad_from_tile(const ttnn::SimpleShape& output_tensor_shape) const;

    const std::string write_to_string() const;
    void print() const;

    Tensor extract_shard(const CoreCoord& core) const;
    Tensor extract_shard(const uint32_t& core_id) const;

    // ======================================================================================
    //                                  Low Level APIs
    // ======================================================================================
    Tensor reshape(const ttnn::SimpleShape& new_shape) const;
    Tensor reshape(const ttnn::Shape& new_shape) const;

    // ======================================================================================
    //                                      Getters
    // ======================================================================================
    const Storage& get_storage() const;
    DataType get_dtype() const;
    Layout get_layout() const;
    const ttnn::SimpleShape& get_logical_shape() const;
    const ttnn::SimpleShape& get_padded_shape() const;
    const TensorSpec& get_tensor_spec() const;

    // [[deprecated("Use get_shape() instead.")]]
    tt::tt_metal::LegacyShape get_legacy_shape() const;
    ttnn::Shape get_shape() const;
    tt::tt_metal::Padding get_padding() const;

    // ======================================================================================
    // Non-Blocking Getters. Query attributes directly, without waiting for worker completion
    // ======================================================================================
    inline const Storage& storage() const { return this->tensor_attributes->storage; };
    inline tt::tt_metal::LegacyShape legacy_shape() const {
        return this->tensor_attributes->tensor_spec.shape().value;
    };
    inline ttnn::Shape shape() const { return this->tensor_attributes->tensor_spec.shape(); };
    inline const ttnn::SimpleShape& logical_shape() const {
        return this->tensor_attributes->tensor_spec.logical_shape();
    };
    inline const ttnn::SimpleShape& padded_shape() const {
        return this->tensor_attributes->tensor_spec.padded_shape();
    };
    inline DataType dtype() const { return this->tensor_attributes->tensor_spec.tensor_layout().get_data_type(); };
    inline Layout layout() const { return this->tensor_attributes->tensor_spec.tensor_layout().get_layout(); };
    inline const TensorSpec& tensor_spec() const { return this->tensor_attributes->tensor_spec; }

    // ======================================================================================
    //                                      Setters
    // ======================================================================================
    inline void set_storage(const Storage& storage) { this->tensor_attributes->storage = storage; }
    // We intend to remove this API once we migrate all ops to compute_output_specs, and provide TensorSpec at creation
    inline void set_tensor_spec(const TensorSpec& tensor_spec) {
        this->tensor_attributes->tensor_spec = tensor_spec;
        this->tensor_attributes->metadata_populated = true;
    }
    // ======================================================================================
    //                                      Extra Helper Functions
    // ======================================================================================
    StorageType storage_type() const;
    const ttnn::SimpleShape strides() const;
    uint32_t volume() const;

    // todo: rename volume to get_volume to indicate that its blocking
    uint32_t get_logical_volume() const;

    bool is_scalar() const;

    bool is_allocated() const;

    bool is_contiguous() const {
        if (this->get_layout() == tt::tt_metal::Layout::ROW_MAJOR) {
            return this->get_legacy_shape() == this->get_legacy_shape().without_padding();
        } else {
            return false;
        }
    }

    // TODO(arakhmati): clean up the methods below
    std::vector<Buffer*> buffers() const {
        auto storage_type = this->storage_type();
        if (storage_type == tt::tt_metal::StorageType::DEVICE) {
            auto storage = std::get<DeviceStorage>(this->get_storage());
            return std::vector<Buffer*>{storage.get_buffer().get()};
        } else if (storage_type == tt::tt_metal::StorageType::MULTI_DEVICE) {
            std::vector<Buffer*> buffers;
            auto storage = std::get<MultiDeviceStorage>(this->get_storage());
            for (const auto& buffer : storage.get_buffers()) {
                buffers.push_back(buffer.get());
            }
            return buffers;
        } else {
            TT_THROW("Cannot get buffers from a tensor with non-device storage.");
        }
    }
    Buffer* buffer() const {
        auto storage_type = this->storage_type();
        TT_FATAL(
            storage_type == tt::tt_metal::StorageType::DEVICE,
            "ttnn::Tensor::buffer(): Expected Tensor with DeviceStorage, got {}",
            storage_type);
        return std::get<DeviceStorage>(this->get_storage()).get_buffer().get();
    }
    DeviceBuffer device_buffer() const { return std::get<DeviceStorage>(this->get_storage()).get_buffer(); }

    Device* device() const {
        if (this->storage_type() == tt::tt_metal::StorageType::DEVICE) {
            auto buffer = this->buffer();
            if (buffer == nullptr) {
                TT_THROW("Cannot get the device from a tensor without an allocated buffer");
            }
            return buffer->device();
        } else if (this->storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE) {
            return this->get_workers().at(0);
        } else {
            TT_THROW("Cannot get the device from a tensor with host storage");
        }
    }

    const MemoryConfig& memory_config() const { return get_tensor_spec().tensor_layout().get_memory_config(); }
    const std::optional<ShardSpec>& shard_spec() const { return this->memory_config().shard_spec; }

    const bool is_sharded() const;

    // Size in bytes of a single element held in tensor
    uint32_t element_size() const;

    static constexpr auto attribute_names = std::forward_as_tuple("storage", "tensor_spec");
    const auto attribute_values() const {
        return std::forward_as_tuple(this->tensor_attributes->storage, this->tensor_attributes->tensor_spec);
    }

    std::vector<uint32_t> host_page_ordering();

    // Main Thread - Wait for all workers in this tensor to populate the entire tensor
    inline void wait_for_tensor_data_populated() const {
        // Stall until all the workers for this tensor
        // have populated the full tensor
        while (this->tensor_attributes->num_workers_completed < this->tensor_attributes->num_shards_to_be_populated) {
        }
    }

    // Main Thread - Wait for the first worker in this tensor to populate the global metadata fields
    inline void wait_for_tensor_metadata_populated() const {
        // First worker is responsible for updating all metadata fields
        // Stall until this worker is done
        while (not this->tensor_attributes->metadata_populated) {
        }
    }

private:
    void init(Storage storage, TensorSpec tensor_spec);
};

Tensor create_device_tensor(const TensorSpec& tensor_spec, Device* device);

[[deprecated]]
Tensor create_device_tensor(
    const ttnn::SimpleShape& shape,
    DataType dtype,
    Layout layout,
    Device* device,
    const MemoryConfig& memory_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED},
    const std::optional<Tile>& tile = std::nullopt);

// TODO: Remove once ALL ops switch over to return ttnn::SimpleShape in compute_output_shapes
[[deprecated("Use create_device_tensor(const TensorSpec&, Device*) instead")]]
Tensor create_device_tensor(
    const ttnn::Shape& shape,
    DataType dtype,
    Layout layout,
    Device* device,
    const MemoryConfig& memory_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED},
    const std::optional<Tile>& tile = std::nullopt);

// template<typename Buffer>
// void *get_host_buffer(const Tensor &tensor);
void* get_raw_host_data_ptr(const Tensor& tensor);

void memcpy(CommandQueue& queue, void* dst, const Tensor& src, const bool blocking = true);
void memcpy(
    CommandQueue& queue,
    void* dst,
    const Tensor& src,
    const size_t offset,
    const size_t size,
    const bool blocking = true);

void memcpy(CommandQueue& queue, Tensor& dst, const void* src);
void memcpy(CommandQueue& queue, Tensor& dst, const void* src, const size_t offset, const size_t size);

void memcpy(CommandQueue& queue, Tensor& dst, const Tensor& src);
void memcpy(CommandQueue& queue, Tensor& dst, const Tensor& src, const size_t offset, const size_t size);

void memcpy(void* dst, const Tensor& src, const bool blocking = true);
void memcpy(void* dst, const Tensor& src, const size_t offset, const size_t size, const bool blocking = true);

void memcpy(Tensor& dst, const void* src);
void memcpy(Tensor& dst, const void* src, const size_t offset, const size_t size);

void memcpy(Tensor& dst, const Tensor& src);
void memcpy(Tensor& dst, const Tensor& src, const size_t offset, const size_t size);

Tensor allocate_tensor_on_devices(
    const ttnn::Shape& shape,
    DataType data_type,
    Layout layout,
    const std::vector<Device*>& devices,
    const MemoryConfig& memory_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED},
    const std::optional<Tile>& tile = std::nullopt);
void write_tensor(
    const Tensor& host_tensor,
    Tensor device_tensor,
    uint8_t cq_id = ttnn::DefaultQueueId,
    const std::vector<SubDeviceId>& sub_device_ids = {});

Tensor set_tensor_id(const Tensor& tensor);

bool validate_worker_modes(const std::vector<Device*>& workers);

}  // namespace tt_metal

}  // namespace tt

namespace ttnn {

using Tensor = tt::tt_metal::Tensor;
using TensorSpec = tt::tt_metal::TensorSpec;

}  // namespace ttnn
