// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <random>
#include <tuple>
#include <variant>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/test_tiles.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "ttnn/any_device.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/reflection.hpp>
#include "types.hpp"

namespace tt {

namespace tt_metal {

namespace distributed {
class MeshDevice;
}

class Tensor {
public:
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
        void increment_main_thread_ref_count(IDevice* worker);

        void decrement_main_thread_ref_count(IDevice* worker);

        uint32_t record_main_thread_ref_count();

        void update_main_thread_ref_count(IDevice* worker, uint32_t ref_count);
    };

    std::optional<std::int64_t> tensor_id = std::nullopt;
    // Shared pointer to all attributes associated with this tensor
    // Can be safely passed between threads when the tensor is copied
    std::shared_ptr<TensorAttributes> tensor_attributes = nullptr;

    // Tensor gets worker queue handle through the device
    std::vector<IDevice*> workers = {};

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
        const ttnn::Shape& logical_shape,
        const ttnn::Shape& padded_shape,
        DataType dtype,
        Layout layout,
        const std::optional<Tile>& tile = std::nullopt);
    Tensor(Storage storage, TensorSpec tensor_spec);

    // Constructors to initialize unpopulated tensor with workers and storage specified. Use this when creating tensor
    // handles in async mode.
    explicit Tensor(
        uint32_t num_buffers, std::optional<DistributedTensorConfig> distributed_tensor_config = std::nullopt);
    explicit Tensor(const std::vector<IDevice*>& workers);

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
        }
        return *this;
    }

    ~Tensor();

    void track_ref_count() { this->tensor_attributes->track_ref_count = true; }

    void perform_cleanup_for_async_mode();

    void populate_buffers_and_metadata(const Tensor& other);

    void deallocate(bool force = false);

    std::vector<IDevice*> get_workers(bool blocking = false) const;

    // Converts a buffer of elements of type `T` to a `Tensor`.
    // Elements in the buffer are assumed to be stored in row-major order. The size of the buffer and the type of the
    // elements have to match `spec`; block float formats such as BFLOAT8_B and BFLOAT4_B require `T` equal `float`.
    //
    // The data in the buffer is copied into a tensor with an owned storage.
    template <typename T>
    static Tensor from_span(
        tt::stl::Span<const T> buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device = std::nullopt);

    // Creates a `Tensor` with storage "borrowed" from the buffer of elements of type `T`.
    //
    // The primary use case for this API is to interop with Python, where `on_creation_callback` and
    // `on_destruction_callback` are specified to be called when the tensor storage is created and destroyed (when
    // making copies of Tensor object):
    //
    // py::object py_tensor = ...;
    // auto on_creation_callback = [t = py_tensor] { t.inc_ref(); };
    // auto on_destruction_callback = [t = py_tensor] { t.dec_ref(); };
    //
    // When working in C++, prefer creating owned tensors, and retaining a reference to the internal buffer, if
    // necessary.
    template <typename T>
    static Tensor from_borrowed_data(
        tt::stl::Span<T> buffer,
        const ttnn::Shape& shape,
        const std::function<void()>& on_creation_callback,
        const std::function<void()>& on_destruction_callback,
        const std::optional<Tile>& tile = std::nullopt);

    // Same as `from_span`, but operates on a vector instead.
    template <typename T>
    static Tensor from_vector(
        const std::vector<T>& buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device = std::nullopt) {
        return from_span(tt::stl::Span<const T>(buffer), spec, device);
    }

    // Same as `from_vector`, but takes in an rvalue. No copies will be made, if the target layout is row-major,
    // physical shape matches logical shape, and no type conversion is needed.
    template <typename T>
    static Tensor from_vector(
        std::vector<T>&& buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device = std::nullopt);

    // Converts a `Tensor` to a `std::vector<T>`.
    // Elements in the vector will be stored in row-major order. The type of the requested vector has to match that of
    // the `Tensor`; block float formats such as BFLOAT8_B and BFLOAT4_B require `T` equal `float`.
    //
    // If the tensor resides on a device, it will be brough back to host.
    template <typename T>
    std::vector<T> to_vector() const;

    Tensor to_device(
        IDevice* target_device,
        const MemoryConfig& mem_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED},
        ttnn::QueueId cq_id = ttnn::DefaultQueueId) const;

    Tensor to_device(
        distributed::MeshDevice* mesh_device,
        const MemoryConfig& mem_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED},
        ttnn::QueueId cq_id = ttnn::DefaultQueueId) const;

    Tensor to_device(
        const std::vector<IDevice*>& workers,
        const MemoryConfig& mem_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED},
        ttnn::QueueId cq_id = ttnn::DefaultQueueId) const;

    Tensor to_layout(Layout target_layout, IDevice* worker = nullptr) const;

    Tensor to_layout(Layout target_layout, distributed::MeshDevice* mesh_device) const;

    Tensor pad(const ttnn::Shape& output_padded_shape, const ttnn::Shape& input_tensor_start, float pad_value) const;

    Tensor cpu(bool blocking = true, ttnn::QueueId cq_id = ttnn::DefaultQueueId) const;

    Tensor unpad(const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) const;

    Tensor pad_to_tile(float pad_value) const;

    Tensor unpad_from_tile(const ttnn::Shape& output_tensor_shape) const;

    const std::string write_to_string() const;
    void print() const;

    Tensor extract_shard(const CoreCoord& core) const;
    Tensor extract_shard(const uint32_t& core_id) const;

    // ======================================================================================
    //                                  Low Level APIs
    // ======================================================================================
    Tensor reshape(const ttnn::Shape& new_shape) const;
    Tensor reshape(const ttnn::Shape& new_logical_shape, const ttnn::Shape& new_padded_shape) const;
    // ======================================================================================
    //                                      Getters
    // ======================================================================================
    const Storage& get_storage() const;
    DataType get_dtype() const;
    Layout get_layout() const;
    const ttnn::Shape& get_logical_shape() const;
    const ttnn::Shape& get_padded_shape() const;
    const TensorSpec& get_tensor_spec() const;

    // ======================================================================================
    // Non-Blocking Getters. Query attributes directly, without waiting for worker completion
    // ======================================================================================
    const Storage& storage() const { return this->tensor_attributes->storage; };
    const ttnn::Shape& logical_shape() const { return this->tensor_attributes->tensor_spec.logical_shape(); };
    const ttnn::Shape& padded_shape() const { return this->tensor_attributes->tensor_spec.padded_shape(); };
    DataType dtype() const { return this->tensor_attributes->tensor_spec.tensor_layout().get_data_type(); };
    Layout layout() const { return this->tensor_attributes->tensor_spec.tensor_layout().get_layout(); };
    const TensorSpec& tensor_spec() const { return this->tensor_attributes->tensor_spec; }

    // ======================================================================================
    //                                      Setters
    // ======================================================================================
    void set_storage(const Storage& storage) { this->tensor_attributes->storage = storage; }
    // We intend to remove this API once we migrate all ops to compute_output_specs, and provide TensorSpec at creation
    void set_tensor_spec(const TensorSpec& tensor_spec) {
        this->tensor_attributes->tensor_spec = tensor_spec;
        this->tensor_attributes->metadata_populated = true;
    }
    // ======================================================================================
    //                                      Extra Helper Functions
    // ======================================================================================
    StorageType storage_type() const;
    bool is_host_tensor() const;
    bool is_device_tensor() const;
    const ttnn::Shape strides() const;
    uint32_t volume() const;

    // todo: rename volume to get_volume to indicate that its blocking
    uint32_t get_logical_volume() const;

    bool is_scalar() const;

    bool is_allocated() const;

    bool is_contiguous() const {
        if (this->get_layout() == tt::tt_metal::Layout::ROW_MAJOR) {
            return this->get_logical_shape() == this->get_padded_shape();
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
    std::shared_ptr<Buffer> device_buffer() const { return std::get<DeviceStorage>(this->get_storage()).get_buffer(); }

    IDevice* device() const {
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
    void wait_for_tensor_data_populated() const {
        // Stall until all the workers for this tensor
        // have populated the full tensor
        while (this->tensor_attributes->num_workers_completed < this->tensor_attributes->num_shards_to_be_populated) {
        }
    }

    // Main Thread - Wait for the first worker in this tensor to populate the global metadata fields
    void wait_for_tensor_metadata_populated() const {
        // First worker is responsible for updating all metadata fields
        // Stall until this worker is done
        while (not this->tensor_attributes->metadata_populated) {
        }
    }

private:
    void init(Storage storage, TensorSpec tensor_spec);
    void deallocate_impl(bool force, bool deallocation_through_destructor);
};

Tensor create_device_tensor(const TensorSpec& tensor_spec, IDevice* device);

[[deprecated]]
Tensor create_device_tensor(
    const ttnn::Shape& shape,
    DataType dtype,
    Layout layout,
    IDevice* device,
    const MemoryConfig& memory_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED},
    const std::optional<Tile>& tile = std::nullopt);

// template<typename Buffer>
// void *get_host_buffer(const Tensor &tensor);
void* get_raw_host_data_ptr(const Tensor& tensor);

void memcpy(
    CommandQueue& queue,
    void* dst,
    const Tensor& src,
    const std::optional<BufferRegion>& region = std::nullopt,
    bool blocking = true);

void memcpy(
    CommandQueue& queue, Tensor& dst, const void* src, const std::optional<BufferRegion>& region = std::nullopt);

void memcpy(
    CommandQueue& queue, Tensor& dst, const Tensor& src, const std::optional<BufferRegion>& region = std::nullopt);

void memcpy(
    void* dst, const Tensor& src, const std::optional<BufferRegion>& region = std::nullopt, bool blocking = true);

void memcpy(Tensor& dst, const void* src, const std::optional<BufferRegion>& region = std::nullopt);

void memcpy(Tensor& dst, const Tensor& src, const std::optional<BufferRegion>& region = std::nullopt);

Tensor allocate_tensor_on_devices(const TensorSpec& spec, const std::vector<IDevice*>& devices);

// Allocates a tensor on a mesh device through mesh buffer.
Tensor allocate_tensor_on_mesh(const TensorSpec& tensor_spec, distributed::MeshDevice* mesh_device);

void write_tensor(const Tensor& host_tensor, Tensor device_tensor, ttnn::QueueId cq_id = ttnn::DefaultQueueId);

Tensor set_tensor_id(const Tensor& tensor);

bool validate_worker_modes(const std::vector<IDevice*>& workers);

}  // namespace tt_metal

}  // namespace tt

namespace ttnn {

using Tensor = tt::tt_metal::Tensor;
using TensorSpec = tt::tt_metal::TensorSpec;

}  // namespace ttnn
