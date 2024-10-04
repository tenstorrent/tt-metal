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
#include "ttnn/common/constants.hpp"
#include "ttnn/tensor/types.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/tile/tile.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/device/mesh_device.hpp"
#include "tt_metal/tt_stl/reflection.hpp"
#include "types.hpp"

namespace tt {

namespace tt_metal {

struct Tensor {
    struct TensorAttributes : public std::enable_shared_from_this<TensorAttributes> {
        Storage storage;
        ttnn::Shape shape;
        DataType dtype;
        Layout layout;
        Tile tile;
        uint32_t num_shards_to_be_populated = 0;
        uint32_t main_thread_ref_count = 0;
        std::atomic<uint32_t> num_sibling_workers_sharing_tensor = 0;
        std::atomic<bool> main_thread_tensor = true;
        std::atomic<bool> metadata_populated = false;
        std::atomic<int> num_workers_completed = 0;
        bool deallocated = false;      // Set to true if device side storage was deallocated
        bool dynamic_storage = false;  // Storage type can change, depending on op behaviour
        bool track_ref_count = false;
        TensorAttributes(const Storage storage, const ttnn::Shape shape, DataType dtype, Layout layout, Tile tile = std::array<uint32_t, 2>{32, 32}) :
            storage(storage), shape(shape), dtype(dtype), layout(layout), tile(tile) {}
        TensorAttributes() :
            shape(std::array<uint32_t, 4>{0xff, 0xff, 0xff, 0xff}), dtype(DataType::INVALID), layout(Layout::INVALID), tile(std::array<uint32_t, 2>{32, 32}) {}
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
        void increment_main_thread_ref_count(Device *worker) {
            if (worker->get_worker_mode() == WorkExecutorMode::ASYNCHRONOUS and worker->in_main_thread()) {
                main_thread_ref_count++;
                if (track_ref_count) {
                    tt::log_info(
                        "Inc Ref Count on tensor {}. Main Thread Ref Count: {}. Total Ref Count: {}.",
                        reinterpret_cast<uint64_t>(this),
                        main_thread_ref_count,
                        shared_from_this().use_count());
                }
            }
        }

        void decrement_main_thread_ref_count(Device *worker) {
            if (worker->get_worker_mode() == WorkExecutorMode::ASYNCHRONOUS and worker->in_main_thread()) {
                main_thread_ref_count--;
                if (track_ref_count) {
                    tt::log_info(
                        "Dec Ref Count on tensor {}. Main Thread Ref Count: {}. Total Ref Count: {}.",
                        reinterpret_cast<uint64_t>(this),
                        main_thread_ref_count,
                        shared_from_this().use_count());
                }
            }
        }

        uint32_t record_main_thread_ref_count() { return main_thread_ref_count; }

        void update_main_thread_ref_count(Device *worker, uint32_t ref_count) {
            if (worker->get_worker_mode() == WorkExecutorMode::ASYNCHRONOUS and worker->in_main_thread()) {
                if (track_ref_count) {
                    tt::log_info(
                        "Update Ref Count on tensor {}. Main Thread Ref Count: {}. Total Ref Count: {}.",
                        reinterpret_cast<uint64_t>(this),
                        main_thread_ref_count,
                        shared_from_this().use_count());
                }
                main_thread_ref_count = ref_count;
            }
        }
    };

    std::optional<std::int64_t> tensor_id = std::nullopt;
    // Shared pointer to all attributes associated with this tensor
    // Can be safely passed between threads when the tensor is copied
    std::shared_ptr<TensorAttributes> tensor_attributes = nullptr;
    // Tensor gets worker queue handle through the device
    std::vector<Device *> workers = {};
    bool deallocate_through_destructor = false;

    // ======================================================================================
    //                                  Hi Level APIs
    // ======================================================================================
    explicit Tensor() :
        tensor_id(std::nullopt),
        tensor_attributes(nullptr),
        workers(std::vector<Device *>{}),
        deallocate_through_destructor(false) {}

    Tensor(const Storage storage, const ttnn::Shape shape, DataType dtype, Layout layout, const std::optional<Tile>& tile = std::nullopt);
    Tensor(const Storage storage, const tt::tt_metal::LegacyShape shape, DataType dtype, Layout layout, const std::optional<Tile>& tile = std::nullopt);

    // Constructor to initialize unpopulated tensor with workers and storage specified. Use this when creating tensor
    // handles in async mode.
    Tensor(
        const std::vector<Device *>& workers,
        uint32_t num_buffers = 0,
        std::optional<DistributedTensorConfig> distributed_tensor_config = std::nullopt) :
        tensor_id(std::nullopt),
        tensor_attributes(std::make_shared<TensorAttributes>()),
        workers(workers),
        deallocate_through_destructor(false) {
        // When creating a device tensor, specify workers.
        // When creating a host tensor, specify num_buffers.
        // If neither are specified, a dummy tensor is being created. Do nothing.
        if (workers.size()) {
            bool in_main_thread_based_on_first_worker = workers.at(0)->in_main_thread();
            for (auto &worker : workers) {
                bool in_main_thread_based_on_curr_worker = worker->in_main_thread();
                TT_FATAL(
                    in_main_thread_based_on_curr_worker == in_main_thread_based_on_first_worker,
                    "in_main_thread() inconsistency found across worker threads. Some worker threads have incorrectly "
                    "assigned main thread IDs.");
            }
            if (in_main_thread_based_on_first_worker) {
                this->tensor_attributes->increment_main_thread_ref_count(this->workers.at(0));
            } else {
                // This tensor is being created from scratch in a worker. Track this and allow it to be explicitly
                // deallocated inside the worker (composite ops do this).
                this->tensor_attributes->main_thread_tensor = false;
            }
            if (workers.size() == 1) {
                this->tensor_attributes->storage = DeviceStorage();
            } else if (workers.size() > 1) {
                this->tensor_attributes->storage = MultiDeviceStorage();
                std::transform(
                    workers.cbegin(),
                    workers.cend(),
                    std::back_inserter(
                        std::get<MultiDeviceStorage>(this->tensor_attributes->storage).ordered_device_ids),
                    [](const Device *worker) { return worker->id(); });
            }
            this->tensor_attributes->num_shards_to_be_populated = workers.size();
        } else if (num_buffers) {
            if (num_buffers == 1) {
                this->tensor_attributes->storage = OwnedStorage();
            } else {
                this->tensor_attributes->storage = MultiDeviceHostStorage();
                // Preallocate buffer and shape vector for MultiDeviceHostStorage
                if (distributed_tensor_config.has_value()) {
                    std::get<MultiDeviceHostStorage>(this->tensor_attributes->storage).strategy =
                        distributed_tensor_config.value();
                }
                std::get<MultiDeviceHostStorage>(this->tensor_attributes->storage).buffers =
                    std::vector<OwnedBuffer>(num_buffers, OwnedBuffer());
                std::get<MultiDeviceHostStorage>(this->tensor_attributes->storage).shapes =
                    std::vector<tt::tt_metal::LegacyShape>(num_buffers, this->tensor_attributes->shape.value);
            }
            this->tensor_attributes->num_shards_to_be_populated = num_buffers;
        }
    }

    Tensor(const Tensor &other) :
        tensor_id(other.tensor_id),
        workers(other.workers),
        tensor_attributes(other.tensor_attributes),
        deallocate_through_destructor(other.deallocate_through_destructor) {
        if (this->workers.size()) {
            if (this->workers.at(0)->in_main_thread()) {
                this->tensor_attributes->increment_main_thread_ref_count(this->workers.at(0));
            }
        }
    }

    Tensor &operator=(const Tensor &other) {
        // Don't self-assign
        this->tensor_id = other.tensor_id;
        if (this->tensor_attributes != other.tensor_attributes) {
            // Update ref count for curr tensor_attr and deallocate if needed
            perform_cleanup_for_async_mode();
            this->workers = other.workers;
            this->tensor_attributes = other.tensor_attributes;
            this->deallocate_through_destructor = other.deallocate_through_destructor;
            if (this->workers.size()) {
                if (this->workers.at(0)->in_main_thread()) {
                    this->tensor_attributes->increment_main_thread_ref_count(this->workers.at(0));
                }
            }
        }
        return *this;
    }

    Tensor(Tensor &&other) noexcept = default;

    Tensor &operator=(Tensor &&other) {
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

    void deepcopy(const Tensor &other);

    void populate_buffers_and_metadata(const Tensor &other);

    void deallocate(bool force = false);

    std::vector<Device *> get_workers(bool blocking = false) const;

    Tensor to(
        Device *target_device,
        const MemoryConfig &mem_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) const;

    Tensor to(
        MeshDevice *mesh_device,
        const MemoryConfig &mem_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) const;

    Tensor to(
        CommandQueue &queue,
        const MemoryConfig &mem_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) const;

    Tensor to(
        const std::vector<Device *> &workers,
        const MemoryConfig &mem_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) const;

    Tensor to(Layout target_layout, Device *worker = nullptr) const;

    Tensor to(Layout target_layout, MeshDevice *mesh_device) const;

    Tensor pad(const tt::tt_metal::LegacyShape &output_tensor_shape, const tt::tt_metal::LegacyShape &input_tensor_start, float pad_value) const;

    Tensor cpu(bool blocking = true, uint8_t cq_id = ttnn::DefaultQueueId) const;

    Tensor cpu_sharded() const;

    Tensor unpad(const tt::tt_metal::LegacyShape &output_tensor_start, const tt::tt_metal::LegacyShape &output_tensor_end) const;

    Tensor pad_to_tile(float pad_value) const;

    Tensor unpad_from_tile(const tt::tt_metal::LegacyShape &output_tensor_shape) const;

    const std::string write_to_string() const;
    void print() const;

    Tensor extract_shard(const CoreCoord &core) const;
    Tensor extract_shard(const uint32_t &core_id) const;

    // ======================================================================================
    //                                  Low Level APIs
    // ======================================================================================
    Tensor reshape(int N, int C, int H, int W) const;
    Tensor reshape(const tt::tt_metal::LegacyShape &new_shape) const;

    // ======================================================================================
    //                                      Getters
    // ======================================================================================
    const Storage &get_storage() const;
    // [[deprecated("Use get_shape() instead.")]]
    const tt::tt_metal::LegacyShape &get_legacy_shape() const;
    const ttnn::Shape &get_shape() const;
    const DataType &get_dtype() const;
    const Layout &get_layout() const;
    const Tile &get_tile() const;

    ttnn::SimpleShape get_logical_shape() const;
    ttnn::SimpleShape get_padded_shape() const;
    tt::tt_metal::Padding get_padding() const;

    // ======================================================================================
    // Non-Blocking Getters. Query attributes directly, without waiting for worker completion
    // ======================================================================================
    inline const Storage &storage() const { return this->tensor_attributes->storage; };
    inline const tt::tt_metal::LegacyShape &legacy_shape() const { return this->tensor_attributes->shape.value; };
    inline const ttnn::Shape &shape() const { return this->tensor_attributes->shape; };
    inline const DataType &dtype() const { return this->tensor_attributes->dtype; };
    inline const Layout &layout() const { return this->tensor_attributes->layout; };
    inline const Tile &tile() const { return this->tensor_attributes->tile; };

    // ======================================================================================
    //                                      Setters
    // ======================================================================================
    inline void set_storage(const Storage &storage) { this->tensor_attributes->storage = storage; }
    inline void set_shape(const ttnn::Shape &shape) { this->tensor_attributes->shape = shape; }
    inline void set_dtype(const DataType &dtype) { this->tensor_attributes->dtype = dtype; }
    inline void set_layout(const Layout &layout) { this->tensor_attributes->layout = layout; }
    inline void set_tile(const Tile &tile) { this->tensor_attributes->tile = tile; }
    // ======================================================================================
    //                                      Extra Helper Functions
    // ======================================================================================
    StorageType storage_type() const;
    const tt::tt_metal::LegacyShape strides() const;
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

    static constexpr auto attribute_names = std::forward_as_tuple("storage", "shape", "dtype", "layout", "tile");
    const auto attribute_values() const {
        return std::forward_as_tuple(this->tensor_attributes->storage, this->tensor_attributes->shape, this->tensor_attributes->dtype, this->tensor_attributes->layout, this->tensor_attributes->tile);
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
};

Tensor create_device_tensor(
    const tt::tt_metal::LegacyShape &shape,
    DataType dtype,
    Layout layout,
    Device *device,
    const MemoryConfig &memory_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED},
    const std::optional<Tile>& tile = std::nullopt);

static Tensor create_device_tensor(
    const ttnn::Shape &shape,
    DataType dtype,
    Layout layout,
    Device *device,
    const MemoryConfig &memory_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED},
    const std::optional<Tile>& tile = std::nullopt) {
    return create_device_tensor(shape.value, dtype, layout, device, memory_config, tile);
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
    const MemoryConfig &memory_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED},
    const std::optional<Tile>& tile = std::nullopt);
Tensor allocate_tensor_on_device(
    const ttnn::Shape &shape,
    DataType data_type,
    Layout layout,
    MeshDevice *mesh_device,
    const MemoryConfig &memory_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED},
    const std::optional<Tile>& tile = std::nullopt);
void write_tensor(Tensor host_tensor, Tensor device_tensor, uint8_t cq_id = ttnn::DefaultQueueId);

// Maps a tensor to the set of devices in the device-mesh that the shards will be distributed across.
std::vector<Device*> distribute_tensor_to_mesh(const Tensor& tensor, MeshDevice& mesh_device);

Tensor set_tensor_id(const Tensor &tensor);

}  // namespace tt_metal

}  // namespace tt

namespace ttnn {

using Tensor = tt::tt_metal::Tensor;

}  // namespace ttnn
