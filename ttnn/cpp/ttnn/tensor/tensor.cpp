// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <memory>
#include <utility>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_constants.hpp>
#include <tt_stl/overloaded.hpp>
#include "tt-metalium/mesh_device_view.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_impl_wrapper.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tracy/Tracy.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include "ttnn/core.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/distributed/api.hpp"

namespace tt::tt_metal {
namespace {

template <typename T>
Tensor create_owned_tensor_from_row_major_data(
    std::vector<T>&& data, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device = std::nullopt) {
    auto physical_data = tensor_impl::encode_tensor_data(std::move(data), spec);

    Tensor output(OwnedStorage{owned_buffer::create(std::move(physical_data))}, spec);

    if (device.has_value()) {
        output = output.to_device(device->get_devices(), spec.memory_config());
    }

    return output;
}

}  // namespace

Tensor::TensorAttributes::TensorAttributes() :
    tensor_spec(
        ttnn::Shape(std::array<uint32_t, 4>{0xff, 0xff, 0xff, 0xff}),
        TensorLayout(DataType::INVALID, PageConfig(Layout::INVALID), MemoryConfig{})) {}

Tensor::TensorAttributes::TensorAttributes(Storage storage, TensorSpec tensor_spec) :
    storage(std::move(storage)), tensor_spec(std::move(tensor_spec)), metadata_populated(true) {}

void Tensor::TensorAttributes::increment_main_thread_ref_count(IDevice* worker) {
    if (worker->get_worker_mode() == WorkExecutorMode::ASYNCHRONOUS and not tt::tt_metal::detail::InWorkerThread()) {
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

void Tensor::TensorAttributes::decrement_main_thread_ref_count(IDevice* worker) {
    if (worker->get_worker_mode() == WorkExecutorMode::ASYNCHRONOUS and not tt::tt_metal::detail::InWorkerThread()) {
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

uint32_t Tensor::TensorAttributes::record_main_thread_ref_count() { return main_thread_ref_count; }

void Tensor::TensorAttributes::update_main_thread_ref_count(IDevice* worker, uint32_t ref_count) {
    if (worker->get_worker_mode() == WorkExecutorMode::ASYNCHRONOUS and not tt::tt_metal::detail::InWorkerThread()) {
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

Tensor::Tensor(
    Storage storage,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    DataType dtype,
    Layout layout,
    const std::optional<Tile>& tile) {
    using namespace tt::constants;

    if (tile.has_value() and  //
        (tile->get_tile_shape()[0] != TILE_WIDTH or tile->get_tile_shape()[1] != TILE_HEIGHT)) {
        tt::log_warning(
            "only matmul op and ccl all-gather currently supports the customized tile shape: {}",
            tile->get_tile_shape());
    }

    const auto memory_config = std::visit(
        tt::stl::overloaded{
            [](const DeviceStorage& s) { return s.memory_config(); },
            [](const MultiDeviceStorage& s) { return s.memory_config(); },
            []<typename Other>(const Other&) { return MemoryConfig{}; }},
        storage);

    init(
        std::move(storage),
        TensorSpec(
            logical_shape,
            TensorLayout::fromPaddedShape(
                dtype, PageConfig(layout, tile), memory_config, logical_shape, padded_shape)));
}

Tensor::Tensor(Storage storage, TensorSpec tensor_spec) { init(std::move(storage), std::move(tensor_spec)); }

void Tensor::init(Storage storage, TensorSpec tensor_spec) {
    tensor_attributes = std::make_shared<TensorAttributes>(std::move(storage), std::move(tensor_spec));

    ZoneScoped;
    std::visit(
        [&](auto&& storage) {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                tensor_attributes->num_shards_to_be_populated = 1;
            } else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_ASSERT(storage.buffer->device() != nullptr);
                workers = {storage.buffer->device()};
                tensor_impl::validate_on_device_dtype_and_layout(
                    storage.buffer->device(),
                    tensor_attributes->tensor_spec.padded_shape(),
                    tensor_attributes->tensor_spec.data_type(),
                    tensor_attributes->tensor_spec.layout());
                // Increment main thread ref count for all tensors on device
                tensor_attributes->increment_main_thread_ref_count(this->workers.at(0));
                // Track if this tensor is being created from scratch in a worker, to allow it to be deallocated inside
                // the worker (composite ops do this).
                tensor_attributes->main_thread_tensor = tt::tt_metal::detail::InMainThread();
                tensor_attributes->num_shards_to_be_populated = 1;
            } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                tensor_attributes->num_shards_to_be_populated = 1;
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                workers.reserve(storage.num_buffers());
                for (int i = 0; i < storage.ordered_device_ids.size(); i++) {
                    auto device_id = storage.ordered_device_ids[i];
                    auto buffer = storage.get_buffer_for_device_id(device_id);
                    TT_ASSERT(buffer->device() != nullptr);
                    TT_ASSERT(buffer->device()->id() == device_id);
                    tensor_impl::validate_on_device_dtype_and_layout(
                        buffer->device(),
                        tensor_attributes->tensor_spec.padded_shape(),
                        tensor_attributes->tensor_spec.data_type(),
                        tensor_attributes->tensor_spec.layout());
                    workers.push_back(buffer->device());
                }
                // Increment main thread ref count for all tensors on cluster
                tensor_attributes->increment_main_thread_ref_count(this->workers.at(0));
                // Track if this tensor is being created from scratch in a worker, to allow it to be deallocated inside
                // the worker (composite ops do this).
                tensor_attributes->main_thread_tensor = tt::tt_metal::detail::InMainThread();
                tensor_attributes->num_shards_to_be_populated = storage.num_buffers();
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                tensor_attributes->num_shards_to_be_populated = storage.num_buffers();
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor_attributes->storage);
    tensor_attributes->num_workers_completed = this->tensor_attributes->num_shards_to_be_populated;
}

Tensor::Tensor(const std::vector<IDevice*>& workers) :
    tensor_attributes(std::make_shared<TensorAttributes>()), workers(workers) {
    if (workers.empty()) {
        return;
    }

    tensor_attributes->storage = [&]() {
        if (workers.size() == 1) {
            return Storage(DeviceStorage());
        }
        MultiDeviceStorage storage;
        std::transform(
            workers.cbegin(),
            workers.cend(),
            std::back_inserter(storage.ordered_device_ids),
            [](const IDevice* worker) { return worker->id(); });
        return Storage(std::move(storage));
    }();
    tensor_attributes->num_shards_to_be_populated = workers.size();
    if (tt::tt_metal::detail::InMainThread()) {
        tensor_attributes->increment_main_thread_ref_count(this->workers.at(0));
    } else {
        // This tensor is being created from scratch in a worker. Track this and allow it to be explicitly
        // deallocated inside the worker (composite ops do this).
        tensor_attributes->main_thread_tensor = false;
    }
}

Tensor::Tensor(uint32_t num_buffers, std::optional<DistributedTensorConfig> distributed_tensor_config) :
    tensor_attributes(std::make_shared<TensorAttributes>()) {
    if (num_buffers == 0) {
        return;
    }

    tensor_attributes->storage = [&]() {
        if (num_buffers == 1) {
            return Storage(OwnedStorage());
        }
        MultiDeviceHostStorage storage;
        if (distributed_tensor_config.has_value()) {
            storage.strategy = distributed_tensor_config.value();
        }
        storage.buffers = std::vector<OwnedBuffer>(num_buffers, OwnedBuffer());
        storage.specs = std::vector<ttnn::TensorSpec>(
            num_buffers,
            TensorSpec(Shape{}, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{})));
        return Storage(std::move(storage));
    }();
    tensor_attributes->num_shards_to_be_populated = num_buffers;
}

Tensor& Tensor::operator=(const Tensor& other) {
    // Don't self-assign
    this->tensor_id = other.tensor_id;
    if (this->tensor_attributes != other.tensor_attributes) {
        // Update ref count for curr tensor_attr and deallocate if needed
        perform_cleanup_for_async_mode();
        this->workers = other.workers;
        this->tensor_attributes = other.tensor_attributes;
        if (this->workers.size()) {
            if (not tt::tt_metal::detail::InWorkerThread()) {
                this->tensor_attributes->increment_main_thread_ref_count(this->workers.at(0));
            }
        }
    }
    return *this;
}

Tensor::Tensor(const Tensor& other) :
    tensor_id(other.tensor_id), workers(other.workers), tensor_attributes(other.tensor_attributes) {
    if (this->workers.size()) {
        if (not tt::tt_metal::detail::InWorkerThread()) {
            this->tensor_attributes->increment_main_thread_ref_count(this->workers.at(0));
        }
    }
}

Tensor::~Tensor() {
    ZoneScoped;
    this->deallocate_impl(/*force=*/false, /*deallocation_through_destructor=*/true);
    // Decrement main thread ref count for all tensors on device
    if (this->workers.size() and this->tensor_attributes) {
        this->tensor_attributes->decrement_main_thread_ref_count(this->workers.at(0));
    }
    tensor_attributes.reset();
}

Tensor::Tensor(
    Storage storage, const ttnn::Shape& shape, DataType dtype, Layout layout, const std::optional<Tile>& tile) :
    Tensor(std::move(storage), /* logical_shape */ shape, /* padded_shape */ shape, dtype, layout, tile) {}

void Tensor::deallocate(bool force) { deallocate_impl(force, /*deallocation_through_destructor=*/false); }

void Tensor::deallocate_impl(bool force, bool deallocation_through_destructor) {
    ZoneScopedN("TensorDeallocate");
    // GraphTracker::instance().track_function_start("Tensor::deallocate", *this, force);
    // Check if the attributes didn't get moved to another tensor.
    // If not, we can deallocate this tensor.
    if (tensor_attributes.use_count() == 0) {
        return;
    }

    auto get_tensor_ref_count = [](const Tensor& tensor) {
        // If owned by the main thread, deallocate this tensor only from the main thread. If owned by worker thread,
        // allow deallocation in worker and use shared_ptr ref count, since this is a thread_local tensor
        return (tensor.workers.at(0)->get_worker_mode() == WorkExecutorMode::SYNCHRONOUS or
                not tensor.tensor_attributes->main_thread_tensor)
                   ? tensor.tensor_attributes.use_count()
                   : tensor.tensor_attributes->main_thread_ref_count;
    };

    std::visit(
        tt::stl::overloaded{
            [this](OwnedStorage& storage) {
                if (this->tensor_attributes.use_count() == 1) {
                    std::visit([](auto&& buffer) { buffer.reset(); }, storage.buffer);
                }
            },
            [force, this](BorrowedStorage& storage) {
                TT_FATAL(not force, "Cannot deallocate tensor with borrowed storage!");
            },
            [this](MultiDeviceHostStorage& storage) {
                if (this->tensor_attributes.use_count() == 1) {
                    for (int i = 0; i < storage.num_buffers(); i++) {
                        std::visit([](auto&& buffer) { buffer.reset(); }, storage.get_buffer(i));
                    }
                }
            },
            [force, this, &get_tensor_ref_count, deallocation_through_destructor](DeviceStorage& storage) {
                if (not this->workers.at(0)->is_initialized()) {
                    return;
                }
                if (tt::tt_metal::detail::InWorkerThread() and this->tensor_attributes->main_thread_tensor) {
                    TT_FATAL(
                        deallocation_through_destructor,
                        "Device tensors created in the main thread cannot be explictly deallocated in worker "
                        "threads.");
                    return;
                }

                if (not this->tensor_attributes->main_thread_tensor) {
                    TT_ASSERT(
                        not this->tensor_attributes->main_thread_ref_count,
                        "main_thread_ref_count for tensors created inside a worker thread must be 0");
                }
                const uint32_t ref_count_to_use = get_tensor_ref_count(*this);
                if ((force or ref_count_to_use == 1) and not this->tensor_attributes->deallocated) {
                    this->tensor_attributes->deallocated = true;
                    this->workers.at(0)->push_work([force, attr = this->tensor_attributes]() mutable {
                        // Cross worker synchronization: If the tensor being deallocated is shared across
                        // workers (ex: all_gather op), wait until all workers are done with this tensor
                        // before deallocating.
                        bool num_threads_sharing_tensor = attr->num_sibling_workers_sharing_tensor;
                        if (num_threads_sharing_tensor) {
                            while (num_threads_sharing_tensor) {
                                num_threads_sharing_tensor = attr->num_sibling_workers_sharing_tensor;
                            }
                        }
                        std::visit(
                            [force, attr](auto&& s) {
                                using type = std::decay_t<decltype(s)>;
                                if constexpr (std::is_same_v<type, DeviceStorage>) {
                                    if (force or s.buffer.use_count() == 1) {
                                        DeallocateBuffer(*(s.buffer));
                                    }
                                    // Safe to reset this buf object since this is the last reference (in
                                    // the main thread) to the tensor attr object holding this buffer. If
                                    // any other tensor handles hold this buffer, it will not be deleted,
                                    // until the last handle goes out of scope or is deallocated.
                                    s.buffer.reset();
                                } else if constexpr (std::is_same_v<type, OwnedStorage>) {
                                    // Manage Dynamic Storage (due to autoformat in async mode): Main thread
                                    // sees this tensor as a device tensor, since worker has not updated
                                    // storage time. When the worker executes the dealloc request, the
                                    // storage type has been appropriately updated to Owned.
                                    TT_ASSERT(
                                        attr->dynamic_storage,
                                        "Tensor storage type changed during runtime (device -> host), but "
                                        "dynamic storage was not marked.");
                                    std::visit([](auto&& buffer) { buffer.reset(); }, s.buffer);
                                }
                            },
                            attr->storage);
                    });
                }
            },
            [force, this, &get_tensor_ref_count, deallocation_through_destructor](MultiDeviceStorage& storage) {
                if (not this->workers.at(0)->is_initialized()) {
                    return;
                }
                if (tt::tt_metal::detail::InWorkerThread() and this->tensor_attributes->main_thread_tensor) {
                    TT_FATAL(
                        deallocation_through_destructor,
                        "Device tensors created in the main thread cannot be explictly deallocated in worker "
                        "threads.");
                    return;
                }
                const uint32_t ref_count_to_use = get_tensor_ref_count(*this);
                if ((force or ref_count_to_use == 1) and not this->tensor_attributes->deallocated) {
                    this->tensor_attributes->deallocated = true;

                    if (storage.mesh_buffer != nullptr) {
                        // TODO: #17215 - Consider if it is possible to retain references to individual device buffers
                        // after mesh buffer was deallocated.
                        storage.mesh_buffer->deallocate();
                    } else {
                        auto dealloc_lambda = std::make_shared<std::function<void(IDevice*)>>(
                            [force, attr = this->tensor_attributes](IDevice* worker) mutable {
                                ZoneScopedN("ShardDeallocate");
                                TT_ASSERT(
                                    std::holds_alternative<MultiDeviceStorage>(attr->storage),
                                    "Unexpected type {}",
                                    tt::stl::get_active_type_name_in_variant(attr->storage));
                                auto& s = std::get<MultiDeviceStorage>(attr->storage);
                                if (s.has_buffer_for_device(worker)) {
                                    auto& device_buffer = s.get_buffer_for_device(worker);
                                    if (force or device_buffer.use_count() == 1) {
                                        DeallocateBuffer(*device_buffer);
                                    }
                                    device_buffer.reset();
                                }
                            });

                        for (auto* worker : this->workers) {
                            worker->push_work([worker, dealloc_lambda]() mutable { (*dealloc_lambda)(worker); });
                        }
                    }
                }
            },
        },
        this->tensor_attributes->storage);
    // GraphTracker::instance().track_function_end();
}

void Tensor::perform_cleanup_for_async_mode() {
    // Used when tensor attributes object for this is reassigned by copy
    // or move assignment operator
    if (this->tensor_attributes) {
        // Object has tensor_attributes that will be reassigned
        if (this->workers.size() and tt::tt_metal::detail::InMainThread() and
            this->workers.at(0)->get_worker_mode() == WorkExecutorMode::ASYNCHRONOUS) {
            // Operator called in main thread with async mode. Main thread Ref Count must be decremented.
            // This is the last tensor in the main thread holding these attributes. Deallocate the buffer
            // for this tensor.
            if (this->tensor_attributes->main_thread_ref_count == 1) {
                this->deallocate();
            }
            this->tensor_attributes->main_thread_ref_count--;
        }
    }
}

void Tensor::populate_buffers_and_metadata(const Tensor& other) {
    ZoneScoped;
    // Applied on a tensor that has an empty storage container initialized.
    this->set_tensor_spec(other.get_tensor_spec());
    // Populate storage container with buffers + shapes
    std::visit(
        [this](auto&& storage) {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage> or std::is_same_v<StorageType, DeviceStorage>) {
                std::get<StorageType>(this->tensor_attributes->storage).insert_buffer(storage.get_buffer());
            } else if constexpr (
                std::is_same_v<StorageType, MultiDeviceHostStorage> or
                std::is_same_v<StorageType, MultiDeviceStorage>) {
                std::get<StorageType>(this->tensor_attributes->storage).buffers = storage.buffers;
                std::get<StorageType>(this->tensor_attributes->storage).specs = storage.specs;
            }
        },
        other.get_storage());  // Non blocking storage query, since this is done for tensors that get created inside the
                               // worker thread
    this->tensor_attributes->num_workers_completed++;
}

std::vector<IDevice*> Tensor::get_workers(bool blocking) const {
    ZoneScoped;
    // Initialize an empty worker vector (remains empty for host side storage)
    std::vector<IDevice*> workers = {};

    if (this->tensor_attributes->dynamic_storage) {
        // Tensor is populated by launch_with_autoformat
        // Storage type can change based on op behaviour, wait until tensor populated.
        this->wait_for_tensor_metadata_populated();
    }

    std::visit(
        [this, blocking, &workers](auto&& storage) {
            using StorageType = std::decay_t<decltype(storage)>;
            // Assign workers only to device tensors
            if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                // Either explictly syncing or workers are pre-populated (this will happen for device tensors if using
                // the correct APIs).
                TT_FATAL(
                    blocking or (this->workers.size() == 1),
                    "Worker Handles for tensor must be populated or blocking = true must be set in get_workers().");
                if (this->workers.size() != 1) {
                    // Not populated - sync.
                    this->wait_for_tensor_data_populated();
                    workers = std::vector<IDevice*>{this->device()};
                } else {
                    // Already populated.
                    workers = this->workers;
                }
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                // Either explictly syncing or workers are pre-populated (this will happen for device tensors if using
                // the correct APIs).
                TT_FATAL(
                    blocking or (this->workers.size()),
                    "Worker Handles for tensor must be populated or blocking = true must be set in get_workers().");
                if (not this->workers.size()) {
                    // Not populated - sync.
                    this->wait_for_tensor_data_populated();
                    workers.reserve(storage.num_buffers());
                    for (int i = 0; i < storage.ordered_device_ids.size(); ++i) {
                        auto device_id = storage.ordered_device_ids[i];
                        workers.push_back(storage.get_buffer_for_device_id(device_id)->device());
                    }
                } else {
                    workers = this->workers;
                }
            }
        },
        this->tensor_attributes->storage);
    return workers;
}

// Getters - Spin until tensor is populated before querying tensor metadata
DataType Tensor::get_dtype() const {
    wait_for_tensor_metadata_populated();
    return dtype();
}
Layout Tensor::get_layout() const {
    wait_for_tensor_metadata_populated();
    return layout();
}

const TensorSpec& Tensor::get_tensor_spec() const {
    wait_for_tensor_metadata_populated();
    return tensor_spec();
}

const ttnn::Shape& Tensor::get_logical_shape() const {
    wait_for_tensor_metadata_populated();
    return logical_shape();
}

const ttnn::Shape& Tensor::get_padded_shape() const {
    wait_for_tensor_metadata_populated();
    return padded_shape();
}

const Storage& Tensor::get_storage() const {
    this->wait_for_tensor_data_populated();
    return this->tensor_attributes->storage;
}

template <>
Tensor Tensor::from_span<float>(
    tt::stl::Span<const float> buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device) {
    size_t volume = spec.logical_shape().volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    switch (spec.data_type()) {
        case DataType::FLOAT32:
            return create_owned_tensor_from_row_major_data(
                std::vector<float>(buffer.begin(), buffer.end()), spec, device);
        case DataType::BFLOAT16: {
            return create_owned_tensor_from_row_major_data(
                std::vector<bfloat16>(buffer.begin(), buffer.end()), spec, device);
        }
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: {
            TT_FATAL(
                spec.tensor_layout().get_layout() == Layout::TILE,
                "Tile layout is required for BFLOAT8_B and BFLOAT4_B");

            // TODO: Implement `encode_tensor_data` in terms of a Span, avoid tilizing the data, as pack_fp32_vec_as_*
            // support row-major input.
            const auto& tile = spec.tensor_layout().get_page_config().get_tile();
            auto physical_data =
                tensor_impl::encode_tensor_data(std::vector<float>(buffer.begin(), buffer.end()), spec);
            std::vector<uint32_t> packed_block_floats =
                spec.data_type() == DataType::BFLOAT8_B
                    ? pack_fp32_vec_as_bfp8_tiles(physical_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile)
                    : pack_fp32_vec_as_bfp4_tiles(physical_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);

            Tensor tensor(OwnedStorage{owned_buffer::create(std::move(packed_block_floats))}, spec);
            if (device.has_value()) {
                tensor = tensor.to_device(device->get_devices(), spec.memory_config());
            }
            return tensor;
        }
        default: {
            TT_THROW("Unsupported data type: {}", spec.data_type());
        }
    }
}

template <typename T>
Tensor Tensor::from_span(tt::stl::Span<const T> buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device) {
    size_t volume = spec.logical_shape().volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    TT_FATAL(
        spec.data_type() == convert_to_data_type<T>(),
        "Unsupported data type: got {}, expected: {}",
        spec.data_type(),
        convert_to_data_type<T>());
    return create_owned_tensor_from_row_major_data(std::vector<T>(buffer.begin(), buffer.end()), spec, device);
}

template <typename T>
Tensor Tensor::from_borrowed_data(
    tt::stl::Span<T> buffer,
    const ttnn::Shape& shape,
    const std::function<void()>& on_creation_callback,
    const std::function<void()>& on_destruction_callback,
    const std::optional<Tile>& tile) {
    size_t volume = shape.volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    BorrowedStorage storage(
        borrowed_buffer::Buffer(buffer.data(), buffer.size()), on_creation_callback, on_destruction_callback);
    return Tensor(std::move(storage), shape, convert_to_data_type<T>(), Layout::ROW_MAJOR, tile);
}

template <>
Tensor Tensor::from_vector<float>(
    std::vector<float>&& buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device) {
    size_t volume = spec.logical_shape().volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    if (spec.data_type() == DataType::FLOAT32) {
        // User `buffer` directly, when no type conversion is needed.
        return create_owned_tensor_from_row_major_data(std::move(buffer), spec, device);
    } else {
        return from_span(tt::stl::Span<const float>(buffer.data(), buffer.size()), spec, device);
    }
}

template <typename T>
Tensor Tensor::from_vector(std::vector<T>&& buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device) {
    size_t volume = spec.logical_shape().volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    TT_FATAL(
        spec.data_type() == convert_to_data_type<T>(),
        "Unsupported data type: got {}, expected: {}",
        spec.data_type(),
        convert_to_data_type<T>());
    return create_owned_tensor_from_row_major_data(std::move(buffer), spec, device);
}

template <>
std::vector<float> Tensor::to_vector<float>() const {
    Tensor cpu_tensor = this->cpu();
    switch (cpu_tensor.get_dtype()) {
        case DataType::BFLOAT16: {
            auto buffer = host_buffer::get_as<bfloat16>(cpu_tensor);
            std::vector<float> physical_data;
            physical_data.reserve(buffer.size());
            std::transform(buffer.begin(), buffer.end(), std::back_inserter(physical_data), [](bfloat16 val) {
                return val.to_float();
            });
            return tensor_impl::decode_tensor_data(std::move(physical_data), cpu_tensor.tensor_spec());
        }
        case DataType::FLOAT32: {
            auto buffer = host_buffer::get_as<float>(cpu_tensor);
            auto physical_data = std::vector<float>(buffer.begin(), buffer.end());
            return tensor_impl::decode_tensor_data(std::move(physical_data), cpu_tensor.tensor_spec());
        }
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: {
            const auto& tile = cpu_tensor.get_tensor_spec().tile();
            auto buffer = host_buffer::get_as<uint32_t>(cpu_tensor);
            auto packed_data = std::vector<uint32_t>(buffer.begin(), buffer.end());
            std::vector<float> unpacked_data =
                cpu_tensor.get_tensor_spec().data_type() == DataType::BFLOAT8_B
                    ? unpack_bfp8_tiles_into_float_vec(
                          packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile)
                    : unpack_bfp4_tiles_into_float_vec(
                          packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);

            return tensor_impl::decode_tensor_data(std::move(unpacked_data), cpu_tensor.tensor_spec());
        }
        default: {
            TT_THROW("Cannot convert tensor to vector for data type: {}", cpu_tensor.get_dtype());
        }
    }
}

template <typename T>
std::vector<T> Tensor::to_vector() const {
    TT_FATAL(
        this->get_dtype() == convert_to_data_type<T>(),
        "Unsupported data type for to_vector: got {}, expected: {}",
        this->get_dtype(),
        convert_to_data_type<T>());
    auto cpu_tensor = this->cpu();
    auto data = host_buffer::get_as<T>(cpu_tensor);
    auto physical_data = std::vector<T>(data.begin(), data.end());
    return tensor_impl::decode_tensor_data(std::move(physical_data), cpu_tensor.tensor_spec());
}

// Instantiate explicitly for the supported types.
template Tensor Tensor::from_span<bfloat16>(
    tt::stl::Span<const bfloat16> buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device);
template Tensor Tensor::from_span<int32_t>(
    tt::stl::Span<const int32_t> buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device);
template Tensor Tensor::from_span<uint8_t>(
    tt::stl::Span<const uint8_t> buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device);
template Tensor Tensor::from_span<uint16_t>(
    tt::stl::Span<const uint16_t> buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device);
template Tensor Tensor::from_span<uint32_t>(
    tt::stl::Span<const uint32_t> buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device);
template Tensor Tensor::from_borrowed_data<float>(
    tt::stl::Span<float> buffer,
    const ttnn::Shape& shape,
    const std::function<void()>& on_creation_callback,
    const std::function<void()>& on_destruction_callback,
    const std::optional<Tile>& tile);
template Tensor Tensor::from_borrowed_data<bfloat16>(
    tt::stl::Span<bfloat16> buffer,
    const ttnn::Shape& shape,
    const std::function<void()>& on_creation_callback,
    const std::function<void()>& on_destruction_callback,
    const std::optional<Tile>& tile);
template Tensor Tensor::from_borrowed_data<int32_t>(
    tt::stl::Span<int32_t> buffer,
    const ttnn::Shape& shape,
    const std::function<void()>& on_creation_callback,
    const std::function<void()>& on_destruction_callback,
    const std::optional<Tile>& tile);
template Tensor Tensor::from_borrowed_data<uint8_t>(
    tt::stl::Span<uint8_t> buffer,
    const ttnn::Shape& shape,
    const std::function<void()>& on_creation_callback,
    const std::function<void()>& on_destruction_callback,
    const std::optional<Tile>& tile);
template Tensor Tensor::from_borrowed_data<uint16_t>(
    tt::stl::Span<uint16_t> buffer,
    const ttnn::Shape& shape,
    const std::function<void()>& on_creation_callback,
    const std::function<void()>& on_destruction_callback,
    const std::optional<Tile>& tile);
template Tensor Tensor::from_borrowed_data<uint32_t>(
    tt::stl::Span<uint32_t> buffer,
    const ttnn::Shape& shape,
    const std::function<void()>& on_creation_callback,
    const std::function<void()>& on_destruction_callback,
    const std::optional<Tile>& tile);
template Tensor Tensor::from_vector<bfloat16>(
    std::vector<bfloat16>&& buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device);
template Tensor Tensor::from_vector<int32_t>(
    std::vector<int32_t>&& buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device);
template Tensor Tensor::from_vector<uint8_t>(
    std::vector<uint8_t>&& buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device);
template Tensor Tensor::from_vector<uint16_t>(
    std::vector<uint16_t>&& buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device);
template Tensor Tensor::from_vector<uint32_t>(
    std::vector<uint32_t>&& buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device);

template std::vector<bfloat16> Tensor::to_vector<bfloat16>() const;
template std::vector<int32_t> Tensor::to_vector<int32_t>() const;
template std::vector<uint8_t> Tensor::to_vector<uint8_t>() const;
template std::vector<uint16_t> Tensor::to_vector<uint16_t>() const;
template std::vector<uint32_t> Tensor::to_vector<uint32_t>() const;

Tensor Tensor::to_device(IDevice* target_device, const MemoryConfig& mem_config, QueueId cq_id) const {
    return tensor_ops::tensor_to_device(*this, target_device, mem_config, cq_id);
}

Tensor Tensor::to_device(distributed::MeshDevice* mesh_device, const MemoryConfig& mem_config, QueueId cq_id) const {
    std::vector<IDevice*> workers_to_use = ttnn::distributed::get_mapped_devices(*this, *mesh_device);
    return tensor_ops::tensor_to_device(*this, workers_to_use, mem_config, cq_id);
}

Tensor Tensor::to_device(const std::vector<IDevice*>& workers, const MemoryConfig& mem_config, QueueId cq_id) const {
    return tensor_ops::tensor_to_device(*this, workers, mem_config, cq_id);
}

Tensor Tensor::cpu(bool blocking, QueueId cq_id) const { return tensor_ops::tensor_cpu(*this, blocking, cq_id); }

Tensor Tensor::extract_shard(const CoreCoord& core) const {
    ZoneScoped;
    const auto& buffer_page_mapping = *this->buffer()->get_buffer_page_mapping();
    auto core_id = buffer_page_mapping.core_to_core_id_.at(core);
    return this->extract_shard(core_id);
}

Tensor Tensor::extract_shard(const uint32_t& core_id) const {
    return tensor_impl::extract_shard_wrapper(*this, core_id);
}

Tensor Tensor::to_layout(Layout target_layout, IDevice* worker) const {
    return tensor_ops::tensor_to_layout(*this, target_layout, worker);
}

Tensor Tensor::to_layout(Layout target_layout, distributed::MeshDevice* mesh_device) const {
    return tensor_ops::tensor_to_layout(*this, target_layout, mesh_device);
}

const std::string Tensor::write_to_string() const { return tensor_impl::to_string_wrapper(*this); }

void Tensor::print() const { tensor_ops::tensor_print(*this); }

Tensor Tensor::pad(
    const ttnn::Shape& output_padded_shape, const ttnn::Shape& input_tensor_start, float pad_value) const {
    return tensor_ops::tensor_pad(*this, output_padded_shape, input_tensor_start, pad_value);
}

Tensor Tensor::unpad(const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) const {
    return tensor_ops::tensor_unpad(*this, output_tensor_start, output_tensor_end);
}

Tensor Tensor::pad_to_tile(float pad_value) const { return tensor_ops::tensor_pad_to_tile(*this, pad_value); }

Tensor Tensor::unpad_from_tile(const ttnn::Shape& output_tensor_shape) const {
    return tensor_ops::tensor_unpad_from_tile(*this, output_tensor_shape);
}

const bool Tensor::is_sharded() const {
    return is_tensor_on_device_or_multidevice(*this) ? this->memory_config().is_sharded() : false;
}

uint32_t Tensor::element_size() const { return tensor_impl::element_size_bytes(this->get_dtype()); }

Tensor Tensor::reshape(const ttnn::Shape& new_shape) const { return tensor_ops::tensor_reshape(*this, new_shape); }

Tensor Tensor::reshape(const ttnn::Shape& new_logical_shape, const ttnn::Shape& new_padded_shape) const {
    return tensor_ops::tensor_reshape(*this, new_logical_shape, new_padded_shape);
}

bool Tensor::is_allocated() const {
    ZoneScoped;
    auto output = std::visit([](auto&& storage) -> bool { return storage.is_allocated(); }, this->get_storage());
    return output;
}

std::vector<uint32_t> Tensor::host_page_ordering() {
    const auto& buffer_page_mapping = *this->buffer()->get_buffer_page_mapping();
    auto cores = buffer_page_mapping.all_cores_;
    auto shard_num_pages = buffer()->shard_spec().num_pages();
    auto num_pages = cores.size() * shard_num_pages;

    std::vector<uint32_t> ret_vec;
    ret_vec.reserve(num_pages);
    for (int page_id = 0; page_id < num_pages; page_id++) {
        if (buffer_page_mapping.dev_page_to_host_page_mapping_[page_id].has_value()) {
            ret_vec.push_back(buffer_page_mapping.dev_page_to_host_page_mapping_[page_id].value());
        }
    }
    return ret_vec;
}

StorageType Tensor::storage_type() const {
    return std::visit(
        tt::stl::overloaded{
            [](const OwnedStorage&) { return StorageType::OWNED; },
            [](const DeviceStorage&) { return StorageType::DEVICE; },
            [](const BorrowedStorage&) { return StorageType::BORROWED; },
            [](const MultiDeviceStorage& s) { return StorageType::MULTI_DEVICE; },
            [](const MultiDeviceHostStorage&) { return StorageType::MULTI_DEVICE_HOST; },
        },
        this->get_storage());
}

bool Tensor::is_host_tensor() const {
    auto type = storage_type();
    return type == StorageType::BORROWED || type == StorageType::OWNED || type == StorageType::MULTI_DEVICE_HOST;
}

bool Tensor::is_device_tensor() const { return !is_host_tensor(); }

const ttnn::Shape Tensor::strides() const {
    return ttnn::Shape(tt::tt_metal::compute_strides(this->get_padded_shape()));
}

uint32_t Tensor::volume() const { return get_padded_shape().volume(); }

uint32_t Tensor::get_logical_volume() const { return get_logical_shape().volume(); }

bool Tensor::is_scalar() const {
    const ttnn::Shape logical_shape = this->get_logical_shape();
    return logical_shape.rank() == 0 || logical_shape.volume() == 1;
}

Tensor create_device_tensor(const TensorSpec& tensor_spec, IDevice* device) {
    ZoneScoped;
    GraphTracker::instance().track_function_start(
        "tt::tt_metal::create_device_tensor",
        tensor_spec.logical_shape(),
        tensor_spec.tensor_layout().get_data_type(),
        tensor_spec.tensor_layout().get_layout(),
        device,
        tensor_spec.tensor_layout().get_memory_config());

    auto device_buffer = tensor_impl::allocate_buffer_on_device(device, tensor_spec);
    auto output = Tensor(DeviceStorage{device_buffer}, tensor_spec);
    output = tt::tt_metal::set_tensor_id(output);

    GraphTracker::instance().track_function_end(output);

    return output;
}

Tensor create_device_tensor(
    const ttnn::Shape& shape,
    DataType data_type,
    Layout layout,
    IDevice* device,
    const MemoryConfig& memory_config,
    const std::optional<Tile>& tile) {
    return create_device_tensor(
        TensorSpec(shape, TensorLayout(data_type, PageConfig(layout, tile), memory_config)), device);
}

namespace detail {
template <typename DataType>
void* get_raw_host_data_ptr(const Tensor& tensor) {
    return std::visit(
        tt::stl::overloaded{
            [](const OwnedStorage& s) {
                auto buffer = owned_buffer::get_as<DataType>(s.buffer);
                return reinterpret_cast<void*>(buffer.data());
            },
            [](const BorrowedStorage& s) {
                if constexpr (
                    std::is_same_v<DataType, float> or std::is_same_v<DataType, bfloat16> or
                    std::is_same_v<DataType, std::uint32_t> or std::is_same_v<DataType, std::int32_t> or
                    std::is_same_v<DataType, std::uint8_t> or std::is_same_v<DataType, std::uint16_t>) {
                    auto buffer = borrowed_buffer::get_as<DataType>(s.buffer);
                    return reinterpret_cast<void*>(buffer.data());
                } else {
                    TT_THROW("Borrowed storage doesn't support this data type");
                }
            },
            [](auto&&) -> void* { TT_THROW("Device storage doesn't support this data type"); },
        },
        tensor.get_storage());
}
}  // namespace detail

void* get_raw_host_data_ptr(const Tensor& tensor) {
    switch (tensor.get_dtype()) {
        case DataType::BFLOAT16: return detail::get_raw_host_data_ptr<bfloat16>(tensor);
        case DataType::FLOAT32: return detail::get_raw_host_data_ptr<float>(tensor);
        case DataType::INT32: return detail::get_raw_host_data_ptr<int32_t>(tensor);
        case DataType::UINT32: return detail::get_raw_host_data_ptr<uint32_t>(tensor);
        case DataType::BFLOAT8_B: return detail::get_raw_host_data_ptr<uint32_t>(tensor);
        case DataType::BFLOAT4_B: return detail::get_raw_host_data_ptr<uint32_t>(tensor);
        case DataType::UINT16: return detail::get_raw_host_data_ptr<uint16_t>(tensor);
        case DataType::UINT8: return detail::get_raw_host_data_ptr<uint8_t>(tensor);
        default: TT_THROW("Unsupported data type");
    }
}

void memcpy(
    CommandQueue& queue, void* dst, const Tensor& src, const std::optional<BufferRegion>& region, bool blocking) {
    TT_FATAL(is_device_tensor(src), "memcpy: src tensor must be on device");

    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE != nullptr) {
        TT_THROW("SLOW_DISPATCH is not supported for memcpy!");
    }

    if (!region.has_value()) {
        EnqueueReadBuffer(queue, src.device_buffer(), dst, blocking);
    } else {
        EnqueueReadSubBuffer(queue, src.device_buffer(), dst, region.value(), blocking);
    }
}

void memcpy(void* dst, const Tensor& src, const std::optional<BufferRegion>& region, bool blocking) {
    memcpy(src.device()->command_queue(), dst, src, region, blocking);
}

void memcpy(CommandQueue& queue, Tensor& dst, const void* src, const std::optional<BufferRegion>& region) {
    TT_FATAL(is_device_tensor(dst), "memcpy: memcpy to non-device tensor is not supported!");

    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE != nullptr) {
        TT_THROW("SLOW_DISPATCH is not supported for memcpy!");
    }

    if (!region.has_value()) {
        EnqueueWriteBuffer(queue, dst.device_buffer(), src, false);
    } else {
        EnqueueWriteSubBuffer(queue, dst.device_buffer(), src, region.value(), false);
    }
}

void memcpy(Tensor& dst, const void* src, const std::optional<BufferRegion>& region) {
    memcpy(dst.device()->command_queue(), dst, src, region);
}

void memcpy(CommandQueue& queue, Tensor& dst, const Tensor& src, const std::optional<BufferRegion>& region) {
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE != nullptr) {
        TT_THROW("SLOW_DISPATCH is not supported for memcpy!");
    }

    TT_ASSERT(dst.get_dtype() == src.get_dtype());
    TT_ASSERT(dst.get_layout() == src.get_layout());

    if (is_cpu_tensor(dst) && is_device_tensor(src)) {
        memcpy(queue, get_raw_host_data_ptr(dst), src, region);
    } else if (is_device_tensor(dst) && is_cpu_tensor(src)) {
        memcpy(queue, dst, get_raw_host_data_ptr(src), region);
    } else {
        TT_THROW("Unsupported memcpy");
    }
}

void memcpy(Tensor& dst, const Tensor& src, const std::optional<BufferRegion>& region) {
    if (is_cpu_tensor(dst) && is_device_tensor(src)) {
        memcpy(src.device()->command_queue(), dst, src, region);
    } else if (is_device_tensor(dst) && is_cpu_tensor(src)) {
        memcpy(dst.device()->command_queue(), dst, src, region);
    } else {
        TT_THROW("Unsupported memcpy");
    }
}

Tensor allocate_tensor_on_devices(const TensorSpec& tensor_spec, const std::vector<IDevice*>& devices) {
    // Top level wrapper to asynchronously create a device tensor (single- or multi-device).
    Tensor device_tensor = Tensor(devices);

    // Save the ref count to later re-set it:
    // 1. device_tensor is copied in the lambda by the main thread, which increments the ref count.
    // 2. The destruction happens in a worker thread, which doesn't decrement the ref count.
    const uint32_t device_tensor_ref_count = device_tensor.tensor_attributes->record_main_thread_ref_count();
    const auto& workers_in_use = device_tensor.get_workers();
    uint32_t num_workers = workers_in_use.size();

    for (int worker_index = 0; worker_index < num_workers; ++worker_index) {
        auto& worker = devices[worker_index];
        worker->push_work([worker, device_tensor, tensor_spec, worker_index]() mutable {
            auto local_tensor = create_device_tensor(tensor_spec, worker);
            insert_buffer_and_shape_for_device(worker, local_tensor, device_tensor, worker_index);

            uint32_t num_workers_completed = (device_tensor.tensor_attributes->num_workers_completed)++;
            if (not num_workers_completed) {
                device_tensor.set_tensor_spec(tensor_spec);
            }
        });
    }
    device_tensor.tensor_attributes->update_main_thread_ref_count(workers_in_use.at(0), device_tensor_ref_count);
    return device_tensor;
}

Tensor allocate_tensor_on_mesh(const TensorSpec& tensor_spec, distributed::MeshDevice* mesh_device) {
    // Allocate a mesh buffer synchronously.
    TT_FATAL(
        tt::tt_metal::detail::InMainThread(), "Allocation of a tensor on mesh must be called from the main thread");
    auto mesh_buffer = tensor_impl::allocate_mesh_buffer_on_device(mesh_device, tensor_spec);
    MultiDeviceStorage multi_device_storage(std::move(mesh_buffer), tensor_spec);
    return Tensor(std::move(multi_device_storage), tensor_spec);
}

void write_tensor(const Tensor& host_tensor, Tensor device_tensor, QueueId cq_id) {
    // Top level wrapper to copy a host tensor to a preallocated device tensor
    TT_ASSERT(device_tensor.workers.size(), "Workers must be specified for device_tensor in write_tensor");

    Tensor async_safe_tensor = copy_borrowed_tensor_in_async_mode(device_tensor.workers.at(0), host_tensor);
    TT_FATAL(
        async_safe_tensor.storage_type() == StorageType::BORROWED or
            async_safe_tensor.storage_type() == StorageType::OWNED or
            async_safe_tensor.storage_type() == StorageType::MULTI_DEVICE_HOST,
        "write_tensor only supports host_tensor to device_tensor data transfer");

    uint32_t host_tensor_ref_count = async_safe_tensor.tensor_attributes->record_main_thread_ref_count();
    uint32_t device_tensor_ref_count = device_tensor.tensor_attributes->record_main_thread_ref_count();

    for (int worker_index = 0; worker_index < device_tensor.workers.size(); ++worker_index) {
        auto& worker = device_tensor.workers[worker_index];
        worker->push_work([cq_id, worker, worker_index, async_safe_tensor, device_tensor]() mutable {
            TT_FATAL(
                device_tensor.storage_type() == StorageType::DEVICE or
                    device_tensor.storage_type() == StorageType::MULTI_DEVICE,
                "write_tensor only supports host_tensor to device_tensor data transfer");
            TT_FATAL(async_safe_tensor.get_logical_shape() == device_tensor.get_logical_shape(), "Error");
            TT_FATAL(async_safe_tensor.get_dtype() == device_tensor.get_dtype(), "Error");
            TT_FATAL(
                async_safe_tensor.get_tensor_spec().page_config() == device_tensor.get_tensor_spec().page_config(),
                "Error");
            std::visit(
                tt::stl::overloaded{
                    [worker, worker_index, cq_id, &async_safe_tensor](const DeviceStorage& device_storage) {
                        // Copying from host to a single device.
                        void* host_data = std::visit(
                            tt::stl::overloaded{
                                [](BorrowedStorage s) {
                                    return std::visit(
                                        [](auto&& b) { return reinterpret_cast<void*>(b.data()); }, s.buffer);
                                },
                                [](OwnedStorage s) {
                                    return std::visit(
                                        [](auto&& b) { return reinterpret_cast<void*>(b.begin()); }, s.buffer);
                                },
                                [](const MultiDeviceHostStorage& host_storage) {
                                    TT_ASSERT(
                                        host_storage.num_buffers() == 1,
                                        "Cannot copy multi-buffer host storage to a single device");
                                    return std::visit(
                                        [](auto&& b) -> void* { return b.begin(); }, host_storage.get_buffer(0));
                                },
                                [](auto&&) -> void* { TT_THROW("Unreachable"); },
                            },
                            async_safe_tensor.get_storage());
                        EnqueueWriteBuffer(
                            worker->command_queue(*cq_id),
                            device_storage.get_buffer(),
                            host_data,
                            /*blocking=*/false);
                    },
                    [worker, worker_index, cq_id, &async_safe_tensor](const MultiDeviceStorage& device_storage) {
                        // Copying from host to multi-device.
                        TT_ASSERT(
                            std::holds_alternative<MultiDeviceHostStorage>(async_safe_tensor.get_storage()),
                            "Unexpected type {}",
                            tt::stl::get_active_type_name_in_variant(async_safe_tensor.get_storage()));
                        auto host_storage = std::get<MultiDeviceHostStorage>(async_safe_tensor.get_storage());
                        void* host_data = std::visit(
                            [](auto&& b) -> void* { return b.begin(); }, host_storage.get_buffer(worker_index));
                        EnqueueWriteBuffer(
                            worker->command_queue(*cq_id),
                            device_storage.get_buffer_for_device(worker),
                            host_data,
                            /*blocking=*/false);
                    },
                    [](auto&& s) { TT_THROW("Unreachable"); }},
                device_tensor.get_storage());
        });
    }
    async_safe_tensor.tensor_attributes->update_main_thread_ref_count(
        device_tensor.workers.at(0), host_tensor_ref_count);
    device_tensor.tensor_attributes->update_main_thread_ref_count(device_tensor.workers.at(0), device_tensor_ref_count);
}

Tensor set_tensor_id(const Tensor& tensor) {
    if (not GraphTracker::instance().is_enabled()) {
        return tensor;
    }
    auto output = tensor;
    output.tensor_id = ttnn::CoreIDs::instance().fetch_and_increment_tensor_id();
    return output;
};

bool validate_worker_modes(const std::vector<IDevice*>& workers) {
    bool worker_modes_match = true;
    auto first_worker_mode = workers.at(0)->get_worker_mode();
    for (auto worker : workers) {
        worker_modes_match &= (worker->get_worker_mode() == first_worker_mode);
    }
    return worker_modes_match;
}

}  // namespace tt::tt_metal
