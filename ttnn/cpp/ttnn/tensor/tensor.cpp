// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <memory>
#include <utility>

#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "impl/buffers/buffer_constants.hpp"
#include "tt_metal/tt_stl/overloaded.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_impl_wrapper.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/distributed/mesh_device.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/graph/graph_tracking.hpp"
#include "ttnn/core.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/distributed/api.hpp"

namespace tt::tt_metal {
namespace {

template <typename T>
Tensor create_owned_tensor_from_row_major_data(
    std::vector<T>&& data, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device = std::nullopt) {
    TensorSpec result_cpu_spec(
        spec.logical_shape(),
        TensorLayout(spec.data_type(), PageConfig(Layout::ROW_MAJOR, spec.tile()), MemoryConfig{}));

    Tensor output(OwnedStorage{owned_buffer::create(std::move(data))}, result_cpu_spec);

    if (spec.layout() == Layout::TILE) {
        // TODO: whenever possible, perform tiliziation on device.
        output = output.to(Layout::TILE);
    }

    if (device.has_value()) {
        output = output.to(device->get_devices(), spec.memory_config());
    }

    return output;
}

// TODO: optimize precomputing multipliers
template <typename T, typename InternalType>
std::vector<T> unpad_tensor_to_vec(const Tensor& cpu_tensor) {
    auto tiled_buffer = host_buffer::get_as<InternalType>(cpu_tensor);
    const auto untiled_shape = cpu_tensor.get_logical_shape();
    const auto tiled_shape = cpu_tensor.get_padded_shape();

    // Calculate total size of the untiled tensor
    size_t total_size = untiled_shape.volume();

    std::vector<T> untiled_data(total_size);

    auto compute_flat_index = [](const std::vector<uint32_t>& indices, const ttnn::SimpleShape& shape) -> uint32_t {
        uint32_t flat_index = 0;
        uint32_t multiplier = 1;
        for (int i = (int)indices.size() - 1; i >= 0; --i) {
            flat_index += indices[i] * multiplier;
            multiplier *= shape[i];
        }
        return flat_index;
    };

    std::vector<uint32_t> indices(tiled_shape.rank(), 0);

    for (size_t idx = 0; idx < total_size; ++idx) {
        uint32_t untiled_index = compute_flat_index(indices, untiled_shape);
        uint32_t tiled_index = compute_flat_index(indices, tiled_shape);
        if constexpr (std::is_same_v<InternalType, bfloat16>) {
            untiled_data[untiled_index] = tiled_buffer[tiled_index].to_float();
        } else {
            untiled_data[untiled_index] = tiled_buffer[tiled_index];
        }

        for (int dim = (int)tiled_shape.rank() - 1; dim >= 0; --dim) {
            if (++indices[dim] < untiled_shape[dim]) {
                break;
            }
            indices[dim] = 0;
        }
    }

    return untiled_data;
}

}  // namespace

Tensor::TensorAttributes::TensorAttributes() :
    tensor_spec(
        ttnn::SimpleShape(std::array<uint32_t, 4>{0xff, 0xff, 0xff, 0xff}),
        TensorLayout(DataType::INVALID, PageConfig(Layout::INVALID), MemoryConfig{})) {}

Tensor::TensorAttributes::TensorAttributes(Storage storage, TensorSpec tensor_spec) :
    storage(std::move(storage)), tensor_spec(std::move(tensor_spec)), metadata_populated(true) {}

void Tensor::TensorAttributes::increment_main_thread_ref_count(Device* worker) {
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

void Tensor::TensorAttributes::decrement_main_thread_ref_count(Device* worker) {
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

void Tensor::TensorAttributes::update_main_thread_ref_count(Device* worker, uint32_t ref_count) {
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
    const ttnn::SimpleShape& logical_shape,
    const ttnn::SimpleShape& padded_shape,
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

Tensor::Tensor(
    Storage storage, const ttnn::Shape& shape, DataType dtype, Layout layout, const std::optional<Tile>& tile) :
    Tensor(std::move(storage), shape.logical_shape(), shape.padded_shape(), dtype, layout, tile) {}

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
                // This tensor is being created from scratch in a worker. Track this and allow it to be explicitly
                // deallocated inside the worker (composite ops do this).
                if (tt::tt_metal::detail::InWorkerThread()) {
                    tensor_attributes->main_thread_tensor = false;
                }
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
                // This tensor is being created from scratch in a worker. Track this and allow it to be explicitly
                // deallocated inside the worker (composite ops do this).
                if (tt::tt_metal::detail::InWorkerThread()) {
                    tensor_attributes->main_thread_tensor = false;
                }
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

Tensor::Tensor(const std::vector<Device*>& workers) :
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
            workers.cbegin(), workers.cend(), std::back_inserter(storage.ordered_device_ids), [](const Device* worker) {
                return worker->id();
            });
        return Storage(std::move(storage));
    }();
    tensor_attributes->num_shards_to_be_populated = workers.size();
    if (!tt::tt_metal::detail::InWorkerThread()) {
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
        storage.shapes = std::vector<ttnn::Shape>(num_buffers, ttnn::Shape{});
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
        this->deallocate_through_destructor = other.deallocate_through_destructor;
        if (this->workers.size()) {
            if (not tt::tt_metal::detail::InWorkerThread()) {
                this->tensor_attributes->increment_main_thread_ref_count(this->workers.at(0));
            }
        }
    }
    return *this;
}

Tensor::Tensor(const Tensor& other) :
    tensor_id(other.tensor_id),
    workers(other.workers),
    tensor_attributes(other.tensor_attributes),
    deallocate_through_destructor(other.deallocate_through_destructor) {
    if (this->workers.size()) {
        if (not tt::tt_metal::detail::InWorkerThread()) {
            this->tensor_attributes->increment_main_thread_ref_count(this->workers.at(0));
        }
    }
}

Tensor::~Tensor() {
    ZoneScoped;
    this->deallocate_through_destructor = true;
    this->deallocate();
    // Decrement main thread ref count for all tensors on device
    if (this->workers.size() and this->tensor_attributes) {
        this->tensor_attributes->decrement_main_thread_ref_count(this->workers.at(0));
    }
    tensor_attributes.reset();
}

Tensor::Tensor(
    Storage storage, const ttnn::SimpleShape& shape, DataType dtype, Layout layout, const std::optional<Tile>& tile) :
    Tensor(std::move(storage), ttnn::Shape(shape.view()), dtype, layout, tile) {}

void Tensor::deallocate(bool force) {
    ZoneScopedN("TensorDeallocate");
    // GraphTracker::instance().track_function_start("Tensor::deallocate", *this, force);
    if (this->tensor_attributes.use_count()) {
        // Check if the attributes didn't get moved to another tensor.
        // If not, we can deallocate this tensor.
        std::visit(
            [force, this](auto& storage) {
                using T = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<T, OwnedStorage>) {
                    if (this->tensor_attributes.use_count() == 1) {
                        std::visit([](auto&& buffer) { buffer.reset(); }, storage.buffer);
                    }
                } else if constexpr (std::is_same_v<T, DeviceStorage>) {
                    if (not this->workers.at(0)->is_initialized()) {
                        return;
                    }
                    if ((not tt::tt_metal::detail::InWorkerThread()) or
                        not this->tensor_attributes->main_thread_tensor) {
                        if (not this->tensor_attributes->main_thread_tensor) {
                            TT_ASSERT(
                                not this->tensor_attributes->main_thread_ref_count,
                                "main_thread_ref_count for tensors created inside a worker thread must be 0");
                        }
                        // If owned by the main thread, deallocate this tensor only from the main thread. If owned by
                        // worker thread, allow deallocation in worker and use shared_ptr ref count, since this is a
                        // thread_local tensor
                        uint32_t ref_count_to_use =
                            (this->workers.at(0)->get_worker_mode() == WorkExecutorMode::SYNCHRONOUS or
                             not this->tensor_attributes->main_thread_tensor)
                                ? this->tensor_attributes.use_count()
                                : this->tensor_attributes->main_thread_ref_count;
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
                    } else {
                        TT_FATAL(
                            this->deallocate_through_destructor,
                            "Device tensors created in the main thread cannot be explictly deallocated in worker "
                            "threads.");
                    }
                } else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                    if (force) {
                        TT_THROW("Cannot deallocate tensor with borrowed storage!");
                    }
                } else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                    if (not this->workers.at(0)->is_initialized()) {
                        return;
                    }
                    if ((not tt::tt_metal::detail::InWorkerThread()) or
                        not this->tensor_attributes->main_thread_tensor) {
                        // If owned by the main thread, deallocate this tensor only from the main thread. If owned by
                        // worker thread, allow deallocation in worker and use shared_ptr ref count, since this is a
                        // thread_local tensor
                        uint32_t ref_count_to_use =
                            (this->workers.at(0)->get_worker_mode() == WorkExecutorMode::SYNCHRONOUS or
                             not this->tensor_attributes->main_thread_tensor)
                                ? this->tensor_attributes.use_count()
                                : this->tensor_attributes->main_thread_ref_count;
                        if ((force or ref_count_to_use == 1) and not this->tensor_attributes->deallocated) {
                            this->tensor_attributes->deallocated = true;
                            auto dealloc_lambda = std::make_shared<std::function<void(Device*)>>(
                                [force, attr = this->tensor_attributes](Device* worker) mutable {
                                    ZoneScopedN("ShardDeallocate");
                                    TT_ASSERT(
                                        std::holds_alternative<tt::tt_metal::MultiDeviceStorage>(attr->storage),
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

                            for (auto worker : this->workers) {
                                worker->push_work([worker, dealloc_lambda]() mutable { (*dealloc_lambda)(worker); });
                            }
                        }
                    } else {
                        TT_FATAL(
                            this->deallocate_through_destructor,
                            "Device tensors created in the main thread cannot be explictly deallocated in worker "
                            "threads.");
                    }
                } else if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                    if (this->tensor_attributes.use_count() == 1) {
                        // Same logic as above for host tensors
                        for (int i = 0; i < storage.num_buffers(); i++) {
                            auto& current_buffer = storage.get_buffer(i);
                            std::visit([](auto&& buffer) { buffer.reset(); }, current_buffer);
                        }
                    }
                } else {
                    raise_unsupported_storage<T>();
                }
            },
            this->tensor_attributes->storage);
    }
    // GraphTracker::instance().track_function_end();
}

void Tensor::perform_cleanup_for_async_mode() {
    // Used when tensor attributes object for this is reassigned by copy
    // or move assignment operator
    if (this->tensor_attributes) {
        // Object has tensor_attributes that will be reassigned
        if (this->workers.size() and (not tt::tt_metal::detail::InWorkerThread()) and
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
                std::get<StorageType>(this->tensor_attributes->storage).shapes = storage.shapes;
            }
        },
        other.get_storage());  // Non blocking storage query, since this is done for tensors that get created inside the
                               // worker thread
    this->tensor_attributes->num_workers_completed++;
}

std::vector<Device*> Tensor::get_workers(bool blocking) const {
    ZoneScoped;
    // Initialize an empty worker vector (remains empty for host side storage)
    std::vector<Device*> workers = {};

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
                    workers = std::vector<Device*>{this->device()};
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
tt::tt_metal::LegacyShape Tensor::get_legacy_shape() const {
    wait_for_tensor_metadata_populated();
    return legacy_shape();
}

ttnn::Shape Tensor::get_shape() const {
    wait_for_tensor_metadata_populated();
    return shape();
}
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

const ttnn::SimpleShape& Tensor::get_logical_shape() const {
    wait_for_tensor_metadata_populated();
    return logical_shape();
}

const ttnn::SimpleShape& Tensor::get_padded_shape() const {
    wait_for_tensor_metadata_populated();
    return padded_shape();
}

tt::tt_metal::Padding Tensor::get_padding() const { return get_legacy_shape().padding(); }

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
            std::vector<bfloat16> bfloat16_data;
            bfloat16_data.reserve(buffer.size());
            std::transform(std::begin(buffer), std::end(buffer), std::back_inserter(bfloat16_data), [](float value) {
                return bfloat16(value);
            });
            return create_owned_tensor_from_row_major_data(std::move(bfloat16_data), spec, device);
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
                tensor = tensor.to(device->get_devices(), spec.memory_config());
            }
            return tensor;
        }
        default: {
            TT_THROW("Unsupported data type for from_span<float>: {}", spec.data_type());
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
        "Unsupported data type for from_span: got {}, expected: {}",
        spec.data_type(),
        convert_to_data_type<T>());
    return create_owned_tensor_from_row_major_data(std::vector<T>(buffer.begin(), buffer.end()), spec, device);
}

template <>
std::vector<float> Tensor::to_vector<float>() const {
    Tensor cpu_tensor = this->cpu();
    switch (cpu_tensor.get_dtype()) {
        case DataType::BFLOAT16: return unpad_tensor_to_vec<float, bfloat16>(cpu_tensor.to(Layout::ROW_MAJOR));
        case DataType::FLOAT32: return unpad_tensor_to_vec<float, float>(cpu_tensor.to(Layout::ROW_MAJOR));
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: {
            const auto& tile = cpu_tensor.get_tensor_spec().tile();
            std::vector<uint32_t> packed_data =
                owned_buffer::get_as<std::uint32_t>(std::get<OwnedStorage>(cpu_tensor.storage()).buffer).get();
            std::vector<float> unpacked_data =
                cpu_tensor.get_tensor_spec().data_type() == DataType::BFLOAT8_B
                    ? unpack_bfp8_tiles_into_float_vec(
                          packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile)
                    : unpack_bfp4_tiles_into_float_vec(
                          packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);

            return tensor_impl::decode_tensor_data(unpacked_data, cpu_tensor.tensor_spec());
        }
        default: {
            TT_THROW("Cannot convert tensor to vector for data type: {}", cpu_tensor.get_dtype());
        }
    }
}

template <typename T>
std::vector<T> Tensor::to_vector() const {
    auto cpu_tensor = this->cpu().to(Layout::ROW_MAJOR);
    TT_FATAL(
        cpu_tensor.get_dtype() == convert_to_data_type<T>(),
        "Unsupported data type for to_vector: got {}, expected: {}",
        cpu_tensor.get_dtype(),
        convert_to_data_type<T>());
    return unpad_tensor_to_vec<T, T>(cpu_tensor);
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

template std::vector<bfloat16> Tensor::to_vector<bfloat16>() const;
template std::vector<int32_t> Tensor::to_vector<int32_t>() const;
template std::vector<uint8_t> Tensor::to_vector<uint8_t>() const;
template std::vector<uint16_t> Tensor::to_vector<uint16_t>() const;
template std::vector<uint32_t> Tensor::to_vector<uint32_t>() const;

Tensor Tensor::to(
    Device* target_device,
    const MemoryConfig& mem_config,
    uint8_t cq_id,
    const std::vector<SubDeviceId>& sub_device_ids) const {
    return tensor_ops::tensor_to(*this, target_device, mem_config, cq_id, sub_device_ids);
}

Tensor Tensor::to(
    distributed::MeshDevice* mesh_device,
    const MemoryConfig& mem_config,
    uint8_t cq_id,
    const std::vector<SubDeviceId>& sub_device_ids) const {
    std::vector<Device*> workers_to_use = ttnn::distributed::get_mapped_devices(*this, *mesh_device);
    return tensor_ops::tensor_to(*this, workers_to_use, mem_config, cq_id, sub_device_ids);
}

Tensor Tensor::to(
    const std::vector<Device*>& workers,
    const MemoryConfig& mem_config,
    uint8_t cq_id,
    const std::vector<SubDeviceId>& sub_device_ids) const {
    return tensor_ops::tensor_to(*this, workers, mem_config, cq_id, sub_device_ids);
}

Tensor Tensor::cpu(bool blocking, uint8_t cq_id, const std::vector<SubDeviceId>& sub_device_ids) const {
    return tensor_ops::tensor_cpu(*this, blocking, cq_id, sub_device_ids);
}

Tensor Tensor::extract_shard(const CoreCoord& core) const {
    ZoneScoped;
    const auto& buffer_page_mapping = *this->buffer()->get_buffer_page_mapping();
    auto core_id = buffer_page_mapping.core_to_core_id_.at(core);
    return this->extract_shard(core_id);
}

Tensor Tensor::extract_shard(const uint32_t& core_id) const {
    return tensor_impl::extract_shard_wrapper(*this, core_id);
}

Tensor Tensor::to(Layout target_layout, Device* worker) const {
    return tensor_ops::tensor_to(*this, target_layout, worker);
}

Tensor Tensor::to(Layout target_layout, distributed::MeshDevice* mesh_device) const {
    return tensor_ops::tensor_to(*this, target_layout, mesh_device);
}

const std::string Tensor::write_to_string() const { return tensor_impl::to_string_wrapper(*this); }

void Tensor::print() const { tensor_ops::tensor_print(*this); }

Tensor Tensor::pad(
    const ttnn::SimpleShape& output_padded_shape, const ttnn::SimpleShape& input_tensor_start, float pad_value) const {
    return tensor_ops::tensor_pad(*this, output_padded_shape, input_tensor_start, pad_value);
}

Tensor Tensor::unpad(const ttnn::SimpleShape& output_tensor_start, const ttnn::SimpleShape& output_tensor_end) const {
    return tensor_ops::tensor_unpad(*this, output_tensor_start, output_tensor_end);
}

Tensor Tensor::pad_to_tile(float pad_value) const { return tensor_ops::tensor_pad_to_tile(*this, pad_value); }

Tensor Tensor::unpad_from_tile(const ttnn::SimpleShape& output_tensor_shape) const {
    return tensor_ops::tensor_unpad_from_tile(*this, output_tensor_shape);
}

const bool Tensor::is_sharded() const {
    return is_tensor_on_device_or_multidevice(*this) ? this->memory_config().is_sharded() : false;
}

uint32_t Tensor::element_size() const { return tensor_impl::element_size_bytes(this->get_dtype()); }

bool Tensor::is_allocated() const {
    ZoneScoped;
    auto output = std::visit([](auto&& storage) -> bool { return storage.is_allocated(); }, this->get_storage());
    return output;
}

std::vector<uint32_t> Tensor::host_page_ordering() {
    const auto& buffer_page_mapping = *this->buffer()->get_buffer_page_mapping();
    auto cores = buffer_page_mapping.all_cores_;
    auto shard_size = buffer()->shard_spec().size();
    auto num_pages = cores.size() * shard_size;

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

const ttnn::SimpleShape Tensor::strides() const {
    return ttnn::SimpleShape(tt::tt_metal::compute_strides(this->get_padded_shape()));
}

uint32_t Tensor::volume() const { return tt::tt_metal::compute_volume(this->get_legacy_shape()); }

uint32_t Tensor::get_logical_volume() const { return get_logical_shape().volume(); }

bool Tensor::is_scalar() const {
    const ttnn::SimpleShape logical_shape = this->get_shape().logical_shape();
    return logical_shape.rank() == 0 || logical_shape.volume() == 1;
}

Tensor create_device_tensor(const TensorSpec& tensor_spec, Device* device) {
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
    const ttnn::SimpleShape& shape,
    DataType data_type,
    Layout layout,
    Device* device,
    const MemoryConfig& memory_config,
    const std::optional<Tile>& tile) {
    return create_device_tensor(
        TensorSpec(shape, TensorLayout(data_type, PageConfig(layout, tile), memory_config)), device);
}

Tensor create_device_tensor(
    const ttnn::Shape& shape,
    DataType data_type,
    Layout layout,
    Device* device,
    const MemoryConfig& memory_config,
    const std::optional<Tile>& tile) {
    return create_device_tensor(
        TensorSpec(
            shape.logical_shape(),
            TensorLayout::fromLegacyPaddedShape(data_type, PageConfig(layout, tile), memory_config, shape)),
        device);
}

namespace detail {
template <typename DataType>
void* get_raw_host_data_ptr(const Tensor& tensor) {
    return std::visit(
        tt::stl::overloaded{
            [](const OwnedStorage& s) {
                auto buffer = owned_buffer::get_as<DataType>(s.buffer);
                return buffer.data();
            },
            [](const BorrowedStorage& s) {
                if constexpr (
                    std::is_same_v<DataType, float> or std::is_same_v<DataType, bfloat16> or
                    std::is_same_v<DataType, std::uint32_t> or std::is_same_v<DataType, std::int32_t> or
                    std::is_same_v<DataType, std::uint8_t> or std::is_same_v<DataType, std::uint16_t>) {
                    auto buffer = borrowed_buffer::get_as<DataType>(s.buffer);
                    return buffer.data();
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
    CommandQueue& queue, void* dst, const Tensor& src, const std::optional<std::size_t> transfer_size, bool blocking) {
    TT_ASSERT(not transfer_size.has_value(), "transfer_size is not supported for memcpy right now!");
    if (not is_device_tensor(src)) {
        TT_THROW("memcpy: src tensor must be on device");
    }

    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE != nullptr) {
        TT_THROW("SLOW_DISPATCH is not supported for memcpy!");
    }
    EnqueueReadBuffer(queue, src.device_buffer(), dst, blocking);
}

void memcpy(void* dst, const Tensor& src, const std::optional<std::size_t> transfer_size, bool blocking) {
    memcpy(src.device()->command_queue(), dst, src, transfer_size, blocking);
}

void memcpy(CommandQueue& queue, Tensor& dst, const void* src, const std::optional<std::size_t> transfer_size) {
    TT_ASSERT(not transfer_size.has_value(), "transfer_size is not supported for memcpy right now!");
    if (not is_device_tensor(dst)) {
        TT_THROW("memcpy: memcpy to non-device tensor is not supported!");
    }
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE != nullptr) {
        TT_THROW("SLOW_DISPATCH is not supported for memcpy!");
    }
    EnqueueWriteBuffer(queue, dst.device_buffer(), src, false);
}

void memcpy(Tensor& dst, const void* src, const std::optional<std::size_t> transfer_size) {
    memcpy(dst.device()->command_queue(), dst, src, transfer_size);
}

void memcpy(CommandQueue& queue, Tensor& dst, const Tensor& src, const std::optional<std::size_t> transfer_size) {
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE != nullptr) {
        TT_THROW("SLOW_DISPATCH is not supported for memcpy!");
    }

    TT_ASSERT(dst.get_dtype() == src.get_dtype());
    TT_ASSERT(dst.get_layout() == src.get_layout());

    if (is_cpu_tensor(dst) && is_device_tensor(src)) {
        memcpy(queue, get_raw_host_data_ptr(dst), src, transfer_size);
    } else if (is_device_tensor(dst) && is_cpu_tensor(src)) {
        memcpy(queue, dst, get_raw_host_data_ptr(src), transfer_size);
    } else {
        TT_THROW("Unsupported memcpy");
    }
}

void memcpy(Tensor& dst, const Tensor& src, const std::optional<std::size_t> transfer_size) {
    if (is_cpu_tensor(dst) && is_device_tensor(src)) {
        memcpy(src.device()->command_queue(), dst, src, transfer_size);
    } else if (is_device_tensor(dst) && is_cpu_tensor(src)) {
        memcpy(dst.device()->command_queue(), dst, src, transfer_size);
    } else {
        TT_THROW("Unsupported memcpy");
    }
}

Tensor allocate_tensor_on_devices(
    const ttnn::Shape& shape,
    DataType data_type,
    Layout layout,
    const std::vector<Device*>& devices,
    const MemoryConfig& memory_config,
    const std::optional<Tile>& tile) {
    // Top level wrapper to asynchronously create a device tensor (single- or multi-device).
    Tensor device_tensor = Tensor(devices);
    TensorSpec tensor_spec(
        shape.logical_shape(),
        TensorLayout::fromLegacyPaddedShape(data_type, PageConfig(layout, tile), memory_config, shape));

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

void write_tensor(
    const Tensor& host_tensor, Tensor device_tensor, uint8_t cq_id, const std::vector<SubDeviceId>& sub_device_ids) {
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
        worker->push_work([cq_id, worker, worker_index, async_safe_tensor, device_tensor, sub_device_ids]() mutable {
            TT_FATAL(
                device_tensor.storage_type() == StorageType::DEVICE or
                    device_tensor.storage_type() == StorageType::MULTI_DEVICE,
                "write_tensor only supports host_tensor to device_tensor data transfer");
            TT_FATAL(async_safe_tensor.get_shape() == device_tensor.get_shape(), "Error");
            TT_FATAL(async_safe_tensor.get_dtype() == device_tensor.get_dtype(), "Error");
            TT_FATAL(
                async_safe_tensor.get_tensor_spec().page_config() == device_tensor.get_tensor_spec().page_config(),
                "Error");
            std::visit(
                tt::stl::overloaded{
                    [worker, worker_index, cq_id, &async_safe_tensor, sub_device_ids](
                        const DeviceStorage& device_storage) {
                        // Copying from host to a single device.
                        void* host_data = std::visit(
                            tt::stl::overloaded{
                                [](BorrowedStorage s) {
                                    return std::visit([](auto&& b) { return b.data(); }, s.buffer);
                                },
                                [](OwnedStorage s) {
                                    return std::visit([](auto&& b) { return static_cast<void*>(b.begin()); }, s.buffer);
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
                            worker->command_queue(cq_id),
                            device_storage.get_buffer(),
                            host_data,
                            /*blocking=*/false,
                            sub_device_ids);
                    },
                    [worker, worker_index, cq_id, &async_safe_tensor, sub_device_ids](
                        const MultiDeviceStorage& device_storage) {
                        // Copying from host to multi-device.
                        TT_ASSERT(
                            std::holds_alternative<MultiDeviceHostStorage>(async_safe_tensor.get_storage()),
                            "Unexpected type {}",
                            tt::stl::get_active_type_name_in_variant(async_safe_tensor.get_storage()));
                        auto host_storage = std::get<MultiDeviceHostStorage>(async_safe_tensor.get_storage());
                        void* host_data = std::visit(
                            [](auto&& b) -> void* { return b.begin(); }, host_storage.get_buffer(worker_index));
                        EnqueueWriteBuffer(
                            worker->command_queue(cq_id),
                            device_storage.get_buffer_for_device(worker),
                            host_data,
                            /*blocking=*/false,
                            sub_device_ids);
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

bool validate_worker_modes(const std::vector<Device*>& workers) {
    bool worker_modes_match = true;
    auto first_worker_mode = workers.at(0)->get_worker_mode();
    for (auto worker : workers) {
        worker_modes_match &= (worker->get_worker_mode() == first_worker_mode);
    }
    return worker_modes_match;
}

}  // namespace tt::tt_metal
