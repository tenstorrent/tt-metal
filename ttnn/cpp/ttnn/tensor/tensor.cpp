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
#include <tt-metalium/buffer_types.hpp>
#include <tt_stl/overloaded.hpp>
#include "storage.hpp"

#include "tt-metalium/mesh_device_view.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_impl_wrapper.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tracy/Tracy.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include "ttnn/core.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/distributed/api.hpp"

namespace tt::tt_metal {
namespace {

template <typename T>
Tensor create_owned_tensor_from_row_major_data(
    std::vector<T>&& data, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device, ttnn::QueueId cq_id) {
    auto physical_data = tensor_impl::encode_tensor_data(std::move(data), spec);

    Tensor output(HostStorage(host_buffer::create(std::move(physical_data))), spec);

    if (device.has_value()) {
        if (auto mesh_device = device->get_mesh_device()) {
            output = output.to_device(mesh_device, spec.memory_config(), cq_id);
        } else {
            output = output.to_device(device->get_devices(), spec.memory_config(), cq_id);
        }
    }

    return output;
}

}  // namespace

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
            if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                if (storage.mesh_buffer != nullptr) {
                    mesh_device_ = storage.mesh_buffer->device();
                }
                workers = {storage.get_device()};
            }
        },
        tensor_attributes->get_storage());
}

Tensor::Tensor(distributed::MeshDevice* mesh_device, TensorSpec spec) :
    tensor_attributes(std::make_shared<TensorAttributes>(DeviceStorage(), std::move(spec))),
    workers({mesh_device}),
    mesh_device_(mesh_device) {
    TT_FATAL(mesh_device_ != nullptr, "Mesh device is nullptr");
}

Tensor::Tensor(const std::vector<IDevice*>& workers, TensorSpec spec) :
    tensor_attributes(std::make_shared<TensorAttributes>(
        workers.empty() ? Storage(HostStorage()) : DeviceStorage(), std::move(spec))),
    workers(workers) {
    TT_FATAL(workers.size() <= 1, "Only single device is supported.");
}

Tensor::Tensor(
    uint32_t num_buffers, TensorSpec spec, std::optional<DistributedTensorConfig> distributed_tensor_config) :
    tensor_attributes(std::make_shared<TensorAttributes>(
        [&]() {
            if (num_buffers <= 1) {
                return Storage(HostStorage());
            }
            MultiDeviceHostStorage storage;
            if (distributed_tensor_config.has_value()) {
                storage.strategy = distributed_tensor_config.value();
            }
            storage.buffers = std::vector<HostBuffer>(num_buffers, HostBuffer());
            storage.specs = std::vector<ttnn::TensorSpec>(
                num_buffers,
                TensorSpec(Shape{}, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{})));
            return Storage(std::move(storage));
        }(),
        std::move(spec))) {}

Tensor& Tensor::operator=(const Tensor& other) {
    // Don't self-assign
    this->tensor_id = other.tensor_id;
    if (this->tensor_attributes != other.tensor_attributes) {
        this->workers = other.workers;
        this->tensor_attributes = other.tensor_attributes;
    }
    this->mesh_device_ = other.mesh_device_;
    return *this;
}

Tensor::Tensor(const Tensor& other) :
    tensor_id(other.tensor_id), workers(other.workers), tensor_attributes(other.tensor_attributes) {
    this->mesh_device_ = other.mesh_device_;
}

Tensor::~Tensor() {
    ZoneScoped;
    this->deallocate_impl(/*force=*/false, /*deallocation_through_destructor=*/true);
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
        return tensor.tensor_attributes.use_count();
    };

    std::visit(
        tt::stl::overloaded{
            [this](HostStorage& storage) {
                if (this->tensor_attributes.use_count() == 1) {
                    storage.buffer.deallocate();
                }
            },
            [this](MultiDeviceHostStorage& storage) {
                if (this->tensor_attributes.use_count() == 1) {
                    for (int i = 0; i < storage.num_buffers(); i++) {
                        storage.get_buffer(i).deallocate();
                    }
                }
            },
            [force, this](DeviceStorage& storage) {
                if ((force or this->tensor_attributes.use_count() == 1)) {
                    if (storage.mesh_buffer and (force or storage.mesh_buffer.use_count() == 1)) {
                        storage.mesh_buffer->deallocate();
                    } else if (storage.buffer and (force or storage.buffer.use_count() == 1)) {
                        DeallocateBuffer(*(storage.buffer));
                    }
                    storage.mesh_buffer.reset();
                    storage.buffer.reset();
                }
            }},
        this->tensor_attributes->get_storage());
    // GraphTracker::instance().track_function_end();
}

std::vector<IDevice*> Tensor::get_workers(bool blocking) const {
    ZoneScoped;
    // Initialize an empty worker vector (remains empty for host side storage)
    std::vector<IDevice*> workers = {};

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
                    workers = std::vector<IDevice*>{this->device()};
                } else {
                    // Already populated.
                    workers = this->workers;
                }
            }
        },
        this->tensor_attributes->get_storage());
    return workers;
}

// Getters - Spin until tensor is populated before querying tensor metadata
DataType Tensor::get_dtype() const { return dtype(); }
Layout Tensor::get_layout() const { return layout(); }

const TensorSpec& Tensor::get_tensor_spec() const { return tensor_spec(); }

const ttnn::Shape& Tensor::get_logical_shape() const { return logical_shape(); }

const ttnn::Shape& Tensor::get_padded_shape() const { return padded_shape(); }

const Storage& Tensor::get_storage() const { return this->tensor_attributes->get_storage(); }

Storage& Tensor::get_storage() { return this->tensor_attributes->get_storage(); }

template <>
Tensor Tensor::from_span<float>(
    tt::stl::Span<const float> buffer,
    const TensorSpec& spec,
    std::optional<ttnn::AnyDevice> device,
    ttnn::QueueId cq_id) {
    size_t volume = spec.logical_shape().volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    switch (spec.data_type()) {
        case DataType::FLOAT32:
            return create_owned_tensor_from_row_major_data(
                std::vector<float>(buffer.begin(), buffer.end()), spec, device, cq_id);
        case DataType::BFLOAT16: {
            return create_owned_tensor_from_row_major_data(
                std::vector<bfloat16>(buffer.begin(), buffer.end()), spec, device, cq_id);
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

            Tensor tensor(HostStorage(host_buffer::create(std::move(packed_block_floats))), spec);
            if (device.has_value()) {
                tensor = tensor.to_device(device->get_devices(), spec.memory_config(), cq_id);
            }
            return tensor;
        }
        default: {
            TT_THROW("Unsupported data type: {}", spec.data_type());
        }
    }
}

template <typename T>
Tensor Tensor::from_span(
    tt::stl::Span<const T> buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device, ttnn::QueueId cq_id) {
    size_t volume = spec.logical_shape().volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    TT_FATAL(
        spec.data_type() == convert_to_data_type<T>(),
        "Unsupported data type: got {}, expected: {}",
        spec.data_type(),
        convert_to_data_type<T>());
    return create_owned_tensor_from_row_major_data(std::vector<T>(buffer.begin(), buffer.end()), spec, device, cq_id);
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
    HostBuffer data_buffer(buffer, MemoryPin(on_creation_callback, on_destruction_callback));
    return Tensor(HostStorage(std::move(data_buffer)), shape, convert_to_data_type<T>(), Layout::ROW_MAJOR, tile);
}

template <>
Tensor Tensor::from_vector<float>(
    std::vector<float>&& buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device, ttnn::QueueId cq_id) {
    size_t volume = spec.logical_shape().volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    if (spec.data_type() == DataType::FLOAT32) {
        // User `buffer` directly, when no type conversion is needed.
        return create_owned_tensor_from_row_major_data(std::move(buffer), spec, device, cq_id);
    } else {
        return from_span(tt::stl::Span<const float>(buffer.data(), buffer.size()), spec, device, cq_id);
    }
}

template <typename T>
Tensor Tensor::from_vector(
    std::vector<T>&& buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device, ttnn::QueueId cq_id) {
    size_t volume = spec.logical_shape().volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    TT_FATAL(
        spec.data_type() == convert_to_data_type<T>(),
        "Unsupported data type: got {}, expected: {}",
        spec.data_type(),
        convert_to_data_type<T>());
    return create_owned_tensor_from_row_major_data(std::move(buffer), spec, device, cq_id);
}

template <>
std::vector<float> Tensor::to_vector<float>(ttnn::QueueId cq_id) const {
    Tensor cpu_tensor = this->cpu(/*blocking=*/true, cq_id);
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
std::vector<T> Tensor::to_vector(ttnn::QueueId cq_id) const {
    TT_FATAL(
        this->get_dtype() == convert_to_data_type<T>(),
        "Unsupported data type for to_vector: got {}, expected: {}",
        this->get_dtype(),
        convert_to_data_type<T>());
    auto cpu_tensor = this->cpu(/*blocking=*/true, cq_id);
    auto data = host_buffer::get_as<T>(cpu_tensor);
    auto physical_data = std::vector<T>(data.begin(), data.end());
    return tensor_impl::decode_tensor_data(std::move(physical_data), cpu_tensor.tensor_spec());
}

// Instantiate explicitly for the supported types.
template Tensor Tensor::from_span<bfloat16>(
    tt::stl::Span<const bfloat16> buffer,
    const TensorSpec& spec,
    std::optional<ttnn::AnyDevice> device,
    ttnn::QueueId cq_id);
template Tensor Tensor::from_span<int32_t>(
    tt::stl::Span<const int32_t> buffer,
    const TensorSpec& spec,
    std::optional<ttnn::AnyDevice> device,
    ttnn::QueueId cq_id);
template Tensor Tensor::from_span<uint8_t>(
    tt::stl::Span<const uint8_t> buffer,
    const TensorSpec& spec,
    std::optional<ttnn::AnyDevice> device,
    ttnn::QueueId cq_id);
template Tensor Tensor::from_span<uint16_t>(
    tt::stl::Span<const uint16_t> buffer,
    const TensorSpec& spec,
    std::optional<ttnn::AnyDevice> device,
    ttnn::QueueId cq_id);
template Tensor Tensor::from_span<uint32_t>(
    tt::stl::Span<const uint32_t> buffer,
    const TensorSpec& spec,
    std::optional<ttnn::AnyDevice> device,
    ttnn::QueueId cq_id);
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
    std::vector<bfloat16>&& buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device, ttnn::QueueId cq_id);
template Tensor Tensor::from_vector<int32_t>(
    std::vector<int32_t>&& buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device, ttnn::QueueId cq_id);
template Tensor Tensor::from_vector<uint8_t>(
    std::vector<uint8_t>&& buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device, ttnn::QueueId cq_id);
template Tensor Tensor::from_vector<uint16_t>(
    std::vector<uint16_t>&& buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device, ttnn::QueueId cq_id);
template Tensor Tensor::from_vector<uint32_t>(
    std::vector<uint32_t>&& buffer, const TensorSpec& spec, std::optional<ttnn::AnyDevice> device, ttnn::QueueId cq_id);

template std::vector<bfloat16> Tensor::to_vector<bfloat16>(ttnn::QueueId cq_id) const;
template std::vector<int32_t> Tensor::to_vector<int32_t>(ttnn::QueueId cq_id) const;
template std::vector<uint8_t> Tensor::to_vector<uint8_t>(ttnn::QueueId cq_id) const;
template std::vector<uint16_t> Tensor::to_vector<uint16_t>(ttnn::QueueId cq_id) const;
template std::vector<uint32_t> Tensor::to_vector<uint32_t>(ttnn::QueueId cq_id) const;

Tensor Tensor::to_device(IDevice* target_device, const MemoryConfig& mem_config, QueueId cq_id) const {
    if (auto mesh_device = dynamic_cast<distributed::MeshDevice*>(target_device)) {
        return to_device(mesh_device, mem_config, cq_id);
    }
    return tensor_ops::tensor_to_device(*this, target_device, mem_config, cq_id);
}

Tensor Tensor::to_device(distributed::MeshDevice* mesh_device, const MemoryConfig& mem_config, QueueId cq_id) const {
    return tensor_ops::tensor_to_device(*this, mesh_device, mem_config, cq_id);
}

Tensor Tensor::to_device(const std::vector<IDevice*>& workers, const MemoryConfig& mem_config, QueueId cq_id) const {
    if (workers.size() == 1) {
        if (auto mesh_device = dynamic_cast<distributed::MeshDevice*>(workers[0])) {
            return to_device(mesh_device, mem_config, cq_id);
        }
    }
    return tensor_ops::tensor_to_device(*this, workers, mem_config, cq_id);
}

Tensor Tensor::cpu(bool blocking, QueueId cq_id) const {
    if (this->mesh_device_.has_value()) {
        return tensor_ops::tensor_cpu(*this, this->mesh_device_.value(), blocking, cq_id);
    }
    return tensor_ops::tensor_cpu(*this, blocking, cq_id);
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

Tensor Tensor::to_layout(Layout target_layout, IDevice* worker) const {
    return tensor_ops::tensor_to_layout(*this, target_layout, worker);
}

Tensor Tensor::to_layout(Layout target_layout, distributed::MeshDevice* mesh_device) const {
    return tensor_ops::tensor_to_layout(*this, target_layout, mesh_device);
}

std::string Tensor::write_to_string() const { return tensor_impl::to_string_wrapper(*this); }

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

bool Tensor::is_sharded() const {
    return tt::tt_metal::is_device_tensor(*this) ? this->memory_config().is_sharded() : false;
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
            [](const HostStorage&) { return StorageType::HOST; },
            [](const DeviceStorage&) { return StorageType::DEVICE; },
            [](const MultiDeviceHostStorage&) { return StorageType::MULTI_DEVICE_HOST; },
        },
        this->get_storage());
}

ttnn::Shape Tensor::strides() const { return ttnn::Shape(tt::tt_metal::compute_strides(this->get_padded_shape())); }

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

    Tensor output;
    if (distributed::MeshDevice* mesh_device = dynamic_cast<distributed::MeshDevice*>(device)) {
        output = allocate_tensor_on_mesh(tensor_spec, mesh_device);
    } else {
        auto device_buffer = tensor_impl::allocate_buffer_on_device(device, tensor_spec);
        output = Tensor(DeviceStorage{device_buffer}, tensor_spec);
    }
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

void memcpy(
    CommandQueue& queue, void* dst, const Tensor& src, const std::optional<BufferRegion>& region, bool blocking) {
    TT_FATAL(is_device_tensor(src), "memcpy: src tensor must be on device");

    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE != nullptr) {
        TT_THROW("SLOW_DISPATCH is not supported for memcpy!");
    }

    if (!region.has_value()) {
        EnqueueReadBuffer(queue, *src.buffer(), dst, blocking);
    } else {
        EnqueueReadSubBuffer(queue, *src.buffer(), dst, region.value(), blocking);
    }
}

void memcpy(
    distributed::MeshCommandQueue& queue,
    void* dst,
    const Tensor& src,
    const std::optional<BufferRegion>& region,
    bool blocking) {
    TT_FATAL(is_device_tensor(src), "memcpy: src tensor must be on device");

    TT_FATAL(queue.device()->num_devices() == 1, "memcpy only supports single device mesh");
    std::vector<distributed::MeshCommandQueue::ShardDataTransfer> shard_data_transfers = {{
        .shard_coord = *distributed::MeshCoordinateRange(queue.device()->shape()).begin(),
        .host_data = dst,
        .region = region,
    }};
    queue.enqueue_read_shards(shard_data_transfers, src.mesh_buffer(), blocking);
}

void memcpy(void* dst, const Tensor& src, const std::optional<BufferRegion>& region, bool blocking) {
    if (auto mesh_device = src.mesh_device()) {
        memcpy(mesh_device->mesh_command_queue(), dst, src, region, blocking);
    } else {
        memcpy(src.device()->command_queue(), dst, src, region, blocking);
    }
}

void memcpy(CommandQueue& queue, Tensor& dst, const void* src, const std::optional<BufferRegion>& region) {
    TT_FATAL(is_device_tensor(dst), "memcpy: memcpy to non-device tensor is not supported!");

    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE != nullptr) {
        TT_THROW("SLOW_DISPATCH is not supported for memcpy!");
    }

    if (!region.has_value()) {
        EnqueueWriteBuffer(queue, *dst.buffer(), src, false);
    } else {
        EnqueueWriteSubBuffer(queue, *dst.buffer(), src, region.value(), false);
    }
}

void memcpy(
    distributed::MeshCommandQueue& queue, Tensor& dst, const void* src, const std::optional<BufferRegion>& region) {
    TT_FATAL(is_device_tensor(dst), "memcpy: memcpy to non-device tensor is not supported!");
    TT_FATAL(queue.device()->num_devices() == 1, "memcpy only supports single device mesh");
    std::vector<distributed::MeshCommandQueue::ShardDataTransfer> shard_data_transfers = {{
        .shard_coord = *distributed::MeshCoordinateRange(queue.device()->shape()).begin(),
        .host_data = const_cast<void*>(src),
        .region = region,
    }};
    queue.enqueue_write_shards(dst.mesh_buffer(), shard_data_transfers, false);
}

void memcpy(Tensor& dst, const void* src, const std::optional<BufferRegion>& region) {
    if (auto mesh_device = dst.mesh_device()) {
        memcpy(dst.mesh_device()->mesh_command_queue(), dst, src, region);
    } else {
        memcpy(dst.device()->command_queue(), dst, src, region);
    }
}

void memcpy(CommandQueue& queue, Tensor& dst, const Tensor& src, const std::optional<BufferRegion>& region) {
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE != nullptr) {
        TT_THROW("SLOW_DISPATCH is not supported for memcpy!");
    }

    TT_ASSERT(dst.get_dtype() == src.get_dtype());
    TT_ASSERT(dst.get_layout() == src.get_layout());

    if (is_cpu_tensor(dst) && is_device_tensor(src)) {
        auto dst_buffer = host_buffer::get_host_buffer(dst);
        memcpy(queue, dst_buffer.view_bytes().data(), src, region);
    } else if (is_device_tensor(dst) && is_cpu_tensor(src)) {
        auto src_buffer = host_buffer::get_host_buffer(src);
        memcpy(queue, dst, src_buffer.view_bytes().data(), region);
    } else {
        TT_THROW("Unsupported memcpy");
    }
}

void memcpy(
    distributed::MeshCommandQueue& queue, Tensor& dst, const Tensor& src, const std::optional<BufferRegion>& region) {
    TT_ASSERT(dst.get_dtype() == src.get_dtype());
    TT_ASSERT(dst.get_layout() == src.get_layout());

    if (is_cpu_tensor(dst) && is_device_tensor(src)) {
        auto dst_buffer = host_buffer::get_host_buffer(dst);
        memcpy(queue, dst_buffer.view_bytes().data(), src, region);
    } else if (is_device_tensor(dst) && is_cpu_tensor(src)) {
        auto src_buffer = host_buffer::get_host_buffer(src);
        memcpy(queue, dst, src_buffer.view_bytes().data(), region);
    } else {
        TT_THROW("Unsupported memcpy");
    }
}

void memcpy(Tensor& dst, const Tensor& src, const std::optional<BufferRegion>& region) {
    if (is_cpu_tensor(dst) && is_device_tensor(src)) {
        if (auto mesh_device = src.mesh_device()) {
            memcpy(mesh_device->mesh_command_queue(), dst, src, region);
        } else {
            memcpy(src.device()->command_queue(), dst, src, region);
        }
    } else if (is_device_tensor(dst) && is_cpu_tensor(src)) {
        if (auto mesh_device = dst.mesh_device()) {
            memcpy(mesh_device->mesh_command_queue(), dst, src, region);
        } else {
            memcpy(dst.device()->command_queue(), dst, src, region);
        }
    } else {
        TT_THROW("Unsupported memcpy");
    }
}

Tensor allocate_tensor_on_devices(const TensorSpec& tensor_spec, const std::vector<IDevice*>& devices) {
    Tensor device_tensor = Tensor(devices, tensor_spec);

    const auto& workers_in_use = device_tensor.get_workers();
    uint32_t num_workers = workers_in_use.size();

    for (int worker_index = 0; worker_index < num_workers; ++worker_index) {
        auto& worker = devices[worker_index];
        auto shard = create_device_tensor(tensor_spec, worker);
        insert_buffer_and_shape_for_device(worker, shard, device_tensor, worker_index);
    }
    return device_tensor;
}

Tensor allocate_tensor_on_mesh(const TensorSpec& tensor_spec, distributed::MeshDevice* mesh_device) {
    // Allocate a mesh buffer synchronously.
    auto mesh_buffer = tensor_impl::allocate_mesh_buffer_on_device(mesh_device, tensor_spec);
    std::vector<std::pair<distributed::MeshCoordinate, TensorSpec>> specs;
    specs.reserve(mesh_device->shape().mesh_size());
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        specs.push_back(std::make_pair(coord, tensor_spec));
    }
    DeviceStorage device_storage(std::move(mesh_buffer), ReplicateTensor(), std::move(specs));
    return Tensor(std::move(device_storage), tensor_spec);
}

void write_tensor(const Tensor& host_tensor, Tensor device_tensor, QueueId cq_id) {
    // Top level wrapper to copy a host tensor to a preallocated device tensor
    TT_ASSERT(device_tensor.workers.size(), "Workers must be specified for device_tensor in write_tensor");

    TT_FATAL(
        host_tensor.storage_type() == StorageType::HOST or host_tensor.storage_type() == StorageType::MULTI_DEVICE_HOST,
        "write_tensor only supports host_tensor to device_tensor data transfer");

    auto& device_storage = std::get<DeviceStorage>(device_tensor.get_storage());
    if (auto mesh_buffer = device_storage.mesh_buffer; mesh_buffer != nullptr) {
        tensor_impl::copy_to_mesh_tensor_wrapper(host_tensor, device_tensor, cq_id);
        return;
    }

    for (int worker_index = 0; worker_index < device_tensor.workers.size(); ++worker_index) {
        TT_FATAL(
            device_tensor.storage_type() == StorageType::DEVICE,
            "write_tensor only supports host_tensor to device_tensor data transfer");
        TT_FATAL(host_tensor.get_logical_shape() == device_tensor.get_logical_shape(), "Error");
        TT_FATAL(host_tensor.get_dtype() == device_tensor.get_dtype(), "Error");
        TT_FATAL(
            host_tensor.get_tensor_spec().page_config() == device_tensor.get_tensor_spec().page_config(), "Error");
        std::visit(
            tt::stl::overloaded{
                [cq_id, &host_tensor, &device_tensor](const DeviceStorage& device_storage) {
                    // Copying from host to a single device.
                    const void* host_data = std::visit(
                        tt::stl::overloaded{
                            [](const HostStorage& host_storage) -> const void* {
                                return host_storage.buffer.view_bytes().data();
                            },
                            [](const MultiDeviceHostStorage& host_storage) -> const void* {
                                TT_ASSERT(
                                    host_storage.num_buffers() == 1,
                                    "Cannot copy multi-buffer host storage to a single device");
                                auto buffer = host_storage.get_buffer(0);
                                return buffer.view_bytes().data();
                            },
                            [](auto&&) -> const void* { TT_THROW("Unreachable"); },
                        },
                        host_tensor.get_storage());
                    if (auto mesh_device = device_tensor.mesh_device()) {
                        tt::tt_metal::memcpy(mesh_device->mesh_command_queue(*cq_id), device_tensor, host_data);
                    } else {
                        tt::tt_metal::memcpy(
                            device_tensor.device()->command_queue(*cq_id), device_tensor, host_data);
                    }
                },
                [](auto&& s) { TT_THROW("Unreachable"); }},
            device_tensor.get_storage());
    }
}

std::vector<IDevice*> Tensor::active_physical_devices() const {
    auto mesh_device = this->mesh_device();
    std::vector<IDevice*> devices = {};
    devices.reserve(this->device_storage().specs.size());
    for (const auto& spec : this->device_storage().specs) {
        devices.push_back(mesh_device->get_device(spec.first));
    }
    return devices;
}

Tensor set_tensor_id(const Tensor& tensor) {
    if (not GraphTracker::instance().is_enabled()) {
        return tensor;
    }
    auto output = tensor;
    output.tensor_id = ttnn::CoreIDs::instance().fetch_and_increment_tensor_id();
    return output;
};

}  // namespace tt::tt_metal
