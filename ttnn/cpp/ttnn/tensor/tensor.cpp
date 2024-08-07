// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <memory>

#include "common/bfloat16.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_impl_wrapper.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

Tensor::Tensor(const Storage storage, const ttnn::Shape shape, DataType dtype, Layout layout) :
    tensor_id{std::nullopt},
    storage{storage},
    shape{shape},
    dtype{dtype},
    layout{layout} {
    ZoneScoped;
    std::visit(
        [&](auto&& storage) {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                // do nothing
            } else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_ASSERT(storage.buffer->device() != nullptr);
                tensor_impl::validate_on_device_dtype_and_layout(storage.buffer->device(), shape.value, dtype, layout);
            } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                // do nothing
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                for (int i = 0; i < storage.ordered_device_ids.size(); i++) {
                    auto device_id = storage.ordered_device_ids[i];
                    auto buffer = storage.get_buffer_for_device_id(device_id);
                    TT_ASSERT(buffer->device() != nullptr);
                    TT_ASSERT(buffer->device()->id() == device_id);
                    tensor_impl::validate_on_device_dtype_and_layout(buffer->device(), shape.value, dtype, layout);
                }
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                // do nothing
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        storage);
}

Tensor::Tensor(const Storage storage, const Shape shape, DataType dtype, Layout layout) :
    Tensor(storage, ttnn::Shape{shape}, dtype, layout) {}

Tensor::~Tensor() {
    ZoneScoped;
    this->deallocate();
}

void Tensor::deallocate(bool force) {
    ZoneScopedN("TensorDeallocate");
    // Check if the attributes didn't get moved to another tensor.
    // If not, we can deallocate this tensor.
    std::visit(
        [force, this](auto& storage) {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, OwnedStorage>) {
                std::visit([](auto&& buffer) { buffer.reset(); }, storage.buffer);
            } else if constexpr (std::is_same_v<T, DeviceStorage>) {
                auto& buffer = storage.buffer;
                if (buffer != nullptr) {
                    auto device = buffer->device();
                    if ((force or buffer.use_count() == 1) and buffer->get_is_allocated()) {
                        device->push_work([=] {
                            tt::log_debug(tt::LogAsync, "Device {}: Deallocating buffer at address 0x{:x}", device->id(), buffer->address());
                            buffer->deallocate();
                            tt::log_debug(tt::LogAsync, "Device {}: Deallocated buffer", device->id());
                        });
                    }
                    buffer.reset();
                }
            } else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                if (force) {
                    TT_THROW("Cannot deallocate tensor with borrowed storage!");
                }
            } else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                for (int i = 0; i < storage.ordered_device_ids.size(); i++) {
                    auto device_id = storage.ordered_device_ids[i];
                    auto buffer = storage.get_buffer_for_device_id(device_id);
                    if(buffer != nullptr) {
                        auto device = buffer->device();
                        if ((force or buffer.use_count() == 1) and buffer->get_is_allocated()) {
                            device->push_work([=] {
                                tt::log_debug(tt::LogAsync, "Device {}: Deallocating buffer at address 0x{:x}", device->id(), buffer->address());
                                buffer->deallocate();
                                tt::log_debug(tt::LogAsync, "Device {}: Deallocated buffer", device->id());
                            });
                        }
                        buffer.reset();
                    }
                }
            } else if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                // Same logic as above for host tensors
                for (int i = 0; i < storage.num_buffers(); i++) {
                    auto& current_buffer = storage.get_buffer(i);
                    std::visit([](auto&& buffer) { buffer.reset(); }, current_buffer);
                }
            } else {
                raise_unsupported_storage<T>();
            }
        },
        this->storage);
}


std::vector<Device*> Tensor::get_workers(bool blocking) const {
    ZoneScoped;
    // Initialize an empty worker vector (remains empty for host side storage)
    std::vector<Device*> workers = {};

    std::visit(
        [this, blocking, &workers](auto&& storage) {
            using StorageType = std::decay_t<decltype(storage)>;
            // Assign workers only to device tensors
            if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                // Either explictly syncing or workers are pre-populated (this will happen for device tensors if using
                // the correct APIs).
                workers = std::vector<Device*>{this->device()};
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                // Either explictly syncing or workers are pre-populated (this will happen for device tensors if using
                // the correct APIs).
                workers.reserve(storage.num_buffers());
                for (int i = 0; i < storage.ordered_device_ids.size(); ++i) {
                    auto device_id = storage.ordered_device_ids[i];
                    workers.push_back(storage.get_buffer_for_device_id(device_id)->device());
                }
            }
        },
        this->storage);
    return workers;
}

// Getters - Spin until tensor is populated before querying tensor metadata
const Shape& Tensor::get_legacy_shape() const {
    return this->shape.value;
}

const ttnn::Shape& Tensor::get_shape() const {
    return this->shape;
}
const DataType& Tensor::get_dtype() const {
    return this->dtype;
}
const Layout& Tensor::get_layout() const {
    return this->layout;
}

const Storage& Tensor::get_storage() const {
    return this->storage;
}

Tensor Tensor::to(CommandQueue& queue, const MemoryConfig& mem_config) const {
    ZoneScoped;
    return tensor_impl::to_device_wrapper(*this, queue.device(), mem_config, std::nullopt);
}

Tensor Tensor::to(Device* target_device, const MemoryConfig& mem_config) const {
    ZoneScoped;
    if (is_tensor_on_device_or_multidevice(*this)) {
        if (this->device() == target_device) {
            return *this;
        } else {
            TT_THROW("Currently do not support moving between devices");
        }
    }
    return tensor_impl::to_device_wrapper(*this, target_device, mem_config, std::nullopt);
}

Tensor Tensor::to(DeviceMesh* device_mesh, const MemoryConfig& mem_config) const {
    ZoneScoped;
    std::vector<Device*> workers_to_use = distribute_tensor_to_mesh(*this, *device_mesh);
    return this->to(workers_to_use, mem_config);
}

Tensor Tensor::to(const std::vector<Device*>& workers, const MemoryConfig& mem_config) const {
    ZoneScoped;

    std::vector<Tensor> shards;
    for (int worker_index = 0; worker_index < workers.size(); ++worker_index) {
        auto& worker = workers[worker_index];
        auto shard = get_shard_for_device(*this, worker, worker_index);
        if (shard.storage_type() == StorageType::OWNED) {
            shard = tensor_impl::to_device_wrapper(shard, worker, mem_config, std::nullopt);
        }
        shards.push_back(shard);
    }
    auto strategy = std::visit(
            [](auto&& storage) -> DistributedTensorConfig {
                using T = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                    return storage.strategy;
                } else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                    return storage.strategy;
                } else {
                    TT_THROW("Unsupported storage type");
                }
            },
            this->get_storage());
    return create_multi_device_tensor(shards, StorageType::MULTI_DEVICE, strategy);
}

Tensor Tensor::cpu(bool blocking) const {
    ZoneScoped;
    if (not is_tensor_on_device_or_multidevice(*this)) {
        return *this;
    }
    return tensor_impl::to_host_wrapper(*this, blocking);
}

Tensor Tensor::cpu_sharded() const {
    ZoneScoped;
    return tensor_impl::to_host_sharded_wrapper(*this);
}

Tensor Tensor::extract_shard(const CoreCoord& core) const {
    ZoneScoped;
    auto buffer_page_mapping = generate_buffer_page_mapping(*this->buffer());
    auto core_id = buffer_page_mapping.core_to_core_id_.at(core);
    return this->extract_shard(core_id);
}

Tensor Tensor::extract_shard(const uint32_t& core_id) const {
    return tensor_impl::extract_shard_wrapper(*this, core_id);
}

Tensor Tensor::to(Layout target_layout, Device* worker) const {
    ZoneScoped;
    // Running without worker threads (non-async)
    TT_ASSERT(
        this->storage_type() != StorageType::DEVICE or
        this->storage_type() != StorageType::MULTI_DEVICE && "Bring tensor to host before converting to target layout");
    return tensor_impl::to_layout_wrapper(*this, target_layout);
}

/*
Tensor create_multi_device_tensor(
    const std::vector<Device *>& workers,
    uint32_t num_buffers = 0,
    std::optional<DistributedTensorConfig> distributed_tensor_config = std::nullopt) {
    // When creating a device tensor, specify workers.
    // When creating a host tensor, specify num_buffers.
    // If neither are specified, a dummy tensor is being created. Do nothing.
    Tensor output;
    if (workers.size()) {
        if (workers.size() == 1) {
            tensor.storage = DeviceStorage();
        } else if (workers.size() > 1) {
            output.storage = MultiDeviceStorage();
            std::transform(
                workers.cbegin(),
                workers.cend(),
                std::back_inserter(
                    std::get<MultiDeviceStorage>(output.storage).ordered_device_ids),
                [](const Device *worker) { return worker->id(); });
        }
    } else if (num_buffers) {
        if (num_buffers == 1) {
            output.storage = OwnedStorage();
        } else {
            output.storage = MultiDeviceHostStorage();
            // Preallocate buffer and shape vector for MultiDeviceHostStorage
            if (distributed_tensor_config.has_value()) {
                std::get<MultiDeviceHostStorage>(output.storage).strategy =
                    distributed_tensor_config.value();
            }
            std::get<MultiDeviceHostStorage>(output.storage).buffers =
                std::vector<OwnedBuffer>(num_buffers, OwnedBuffer());
            std::get<MultiDeviceHostStorage>(output.storage).shapes =
                std::vector<Shape>(num_buffers, output.shape.value);
        };
    }
    return output;
}
*/

Tensor Tensor::to(Layout target_layout, DeviceMesh* device_mesh) const {
    ZoneScoped;
    if (device_mesh) {
        auto workers = distribute_tensor_to_mesh(*this, *device_mesh);

        if (std::holds_alternative<MultiDeviceHostStorage>(this->get_storage())) {
            auto& host_storage = std::get<MultiDeviceHostStorage>(this->get_storage());
            auto distributed_config = host_storage.strategy;

            std::vector<Tensor> shards;
            for (int worker_index = 0; worker_index < workers.size(); ++worker_index) {
                auto& worker = workers[worker_index];
                auto shard = get_shard_for_device(*this, worker, worker_index);
                shard = tensor_impl::to_layout_wrapper(shard, target_layout);
                shards.push_back(shard);
            }
            return create_multi_device_tensor(shards, StorageType::MULTI_DEVICE_HOST, host_storage.strategy);
        } else {
            TT_THROW("to(layout) must be called on host tensors with MULTI_DEVICE_HOST_STORAGE when multiple workers are "
                     "specified");
        }
    }
    // Running without worker threads (non-async)
    TT_ASSERT(
        this->storage_type() != StorageType::DEVICE or
        this->storage_type() != StorageType::MULTI_DEVICE && "Bring tensor to host before converting to target layout");
    return tensor_impl::to_layout_wrapper(*this, target_layout);
}

const std::string Tensor::write_to_string() const { return tensor_impl::to_string_wrapper(*this); }

void Tensor::print() const { std::cout << write_to_string() << std::endl; }

Tensor Tensor::pad(const Shape& output_tensor_shape, const Shape& input_tensor_start, float pad_value) const {
    ZoneScoped;
    TT_ASSERT(
        this->storage_type() == StorageType::OWNED or this->storage_type() == StorageType::MULTI_DEVICE_HOST or
        this->storage_type() == StorageType::BORROWED && "Tensor must be on host for padding");
    TT_ASSERT(this->get_layout() == Layout::ROW_MAJOR && "Tensor layout must be ROW_MAJOR for padding");

    auto input_shape = this->get_legacy_shape();
    auto dimensions_pads = std::vector<Padding::PadDimension>();
    for (auto index = 0; index < input_shape.rank(); index++) {
        auto front = input_tensor_start[index];
        auto back = output_tensor_shape[index] - (input_tensor_start[index] + input_shape[index]);
        dimensions_pads.push_back(Padding::PadDimension{.front = front, .back = back});
    }
    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
    auto output_shape_with_padding = Shape(output_tensor_shape, padding);

    return tensor_impl::pad_wrapper(*this, output_shape_with_padding, input_tensor_start, pad_value);
}

Tensor Tensor::unpad(const Shape& output_tensor_start, const Shape& output_tensor_end) const {
    ZoneScoped;
    TT_ASSERT(this->get_layout() == Layout::ROW_MAJOR && "Tensor layout must be ROW_MAJOR for unpadding");
    return tensor_impl::unpad_wrapper(*this, output_tensor_start, output_tensor_end);
}

Tensor Tensor::pad_to_tile(float pad_value) const {
    ZoneScoped;
    uint32_t height = this->get_legacy_shape()[-2];
    uint32_t width = this->get_legacy_shape()[-1];
    uint32_t padded_height = round_up(height, TILE_HEIGHT);
    uint32_t padded_width = round_up(width, TILE_WIDTH);

    std::vector<uint32_t> shape;
    std::vector<uint32_t> padded_shape;
    std::vector<uint32_t> input_tensor_start;

    for (auto index = 0; index < this->get_legacy_shape().rank() - 2; index++) {
        shape.push_back(this->get_legacy_shape().without_padding()[index]);
        padded_shape.push_back(this->get_legacy_shape()[index]);
        input_tensor_start.push_back(0);
    }

    shape.push_back(height);
    shape.push_back(width);
    padded_shape.push_back(padded_height);
    padded_shape.push_back(padded_width);
    input_tensor_start.push_back(0);
    input_tensor_start.push_back(0);

    return this->pad(Shape(shape, padded_shape), Shape{input_tensor_start}, pad_value);
}

Tensor Tensor::unpad_from_tile(const Shape& output_tensor_shape) const {
    ZoneScoped;

    for (auto index = 0; index < this->get_legacy_shape().rank() - 2; index++) {
        TT_ASSERT(
            this->get_legacy_shape().without_padding()[index] == output_tensor_shape[index],
            "Input shape must match output shape apart from last 2 dims");
    }
    TT_ASSERT(
        this->get_legacy_shape()[-2] % TILE_HEIGHT == 0 && this->get_legacy_shape()[-1] % TILE_WIDTH == 0,
        "Last 2 dims of input shape must be multiples of 32");
    TT_ASSERT(
        this->get_legacy_shape()[-2] - TILE_HEIGHT < output_tensor_shape[-2] &&
            this->get_legacy_shape()[-1] - TILE_WIDTH < output_tensor_shape[-1],
        "Last 2 dims of output must be within range to have been padded to input");
    std::vector<uint32_t> output_tensor_start{};
    std::vector<uint32_t> output_tensor_end{};
    for (auto index = 0; index < this->get_legacy_shape().rank(); index++) {
        output_tensor_start.push_back(0);
        output_tensor_end.push_back(output_tensor_shape[index] - 1);
    }
    return this->unpad(output_tensor_start, output_tensor_end);
}

const bool Tensor::is_sharded() const {
    return is_tensor_on_device_or_multidevice(*this) ? this->memory_config().is_sharded() : false;
}

uint32_t Tensor::element_size() const { return tensor_impl::element_size_bytes(this->get_dtype()); }

Tensor Tensor::reshape(int N, int C, int H, int W) const {
    ZoneScoped;
    auto new_shape = infer_dims_for_reshape(N, C, H, W, this->volume());
    return this->reshape(new_shape);
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    ZoneScoped;
    TT_ASSERT(
        this->volume() == tt::tt_metal::compute_volume(new_shape),
        "{} != {}",
        this->volume(),
        tt::tt_metal::compute_volume(new_shape));
    if (this->get_layout() == Layout::TILE) {
        TT_ASSERT(
            new_shape[-2] % TILE_HEIGHT == 0 && new_shape[-1] % TILE_WIDTH == 0 &&
            "Expected a multiple of 32 for H, W (or -1 evaluating to such) in Tensor::reshape()!");
    }
    return std::visit(
        [this, &new_shape](auto&& storage) -> Tensor {
            using T = std::decay_t<decltype(storage)>;
            const auto& tensor = *this;
            if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                auto updated_storage = std::get<T>(tensor.get_storage());
                for (int i = 0; i < updated_storage.shapes.size(); i++) {
                    updated_storage.shapes[i] = new_shape;
                }
                return Tensor(updated_storage, new_shape, tensor.get_dtype(), tensor.get_layout());
            }
            if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                MultiDeviceStorage updated_storage = std::get<T>(tensor.get_storage());
                std::unordered_map<int, Shape> new_shapes;

                for (auto device_id : updated_storage.ordered_device_ids) {
                    new_shapes.insert({device_id, new_shape});
                }
                updated_storage.shapes = new_shapes;
                return Tensor(updated_storage, new_shape, tensor.get_dtype(), tensor.get_layout());
            } else {
                return Tensor(tensor.get_storage(), new_shape, tensor.get_dtype(), tensor.get_layout());
            }
        },
        this->get_storage());
}

bool Tensor::is_allocated() const {
    ZoneScoped;
    return std::visit(
        [](auto&& storage) -> bool {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, OwnedStorage>) {
                return std::visit([](auto&& buffer) -> bool { return buffer.is_allocated(); }, storage.buffer);
            } else if constexpr (std::is_same_v<T, DeviceStorage>) {
                return bool(storage.buffer) and storage.buffer->get_is_allocated();
            } else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                return true;
            } else if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                bool is_allocated = true;
                for (int i = 0; i < storage.num_buffers(); i++) {
                    is_allocated &=
                        std::visit([](auto&& buffer) -> bool { return buffer.is_allocated(); }, storage.get_buffer(i));
                }
                return is_allocated;
            } else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                bool is_allocated = true;
                for (int i = 0; i < storage.ordered_device_ids.size(); ++i) {
                    auto device_id = storage.ordered_device_ids[i];
                    const auto& buffer = storage.get_buffer_for_device_id(device_id);
                    is_allocated &= bool(buffer) and buffer->size() > 0;
                }
                return is_allocated;
            } else {
                raise_unsupported_storage<T>();
            }
        },
        this->get_storage());
}

std::vector<uint32_t> Tensor::host_page_ordering() {
    auto buffer_page_mapping = generate_buffer_page_mapping(*this->buffer());
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
        [](auto&& storage) -> StorageType {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, OwnedStorage>) {
                return StorageType::OWNED;
            } else if constexpr (std::is_same_v<T, DeviceStorage>) {
                return StorageType::DEVICE;
            } else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                return StorageType::BORROWED;
            } else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                return StorageType::MULTI_DEVICE;
            } else if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                return StorageType::MULTI_DEVICE_HOST;
            } else {
                raise_unsupported_storage<T>();
            }
        },
        this->get_storage());
}

namespace detail {
const Shape compute_strides(const Shape& shape) {
    auto num_elements = compute_volume(shape);
    std::vector<std::uint32_t> strides;
    for (std::int32_t index = 0; index < shape.rank(); index++) {
        num_elements /= shape[index];
        strides.push_back(num_elements);
    }
    return strides;
}
}  // namespace detail

const Shape Tensor::strides() const { return detail::compute_strides(this->get_legacy_shape()); }

uint32_t Tensor::volume() const { return tt::tt_metal::compute_volume(this->get_legacy_shape()); }

uint32_t Tensor::intended_volume() const { return tt::tt_metal::compute_volume(this->get_shape()); }

Tensor create_device_tensor(
    const Shape& shape, DataType data_type, Layout layout, Device* device, const MemoryConfig& memory_config) {
    ZoneScoped;
    if (memory_config.is_sharded()) {
        TT_ASSERT(memory_config.shard_spec.has_value());

        auto& shard_spec = memory_config.shard_spec.value();
        auto& shard_shape = shard_spec.shape;

        auto width = shape[-1];
        auto other_dims = 1;
        for (int i = 0; i < shape.rank() - 1; i++) {
            other_dims *= shape[i];
        }

        auto element_size = tensor_impl::element_size_bytes(data_type);
        auto page_shape = tensor_impl::get_sharded_page_shape(layout, data_type, shard_spec.shape);
        std::array<uint32_t, 2> tensor2d_size = {other_dims / page_shape[0], width / page_shape[1]};
        ShardSpecBuffer shard_spec_buffer(shard_spec, page_shape, tensor2d_size);
        uint32_t packed_size_in_bytes =
            tensor_impl::packed_buffer_size_bytes_wrapper(data_type, compute_buffer_size(shape, data_type));
        auto device_buffer = tensor_impl::allocate_buffer_on_device(
            packed_size_in_bytes, device, shape, data_type, layout, memory_config, shard_spec_buffer);
        return Tensor(DeviceStorage{device_buffer}, shape, data_type, layout);
    } else {
        uint32_t packed_size_in_bytes =
            tensor_impl::packed_buffer_size_bytes_wrapper(data_type, compute_buffer_size(shape, data_type));
        auto device_buffer = tensor_impl::allocate_buffer_on_device(
            packed_size_in_bytes, device, shape, data_type, layout, memory_config, std::nullopt);
        return Tensor(DeviceStorage{device_buffer}, shape, data_type, layout);
    }
}

namespace detail {
template <typename DataType>
void* get_raw_host_data_ptr(const Tensor& tensor) {
    return std::visit(
        [](auto&& storage) -> void* {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                auto buffer = owned_buffer::get_as<DataType>(storage.buffer);
                return buffer.data();
            } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                if constexpr (
                    std::is_same_v<DataType, float> or std::is_same_v<DataType, bfloat16> or
                    std::is_same_v<DataType, std::uint32_t> or std::is_same_v<DataType, std::int32_t> or
                    std::is_same_v<DataType, std::uint8_t> or std::is_same_v<DataType, std::uint16_t>) {
                    auto buffer = borrowed_buffer::get_as<DataType>(storage.buffer);
                    return buffer.data();
                } else {
                    TT_THROW("Borrowed storage doesn't support this data type");
                }
            } else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                TT_THROW("Device storage isn't supported");
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.get_storage());
}
}  // namespace detail

void* get_raw_host_data_ptr(const Tensor& tensor) {
    switch (tensor.get_dtype()) {
        case DataType::BFLOAT16:
            return detail::get_raw_host_data_ptr<bfloat16>(tensor);
        case DataType::FLOAT32:
            return detail::get_raw_host_data_ptr<float>(tensor);
        case DataType::INT32:
            return detail::get_raw_host_data_ptr<int32_t>(tensor);
        case DataType::UINT32:
            return detail::get_raw_host_data_ptr<uint32_t>(tensor);
        case DataType::BFLOAT8_B:
            return detail::get_raw_host_data_ptr<uint32_t>(tensor);
        case DataType::BFLOAT4_B:
            return detail::get_raw_host_data_ptr<uint32_t>(tensor);
        case DataType::UINT16:
            return detail::get_raw_host_data_ptr<uint16_t>(tensor);
        case DataType::UINT8:
            return detail::get_raw_host_data_ptr<uint8_t>(tensor);
        default:
            TT_THROW("Unsupported data type");
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

Tensor allocate_tensor_on_device(
    const ttnn::Shape& shape, DataType data_type, Layout layout, Device* device, const MemoryConfig& memory_config) {
    return create_device_tensor(shape.value, data_type, layout, device, memory_config);
}

Tensor allocate_tensor_on_device(
    const ttnn::Shape& shape,
    DataType data_type,
    Layout layout,
    DeviceMesh* device_mesh,
    const MemoryConfig& memory_config) {
    TT_THROW("allocate_tensor_on_device Not implemented");
    return Tensor{{}, shape, data_type, layout};
    // return create_device_tensor(shape.value, data_type, layout, worker, memory_config, true);
}

void write_tensor(const Tensor& host_tensor, Tensor& device_tensor, uint8_t cq_id) {
    std::visit(
        [&](auto&& device_tensor_storage) {
            void* host_data = nullptr;
            using StorageType = std::decay_t<decltype(device_tensor_storage)>;
            if constexpr (std::is_same_v<DeviceStorage, StorageType>) {
                if (std::holds_alternative<BorrowedStorage>(host_tensor.get_storage())) {
                    // Handle case when writing borrowed tensor single device tensor (only allowed for sync
                    // mode)
                    auto host_storage = std::get<BorrowedStorage>(host_tensor.get_storage());
                    std::visit([&host_data](auto&& b) { host_data = b.data(); }, host_storage.buffer);
                } else {
                    TT_ASSERT(std::holds_alternative<OwnedStorage>(host_tensor.get_storage()), fmt::format("Unexpected type {} in {}:{} ",tt::stl::get_active_type_name_in_variant(host_tensor.get_storage()),__FILE__, __LINE__));
                    auto host_storage = std::get<OwnedStorage>(host_tensor.get_storage());
                    std::visit([&host_data](auto&& b) { host_data = b.begin(); }, host_storage.get_buffer());
                }
                auto device = device_tensor.device();
                EnqueueWriteBuffer(device->command_queue(cq_id), device_tensor_storage.get_buffer(), host_data, false);
            } else if constexpr (std::is_same_v<MultiDeviceStorage, StorageType>) {
                auto host_storage = std::get<MultiDeviceHostStorage>(host_tensor.get_storage());
                auto workers = device_tensor.get_workers();
                for (int worker_index = 0; worker_index < workers.size(); ++worker_index) {
                    auto worker = workers[worker_index];
                    std::visit(
                        [worker_index, &host_data](auto&& b) { host_data = b.begin(); },
                        host_storage.get_buffer(worker_index));
                    EnqueueWriteBuffer(
                        worker->command_queue(cq_id), device_tensor_storage.get_buffer_for_device(worker), host_data, false);
                }
            }
        },
        device_tensor.get_storage());
}


std::vector<Device*> distribute_tensor_to_mesh(const Tensor& tensor, DeviceMesh& device_mesh) {
    auto get_multi_device_workers = [&](const std::vector<Device*>& workers) {
        if (std::holds_alternative<MultiDeviceStorage>(tensor.get_storage()) or
            std::holds_alternative<MultiDeviceHostStorage>(tensor.get_storage())) {
            return std::vector<Device*>(workers.begin(), workers.begin() + num_buffers_in_tensor(tensor));
        }
        return workers;
    };

    if (device_mesh.get_view() != nullptr and std::holds_alternative<MultiDeviceHostStorage>(tensor.get_storage())) {
        const auto& host_storage = std::get<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage());

        return std::visit([&](const auto& strategy) {
            using StrategyType = std::decay_t<decltype(strategy)>;
            if constexpr (std::is_same_v<StrategyType, ShardTensor2D>) {
                auto mesh_view = device_mesh.get_view();
                return mesh_view->get_devices(strategy.shard_mesh);
            } else {
                return get_multi_device_workers(device_mesh.get_devices());
            }
        }, host_storage.strategy);
    } else if (std::holds_alternative<MultiDeviceStorage>(tensor.get_storage())) {
        return tensor.get_workers();
    } else {
        return get_multi_device_workers(device_mesh.get_devices());
    }
}


}  // namespace tt_metal

}  // namespace tt
