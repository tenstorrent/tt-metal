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
#include "tt_stl/span.hpp"
#include "ttnn/tensor/storage.hpp"

#include "tt-metalium/mesh_device_view.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_impl_wrapper.hpp"
#include <tt-metalium/host_buffer.hpp>
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
HostBuffer create_host_buffer_from_row_major_data(tt::stl::Span<const T> data, const TensorSpec& spec, T pad_value) {
    return tensor_impl::logical_matches_physical(spec)
               ? HostBuffer(std::vector<T>(data.begin(), data.end()))
               : HostBuffer(tensor_impl::encode_tensor_data(data, spec, pad_value));
}

template <typename T>
HostBuffer create_host_buffer_from_row_major_data(std::vector<T>&& data, const TensorSpec& spec, T pad_value) {
    return tensor_impl::logical_matches_physical(spec)
               ? HostBuffer(std::move(data))
               : HostBuffer(tensor_impl::encode_tensor_data(tt::stl::make_const_span(data), spec, pad_value));
}

template <typename T>
Tensor create_tensor_from_row_major_data(
    auto&& data, const TensorSpec& spec, distributed::MeshDevice* device, ttnn::QueueId cq_id, T pad_value) {
    Tensor tensor(create_host_buffer_from_row_major_data(std::forward<decltype(data)>(data), spec, pad_value), spec);

    return (device != nullptr) ? tensor.to_device(device, spec.memory_config(), cq_id) : tensor;
}

}  // namespace

Tensor::Tensor(
    HostBuffer buffer, const ttnn::Shape& shape, DataType dtype, Layout layout, const std::optional<Tile>& tile) :
    Tensor(std::move(buffer), /* logical_shape */ shape, /* padded_shape */ shape, dtype, layout, tile) {}

Tensor::Tensor(
    HostBuffer buffer,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    DataType dtype,
    Layout layout,
    const std::optional<Tile>& tile) {
    using namespace tt::constants;

    if (tile.has_value() and  //
        (tile->get_tile_shape()[0] != TILE_WIDTH or tile->get_tile_shape()[1] != TILE_HEIGHT)) {
        log_warning(
            tt::LogTTNN,
            "only matmul op and ccl all-gather currently supports the customized tile shape: {}",
            tile->get_tile_shape());
    }

    init(
        Storage(HostStorage(std::move(buffer))),
        TensorSpec(
            logical_shape,
            TensorLayout::fromPaddedShape(
                dtype, PageConfig(layout, tile), MemoryConfig{}, logical_shape, padded_shape)),
        ReplicateTensor{});
}

Tensor::Tensor(HostBuffer storage, TensorSpec tensor_spec) :
    Tensor(Storage(HostStorage(std::move(storage))), std::move(tensor_spec), ReplicateTensor{}) {}

Tensor::Tensor(Storage storage, TensorSpec tensor_spec, DistributedTensorConfig distributed_tensor_config) {
    init(Storage(std::move(storage)), std::move(tensor_spec), std::move(distributed_tensor_config));
}

void Tensor::init(Storage storage, TensorSpec tensor_spec, DistributedTensorConfig distributed_tensor_config) {
    tensor_attributes = std::make_shared<TensorAttributes>(
        std::move(storage), std::move(tensor_spec), std::move(distributed_tensor_config));

    if (auto* device_storage = std::get_if<DeviceStorage>(&tensor_attributes->get_storage());
        device_storage != nullptr && device_storage->mesh_buffer != nullptr) {
        mesh_device_ = device_storage->mesh_buffer->device();
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    this->tensor_id = other.tensor_id;
    if (this->tensor_attributes != other.tensor_attributes) {
        this->tensor_attributes = other.tensor_attributes;
    }
    this->mesh_device_ = other.mesh_device_;
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    this->tensor_id = std::move(other.tensor_id);
    if (this->tensor_attributes != other.tensor_attributes) {
        this->tensor_attributes = std::move(other.tensor_attributes);
    }
    this->mesh_device_ = std::move(other.mesh_device_);
    return *this;
}

Tensor::Tensor(const Tensor& other) : tensor_id(other.tensor_id), tensor_attributes(other.tensor_attributes) {
    this->mesh_device_ = other.mesh_device_;
}

Tensor::~Tensor() {
    ZoneScoped;
    this->deallocate_impl(/*force=*/false);
}

void Tensor::deallocate(bool force) { deallocate_impl(force); }

void Tensor::deallocate_impl(bool force) {
    auto can_deallocate = []<typename T>(const std::shared_ptr<T>& shared_resource, bool force) {
        // It is safe to deallocate a shared resource, if either it is not shared or `force` is set.
        return shared_resource.use_count() == 1 ||  //
               (shared_resource.use_count() > 1 && force);
    };

    ZoneScopedN("TensorDeallocate");
    // GraphTracker::instance().track_function_start("Tensor::deallocate", *this, force);
    if (can_deallocate(tensor_attributes, force)) {
        std::visit(
            tt::stl::overloaded{
                [](HostStorage&) {},
                [this, force, &can_deallocate](DeviceStorage& storage) {
                    if (can_deallocate(storage.mesh_buffer, force)) {
                        storage.mesh_buffer->deallocate();
                    } else if (can_deallocate(storage.buffer, force)) {
                        DeallocateBuffer(*(storage.buffer));
                    }
                    storage.mesh_buffer.reset();
                    storage.buffer.reset();
                }},
            this->tensor_attributes->get_storage());
    }
    // GraphTracker::instance().track_function_end();
}

DataType Tensor::get_dtype() const { return dtype(); }
Layout Tensor::get_layout() const { return layout(); }

const TensorSpec& Tensor::get_tensor_spec() const { return tensor_spec(); }

const ttnn::Shape& Tensor::get_logical_shape() const { return logical_shape(); }

const ttnn::Shape& Tensor::get_padded_shape() const { return padded_shape(); }

uint64_t Tensor::get_logical_volume() const { return logical_shape().volume(); }
uint32_t Tensor::volume() const { return padded_shape().volume(); }

Storage& Tensor::get_storage() { return this->tensor_attributes->get_storage(); }

const DistributedTensorConfig& Tensor::get_distributed_tensor_config() const {
    return this->tensor_attributes->get_distributed_tensor_config();
}

const Storage& Tensor::get_storage() const { return this->tensor_attributes->get_storage(); }

template <>
Tensor Tensor::from_span<float>(
    tt::stl::Span<const float> buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    float pad_value) {
    ZoneScoped;
    size_t volume = spec.logical_shape().volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    switch (spec.data_type()) {
        case DataType::FLOAT32: return create_tensor_from_row_major_data(buffer, spec, device, cq_id, pad_value);
        case DataType::BFLOAT16: {
            return create_tensor_from_row_major_data(
                std::vector<bfloat16>(buffer.begin(), buffer.end()),
                spec,
                device,
                cq_id,
                static_cast<bfloat16>(pad_value));
        }
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: {
            TT_FATAL(
                spec.tensor_layout().get_layout() == Layout::TILE,
                "Tile layout is required for BFLOAT8_B and BFLOAT4_B");

            // TODO: Implement `encode_tensor_data` in terms of a Span, avoid tilizing the data, as pack_fp32_vec_as_*
            // support row-major input.
            const auto& tile = spec.tensor_layout().get_page_config().get_tile();
            auto physical_data = tensor_impl::encode_tensor_data(buffer, spec, pad_value);
            std::vector<uint32_t> packed_block_floats =
                spec.data_type() == DataType::BFLOAT8_B
                    ? pack_fp32_vec_as_bfp8_tiles(physical_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile)
                    : pack_fp32_vec_as_bfp4_tiles(physical_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);

            Tensor tensor(HostBuffer(std::move(packed_block_floats)), spec);
            if (device != nullptr) {
                tensor = tensor.to_device(device, spec.memory_config(), cq_id);
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
    tt::stl::Span<const T> buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    T pad_value) {
    ZoneScoped;
    size_t volume = spec.logical_shape().volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    TT_FATAL(
        spec.data_type() == convert_to_data_type<T>(),
        "Unsupported data type: got {}, expected: {}",
        spec.data_type(),
        convert_to_data_type<T>());
    return create_tensor_from_row_major_data(buffer, spec, device, cq_id, pad_value);
}

template <typename T>
Tensor Tensor::from_borrowed_data(
    tt::stl::Span<T> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<Tile>& tile) {
    size_t volume = shape.volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    return Tensor(HostBuffer(buffer, std::move(buffer_pin)), shape, convert_to_data_type<T>(), Layout::ROW_MAJOR, tile);
}

template <>
Tensor Tensor::from_vector<float>(
    std::vector<float>&& buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    float pad_value) {
    ZoneScoped;
    size_t volume = spec.logical_shape().volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    if (spec.data_type() == DataType::FLOAT32) {
        // User `buffer` directly, when no type conversion is needed.
        return create_tensor_from_row_major_data(std::move(buffer), spec, device, cq_id, pad_value);
    }
    return from_span(tt::stl::Span<const float>(buffer.data(), buffer.size()), spec, device, cq_id, pad_value);
}

template <typename T>
Tensor Tensor::from_vector(
    std::vector<T>&& buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    T pad_value) {
    ZoneScoped;
    size_t volume = spec.logical_shape().volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    TT_FATAL(
        spec.data_type() == convert_to_data_type<T>(),
        "Unsupported data type: got {}, expected: {}",
        spec.data_type(),
        convert_to_data_type<T>());
    return create_tensor_from_row_major_data(std::move(buffer), spec, device, cq_id, pad_value);
}

template <>
std::vector<float> Tensor::to_vector<float>(ttnn::QueueId cq_id) const {
    ZoneScoped;
    Tensor cpu_tensor = this->cpu(/*blocking=*/true, cq_id);
    switch (cpu_tensor.dtype()) {
        case DataType::BFLOAT16: {
            auto buffer = host_buffer::get_as<bfloat16>(cpu_tensor);
            std::vector<float> physical_data;
            physical_data.reserve(buffer.size());
            std::transform(buffer.begin(), buffer.end(), std::back_inserter(physical_data), [](bfloat16 val) {
                return val.to_float();
            });
            if (tensor_impl::logical_matches_physical(cpu_tensor.tensor_spec())) {
                return physical_data;
            }
            return tensor_impl::decode_tensor_data(tt::stl::make_const_span(physical_data), cpu_tensor.tensor_spec());
        }
        case DataType::FLOAT32: {
            auto buffer = host_buffer::get_as<const float>(cpu_tensor);
            return tensor_impl::decode_tensor_data(buffer, cpu_tensor.tensor_spec());
        }
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: {
            const auto& tile = cpu_tensor.tensor_spec().tile();
            auto buffer = host_buffer::get_as<const uint32_t>(cpu_tensor);
            std::vector<float> unpacked_data =
                cpu_tensor.tensor_spec().data_type() == DataType::BFLOAT8_B
                    ? unpack_bfp8_tiles_into_float_vec(buffer, /*row_major_output=*/false, /*is_exp_a=*/false, tile)
                    : unpack_bfp4_tiles_into_float_vec(buffer, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
            return tensor_impl::decode_tensor_data(tt::stl::make_const_span(unpacked_data), cpu_tensor.tensor_spec());
        }
        default: {
            TT_THROW("Cannot convert tensor to vector for data type: {}", cpu_tensor.dtype());
        }
    }
}

template <typename T>
std::vector<T> Tensor::to_vector(ttnn::QueueId cq_id) const {
    ZoneScoped;
    TT_FATAL(
        this->dtype() == convert_to_data_type<T>(),
        "Unsupported data type for to_vector: got {}, expected: {}",
        this->dtype(),
        convert_to_data_type<T>());
    auto cpu_tensor = this->cpu(/*blocking=*/true, cq_id);
    auto data = host_buffer::get_as<const T>(cpu_tensor);
    if (tensor_impl::logical_matches_physical(cpu_tensor.tensor_spec())) {
        return std::vector<T>(data.begin(), data.end());
    }
    return tensor_impl::decode_tensor_data(data, cpu_tensor.tensor_spec());
}

// Instantiate explicitly for the supported types.
template Tensor Tensor::from_span<bfloat16>(
    tt::stl::Span<const bfloat16> buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    bfloat16 pad_value);
template Tensor Tensor::from_span<int32_t>(
    tt::stl::Span<const int32_t> buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    int32_t pad_value);
template Tensor Tensor::from_span<uint8_t>(
    tt::stl::Span<const uint8_t> buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    uint8_t pad_value);
template Tensor Tensor::from_span<uint16_t>(
    tt::stl::Span<const uint16_t> buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    uint16_t pad_value);
template Tensor Tensor::from_span<uint32_t>(
    tt::stl::Span<const uint32_t> buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    uint32_t pad_value);
template Tensor Tensor::from_borrowed_data<float>(
    tt::stl::Span<float> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<Tile>& tile);
template Tensor Tensor::from_borrowed_data<bfloat16>(
    tt::stl::Span<bfloat16> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<Tile>& tile);
template Tensor Tensor::from_borrowed_data<int32_t>(
    tt::stl::Span<int32_t> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<Tile>& tile);
template Tensor Tensor::from_borrowed_data<uint8_t>(
    tt::stl::Span<uint8_t> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<Tile>& tile);
template Tensor Tensor::from_borrowed_data<uint16_t>(
    tt::stl::Span<uint16_t> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<Tile>& tile);
template Tensor Tensor::from_borrowed_data<uint32_t>(
    tt::stl::Span<uint32_t> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<Tile>& tile);
template Tensor Tensor::from_vector<bfloat16>(
    std::vector<bfloat16>&& buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    bfloat16 pad_value);
template Tensor Tensor::from_vector<int32_t>(
    std::vector<int32_t>&& buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    int32_t pad_value);
template Tensor Tensor::from_vector<uint8_t>(
    std::vector<uint8_t>&& buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    uint8_t pad_value);
template Tensor Tensor::from_vector<uint16_t>(
    std::vector<uint16_t>&& buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    uint16_t pad_value);
template Tensor Tensor::from_vector<uint32_t>(
    std::vector<uint32_t>&& buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    uint32_t pad_value);

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

Tensor Tensor::cpu(bool blocking, QueueId cq_id) const { return tensor_ops::tensor_cpu(*this, blocking, cq_id); }

Tensor Tensor::extract_shard(const CoreCoord& core) const {
    ZoneScoped;
    const auto& buffer_page_mapping = *this->buffer()->get_buffer_page_mapping();
    auto core_id = buffer_page_mapping.core_to_core_id.at(core);
    return this->extract_shard(core_id);
}

Tensor Tensor::extract_shard(const uint32_t& core_id) const {
    return tensor_impl::extract_shard_wrapper(*this, core_id);
}

Tensor Tensor::to_layout(Layout target_layout) const { return tensor_ops::tensor_to_layout(*this, target_layout); }

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

uint32_t Tensor::element_size() const { return tensor_impl::element_size_bytes(this->dtype()); }

Tensor Tensor::reshape(const ttnn::Shape& new_shape) const { return tensor_ops::tensor_reshape(*this, new_shape); }

Tensor Tensor::reshape(const ttnn::Shape& new_logical_shape, const ttnn::Shape& new_padded_shape) const {
    return tensor_ops::tensor_reshape(*this, new_logical_shape, new_padded_shape);
}

bool Tensor::is_allocated() const {
    ZoneScoped;
    auto output = std::visit(
        tt::stl::overloaded{
            [](const DeviceStorage& storage) { return storage.is_allocated(); },
            [](const HostStorage&) { return true; },
        },
        this->storage());
    return output;
}

StorageType Tensor::storage_type() const {
    return std::visit(
        tt::stl::overloaded{
            [](const HostStorage&) { return StorageType::HOST; },
            [](const DeviceStorage&) { return StorageType::DEVICE; },
        },
        this->storage());
}

ttnn::Shape Tensor::strides() const { return ttnn::Shape(tt::tt_metal::compute_strides(this->padded_shape())); }

uint64_t Tensor::logical_volume() const { return logical_shape().volume(); }
uint64_t Tensor::physical_volume() const { return padded_shape().volume(); }

bool Tensor::is_scalar() const {
    const ttnn::Shape logical_shape = this->logical_shape();
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
        output = allocate_tensor_on_device(tensor_spec, mesh_device);
    } else {
        auto device_buffer = tensor_impl::allocate_buffer_on_device(device, tensor_spec);
        output = Tensor(DeviceStorage{device_buffer}, tensor_spec, ReplicateTensor{});
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
    ZoneScoped;
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
    ZoneScoped;
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
    ZoneScoped;
    if (auto mesh_device = src.mesh_device()) {
        memcpy(mesh_device->mesh_command_queue(), dst, src, region, blocking);
    } else {
        memcpy(src.device()->command_queue(), dst, src, region, blocking);
    }
}

void memcpy(CommandQueue& queue, Tensor& dst, const void* src, const std::optional<BufferRegion>& region) {
    ZoneScoped;
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
    ZoneScoped;
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
    ZoneScoped;
    if (auto mesh_device = dst.mesh_device()) {
        memcpy(mesh_device->mesh_command_queue(), dst, src, region);
    } else {
        memcpy(dst.device()->command_queue(), dst, src, region);
    }
}

void memcpy(CommandQueue& queue, Tensor& dst, const Tensor& src, const std::optional<BufferRegion>& region) {
    ZoneScoped;
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE != nullptr) {
        TT_THROW("SLOW_DISPATCH is not supported for memcpy!");
    }

    TT_ASSERT(dst.dtype() == src.dtype());
    TT_ASSERT(dst.layout() == src.layout());

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
    ZoneScoped;
    TT_ASSERT(dst.dtype() == src.dtype());
    TT_ASSERT(dst.layout() == src.layout());

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
    ZoneScoped;
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

Tensor allocate_tensor_on_device(const TensorSpec& tensor_spec, distributed::MeshDevice* device) {
    auto mesh_buffer = tensor_impl::allocate_mesh_buffer_on_device(device, tensor_spec);
    std::vector<distributed::MeshCoordinate> coords;
    coords.reserve(device->shape().mesh_size());
    for (const auto& coord : distributed::MeshCoordinateRange(device->shape())) {
        coords.push_back(coord);
    }
    DeviceStorage device_storage(std::move(mesh_buffer), std::move(coords));
    return Tensor(std::move(device_storage), tensor_spec, ReplicateTensor{});
}

Tensor allocate_tensor_on_host(const TensorSpec& tensor_spec, distributed::MeshDevice* device) {
    auto distributed_host_buffer = DistributedHostBuffer::create(device->shape());
    for (const auto& coord : distributed::MeshCoordinateRange(device->shape())) {
        distributed_host_buffer.emplace_shard(coord, []() { return HostBuffer(); });
    }

    distributed_host_buffer = distributed_host_buffer.transform(
        [&](const HostBuffer&) { return tensor_impl::allocate_host_buffer(tensor_spec); },
        DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);
    return Tensor(HostStorage(std::move(distributed_host_buffer)), tensor_spec, ReplicateTensor{});
}

void write_tensor(const Tensor& src, Tensor& dst, bool blocking, QueueId cq_id) {
    ZoneScoped;
    TT_FATAL(
        (is_device_tensor(src) && is_cpu_tensor(dst)) ||    // device to host
            (is_cpu_tensor(src) && is_device_tensor(dst)),  // host to device
        "Unsupported data transfer direction; source storage type: {}, destination storage type: {}",
        src.storage_type(),
        dst.storage_type());

    if (is_device_tensor(src)) {
        tensor_impl::copy_to_host_tensor_wrapper(src, dst, blocking, cq_id);
        return;
    }

    if (auto mesh_buffer = dst.device_storage().mesh_buffer; mesh_buffer != nullptr) {
        TT_FATAL(!blocking, "Blocking is not supported for host to device copy");
        tensor_impl::copy_to_device_tensor_wrapper(src, dst, cq_id);
        return;
    }

    TT_FATAL(src.logical_shape() == dst.logical_shape(), "Error");
    TT_FATAL(src.dtype() == dst.dtype(), "Error");
    TT_FATAL(src.tensor_spec().page_config() == dst.tensor_spec().page_config(), "Error");
    std::visit(
        tt::stl::overloaded{
            [cq_id, &src, &dst](const DeviceStorage& device_storage) {
                // Copying from host to a single device.
                const void* host_data = std::visit(
                    tt::stl::overloaded{
                        [](const HostStorage& host_storage) -> const void* {
                            TT_FATAL(
                                host_storage.buffer().shape() == distributed::MeshShape(1, 1),
                                "Can't get a single buffer from host storage distributed over mesh shape {}",
                                host_storage.buffer().shape());
                            return host_storage.buffer()
                                .get_shard(distributed::MeshCoordinate(0, 0))
                                ->view_bytes()
                                .data();
                        },
                        [](auto&&) -> const void* { TT_THROW("Unreachable"); },
                    },
                    src.storage());
                if (auto mesh_device = dst.mesh_device()) {
                    tt::tt_metal::memcpy(mesh_device->mesh_command_queue(*cq_id), dst, host_data);
                } else {
                    tt::tt_metal::memcpy(dst.device()->command_queue(*cq_id), dst, host_data);
                }
            },
            [](auto&& s) { TT_THROW("Unreachable"); }},
        dst.storage());
}

Tensor set_tensor_id(const Tensor& tensor) {
    if (not GraphTracker::instance().is_enabled()) {
        return tensor;
    }
    auto output = tensor;
    output.tensor_id = ttnn::CoreIDs::instance().fetch_and_increment_tensor_id();
    return output;
};

Storage& Tensor::storage() { return this->tensor_attributes->get_storage(); }

const Storage& Tensor::storage() const { return this->tensor_attributes->get_storage(); }

const ttnn::Shape& Tensor::logical_shape() const { return this->tensor_attributes->get_tensor_spec().logical_shape(); }

const ttnn::Shape& Tensor::padded_shape() const { return this->tensor_attributes->get_tensor_spec().padded_shape(); }

DataType Tensor::dtype() const { return this->tensor_attributes->get_tensor_spec().tensor_layout().get_data_type(); }

Layout Tensor::layout() const { return this->tensor_attributes->get_tensor_spec().tensor_layout().get_layout(); }

const TensorSpec& Tensor::tensor_spec() const { return this->tensor_attributes->get_tensor_spec(); }

Buffer* Tensor::buffer() const { return device_storage().get_buffer(); }

const DeviceStorage& Tensor::device_storage() const& {
    const auto* device_storage = std::get_if<DeviceStorage>(&this->storage());
    TT_FATAL(device_storage != nullptr, "Expected Tensor with DeviceStorage, got {}", this->storage_type());
    return *device_storage;
}

const HostStorage& Tensor::host_storage() const& {
    const auto* host_storage = std::get_if<HostStorage>(&this->storage());
    TT_FATAL(host_storage != nullptr, "Expected Tensor with HostStorage, got {}", this->storage_type());
    return *host_storage;
}

distributed::MeshDevice* Tensor::mesh_device() const {
    if (this->mesh_device_.has_value()) {
        return this->mesh_device_.value();
    }
    return nullptr;
}

std::shared_ptr<distributed::MeshBuffer> Tensor::mesh_buffer() const { return device_storage().get_mesh_buffer(); }

IDevice* Tensor::device() const {
    if (this->mesh_device_.has_value()) {
        return this->mesh_device_.value();
    }
    if (this->storage_type() == tt::tt_metal::StorageType::DEVICE) {
        auto buffer = this->buffer();
        if (buffer == nullptr) {
            TT_THROW("Cannot get the device from a tensor without an allocated buffer");
        }
        return buffer->device();
    } else {
        TT_THROW("Cannot get the device from a tensor with host storage");
    }
}

const MemoryConfig& Tensor::memory_config() const { return tensor_spec().tensor_layout().get_memory_config(); }

const std::optional<ShardSpec>& Tensor::shard_spec() const { return this->memory_config().shard_spec(); }

const std::optional<NdShardSpec>& Tensor::nd_shard_spec() const { return this->memory_config().nd_shard_spec(); }

const DistributedTensorConfig& Tensor::distributed_tensor_config() const {
    return this->tensor_attributes->get_distributed_tensor_config();
}

}  // namespace tt::tt_metal
