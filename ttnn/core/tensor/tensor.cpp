// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <memory>
#include <utility>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt_stl/overloaded.hpp>
#include "tt_stl/small_vector.hpp"
#include "tt_stl/span.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/storage.hpp"

#include "tt-metalium/mesh_device_view.hpp"
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
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>
#include <ttnn/operations/core/core.hpp>

namespace tt::tt_metal {
namespace {

template <typename T>
HostBuffer create_host_buffer_from_row_major_data(std::vector<T>&& data, const TensorSpec& spec, T pad_value) {
    return tensor_impl::logical_matches_physical(spec)
               ? HostBuffer(std::move(data))
               : HostBuffer(tensor_impl::encode_tensor_data(tt::stl::make_const_span(data), spec, pad_value));
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
    const std::optional<Tile>& tile) :
    Tensor(
        std::move(buffer),
        TensorSpec(
            logical_shape,
            TensorLayout::fromPaddedShape(
                dtype, PageConfig(layout, tile), MemoryConfig{}, logical_shape, padded_shape))) {
    using namespace tt::constants;
    if (tile.has_value() and  //
        (tile->get_tile_shape()[0] != TILE_WIDTH or tile->get_tile_shape()[1] != TILE_HEIGHT)) {
        log_warning(
            tt::LogTTNN,
            "only matmul op and ccl all-gather currently supports the customized tile shape: {}",
            tile->get_tile_shape());
    }
}

Tensor::Tensor(HostBuffer buffer, TensorSpec tensor_spec) :
    Tensor(Storage(HostStorage(std::move(buffer))), std::move(tensor_spec), TensorTopology{}) {}

Tensor::Tensor(Storage storage, TensorSpec tensor_spec, TensorTopology tensor_topology) {
    init(Storage(std::move(storage)), std::move(tensor_spec), std::move(tensor_topology));
}

void Tensor::init(Storage storage, TensorSpec tensor_spec, TensorTopology tensor_topology) {
    tensor_attributes =
        std::make_shared<TensorAttributes>(std::move(storage), std::move(tensor_spec), std::move(tensor_topology));

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
    this->tensor_id = other.tensor_id;
    if (this->tensor_attributes != other.tensor_attributes) {
        this->tensor_attributes = std::move(other.tensor_attributes);
    }
    this->mesh_device_ = other.mesh_device_;
    return *this;
}

Tensor::Tensor(const Tensor& other) = default;

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
                    }
                    storage.mesh_buffer.reset();
                }},
            this->tensor_attributes->get_storage());
    }
    // GraphTracker::instance().track_function_end();
}

template <typename T>
Tensor Tensor::from_span(
    tt::stl::Span<const T> buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    T pad_value) {
    ZoneScoped;
    return from_vector(std::vector<T>(buffer.begin(), buffer.end()), spec, device, cq_id, pad_value);
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

template <typename T>
Tensor Tensor::from_vector(
    std::vector<T>&& buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    T pad_value) {
    ZoneScoped;
    size_t volume = spec.logical_shape().volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    if (spec.data_type() == DataType::BFLOAT8_B || spec.data_type() == DataType::BFLOAT4_B) {
        TT_FATAL(spec.layout() == Layout::TILE, "Block float types are only supported in TILE layout");
    }

    // Create host tensor with DataType matching buffer
    auto buffer_dtype = convert_to_data_type<T>();
    auto buffer_spec =
        TensorSpec(spec.logical_shape(), TensorLayout(buffer_dtype, spec.page_config(), spec.memory_config()));
    auto res = Tensor(create_host_buffer_from_row_major_data(std::move(buffer), buffer_spec, pad_value), buffer_spec);
    // Convert to datatype from original spec
    res = ttnn::to_dtype(res, spec.data_type());
    if (device) {
        res = res.to_device(device, spec.memory_config(), cq_id);
    }
    return res;
}

template <>
std::vector<float> Tensor::to_vector<float>(std::optional<ttnn::QueueId> cq_id) const {
    ZoneScoped;
    Tensor cpu_tensor = this->cpu(/*blocking=*/true, cq_id);
    switch (cpu_tensor.dtype()) {
        case DataType::BFLOAT16: {
            auto buffer = host_buffer::get_as<bfloat16>(cpu_tensor);
            std::vector<float> physical_data;
            physical_data.reserve(buffer.size());
            std::transform(buffer.begin(), buffer.end(), std::back_inserter(physical_data), [](bfloat16 val) {
                return static_cast<float>(val);
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
std::vector<T> Tensor::to_vector(std::optional<ttnn::QueueId> cq_id) const {
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

template <typename T>
T Tensor::item(std::optional<ttnn::QueueId> cq_id) const {
    ZoneScoped;
    TT_FATAL(
        this->logical_shape().volume() == 1,
        "tensor.item() requires tensor to have exactly one element, but got {} elements",
        this->logical_shape().volume());

    // Use existing infrastructure: to_vector() already handles multi-device and host tensors correctly
    // by calling cpu() internally when needed
    auto vector_data = this->to_vector<T>(cq_id);
    return vector_data[0];
}

// Instantiate explicitly for the supported types.
template Tensor Tensor::from_span<bfloat16>(
    tt::stl::Span<const bfloat16> buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    bfloat16 pad_value);
template Tensor Tensor::from_span<float>(
    tt::stl::Span<const float> buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    float pad_value);
template Tensor Tensor::from_span<int32_t>(
    tt::stl::Span<const int32_t> buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    int32_t pad_value);
template Tensor Tensor::from_span<uint8_t>(
    tt::stl::Span<const uint8_t> buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    uint8_t pad_value);
template Tensor Tensor::from_span<uint16_t>(
    tt::stl::Span<const uint16_t> buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    uint16_t pad_value);
template Tensor Tensor::from_span<uint32_t>(
    tt::stl::Span<const uint32_t> buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
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
    std::optional<ttnn::QueueId> cq_id,
    bfloat16 pad_value);
template Tensor Tensor::from_vector<float>(
    std::vector<float>&& buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    float pad_value);
template Tensor Tensor::from_vector<int32_t>(
    std::vector<int32_t>&& buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    int32_t pad_value);
template Tensor Tensor::from_vector<uint8_t>(
    std::vector<uint8_t>&& buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    uint8_t pad_value);
template Tensor Tensor::from_vector<uint16_t>(
    std::vector<uint16_t>&& buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    uint16_t pad_value);
template Tensor Tensor::from_vector<uint32_t>(
    std::vector<uint32_t>&& buffer,
    const TensorSpec& spec,
    distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    uint32_t pad_value);

template std::vector<bfloat16> Tensor::to_vector<bfloat16>(std::optional<ttnn::QueueId> cq_id) const;
template std::vector<int32_t> Tensor::to_vector<int32_t>(std::optional<ttnn::QueueId> cq_id) const;
template std::vector<uint8_t> Tensor::to_vector<uint8_t>(std::optional<ttnn::QueueId> cq_id) const;
template std::vector<uint16_t> Tensor::to_vector<uint16_t>(std::optional<ttnn::QueueId> cq_id) const;
template std::vector<uint32_t> Tensor::to_vector<uint32_t>(std::optional<ttnn::QueueId> cq_id) const;

template float Tensor::item<float>(std::optional<ttnn::QueueId> cq_id) const;
template bfloat16 Tensor::item<bfloat16>(std::optional<ttnn::QueueId> cq_id) const;
template int32_t Tensor::item<int32_t>(std::optional<ttnn::QueueId> cq_id) const;
template uint8_t Tensor::item<uint8_t>(std::optional<ttnn::QueueId> cq_id) const;
template uint16_t Tensor::item<uint16_t>(std::optional<ttnn::QueueId> cq_id) const;
template uint32_t Tensor::item<uint32_t>(std::optional<ttnn::QueueId> cq_id) const;

Tensor Tensor::to_device(
    distributed::MeshDevice* mesh_device,
    ttsl::optional_reference<const MemoryConfig> mem_config,
    std::optional<ttnn::QueueId> cq_id) const {
    return tensor_ops::tensor_to_device(*this, mesh_device, mem_config, cq_id);
}

Tensor Tensor::cpu(bool blocking, std::optional<ttnn::QueueId> cq_id) const {
    return tensor_ops::tensor_cpu(*this, blocking, cq_id);
}

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

ttnn::Shape Tensor::strides() const {
    auto s = tt::tt_metal::compute_strides(this->padded_shape());
    return ttnn::Shape(tt::stl::SmallVector<uint32_t>(s.begin(), s.end()));
}

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
    distributed::MeshDevice* mesh_device = dynamic_cast<distributed::MeshDevice*>(device);
    output = allocate_tensor_on_device(tensor_spec, mesh_device);
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
    auto mesh_device = src.device();
    TT_FATAL(mesh_device, "Tensor must be on device");
    memcpy(mesh_device->mesh_command_queue(), dst, src, region, blocking);
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
    auto mesh_device = dst.device();
    TT_FATAL(mesh_device, "Tensor must be on device");
    memcpy(mesh_device->mesh_command_queue(), dst, src, region);
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
        auto mesh_device = src.device();
        memcpy(mesh_device->mesh_command_queue(), dst, src, region);
    } else if (is_device_tensor(dst) && is_cpu_tensor(src)) {
        auto mesh_device = dst.device();
        memcpy(mesh_device->mesh_command_queue(), dst, src, region);
    } else {
        TT_THROW("Unsupported memcpy");
    }
}

Tensor allocate_tensor_on_device(const TensorSpec& tensor_spec, distributed::MeshDevice* device) {
    auto mesh_buffer = tensor_impl::allocate_device_buffer(device, tensor_spec);
    std::vector<distributed::MeshCoordinate> coords;
    coords.reserve(device->shape().mesh_size());
    for (const auto& coord : distributed::MeshCoordinateRange(device->shape())) {
        coords.push_back(coord);
    }
    DeviceStorage device_storage(std::move(mesh_buffer), coords);
    // TODO (#25340): Implement correct logic and add test for this
    ttsl::SmallVector<distributed::MeshMapperConfig::Placement> placements(device->shape().dims());
    for (size_t i = 0; i < device->shape().dims(); i++) {
        placements[i] = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
    }

    auto tensor_topology = TensorTopology{device->shape(), placements, coords};
    return Tensor(std::move(device_storage), tensor_spec, tensor_topology);
}

Tensor allocate_tensor_on_host(const TensorSpec& tensor_spec, distributed::MeshDevice* device) {
    auto distributed_host_buffer = DistributedHostBuffer::create(device->get_view());

    std::vector<distributed::MeshCoordinate> coords;
    coords.reserve(device->shape().mesh_size());
    for (const auto& coord : distributed::MeshCoordinateRange(device->shape())) {
        coords.push_back(coord);
    }

    distributed_host_buffer.emplace_shards(
        coords,
        [&](const auto&) { return tensor_impl::allocate_host_buffer(tensor_spec); },
        DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);

    // TODO (#25340): Implement correct logic and add test for this
    return Tensor(HostStorage(std::move(distributed_host_buffer)), tensor_spec, TensorTopology{});
}

void write_tensor(const Tensor& src, Tensor& dst, bool blocking, std::optional<ttnn::QueueId> cq_id) {
    ZoneScoped;
    TT_FATAL(
        (is_device_tensor(src) && is_cpu_tensor(dst)) ||    // device to host
            (is_cpu_tensor(src) && is_device_tensor(dst)),  // host to device
        "Unsupported data transfer direction; source storage type: {}, destination storage type: {}",
        src.storage_type(),
        dst.storage_type());

    if (is_device_tensor(src)) {
        tensor_impl::copy_to_host_wrapper(src, dst, blocking, cq_id);
        return;
    }

    TT_FATAL(src.logical_shape() == dst.logical_shape(), "Error");
    TT_FATAL(src.dtype() == dst.dtype(), "Error");
    TT_FATAL(src.tensor_spec().page_config() == dst.tensor_spec().page_config(), "Error");

    auto mesh_buffer = dst.device_storage().mesh_buffer;
    TT_FATAL(!blocking, "Blocking is not supported for host to device copy");
    tensor_impl::copy_to_device_wrapper(src, dst, cq_id);
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

distributed::MeshDevice* Tensor::device() const {
    if (this->mesh_device_.has_value()) {
        return this->mesh_device_.value();
    }
    return nullptr;
}

std::shared_ptr<distributed::MeshBuffer> Tensor::mesh_buffer() const { return device_storage().get_mesh_buffer(); }

const MemoryConfig& Tensor::memory_config() const { return tensor_spec().tensor_layout().get_memory_config(); }

const std::optional<ShardSpec>& Tensor::shard_spec() const { return this->memory_config().shard_spec(); }

const std::optional<NdShardSpec>& Tensor::nd_shard_spec() const { return this->memory_config().nd_shard_spec(); }

const TensorTopology& Tensor::tensor_topology() const { return this->tensor_attributes->get_tensor_topology(); }

}  // namespace tt::tt_metal

// host data to tensor conversion implementation

using namespace tt::tt_metal;
namespace {

// Host buffer does not have an API to get the original number of elements,
// but in context of the conversion from python it is possible to use
// the type ID and the set of expected types.
DataType map_hostbuffer_type_to_datatype(const HostBuffer& buffer) {
    const auto& type_info = buffer.type_info();

    if (type_info == typeid(bfloat16)) {
        return DataType::BFLOAT16;
    } else if (type_info == typeid(float)) {
        return DataType::FLOAT32;
    } else if (type_info == typeid(uint32_t)) {
        return DataType::UINT32;
    } else if (type_info == typeid(uint8_t)) {
        return DataType::UINT8;
    } else if (type_info == typeid(uint16_t)) {
        return DataType::UINT16;
    } else if (type_info == typeid(int32_t)) {
        return DataType::INT32;
    } else {
        TT_THROW("Unsupported type in HostBuffer: {}", buffer.type_info().name());
    }
}

std::size_t get_element_count(const HostBuffer& buffer) {
    auto data_type = map_hostbuffer_type_to_datatype(buffer);
    auto byte_span = buffer.view_bytes();
    switch (data_type) {
        case DataType::BFLOAT16: return byte_span.size() / sizeof(bfloat16);
        case DataType::FLOAT32: return byte_span.size() / sizeof(float);
        case DataType::UINT32: return byte_span.size() / sizeof(uint32_t);
        case DataType::UINT8: return byte_span.size() / sizeof(uint8_t);
        case DataType::UINT16: return byte_span.size() / sizeof(uint16_t);
        case DataType::INT32: return byte_span.size() / sizeof(int32_t);
        default: TT_FATAL(false, "Unhandled DataType in get_element_count");
    }
}

struct TensorPreparedConversion {
    /// Use this layout to construct the initial tensor -- extra conversion might be done
    /// after the tensor has been moved to device.
    Layout construct_with_layout = Layout::TILE;
    DataType host_convert_data_type = DataType::INVALID;
};

#define py_log(...) \
    std::cout << fmt::format("{}:{} {} {}", __FILE__, __LINE__, __func__, fmt::format(__VA_ARGS__)) << std::endl;

template <typename T>
Tensor create_typed_tt_tensor_from_host_data(
    const HostBuffer& host_data,
    const ttnn::Shape& tensor_shape,
    const TensorLayout& tensor_layout,
    ttnn::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    T pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper) {
    TT_FATAL(
        !tensor_layout.get_memory_config().is_sharded() || tensor_layout.get_memory_config().shard_spec().has_value() ||
            tensor_layout.get_memory_config().nd_shard_spec().has_value(),
        "Sharded tensors must have a shard spec when converting to tt tensors!");

    TT_FATAL(
        host_data.type_info() == typeid(T),
        "Mismatch between the host buffer data type and the target tensor data: host buffer is {} and the target is {}",
        host_data.type_info().name(),
        typeid(T).name());

    tt::stl::Span<T> pydata_span(
        const_cast<T*>(reinterpret_cast<const T*>(host_data.view_bytes().data())), tensor_shape.volume());

    if (mesh_mapper == nullptr) {
        // Create a single tt tensor from the pydata.
        const TensorSpec tensor_spec(tensor_shape, tensor_layout);
        Tensor output;
        if (const bool pydata_borrowable = tensor_spec.layout() == Layout::ROW_MAJOR &&
                                           tensor_spec.physical_shape() == tensor_spec.logical_2d_shape() &&
                                           tensor_spec.data_type() == convert_to_data_type<T>();
            pydata_borrowable) {
            output = Tensor(
                host_data,
                tensor_shape,
                tensor_layout.get_data_type(),
                tensor_layout.get_layout(),
                tensor_layout.get_tile());
        } else {
            const bool is_custom_bfloat =
                std::is_same_v<T, float> && (tensor_layout.get_data_type() == DataType::BFLOAT4_B ||
                                             tensor_layout.get_data_type() == DataType::BFLOAT8_B);

            if (is_custom_bfloat) {
                // Using already implemented logic for the bfloat4/8
                output =
                    Tensor::from_span(tt::stl::make_const_span(pydata_span), tensor_spec, device, cq_id, pad_value);
            } else {
                // Otherwise construct the tensor from the host buffer directly, or through encoding. Calling
                // `make_span` for other cases is inefficient here, as it will create a new host buffer from the span.
                if (tensor_impl::logical_matches_physical(tensor_spec)) {
                    output = Tensor(host_data, tensor_spec);
                } else {
                    output = Tensor(
                        HostBuffer(tensor_impl::encode_tensor_data(
                            tt::stl::make_const_span(pydata_span), tensor_spec, pad_value)),
                        tensor_spec);
                }
            }
        }
        if (device != nullptr) {
            output = output.to_device(device, tensor_spec.memory_config(), cq_id);
        }
        return output;
    } else {
        // Shard pydata across mesh and apply `tensor_layout` at each shard.
        // Shapes of multi device shards will be derived automatically.
        return ttnn::distributed::create_distributed_tensor(
            pydata_span,
            tensor_shape,
            host_data.pin(),
            tensor_layout,
            *mesh_mapper,
            device != nullptr ? std::make_optional(std::ref(*device)) : std::nullopt,
            cq_id,
            pad_value);
    }
}

Tensor create_tt_tensor_from_host_data(
    const HostBuffer& host_data,
    const ttnn::Shape& tensor_shape,
    const TensorLayout& tensor_layout,
    ttnn::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    float pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper) {
    auto create_concrete = [&]<typename T>() {
        return create_typed_tt_tensor_from_host_data<T>(
            host_data, tensor_shape, tensor_layout, device, cq_id, pad_value, mesh_mapper);
    };
    switch (tensor_layout.get_data_type()) {
        case DataType::UINT8: return create_concrete.operator()<uint8_t>();
        case DataType::UINT16: return create_concrete.operator()<uint16_t>();
        case DataType::INT32: return create_concrete.operator()<int32_t>();
        case DataType::UINT32: return create_concrete.operator()<uint32_t>();
        case DataType::FLOAT32: return create_concrete.operator()<float>();
        case DataType::BFLOAT16: return create_concrete.operator()<bfloat16>();
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: {
            return create_concrete.operator()<float>();
        }
        case DataType::INVALID: {
            TT_THROW("Unsupported DataType: {}", tensor_layout.get_data_type());
        }
    }

    TT_THROW("Unsupported DataType: {}", tensor_layout.get_data_type());
}

struct HostBufferConversionInput {
    host_buffer_data_type host_type;
    DataType target_type;
    Layout layout;

    bool operator==(const HostBufferConversionInput& other) const {
        return host_type == other.host_type && target_type == other.target_type && layout == other.layout;
    }
};

struct HostBufferConversionInputHash {
    std::size_t operator()(const HostBufferConversionInput& input) const {
        std::size_t h1 = std::hash<int>{}(static_cast<int>(input.host_type));
        std::size_t h2 = std::hash<int>{}(static_cast<int>(input.target_type));
        std::size_t h3 = std::hash<int>{}(static_cast<int>(input.layout));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

std::optional<TensorPreparedConversion> prepare_tensor_conversion(
    const host_buffer_data_type& host_data_type, const TensorLayout& tensor_layout, bool has_device) {
    // Early exit conditions -- on-device strategy is not supported

    if (!has_device ||
        // Device is required
        tensor_layout.get_memory_config().is_sharded() ||
        // Sharded tensor handling and on-device type-casting cannot be done with the regular strategy
        (((tensor_layout.get_tile().get_tile_shape()[0] % tt::constants::TILE_WIDTH) != 0) ||
         ((tensor_layout.get_tile().get_tile_shape()[1] % tt::constants::TILE_HEIGHT) != 0))
        // on-device tiling operation expects 32x32 row
    ) {
        return std::nullopt;
    }

    // High-level overview of the conversion strategy logic.
    //
    // Not all mappings improve performance if they are done on device: the type conversion itself is not the most
    // expensive part of the conversion, it is ROW->TILE conversion. If done on host, it might be ~10 times slower than
    // device. But due to existing issues with some on-device operators, only the mappings below can be safely done on
    // device, without the loss of precision.
    //
    // Edge cases that require host-side conversion due to known bugs:
    //    - int32 tensors with retiling can lose precision https://github.com/tenstorrent/tt-metal/issues/23407,
    //      although the size is not stable. `(32, 32, 64, 64)` Can trigger the bug as well.
    //    - uint8 typecast missing device support https://github.com/tenstorrent/tt-metal/issues/21682
    //    - float32 precision loss when changing layout https://github.com/tenstorrent/tt-metal/issues/23405
    //    - bfloat16 to bfloat4b/bfloat8b conversions can zero half the tensor in some conditions.
    //      The test triggering this bug is test_matmul.py::test_tiny_tiles_bfloat
    //
    // Based on the benchmark data, not all conversion pairings have performance improvements
    // when done on host. Additionally, some types cannot be stored in ROW-MAJOR form, like bfloat8, meaning that
    // on-host conversion to TILE is mandatory for the TTNN tensor creation.
    //
    // To extend the conversion map once the aforementioned bugs are resolved:
    //
    // - `construct_with_layout` constrols which layout should be used for the host-side tensor construction. For
    //   performance reasons the ROW-MAJOR is the most optimal one.
    // - `host_side_conversion` to show whether on-device type casting is necessary or not.
    //   If not, the tensor will be created using torch (or on-host converted torch data) and optionally changed to the
    //   right layout.

    // Mapping
    // `{input_torch_type, expected_ttnn_type, expected_layout}` -> `{on-host_tensor_layout, on-host_tensor_data_type,
    // torch_data_conversion}`

    static std::unordered_map<HostBufferConversionInput, TensorPreparedConversion, HostBufferConversionInputHash>
        conversion_map = {
    // clang-format off

            // At the moment there are no cases that can be safely implemented with on-device
            // conversion, and bfloat16 cases are to be implemented in a follow-up PR to avoid
            // breaking too many tests in a scope of a single PR. The conversion mappings below
            // can be enabled and updated as related bugs with type/layout conversion are fixed
            // in the other parts of the library

            // The mapping structure is
            // {<Input-Type>, <Target-Type>, <Target-Layout>} -> {<Layout-To-Construct-On-Host>, <Type-To-Cast-On-Host>}

#if false
            {{host_buffer_data_type::BFLOAT16,     DataType::BFLOAT16,  Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::BFLOAT16, Layout::ROW_MAJOR},  {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::BFLOAT4_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::BFLOAT8_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::FLOAT32,   Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::FLOAT32,   Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::INT32,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::INT32,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::UINT16,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::UINT16,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::UINT32,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::UINT32,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::UINT8,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::UINT8,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::BFLOAT16,  Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::BFLOAT16,  Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::BFLOAT4_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::BFLOAT8_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::FLOAT32,   Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::FLOAT32,   Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::INT32,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::INT32,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::FLOAT16,      DataType::UINT16,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::UINT16,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::UINT32,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::UINT32,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::UINT32 }},
            {{host_buffer_data_type::FLOAT16,      DataType::UINT8,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::UINT8,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT32,      DataType::BFLOAT16,  Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::BFLOAT16,  Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::BFLOAT16,  Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::BFLOAT4_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::BFLOAT4_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::BFLOAT8_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::BFLOAT8_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::FLOAT32,   Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::FLOAT32,   Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::FLOAT32,   Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::INT32,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::INT32,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::UINT16,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::UINT16,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::UINT16,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::UINT32,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::UINT32,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::UINT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::UINT8,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::UINT8,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::UINT8,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::BFLOAT8_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::BFLOAT4_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::BFLOAT16,  Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::FLOAT32,   Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::UINT8,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::UINT16,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::UINT32,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::INT32,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::BFLOAT8_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::BFLOAT4_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::BFLOAT16,  Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::FLOAT32,   Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::UINT8,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::UINT16,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::UINT32,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::INT32,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::INT16,        DataType::BFLOAT16,  Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::BFLOAT16,  Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::BFLOAT4_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::BFLOAT8_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::FLOAT32,   Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::FLOAT32,   Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::INT32,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::INT32,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::UINT16,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::UINT16,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::UINT32,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::UINT32,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::UINT8,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::UINT8,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::BFLOAT16,  Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::BFLOAT16,  Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::BFLOAT4_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::BFLOAT8_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::FLOAT32,   Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::FLOAT32,   Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::INT32,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::INT32,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::UINT16,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::UINT16,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::UINT32,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::UINT32,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::UINT8,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::UINT8,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::BFLOAT16,  Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::INT64,        DataType::BFLOAT16,  Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::BFLOAT4_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::BFLOAT8_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::FLOAT32,   Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::FLOAT32,   Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::INT32,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::INT32,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::UINT16,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::UINT16,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::UINT32,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::UINT32,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::UINT8,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::UINT8 }},
            {{host_buffer_data_type::INT64,        DataType::UINT8,     Layout::TILE},      {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::BFLOAT16,  Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::UINT8,        DataType::BFLOAT16,  Layout::TILE},      {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::BFLOAT4_B, Layout::TILE},      {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::BFLOAT8_B, Layout::TILE},      {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::FLOAT32,   Layout::ROW_MAJOR}, {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::FLOAT32,   Layout::TILE},      {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::INT32,     Layout::ROW_MAJOR}, {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::INT32,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::UINT8,        DataType::UINT16,    Layout::ROW_MAJOR}, {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::UINT16,    Layout::TILE},      {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::UINT32,    Layout::ROW_MAJOR}, {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::UINT32,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::UINT8,        DataType::UINT8,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::UINT8,     Layout::TILE},      {Layout::TILE,      DataType::UINT8 }},
#endif

            // clang-format on
        };

    HostBufferConversionInput input{
        .host_type = host_data_type,
        .target_type = tensor_layout.get_data_type(),
        .layout = tensor_layout.get_layout(),
    };

    auto it = conversion_map.find(input);
    if (it == conversion_map.end()) {
        return std::nullopt;
    } else {
        return it->second;
    }
}
}  // namespace

Tensor tt::tt_metal::convert_python_tensor_to_tt_tensor(
    const ttnn::Shape& tensor_shape,
    const TensorLayout& tensor_layout,
    const host_buffer_data_type& host_data_type,
    std::function<HostBuffer(DataType)> get_host_data,
    ttnn::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    float pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper) {
    ZoneScoped;

    auto strategy = prepare_tensor_conversion(host_data_type, tensor_layout, device != nullptr);
    Tensor output;

    GraphTracker::instance().track_function_start(
        "tt::tt_metal::detail::convert_python_tensor_to_tt_tensor",
        tensor_layout.get_data_type(),
        tensor_layout.get_layout(),
        tensor_layout.get_tile(),
        tensor_layout.get_memory_config(),
        device,
        cq_id,
        pad_value,
        mesh_mapper);

    DataType host_dtype;
    if (strategy) {
        host_dtype = strategy->host_convert_data_type;
    } else {
        if (tensor_layout.get_data_type() == DataType::BFLOAT4_B ||
            tensor_layout.get_data_type() == DataType::BFLOAT8_B) {
            host_dtype = DataType::FLOAT32;
        } else {
            host_dtype = tensor_layout.get_data_type();
        }
    }

    HostBuffer host_data = get_host_data(host_dtype);

    TT_FATAL(
        get_element_count(host_data) == tensor_shape.volume(),
        "Number of elements from python tensor {} must match volume of shape {}!",
        get_element_count(host_data),
        tensor_shape.volume());

    if (strategy && !host_data.view_bytes().empty()) {
        // to tile the tensor it must have non-zero volume or a sufficient rank -- if this fails
        // the tensor must be constructed on host.
        output = create_tt_tensor_from_host_data(
            host_data,
            tensor_shape,
            TensorLayout(
                strategy->host_convert_data_type,
                PageConfig(strategy->construct_with_layout, tensor_layout.get_tile()),
                tensor_layout.get_memory_config()),
            device,
            cq_id,
            pad_value,
            mesh_mapper);

        output = tt::tt_metal::set_tensor_id(output);

        auto set_layout = [&](Layout target) {
            if (output.layout() != target) {
                output = ttnn::to_layout(output, target, std::nullopt, tensor_layout.get_memory_config());
            }
        };

        if (output.dtype() != tensor_layout.get_data_type()) {
            // Need to perform final data conversion on device, typecast requires TILE layout.
            set_layout(Layout::TILE);
            output = ttnn::typecast(output, tensor_layout.get_data_type());
        }

        set_layout(tensor_layout.get_layout());
    } else {
        // Convert on host
        if (tensor_layout.get_data_type() == DataType::BFLOAT8_B ||
            tensor_layout.get_data_type() == DataType::BFLOAT4_B) {
            TT_FATAL(
                tensor_layout.get_layout() == Layout::TILE,
                "Tile layout is required for tensor of type bfloat8_b or bfloat4_b; got {}.",
                tensor_layout.get_layout());
        }

        output = create_tt_tensor_from_host_data(
            host_data, tensor_shape, tensor_layout, device, cq_id, pad_value, mesh_mapper);
        output = tt::tt_metal::set_tensor_id(output);
    }

    GraphTracker::instance().track_function_end(output);

    return output;
}
