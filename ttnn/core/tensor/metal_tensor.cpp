// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/metal_tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl_wrapper.hpp"
#include "tensor_ops.hpp"

namespace tt::tt_metal {
namespace {

template <typename T>
HostBuffer create_host_buffer_from_row_major_data(std::vector<T>&& data, const TensorSpec& spec, T pad_value) {
    return tensor_impl::logical_matches_physical(spec)
               ? HostBuffer(std::move(data))
               : HostBuffer(tensor_impl::encode_tensor_data(tt::stl::make_const_span(data), spec, pad_value));
}

}  // namespace

namespace metal_tensor {
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

Tensor::~Tensor() { this->deallocate_impl(/*force=*/false); }

void Tensor::deallocate(bool force) { deallocate_impl(force); }

void Tensor::deallocate_impl(bool force) {
    auto can_deallocate = []<typename T>(const std::shared_ptr<T>& shared_resource, bool force) {
        // It is safe to deallocate a shared resource, if either it is not shared or `force` is set.
        return shared_resource.use_count() == 1 ||  //
               (shared_resource.use_count() > 1 && force);
    };

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
    // TODO: to_dtype should be metal_tensor function, remove this ugly workaround
    res = ttnn::to_dtype(ttnn::Tensor(res), spec.data_type()).get_materialized_tensor();
    // res = ttnn::to_dtype(res, spec.data_type());
    if (device) {
        res = res.to_device(device, spec.memory_config(), cq_id);
    }
    return res;
}

template <>
std::vector<float> Tensor::to_vector<float>(std::optional<ttnn::QueueId> cq_id) const {
    Tensor cpu_tensor = this->cpu(/*blocking=*/true, cq_id);
    switch (cpu_tensor.dtype()) {
        case DataType::BFLOAT16: {
            // TODO: host_buffer::get_as should be metal_tensor function, remove this ugly workaround
            auto buffer = host_buffer::get_as<bfloat16>(ttnn::Tensor(cpu_tensor));
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
            // TODO: host_buffer::get_as should be metal_tensor function, remove this ugly workaround
            auto buffer = host_buffer::get_as<const float>(ttnn::Tensor(cpu_tensor));
            return tensor_impl::decode_tensor_data(buffer, cpu_tensor.tensor_spec());
        }
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: {
            const auto& tile = cpu_tensor.tensor_spec().tile();
            // TODO: host_buffer::get_as should be metal_tensor function, remove this ugly workaround
            auto buffer = host_buffer::get_as<const uint32_t>(ttnn::Tensor(cpu_tensor));
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
    TT_FATAL(
        this->dtype() == convert_to_data_type<T>(),
        "Unsupported data type for to_vector: got {}, expected: {}",
        this->dtype(),
        convert_to_data_type<T>());
    auto cpu_tensor = this->cpu(/*blocking=*/true, cq_id);
    // TODO: host_buffer::get_as should be metal_tensor function, remove this ugly workaround
    auto data = host_buffer::get_as<const T>(ttnn::Tensor(cpu_tensor));
    if (tensor_impl::logical_matches_physical(cpu_tensor.tensor_spec())) {
        return std::vector<T>(data.begin(), data.end());
    }
    return tensor_impl::decode_tensor_data(data, cpu_tensor.tensor_spec());
}

template <typename T>
T Tensor::item(std::optional<ttnn::QueueId> cq_id) const {
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
    // TODO: tensor_to_device should be metal_tensor function, remove this ugly workaround
    return tensor_ops::tensor_to_device(ttnn::Tensor(*this), mesh_device, mem_config, cq_id).get_materialized_tensor();
}

Tensor Tensor::cpu(bool blocking, std::optional<ttnn::QueueId> cq_id) const {
    // TODO: tensor_cpu should be metal_tensor function, remove this ugly workaround
    return tensor_ops::tensor_cpu(ttnn::Tensor(*this), blocking, cq_id).get_materialized_tensor();
}

Tensor Tensor::extract_shard(const CoreCoord& core) const {
    ZoneScoped;
    const auto& buffer_page_mapping = *this->buffer()->get_buffer_page_mapping();
    auto core_id = buffer_page_mapping.core_to_core_id.at(core);
    return this->extract_shard(core_id);
}

Tensor Tensor::extract_shard(const uint32_t& core_id) const {
    // TODO: extract_shard should be metal_tensor function, remove this ugly workaround
    return tensor_impl::extract_shard_wrapper(ttnn::Tensor(*this), core_id).get_materialized_tensor();
}

Tensor Tensor::to_layout(Layout target_layout) const {
    // TODO: tensor_to_layout should be metal_tensor function, remove this ugly workaround
    return tensor_ops::tensor_to_layout(ttnn::Tensor(*this), target_layout).get_materialized_tensor();
}

std::string Tensor::write_to_string() const {
    // TODO: to_string should be metal_tensor function, remove this ugly workaround
    return tensor_impl::to_string_wrapper(ttnn::Tensor(*this));
}

void Tensor::print() const {
    // TODO: tensor_print should be metal_tensor function, remove this ugly workaround
    tensor_ops::tensor_print(ttnn::Tensor(*this));
}

Tensor Tensor::pad(
    const ttnn::Shape& output_padded_shape, const ttnn::Shape& input_tensor_start, float pad_value) const {
    // TODO: tensor_pad should be metal_tensor function, remove this ugly workaround
    return tensor_ops::tensor_pad(ttnn::Tensor(*this), output_padded_shape, input_tensor_start, pad_value)
        .get_materialized_tensor();
}

Tensor Tensor::unpad(const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) const {
    // TODO: tensor_unpad should be metal_tensor function, remove this ugly workaround
    return tensor_ops::tensor_unpad(ttnn::Tensor(*this), output_tensor_start, output_tensor_end)
        .get_materialized_tensor();
}

Tensor Tensor::pad_to_tile(float pad_value) const {
    // TODO: tensor_pad_to_tile should be metal_tensor function, remove this ugly workaround
    return tensor_ops::tensor_pad_to_tile(ttnn::Tensor(*this), pad_value).get_materialized_tensor();
}

Tensor Tensor::unpad_from_tile(const ttnn::Shape& output_tensor_shape) const {
    // TODO: tensor_unpad_from_tile should be metal_tensor function, remove this ugly workaround
    return tensor_ops::tensor_unpad_from_tile(ttnn::Tensor(*this), output_tensor_shape).get_materialized_tensor();
}

bool Tensor::is_sharded() const {
    // TODO: is_device_tensor should be metal_tensor function, remove this ugly workaround
    return tt::tt_metal::is_device_tensor(ttnn::Tensor(*this)) ? this->memory_config().is_sharded() : false;
}

uint32_t Tensor::element_size() const { return tensor_impl::element_size_bytes(this->dtype()); }

Tensor Tensor::reshape(const ttnn::Shape& new_shape) const {
    // TODO: tensor_reshape should be metal_tensor function, remove this ugly workaround
    return tensor_ops::tensor_reshape(ttnn::Tensor(*this), new_shape).get_materialized_tensor();
}

Tensor Tensor::reshape(const ttnn::Shape& new_logical_shape, const ttnn::Shape& new_padded_shape) const {
    // TODO: tensor_reshape should be metal_tensor function, remove this ugly workaround
    return tensor_ops::tensor_reshape(ttnn::Tensor(*this), new_logical_shape, new_padded_shape)
        .get_materialized_tensor();
}

Tensor Tensor::with_tensor_topology(TensorTopology tensor_topology) const {
    Tensor result = *this;
    result.tensor_attributes =
        std::make_shared<TensorAttributes>(tensor_attributes->with_tensor_topology(std::move(tensor_topology)));
    return result;
}

bool Tensor::is_allocated() const {
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

}  // namespace metal_tensor
}  // namespace tt::tt_metal
