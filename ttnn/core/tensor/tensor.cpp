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
#include "ttnn/experimental/lazy/evaluation_manager.hpp"

using LazyTensor = ttnn::experimental::lazy::LazyTensor;

namespace tt::tt_metal {
namespace {

template <typename T>
HostBuffer create_host_buffer_from_row_major_data(std::vector<T>&& data, const TensorSpec& spec, T pad_value) {
    return tensor_impl::logical_matches_physical(spec)
               ? HostBuffer(std::move(data))
               : HostBuffer(tensor_impl::encode_tensor_data(tt::stl::make_const_span(data), spec, pad_value));
}

}  // namespace

Tensor::Tensor(std::shared_ptr<LazyTensor> lazy_tensor) : lazy_tensor_(std::move(lazy_tensor)) {}

Tensor::Tensor(const tt::tt_metal::metal_tensor::Tensor& metal_tensor) :
    lazy_tensor_(LazyTensor::make_materialized_tensor(metal_tensor)) {}

Tensor::Tensor(tt::tt_metal::metal_tensor::Tensor&& metal_tensor) :
    lazy_tensor_(LazyTensor::make_materialized_tensor(metal_tensor)) {}

Tensor::Tensor(tt::tt_metal::Storage storage, TensorSpec tensor_spec, tt::tt_metal::TensorTopology tensor_topology) :
    lazy_tensor_(LazyTensor::make_materialized_tensor(
        tt::tt_metal::metal_tensor::Tensor(std::move(storage), std::move(tensor_spec), std::move(tensor_topology)))) {}

Tensor::Tensor(
    tt::tt_metal::HostBuffer buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::DataType dtype,
    tt::tt_metal::Layout layout,
    const std::optional<tt::tt_metal::Tile>& tile) :
    lazy_tensor_(LazyTensor::make_materialized_tensor(
        tt::tt_metal::metal_tensor::Tensor(std::move(buffer), shape, dtype, layout, tile))) {}

Tensor::Tensor(
    tt::tt_metal::HostBuffer buffer,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    tt::tt_metal::DataType dtype,
    tt::tt_metal::Layout layout,
    const std::optional<tt::tt_metal::Tile>& tile) :
    lazy_tensor_(LazyTensor::make_materialized_tensor(
        tt::tt_metal::metal_tensor::Tensor(std::move(buffer), logical_shape, padded_shape, dtype, layout, tile))) {}

Tensor::Tensor(tt::tt_metal::HostBuffer buffer, TensorSpec tensor_spec) :
    lazy_tensor_(LazyTensor::make_materialized_tensor(
        tt::tt_metal::metal_tensor::Tensor(std::move(buffer), std::move(tensor_spec)))) {}

template <typename T>
Tensor Tensor::from_span(
    tt::stl::Span<const T> buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    T pad_value) {
    return Tensor(tt::tt_metal::metal_tensor::Tensor::from_span(buffer, spec, device, cq_id, pad_value));
}

template <typename T>
Tensor Tensor::from_borrowed_data(
    tt::stl::Span<T> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<tt::tt_metal::Tile>& tile) {
    return Tensor(tt::tt_metal::metal_tensor::Tensor::from_borrowed_data(buffer, shape, std::move(buffer_pin), tile));
}

template <typename T>
Tensor Tensor::from_vector(
    std::vector<T>&& buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    T pad_value) {
    return Tensor(tt::tt_metal::metal_tensor::Tensor::from_vector(std::move(buffer), spec, device, cq_id, pad_value));
}

template <typename T>
std::vector<T> Tensor::to_vector(std::optional<ttnn::QueueId> cq_id) const {
    return get_materialized_tensor().to_vector<T>(cq_id);
}

template <typename T>
T Tensor::item(std::optional<ttnn::QueueId> cq_id) const {
    return get_materialized_tensor().item<T>(cq_id);
}

Tensor Tensor::to_device(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    ttsl::optional_reference<const tt::tt_metal::MemoryConfig> mem_config,
    std::optional<ttnn::QueueId> cq_id) const {
    return Tensor(get_materialized_tensor().to_device(mesh_device, mem_config, cq_id));
}

Tensor Tensor::to_layout(tt::tt_metal::Layout target_layout) const {
    return Tensor(get_materialized_tensor().to_layout(target_layout));
}

Tensor Tensor::pad(
    const ttnn::Shape& output_padded_shape, const ttnn::Shape& input_tensor_start, float pad_value) const {
    return Tensor(get_materialized_tensor().pad(output_padded_shape, input_tensor_start, pad_value));
}

Tensor Tensor::cpu(bool blocking, std::optional<ttnn::QueueId> cq_id) const {
    return Tensor(get_materialized_tensor().cpu(blocking, cq_id));
}

Tensor Tensor::unpad(const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) const {
    return Tensor(get_materialized_tensor().unpad(output_tensor_start, output_tensor_end));
}

Tensor Tensor::pad_to_tile(float pad_value) const { return Tensor(get_materialized_tensor().pad_to_tile(pad_value)); }

Tensor Tensor::unpad_from_tile(const ttnn::Shape& output_tensor_shape) const {
    return Tensor(get_materialized_tensor().unpad_from_tile(output_tensor_shape));
}

std::string Tensor::write_to_string() const { return get_materialized_tensor().write_to_string(); }

void Tensor::print() const { get_materialized_tensor().print(); }

void Tensor::deallocate(bool force) { get_materialized_tensor().deallocate(force); }

Tensor Tensor::extract_shard(const tt::tt_metal::CoreCoord& core) const {
    return Tensor(get_materialized_tensor().extract_shard(core));
}

Tensor Tensor::extract_shard(const uint32_t& core_id) const {
    return Tensor(get_materialized_tensor().extract_shard(core_id));
}

Tensor Tensor::reshape(const ttnn::Shape& new_shape) const {
    return Tensor(get_materialized_tensor().reshape(new_shape));
}

Tensor Tensor::reshape(const ttnn::Shape& new_logical_shape, const ttnn::Shape& new_padded_shape) const {
    return Tensor(get_materialized_tensor().reshape(new_logical_shape, new_padded_shape));
}

Tensor Tensor::with_tensor_topology(tt::tt_metal::TensorTopology tensor_topology) const {
    return Tensor(get_materialized_tensor().with_tensor_topology(std::move(tensor_topology)));
}

const tt::tt_metal::Storage& Tensor::storage() const { return get_materialized_tensor().storage(); }

tt::tt_metal::Storage& Tensor::storage() { return get_materialized_tensor().storage(); }

tt::tt_metal::DataType Tensor::dtype() const { return lazy_tensor_->tensor_spec().tensor_layout().get_data_type(); }

tt::tt_metal::Layout Tensor::layout() const { return lazy_tensor_->tensor_spec().tensor_layout().get_layout(); }

const ttnn::Shape& Tensor::logical_shape() const { return lazy_tensor_->tensor_spec().logical_shape(); }

const ttnn::Shape& Tensor::padded_shape() const { return lazy_tensor_->tensor_spec().padded_shape(); }

const TensorSpec& Tensor::tensor_spec() const { return lazy_tensor_->tensor_spec(); }

uint64_t Tensor::logical_volume() const { return lazy_tensor_->tensor_spec().logical_shape().volume(); }

uint64_t Tensor::physical_volume() const { return lazy_tensor_->tensor_spec().padded_shape().volume(); }

const tt::tt_metal::MemoryConfig& Tensor::memory_config() const {
    return lazy_tensor_->tensor_spec().tensor_layout().get_memory_config();
}

const tt::tt_metal::TensorTopology& Tensor::tensor_topology() const {
    // TODO: Should be available without materialization
    return get_materialized_tensor().tensor_topology();
}

const std::optional<tt::tt_metal::ShardSpec>& Tensor::shard_spec() const {
    return lazy_tensor_->tensor_spec().tensor_layout().get_memory_config().shard_spec();
}

const std::optional<tt::tt_metal::NdShardSpec>& Tensor::nd_shard_spec() const {
    return lazy_tensor_->tensor_spec().tensor_layout().get_memory_config().nd_shard_spec();
}

tt::tt_metal::StorageType Tensor::storage_type() const { return lazy_tensor_->storage_type(); }

ttnn::Shape Tensor::strides() const {
    // TODO: remove duplication with tensor.cpp
    auto s = tt::tt_metal::compute_strides(this->padded_shape());
    return ttnn::Shape(tt::stl::SmallVector<uint32_t>(s.begin(), s.end()));
}

bool Tensor::is_scalar() const {
    const ttnn::Shape logical_shape = this->logical_shape();
    return logical_shape.rank() == 0 || logical_shape.volume() == 1;
}

bool Tensor::is_allocated() const {
    return lazy_tensor_->is_materialized() ? get_materialized_tensor().is_allocated() : false;
}

tt::tt_metal::Buffer* Tensor::buffer() const { return get_materialized_tensor().buffer(); }
uint32_t Tensor::buffer_alignment() const { return lazy_tensor_->buffer_alignment(); }

const tt::tt_metal::DeviceStorage& Tensor::device_storage() const& {
    return get_materialized_tensor().device_storage();
}

const tt::tt_metal::HostStorage& Tensor::host_storage() const& { return get_materialized_tensor().host_storage(); }

std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> Tensor::mesh_buffer() const {
    return get_materialized_tensor().mesh_buffer();
}

tt::tt_metal::distributed::MeshDevice* Tensor::device() const { return lazy_tensor_->device(); }

bool Tensor::is_sharded() const {
    // return get_materialized_tensor().is_sharded();
    // TODO: This should check if tensor is on the device
    return memory_config().is_sharded();
}

uint32_t Tensor::element_size() const { return tensor_impl::element_size_bytes(this->dtype()); }

// ttnn Tensor-only methods / constructors
tt::tt_metal::metal_tensor::Tensor& Tensor::get_materialized_tensor() {
    TT_FATAL(lazy_tensor_->is_materialized(), "Tensor is not materialized");
    return lazy_tensor_->materialized_tensor();
}

const tt::tt_metal::metal_tensor::Tensor& Tensor::get_materialized_tensor() const {
    TT_FATAL(lazy_tensor_->is_materialized(), "Tensor is not materialized");
    return lazy_tensor_->materialized_tensor();
}

std::shared_ptr<TensorAttributes> Tensor::tensor_attributes() const {
    return get_materialized_tensor().tensor_attributes;
}

Tensor Tensor::make_lazy_tensor(
    const std::vector<Tensor>& op_inputs,
    const std::shared_ptr<ttnn::experimental::lazy::LazyOperation>& op,
    TensorSpec tensor_spec) {
    std::vector<std::shared_ptr<LazyTensor>> lazy_op_inputs;
    std::transform(op_inputs.begin(), op_inputs.end(), std::back_inserter(lazy_op_inputs), [](const Tensor& tensor) {
        return tensor.lazy();
    });
    return Tensor(LazyTensor::make_lazy_tensor(lazy_op_inputs, op, std::move(tensor_spec)));
}

std::vector<Tensor> Tensor::make_lazy_tensors(
    const std::vector<Tensor>& op_inputs,
    const std::shared_ptr<ttnn::experimental::lazy::LazyOperation>& op,
    const std::vector<TensorSpec>& tensor_specs) {
    std::vector<std::shared_ptr<LazyTensor>> lazy_op_inputs;
    std::transform(op_inputs.begin(), op_inputs.end(), std::back_inserter(lazy_op_inputs), [](const Tensor& tensor) {
        return tensor.lazy();
    });
    auto lazy_tensors = LazyTensor::make_lazy_tensors(lazy_op_inputs, op, tensor_specs);
    std::vector<Tensor> tensors;
    tensors.reserve(lazy_tensors.size());
    for (const auto& lazy_tensor : lazy_tensors) {
        tensors.push_back(Tensor(lazy_tensor));
    }
    return tensors;
}

const std::shared_ptr<LazyTensor>& Tensor::lazy() const { return lazy_tensor_; }

// TODO: Rename to eval
void Tensor::evaluate() { ttnn::experimental::lazy::evaluate(lazy_tensor_); }

template Tensor Tensor::from_span<bfloat16>(
    tt::stl::Span<const bfloat16> buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    bfloat16 pad_value);
template Tensor Tensor::from_span<float>(
    tt::stl::Span<const float> buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    float pad_value);
template Tensor Tensor::from_span<int32_t>(
    tt::stl::Span<const int32_t> buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    int32_t pad_value);
template Tensor Tensor::from_span<uint8_t>(
    tt::stl::Span<const uint8_t> buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    uint8_t pad_value);
template Tensor Tensor::from_span<uint16_t>(
    tt::stl::Span<const uint16_t> buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    uint16_t pad_value);
template Tensor Tensor::from_span<uint32_t>(
    tt::stl::Span<const uint32_t> buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    uint32_t pad_value);
template Tensor Tensor::from_borrowed_data<float>(
    tt::stl::Span<float> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<tt::tt_metal::Tile>& tile);
template Tensor Tensor::from_borrowed_data<bfloat16>(
    tt::stl::Span<bfloat16> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<tt::tt_metal::Tile>& tile);
template Tensor Tensor::from_borrowed_data<int32_t>(
    tt::stl::Span<int32_t> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<tt::tt_metal::Tile>& tile);
template Tensor Tensor::from_borrowed_data<uint8_t>(
    tt::stl::Span<uint8_t> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<tt::tt_metal::Tile>& tile);
template Tensor Tensor::from_borrowed_data<uint16_t>(
    tt::stl::Span<uint16_t> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<tt::tt_metal::Tile>& tile);
template Tensor Tensor::from_borrowed_data<uint32_t>(
    tt::stl::Span<uint32_t> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<tt::tt_metal::Tile>& tile);

template Tensor Tensor::from_vector<bfloat16>(
    std::vector<bfloat16>&& buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    bfloat16 pad_value);
template Tensor Tensor::from_vector<float>(
    std::vector<float>&& buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    float pad_value);
template Tensor Tensor::from_vector<int32_t>(
    std::vector<int32_t>&& buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    int32_t pad_value);
template Tensor Tensor::from_vector<uint8_t>(
    std::vector<uint8_t>&& buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    uint8_t pad_value);
template Tensor Tensor::from_vector<uint16_t>(
    std::vector<uint16_t>&& buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    uint16_t pad_value);
template Tensor Tensor::from_vector<uint32_t>(
    std::vector<uint32_t>&& buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    uint32_t pad_value);

template std::vector<bfloat16> Tensor::to_vector<bfloat16>(std::optional<ttnn::QueueId> cq_id) const;
template std::vector<float> Tensor::to_vector<float>(std::optional<ttnn::QueueId> cq_id) const;
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

Tensor create_device_tensor(const TensorSpec& tensor_spec, IDevice* device) {
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

}  // namespace tt::tt_metal
