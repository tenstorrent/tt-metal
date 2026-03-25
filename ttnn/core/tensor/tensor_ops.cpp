// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include "tensor/tensor_ops.hpp"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <cstdint>

#include <tt-metalium/bfloat16.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tracy/Tracy.hpp>
#include "ttnn/graph/graph_serialization.hpp"

namespace {

tt::tt_metal::Tensor allocate_tensor_on_device(
    const tt::tt_metal::TensorSpec& tensor_spec, tt::tt_metal::distributed::MeshDevice* device) {
    using namespace tt::tt_metal;
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
}  // namespace

namespace tt::tt_metal {

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
    return Tensor(HostTensor(std::move(distributed_host_buffer), tensor_spec, TensorTopology{}));
}

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
}  // namespace tt::tt_metal

namespace tt::tt_metal {

Tensor to_device(
    const Tensor& input_tensor,
    distributed::MeshDevice* mesh_device,
    ttsl::optional_reference<const MemoryConfig> mem_config,
    std::optional<QueueId> cq_id) {
    GraphTracker::instance().track_function_start("Tensor::to_device", input_tensor, mesh_device, mem_config);
    if (input_tensor.storage_type() == StorageType::DEVICE) {
        TT_ASSERT(input_tensor.device() == mesh_device, "Currently do not support moving between devices");
        GraphTracker::instance().track_function_end(input_tensor);
        return input_tensor;
    }
    auto device_tensor = tensor_impl::to_device(input_tensor, mesh_device, mem_config, cq_id);
    GraphTracker::instance().track_function_end(device_tensor);
    return device_tensor;
}

void copy_to_device(const Tensor& host_tensor, Tensor& device_tensor, std::optional<tt::tt_metal::QueueId> cq_id) {
    GraphTracker::instance().track_function_start("tt::tt_metal::copy_to_device", host_tensor, device_tensor, cq_id);
    tensor_impl::copy_to_device(host_tensor, device_tensor, cq_id);
    device_tensor = tt::tt_metal::set_tensor_id(device_tensor);
    GraphTracker::instance().track_function_end(device_tensor);
}

void copy_to_device(
    distributed::MeshCommandQueue& queue,
    const std::byte* src,
    Tensor& device_tensor,
    const std::optional<BufferRegion>& region) {
    GraphTracker::instance().track_function_start("tt::tt_metal::copy_to_device", queue, src, device_tensor, region);
    tensor_impl::copy_to_device(queue, src, device_tensor, region);
    GraphTracker::instance().track_function_end(device_tensor);
}

void copy_to_host(
    distributed::MeshCommandQueue& queue,
    const Tensor& device_tensor,
    std::byte* dst,
    const std::optional<BufferRegion>& region,
    bool blocking) {
    GraphTracker::instance().track_function_start(
        "tt::tt_metal::copy_to_host", queue, device_tensor, dst, region, blocking);
    tensor_impl::copy_to_host(queue, device_tensor, dst, region, blocking);
    GraphTracker::instance().track_function_end(device_tensor);
}

void copy_to_host(const Tensor& device_tensor, Tensor& host_tensor, bool blocking, std::optional<QueueId> cq_id) {
    GraphTracker::instance().track_function_start(
        "tt::tt_metal::copy_to_host", device_tensor, host_tensor, blocking, cq_id);
    tensor_impl::copy_to_host(device_tensor, host_tensor, blocking, cq_id);
    GraphTracker::instance().track_function_end(host_tensor);
}

Tensor cpu(const Tensor& input_tensor, bool blocking, std::optional<QueueId> cq_id) {
    if (input_tensor.storage_type() != StorageType::DEVICE) {
        return input_tensor;
    }

    GraphTracker::instance().track_function_start("Tensor::cpu", input_tensor, blocking);

    auto output = tensor_impl::to_host(input_tensor, blocking, cq_id);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor to_layout(const Tensor& input_tensor, Layout target_layout) {
    GraphTracker::instance().track_function_start("Tensor::to_layout", input_tensor, target_layout);
    TT_FATAL(is_cpu_tensor(input_tensor), "Tensor must be on host for to_layout conversion");
    Tensor output = Tensor(tensor_impl::to_layout(input_tensor.host_tensor(), target_layout));
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor pad(
    const Tensor& input_tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
    float pad_value) {
    GraphTracker::instance().track_function_start(
        "Tensor::pad", input_tensor, output_padded_shape, input_tensor_start, pad_value);
    TT_FATAL(is_cpu_tensor(input_tensor), "Tensor must be on host for padding");
    // TODO: Flip to assert when we remove use cases in python and c++
    if (input_tensor.layout() != Layout::ROW_MAJOR) {
        log_warning(
            tt::LogOp,
            "Tensor layout {} must be ROW_MAJOR for padding! Returning original tensor!",
            input_tensor.layout());
        return input_tensor;
    }

    auto output =
        Tensor(tensor_impl::pad(input_tensor.host_tensor(), output_padded_shape, input_tensor_start, pad_value));
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor unpad(
    const Tensor& input_tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const tt::tt_metal::Shape& output_tensor_end) {
    GraphTracker::instance().track_function_start(
        "Tensor::unpad", input_tensor, output_tensor_start, output_tensor_end);
    TT_FATAL(is_cpu_tensor(input_tensor), "Tensor must be on host for unpadding");
    auto output = Tensor(tensor_impl::unpad(input_tensor.host_tensor(), output_tensor_start, output_tensor_end));
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor pad_to_tile(const Tensor& input_tensor, float pad_value) {
    GraphTracker::instance().track_function_start("Tensor::pad_to_tile", input_tensor, pad_value);
    TT_FATAL(is_cpu_tensor(input_tensor), "Tensor must be on host for pad_to_tile conversion");
    auto output = Tensor(tensor_impl::pad_to_tile(input_tensor.host_tensor(), pad_value));
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor unpad_from_tile(const Tensor& input_tensor, const tt::tt_metal::Shape& output_tensor_shape) {
    GraphTracker::instance().track_function_start("Tensor::unpad_from_tile", input_tensor, output_tensor_shape);
    TT_FATAL(is_cpu_tensor(input_tensor), "Tensor must be on host for unpad_from_tile conversion");
    auto output = Tensor(tensor_impl::unpad_from_tile(input_tensor.host_tensor(), output_tensor_shape));
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

// ======================================================================================
//                                  .tensor_view()
// ======================================================================================

// TODO(river): This will be moved to runtime after MeshTensor is in.
Tensor view_device(const Tensor& input_tensor, const Shape& new_logical_shape, const Shape& new_padded_shape) {
    // Just edit shape if shape has a 0 dimension
    if (input_tensor.logical_volume() == 0) {
        TT_FATAL(new_logical_shape.volume() == 0, "Tensor volume is 0, but shape's volume is not");
    }
    bool is_row_major = input_tensor.layout() == Layout::ROW_MAJOR;
    bool changing_last_dim = new_padded_shape[-1] != input_tensor.padded_shape()[-1];
    const auto& input_memory_config = input_tensor.memory_config();
    TT_FATAL(
        !input_memory_config.is_sharded() || !changing_last_dim ||
            input_memory_config.shard_spec()->shape[1] == input_tensor.padded_shape()[-1],
        "Changing the last dimension of a sharded tensor is not supported unless the shard width matches the input "
        "last dimension. "
        "Input shape: {}, New shape: {}, Shard width: {}",
        input_tensor.padded_shape(),
        new_padded_shape,
        input_memory_config.shard_spec()->shape[1]);

    auto output_memory_config = input_memory_config;
    if (is_row_major && input_memory_config.is_sharded() && changing_last_dim) {
        auto shard_spec = input_memory_config.shard_spec().value();
        auto shard_volume = shard_spec.numel();
        shard_spec.shape[1] = new_padded_shape[-1];  // update output shard to match new shard width
        shard_spec.shape[0] = shard_volume / shard_spec.shape[1];
        output_memory_config =
            MemoryConfig{input_memory_config.memory_layout(), input_memory_config.buffer_type(), shard_spec};
    }

    auto new_spec = tt::tt_metal::TensorSpec(
        new_logical_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            input_tensor.tensor_spec().page_config(),
            output_memory_config,
            new_logical_shape,
            new_padded_shape));

    // TODO (#25340): Review tensor topology logic for reshape
    if (input_tensor.layout() != Layout::ROW_MAJOR || !changing_last_dim) {
        return Tensor(input_tensor.device_storage(), new_spec, input_tensor.tensor_topology());
    }
    if (!input_tensor.memory_config().is_sharded()) {
        auto device_storage = input_tensor.device_storage();
        auto* device_buffer = device_storage.get_buffer();
        auto page_size_bytes = new_spec.compute_page_size_bytes();
        device_buffer->set_page_size(page_size_bytes);
        return Tensor(std::move(device_storage), new_spec, input_tensor.tensor_topology());
    }

    tt::tt_metal::ShardSpec new_shard_spec = output_memory_config.shard_spec().value();
    std::array<uint32_t, 2> shard_page_shape = {1, new_shard_spec.shape[1]};
    std::array<uint32_t, 2> tensor2d_shape_in_pages = {
        new_spec.physical_shape().height() / shard_page_shape[0],
        new_spec.physical_shape().width() / shard_page_shape[1]};
    tt::tt_metal::ShardSpecBuffer new_shard_spec_buffer =
        tt::tt_metal::ShardSpecBuffer(new_shard_spec, shard_page_shape, tensor2d_shape_in_pages);

    tt::tt_metal::Shape tensor_shape_pages(tensor2d_shape_in_pages);
    tt::tt_metal::Shape shard_shape_pages(new_shard_spec_buffer.shape_in_pages());
    tt::tt_metal::BufferDistributionSpec new_buffer_dist_spec = tt::tt_metal::BufferDistributionSpec(
        tensor_shape_pages, shard_shape_pages, new_shard_spec.grid, new_shard_spec.orientation);

    auto device_local_config = input_tensor.mesh_buffer().device_local_config();
    auto& sharding_args = device_local_config.sharding_args;
    tt::tt_metal::BufferShardingArgs new_sharding_args(
        new_buffer_dist_spec, new_shard_spec_buffer, sharding_args.buffer_layout());

    tt::tt_metal::distributed::DeviceLocalBufferConfig new_device_config = {
        .page_size = new_spec.compute_page_size_bytes(),
        .buffer_type = device_local_config.buffer_type,
        .sharding_args = new_sharding_args,
        .bottom_up = device_local_config.bottom_up};

    auto view_mesh_buffer = tt::tt_metal::distributed::MeshBuffer::create(
        input_tensor.mesh_buffer().global_config(),
        new_device_config,
        input_tensor.device(),
        input_tensor.mesh_buffer().address());
    tt::tt_metal::DeviceStorage view_storage(
        view_mesh_buffer, input_tensor.device_storage().coords, input_tensor.device_storage().get_root_mesh_buffer());

    return Tensor(view_storage, new_spec, input_tensor.tensor_topology());
}

Tensor view(const Tensor& input_tensor, const Shape& new_logical_shape, const Shape& new_padded_shape) {
    tt::tt_metal::GraphTracker::instance().track_function_start(
        "Tensor::reshape", input_tensor, new_logical_shape, new_padded_shape);

    Tensor output;
    if (is_cpu_tensor(input_tensor)) {
        output = Tensor(tensor_impl::view(input_tensor.host_tensor(), new_logical_shape, new_padded_shape));
    } else {
        output = view_device(input_tensor, new_logical_shape, new_padded_shape);
    }

    output = tt::tt_metal::set_tensor_id(output);
    tt::tt_metal::GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor view(const Tensor& input_tensor, const Shape& new_shape) { return view(input_tensor, new_shape, new_shape); }

// ======================================================================================
//                                  .tensor_reshape()
// ======================================================================================
Tensor reshape(
    const Tensor& input_tensor,
    const tt::tt_metal::Shape& new_logical_shape,
    const tt::tt_metal::Shape& new_padded_shape) {
    return view(input_tensor, new_logical_shape, new_padded_shape);
}

Tensor reshape(const Tensor& input_tensor, const tt::tt_metal::Shape& new_shape) {
    return reshape(input_tensor, new_shape, new_shape);
}

Tensor to_dtype(const Tensor& input_tensor, DataType dtype) {
    GraphTracker::instance().track_function_start("tt::tt_metal::to_dtype", input_tensor, dtype);
    auto output_tensor = Tensor(tensor_impl::to_dtype(input_tensor.host_tensor(), dtype));
    GraphTracker::instance().track_function_end(output_tensor);
    return output_tensor;
}

std::string to_string(const Tensor& tensor) { return tensor_impl::to_string(tensor); }

}  // namespace tt::tt_metal
