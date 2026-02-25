// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include "tensor/tensor_ops.hpp"

#include "tt-metalium/experimental/tensor/host_tensor.hpp"
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
    return Tensor(HostStorage(std::move(distributed_host_buffer)), tensor_spec, TensorTopology{});
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
    output = Tensor(tensor_impl::allocate_tensor_on_device(tensor_spec, mesh_device));
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

MeshTensor create_device_metal_tensor(const TensorSpec& tensor_spec, distributed::MeshDevice* mesh_device) {
    MeshTensor output = tensor_impl::allocate_tensor_on_device(tensor_spec, mesh_device);
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

    TT_FATAL(mesh_device != nullptr, "Need target device in order to move tensor to device!");
    auto cq_id_int = tt::tt_metal::raw_optional(cq_id);
    distributed::MeshCommandQueue& mesh_cq = mesh_device->mesh_command_queue(cq_id_int);
    auto device_tensor = Tensor(tensor_impl::to_device(mesh_cq, input_tensor.host_tensor(), mem_config));

    GraphTracker::instance().track_function_end(device_tensor);
    return device_tensor;
}

void copy_to_device(const Tensor& host_tensor, Tensor& device_tensor, std::optional<tt::tt_metal::QueueId> cq_id) {
    TT_FATAL(host_tensor.storage_type() == StorageType::HOST, "Source tensor is not on host.");
    TT_FATAL(device_tensor.storage_type() == StorageType::DEVICE, "Destination tensor is not on device.");

    GraphTracker::instance().track_function_start("tt::tt_metal::copy_to_device", host_tensor, device_tensor, cq_id);

    auto cq_id_int = tt::tt_metal::raw_optional(cq_id);
    distributed::MeshCommandQueue& mesh_cq = device_tensor.device()->mesh_command_queue(cq_id_int);
    tensor_impl::copy_to_device(mesh_cq, host_tensor.host_tensor(), device_tensor.device_tensor());

    device_tensor = tt::tt_metal::set_tensor_id(device_tensor);
    GraphTracker::instance().track_function_end(device_tensor);
}

void copy_to_device(
    distributed::MeshCommandQueue& queue,
    const std::byte* src,
    Tensor& device_tensor,
    const std::optional<BufferRegion>& region) {
    TT_FATAL(
        device_tensor.storage_type() == StorageType::DEVICE, "copy_to_device: destination tensor must be on device");
    GraphTracker::instance().track_function_start("tt::tt_metal::copy_to_device", queue, src, device_tensor, region);
    tensor_impl::copy_to_device(queue, src, device_tensor.device_tensor(), region);
    GraphTracker::instance().track_function_end(device_tensor);
}

void copy_to_host(
    distributed::MeshCommandQueue& queue,
    const Tensor& device_tensor,
    std::byte* dst,
    const std::optional<BufferRegion>& region,
    bool blocking) {
    TT_FATAL(device_tensor.storage_type() == StorageType::DEVICE, "copy_to_host: source tensor must be on device");
    GraphTracker::instance().track_function_start(
        "tt::tt_metal::copy_to_host", queue, device_tensor, dst, region, blocking);
    tensor_impl::copy_to_host(queue, device_tensor.device_tensor(), dst, region, blocking);
    GraphTracker::instance().track_function_end(device_tensor);
}

void copy_to_host(const Tensor& device_tensor, Tensor& host_tensor, bool blocking, std::optional<QueueId> cq_id) {
    GraphTracker::instance().track_function_start(
        "tt::tt_metal::copy_to_host", device_tensor, host_tensor, blocking, cq_id);
    TT_FATAL(device_tensor.storage_type() == StorageType::DEVICE, "Source tensor is not on device.");
    TT_FATAL(host_tensor.storage_type() == StorageType::HOST, "Destination tensor is not on host.");

    auto cq_id_int = tt::tt_metal::raw_optional(cq_id);
    distributed::MeshCommandQueue& mesh_cq = device_tensor.device()->mesh_command_queue(cq_id_int);

    tensor_impl::copy_to_host(mesh_cq, device_tensor.device_tensor(), host_tensor.host_tensor(), blocking);
    GraphTracker::instance().track_function_end(host_tensor);
}

Tensor cpu(const Tensor& input_tensor, bool blocking, std::optional<QueueId> cq_id) {
    if (input_tensor.storage_type() != StorageType::DEVICE) {
        return input_tensor;
    }

    GraphTracker::instance().track_function_start("Tensor::cpu", input_tensor, blocking);

    auto cq_id_int = tt::tt_metal::raw_optional(cq_id);
    distributed::MeshCommandQueue& mesh_cq = input_tensor.device()->mesh_command_queue(cq_id_int);
    auto output = Tensor(tensor_impl::to_host(mesh_cq, input_tensor.device_tensor(), blocking));

    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor to_layout(const Tensor& input_tensor, Layout target_layout) {
    GraphTracker::instance().track_function_start("Tensor::to_layout", input_tensor, target_layout);
    TT_FATAL(
        input_tensor.storage_type() != StorageType::DEVICE, "Bring tensor to host before converting to target layout");
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
    TT_FATAL(is_cpu_tensor(input_tensor), "Tensor must be on host for padding");

    GraphTracker::instance().track_function_start(
        "Tensor::pad", input_tensor, output_padded_shape, input_tensor_start, pad_value);
    // TODO: Flip to assert when we remove use cases in python and c++
    if (input_tensor.layout() != Layout::ROW_MAJOR) {
        log_warning(
            tt::LogOp,
            "Tensor layout {} must be ROW_MAJOR for padding! Returning original tensor!",
            input_tensor.layout());
        return input_tensor;
    }

    Tensor output =
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
    TT_ASSERT(input_tensor.layout() == Layout::ROW_MAJOR && "Tensor layout must be ROW_MAJOR for unpadding");
    Tensor output = Tensor(tensor_impl::unpad(input_tensor.host_tensor(), output_tensor_start, output_tensor_end));
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor pad_to_tile(const Tensor& input_tensor, float pad_value) {
    GraphTracker::instance().track_function_start("Tensor::pad_to_tile", input_tensor, pad_value);
    Tensor output = Tensor(tensor_impl::pad_to_tile(input_tensor.host_tensor(), pad_value));
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor unpad_from_tile(const Tensor& input_tensor, const tt::tt_metal::Shape& output_tensor_shape) {
    GraphTracker::instance().track_function_start("Tensor::unpad_from_tile", input_tensor, output_tensor_shape);
    Tensor output = Tensor(tensor_impl::unpad_from_tile(input_tensor.host_tensor(), output_tensor_shape));
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

// ======================================================================================
//                                  .tensor_view()
// ======================================================================================

Tensor view(const Tensor& input_tensor, const Shape& new_logical_shape, const Shape& new_padded_shape) {
    GraphTracker::instance().track_function_start("Tensor::reshape", input_tensor, new_logical_shape, new_padded_shape);

    Tensor output;
    if (is_device_tensor(input_tensor)) {
        output = Tensor(tensor_impl::view(input_tensor.device_tensor(), new_logical_shape, new_padded_shape));
    } else {
        output = Tensor(tensor_impl::view(input_tensor.host_tensor(), new_logical_shape, new_padded_shape));
    }

    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
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
    Tensor output_tensor = Tensor(tensor_impl::to_dtype(input_tensor.host_tensor(), dtype));
    GraphTracker::instance().track_function_end(output_tensor);
    return output_tensor;
}

std::string to_string(const Tensor& tensor) { return tensor_impl::to_string(tensor); }

}  // namespace tt::tt_metal
