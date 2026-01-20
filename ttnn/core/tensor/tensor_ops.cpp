// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
#include <tt-metalium/graph_tracking.hpp>

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
    TT_FATAL(
        input_tensor.storage_type() != StorageType::DEVICE, "Bring tensor to host before converting to target layout");
    Tensor output = tensor_impl::to_layout(input_tensor, target_layout);
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
    TT_ASSERT(is_cpu_tensor(input_tensor), "Tensor must be on host for padding");
    // TODO: Flip to assert when we remove use cases in python and c++
    if (input_tensor.layout() != Layout::ROW_MAJOR) {
        log_warning(
            tt::LogOp,
            "Tensor layout {} must be ROW_MAJOR for padding! Returning original tensor!",
            input_tensor.layout());
        return input_tensor;
    }

    auto output = tensor_impl::pad(input_tensor, output_padded_shape, input_tensor_start, pad_value);
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
    auto output = tensor_impl::unpad(input_tensor, output_tensor_start, output_tensor_end);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor pad_to_tile(const Tensor& input_tensor, float pad_value) {
    GraphTracker::instance().track_function_start("Tensor::pad_to_tile", input_tensor, pad_value);
    uint32_t height = input_tensor.padded_shape()[-2];
    uint32_t width = input_tensor.padded_shape()[-1];
    uint32_t padded_height = round_up(height, constants::TILE_HEIGHT);
    uint32_t padded_width = round_up(width, constants::TILE_WIDTH);

    ttsl::SmallVector<uint32_t> padded_shape;
    ttsl::SmallVector<uint32_t> input_tensor_start;

    for (auto index = 0; index < static_cast<int>(input_tensor.padded_shape().rank()) - 2; index++) {
        padded_shape.push_back(input_tensor.padded_shape()[index]);
        input_tensor_start.push_back(0);
    }

    padded_shape.push_back(padded_height);
    padded_shape.push_back(padded_width);
    input_tensor_start.push_back(0);
    input_tensor_start.push_back(0);

    auto output = input_tensor.pad(
        tt::tt_metal::Shape(std::move(padded_shape)), tt::tt_metal::Shape{std::move(input_tensor_start)}, pad_value);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor unpad_from_tile(const Tensor& input_tensor, const tt::tt_metal::Shape& output_tensor_shape) {
    GraphTracker::instance().track_function_start("Tensor::unpad_from_tile", input_tensor, output_tensor_shape);

    for (auto index = -3; index >= -static_cast<int>(input_tensor.padded_shape().rank()); index--) {
        TT_ASSERT(
            input_tensor.logical_shape()[index] == output_tensor_shape[index],
            "Input shape must match output shape apart from last 2 dims");
    }
    TT_ASSERT(
        input_tensor.padded_shape()[-2] % constants::TILE_HEIGHT == 0 &&
            input_tensor.padded_shape()[-1] % constants::TILE_WIDTH == 0,
        "Last 2 dims of input shape must be multiples of 32");
    TT_ASSERT(
        input_tensor.padded_shape()[-2] < output_tensor_shape[-2] + constants::TILE_HEIGHT &&
            input_tensor.padded_shape()[-1] < output_tensor_shape[-1] + constants::TILE_WIDTH,
        "Last 2 dims of output must be within range to have been padded to input");
    Shape output_tensor_start(ttsl::SmallVector<uint32_t>(input_tensor.padded_shape().rank(), 0));
    Shape output_tensor_end(ttsl::SmallVector<uint32_t>(input_tensor.padded_shape().rank(), 1));
    for (int index = -1; index >= -static_cast<int>(output_tensor_shape.rank()); index--) {
        output_tensor_end[index] = output_tensor_shape[index];
    }
    auto output = input_tensor.unpad(output_tensor_start, output_tensor_end);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

// ======================================================================================
//                                  .tensor_view()
// ======================================================================================
Tensor view(const Tensor& input_tensor, const Shape& new_logical_shape, const Shape& new_padded_shape) {
    tt::tt_metal::GraphTracker::instance().track_function_start(
        "Tensor::reshape", input_tensor, new_logical_shape, new_padded_shape);

    auto infer_output_memory_config = [](const MemoryConfig& input_memory_config,
                                         const tt::tt_metal::Shape& output_padded_shape) -> MemoryConfig {
        if (input_memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
            auto shard_spec = input_memory_config.shard_spec().value();
            shard_spec.shape[1] = output_padded_shape[-1];  // update output shard to match new shard width
            return MemoryConfig{input_memory_config.memory_layout(), input_memory_config.buffer_type(), shard_spec};
        }
        return input_memory_config;
    };

    // Just edit shape if shape has a 0 dimension
    if (input_tensor.logical_volume() == 0) {
        TT_FATAL(new_logical_shape.volume() == 0, "Tensor volume is 0, but shape's volume is not");
    }

    const auto output_memory_config = infer_output_memory_config(input_tensor.memory_config(), new_padded_shape);
    auto new_spec = tt::tt_metal::TensorSpec(
        new_logical_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            input_tensor.tensor_spec().page_config(),
            output_memory_config,
            new_logical_shape,
            new_padded_shape));

    // TODO (#25340): Review tensor topology logic for reshape
    auto output = std::visit(
        [&input_tensor, &new_spec, &new_logical_shape, &new_padded_shape](auto&& storage) -> Tensor {
            using T = std::decay_t<decltype(storage)>;
            const auto& tensor = input_tensor;

            if constexpr (std::is_same_v<T, tt::tt_metal::DeviceStorage>) {
                auto device_storage = std::get<tt::tt_metal::DeviceStorage>(tensor.storage());
                if (input_tensor.layout() != Layout::ROW_MAJOR) {
                    return Tensor(std::move(device_storage), new_spec, tensor.tensor_topology());
                }
                if (tensor.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
                    auto* device_buffer = device_storage.get_buffer();
                    const auto& tensor_spec = tensor.tensor_spec();
                    auto page_size_bytes = tensor_spec.compute_page_size_bytes();
                    device_buffer->set_page_size(page_size_bytes);
                    return Tensor(std::move(device_storage), new_spec, tensor.tensor_topology());
                }

                auto* device_buffer = device_storage.get_buffer();
                tt::tt_metal::ShardSpecBuffer shard_spec_buffer = device_buffer->shard_spec();

                auto shard_spec = shard_spec_buffer.tensor_shard_spec;
                auto shard_shape = shard_spec.shape;

                uint32_t mul_div;
                if (new_logical_shape[-1] == 0 || shard_shape[1] == 0) {
                    mul_div = 0;
                } else {
                    mul_div = new_logical_shape[-1] > shard_shape[1] ? (new_logical_shape[-1] / shard_shape[1])
                                                                     : (shard_shape[1] / new_logical_shape[-1]);
                }

                shard_spec.shape[0] =
                    new_logical_shape[-1] > shard_shape[1] ? shard_shape[0] / mul_div : shard_shape[0] * mul_div;
                shard_spec.shape[1] = new_logical_shape[-1];

                MemoryConfig mem_config = input_tensor.memory_config().with_shard_spec(shard_spec);

                auto upd_spec = tt::tt_metal::TensorSpec(
                    new_logical_shape,
                    TensorLayout::fromPaddedShape(
                        input_tensor.dtype(),
                        input_tensor.tensor_spec().page_config(),
                        mem_config,
                        new_logical_shape,
                        new_padded_shape));

                shard_spec_buffer.page_shape = {1, new_logical_shape[-1]};
                shard_spec_buffer.tensor2d_shape_in_pages = {
                    upd_spec.physical_shape().height() / shard_spec_buffer.page_shape[0],
                    upd_spec.physical_shape().width() / shard_spec_buffer.page_shape[1]};
                shard_spec_buffer.set_shard_spec(shard_spec);
                device_buffer->set_shard_spec(shard_spec_buffer);

                auto page_size_bytes = upd_spec.compute_page_size_bytes();
                device_buffer->set_page_size(page_size_bytes);

                return Tensor(std::move(device_storage), upd_spec, tensor.tensor_topology());

            } else if constexpr (std::is_same_v<T, tt::tt_metal::HostStorage>) {
                return Tensor(tensor.storage(), new_spec, tensor.tensor_topology());
            } else {
                static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported storage type");
            }
        },
        input_tensor.storage());
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
    auto output_tensor = tensor_impl::to_dtype(input_tensor, dtype);
    GraphTracker::instance().track_function_end(output_tensor);
    return output_tensor;
}

std::string to_string(const Tensor& tensor) { return tensor_impl::to_string(tensor); }

}  // namespace tt::tt_metal
