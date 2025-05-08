// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_ops.hpp"

#include "tt_stl/overloaded.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <memory>

#include <tt-metalium/bfloat16.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_impl_wrapper.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tracy/Tracy.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/core.hpp"

#include "cpp/ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "cpp/ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace tt::tt_metal::tensor_ops {

Tensor tensor_to_device(
    const Tensor& input_tensor, IDevice* target_device, const MemoryConfig& mem_config, QueueId cq_id) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::to_device", input_tensor, target_device, mem_config);
    if (input_tensor.storage_type() == StorageType::DEVICE) {
        TT_FATAL(input_tensor.device() == target_device, "Currently do not support moving between devices");
        GraphTracker::instance().track_function_end(input_tensor);
        return input_tensor;
    }
    auto device_tensor = tensor_impl::to_device_wrapper(input_tensor, target_device, mem_config, cq_id);
    device_tensor = tt::tt_metal::set_tensor_id(device_tensor);
    GraphTracker::instance().track_function_end(device_tensor);
    return device_tensor;
}

Tensor tensor_to_device(
    const Tensor& input_tensor, distributed::MeshDevice* mesh_device, const MemoryConfig& mem_config, QueueId cq_id) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::to_device", input_tensor, mesh_device, mem_config);
    if (input_tensor.storage_type() == StorageType::DEVICE) {
        TT_ASSERT(input_tensor.mesh_device() == mesh_device, "Currently do not support moving between devices");
        GraphTracker::instance().track_function_end(input_tensor);
        return input_tensor;
    }
    auto device_tensor = tensor_impl::to_device_mesh_tensor_wrapper(input_tensor, mesh_device, mem_config, cq_id);
    GraphTracker::instance().track_function_end(device_tensor);
    return device_tensor;
}

Tensor tensor_cpu(const Tensor& input_tensor, bool blocking, QueueId cq_id) {
    if (input_tensor.storage_type() != StorageType::DEVICE) {
        return input_tensor;
    }

    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::cpu", input_tensor, blocking);

    if (input_tensor.mesh_device_.has_value()) {
        auto output = tensor_impl::to_host_mesh_tensor_wrapper(input_tensor, blocking, cq_id);
        output = tt::tt_metal::set_tensor_id(output);
        GraphTracker::instance().track_function_end(output);
        return output;
    }

    auto workers = input_tensor.get_workers(blocking);
    if (not workers.size()) {
        // Tensor is on host and does not have a worker group.
        // Return immediately. If this is a result of .cpu() called twice,
        // tensor accessors will stall until tensor is populated.
        auto output = tt::tt_metal::set_tensor_id(input_tensor);
        GraphTracker::instance().track_function_end(output);
        return output;
    }

    TT_FATAL(workers.size() == 1, "Unexpected number of workers");
    Tensor host_tensor = tensor_impl::to_host_wrapper(input_tensor, blocking, cq_id);
    host_tensor = tt::tt_metal::set_tensor_id(host_tensor);
    GraphTracker::instance().track_function_end(host_tensor);
    return host_tensor;
}

Tensor tensor_to_layout(const Tensor& input_tensor, Layout target_layout, IDevice* worker) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::to_layout", input_tensor, target_layout, worker);
    TT_ASSERT(
        input_tensor.storage_type() != StorageType::DEVICE, "Bring tensor to host before converting to target layout");
    Tensor output = tensor_impl::to_layout_wrapper(input_tensor, target_layout);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor tensor_to_layout(const Tensor& input_tensor, Layout target_layout, distributed::MeshDevice* mesh_device) {
    ZoneScoped;
    TT_FATAL(
        is_cpu_tensor(input_tensor) || is_multi_device_host_tensor(input_tensor),
        "to(layout) must be called on host tensors with MULTI_DEVICE_HOST_STORAGE when multiple "
        "workers "
        "are specified");

    GraphTracker::instance().track_function_start("Tensor::to_layout", input_tensor, target_layout, mesh_device);
    if (mesh_device) {
        // Mesh Device provided - have a handle to the thread-pool
        Tensor tensor_modified_layout = std::visit(
            tt::stl::overloaded{
                [&](const HostStorage& s) { return tensor_impl::to_layout_wrapper(input_tensor, target_layout); },
                [&](const MultiDeviceHostStorage& s) {
                    std::vector<Tensor> shards(s.buffers.size());
                    for (std::size_t shard_idx = 0; shard_idx < s.buffers.size(); ++shard_idx) {
                        // Multi-Thread Host tilization of shards.
                        mesh_device->enqueue_to_thread_pool([shard_idx, &s, &shards, target_layout]() {
                            ZoneScopedN("HostTilize");
                            Tensor shard(s.buffers[shard_idx], s.specs[shard_idx]);
                            shards[shard_idx] = tensor_impl::to_layout_wrapper(shard, target_layout);
                        });
                    }
                    mesh_device->wait_for_thread_pool();
                    return ttnn::distributed::aggregate_as_tensor(shards, s.strategy);
                },
                [&](const DeviceStorage& s) -> Tensor { TT_THROW("Unexpected storage type"); },
            },
            input_tensor.get_storage());

        tensor_modified_layout = tt::tt_metal::set_tensor_id(tensor_modified_layout);
        GraphTracker::instance().track_function_end(tensor_modified_layout);
        return tensor_modified_layout;
    }

    // Running without worker threads (non-async)
    auto output = tensor_impl::to_layout_wrapper(input_tensor, target_layout);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

void tensor_print(const Tensor& input_tensor) {
    GraphTracker::instance().track_function_start("Tensor::print", input_tensor);
    std::cout << input_tensor.write_to_string() << std::endl;
    GraphTracker::instance().track_function_end();
}

Tensor tensor_pad(
    const Tensor& input_tensor,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    float pad_value) {
    ZoneScoped;
    GraphTracker::instance().track_function_start(
        "Tensor::pad", input_tensor, output_padded_shape, input_tensor_start, pad_value);
    TT_ASSERT(
        is_cpu_tensor(input_tensor) || is_multi_device_host_tensor(input_tensor), "Tensor must be on host for padding");
    // TODO: Flip to assert when we remove use cases in python and c++
    if (input_tensor.get_layout() != Layout::ROW_MAJOR) {
        log_warning(
            tt::LogOp,
            "Tensor layout {} must be ROW_MAJOR for padding! Returning original tensor!",
            input_tensor.get_layout());
        return input_tensor;
    }

    auto output = tensor_impl::pad_wrapper(input_tensor, output_padded_shape, input_tensor_start, pad_value);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor tensor_unpad(
    const Tensor& input_tensor, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) {
    ZoneScoped;
    GraphTracker::instance().track_function_start(
        "Tensor::unpad", input_tensor, output_tensor_start, output_tensor_end);
    TT_ASSERT(input_tensor.get_layout() == Layout::ROW_MAJOR && "Tensor layout must be ROW_MAJOR for unpadding");
    auto output = tensor_impl::unpad_wrapper(input_tensor, output_tensor_start, output_tensor_end);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor tensor_pad_to_tile(const Tensor& input_tensor, float pad_value) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::pad_to_tile", input_tensor, pad_value);
    uint32_t height = input_tensor.get_padded_shape()[-2];
    uint32_t width = input_tensor.get_padded_shape()[-1];
    uint32_t padded_height = round_up(height, constants::TILE_HEIGHT);
    uint32_t padded_width = round_up(width, constants::TILE_WIDTH);

    ttnn::SmallVector<uint32_t> padded_shape;
    ttnn::SmallVector<uint32_t> input_tensor_start;

    for (auto index = 0; index < static_cast<int>(input_tensor.get_padded_shape().rank()) - 2; index++) {
        padded_shape.push_back(input_tensor.get_padded_shape()[index]);
        input_tensor_start.push_back(0);
    }

    padded_shape.push_back(padded_height);
    padded_shape.push_back(padded_width);
    input_tensor_start.push_back(0);
    input_tensor_start.push_back(0);

    auto output =
        input_tensor.pad(ttnn::Shape(std::move(padded_shape)), ttnn::Shape{std::move(input_tensor_start)}, pad_value);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor tensor_unpad_from_tile(const Tensor& input_tensor, const ttnn::Shape& output_tensor_shape) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::unpad_from_tile", input_tensor, output_tensor_shape);

    for (auto index = -3; index >= -static_cast<int>(input_tensor.get_padded_shape().rank()); index--) {
        TT_ASSERT(
            input_tensor.get_logical_shape()[index] == output_tensor_shape[index],
            "Input shape must match output shape apart from last 2 dims");
    }
    TT_ASSERT(
        input_tensor.get_padded_shape()[-2] % constants::TILE_HEIGHT == 0 &&
            input_tensor.get_padded_shape()[-1] % constants::TILE_WIDTH == 0,
        "Last 2 dims of input shape must be multiples of 32");
    TT_ASSERT(
        input_tensor.get_padded_shape()[-2] < output_tensor_shape[-2] + constants::TILE_HEIGHT &&
            input_tensor.get_padded_shape()[-1] < output_tensor_shape[-1] + constants::TILE_WIDTH,
        "Last 2 dims of output must be within range to have been padded to input");
    Shape output_tensor_start(ttnn::SmallVector<uint32_t>(input_tensor.padded_shape().rank(), 0));
    Shape output_tensor_end(ttnn::SmallVector<uint32_t>(input_tensor.padded_shape().rank(), 1));
    for (int index = -1; index >= -static_cast<int>(output_tensor_shape.rank()); index--) {
        output_tensor_end[index] = output_tensor_shape[index];
    }
    auto output = input_tensor.unpad(output_tensor_start, output_tensor_end);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor tensor_reshape(
    const Tensor& input_tensor, const ttnn::Shape& new_logical_shape, const ttnn::Shape& new_padded_shape) {
    return ttnn::reshape(input_tensor, new_logical_shape, new_padded_shape);
}

Tensor tensor_reshape(const Tensor& input_tensor, const ttnn::Shape& new_shape) {
    return ttnn::reshape(input_tensor, new_shape);
}

}  // namespace tt::tt_metal::tensor_ops
