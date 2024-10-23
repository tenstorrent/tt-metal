// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_ops.hpp"

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
#include "tt_metal/graph/graph_tracking.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/core.hpp"


namespace{
    inline void SynchronizeWorkerThreads(const std::vector<Device*>& workers) {
        // Push empty work to threads and ensure its been picked up
        for (auto target_device : workers) {
            target_device->work_executor.push_work([](){});
        }
        // Block until work has been picked up, to flush the queue
        for (auto target_device : workers) {
            while(not target_device->work_executor.worker_queue.empty());
        }
    }
}


namespace tt::tt_metal::tensor_ops {

Tensor tensor_to(const Tensor& input_tensor, Device* target_device, const MemoryConfig& mem_config) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::to", input_tensor, target_device, mem_config);
    // Tensor can be using borrowed storage. If so, when running in async mode, copy this tensor to owned storage.
    Tensor async_safe_tensor = copy_borrowed_tensor_in_async_mode(target_device, input_tensor);
    // Populate device storage outside of thread, so that downstream
    // functions running in main can get storage type without blocking
    Tensor device_tensor({target_device});
    // Record main thread ref count for tensors before pushing to queue.
    uint32_t device_tensor_ref_count = device_tensor.tensor_attributes->record_main_thread_ref_count();
    uint32_t original_tensor_ref_count = async_safe_tensor.tensor_attributes->record_main_thread_ref_count();
    target_device->push_work([async_safe_tensor, device_tensor, mem_config, target_device]() mutable {
        if (async_safe_tensor.storage_type() == StorageType::DEVICE) {
            TT_ASSERT(async_safe_tensor.device() == target_device && "Currently do not support moving between devices");
            device_tensor.populate_buffers_and_metadata(async_safe_tensor);
        } else {
            tensor_impl::validate_on_device_dtype_and_layout(
                target_device,
                async_safe_tensor.get_padded_shape(),
                async_safe_tensor.get_dtype(),
                async_safe_tensor.get_layout());
            auto local_tensor =
                tensor_impl::to_device_wrapper(async_safe_tensor, target_device, mem_config, std::nullopt);
            // Populate device tensor
            device_tensor.populate_buffers_and_metadata(local_tensor);
        }
    });
    // Update main thread ref count for tensors after pushing to queue (update original tensor and returned tensor,
    // since both can be on device).
    device_tensor.tensor_attributes->update_main_thread_ref_count(device_tensor.workers.at(0), device_tensor_ref_count);
    async_safe_tensor.tensor_attributes->update_main_thread_ref_count(
        device_tensor.workers.at(0), original_tensor_ref_count);
    device_tensor = tt::tt_metal::set_tensor_id(device_tensor);
    GraphTracker::instance().track_function_end(device_tensor);
    return device_tensor;
}

Tensor tensor_to(const Tensor& input_tensor, const std::vector<Device*>& workers, const MemoryConfig& mem_config) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::to", input_tensor, workers, mem_config);
    TT_FATAL(
        validate_worker_modes(workers), "All device threads/workers must be running in the same mode (ASYNC or SYNC)");
    Tensor device_tensor = Tensor(workers);
    uint32_t device_tensor_ref_count = device_tensor.tensor_attributes->record_main_thread_ref_count();
    uint32_t original_tensor_ref_count = input_tensor.tensor_attributes->record_main_thread_ref_count();
    uint32_t num_workers = workers.size();
    for (int worker_index = 0; worker_index < workers.size(); ++worker_index) {
        auto& worker = workers[worker_index];
        worker->push_work([worker, input_tensor, device_tensor, mem_config, num_workers, worker_index]() mutable {
            auto shard = get_shard_for_device(input_tensor, worker, worker_index);
            if (shard.storage_type() == StorageType::OWNED) {
                shard = tensor_impl::to_device_wrapper(shard, worker, mem_config, std::nullopt);
            }
            insert_buffer_and_shape_for_device(worker, shard, device_tensor, worker_index);
            uint32_t num_workers_completed = (device_tensor.tensor_attributes->num_workers_completed)++;
            if (not num_workers_completed) {
                device_tensor.set_shape(input_tensor.get_shape());
                device_tensor.set_dtype(input_tensor.get_dtype());
                device_tensor.set_layout(input_tensor.get_layout());
                device_tensor.set_tile(input_tensor.get_tile());
                device_tensor.tensor_attributes->metadata_populated = true;
            }
        });
    }
    device_tensor.tensor_attributes->update_main_thread_ref_count(workers.at(0), device_tensor_ref_count);
    input_tensor.tensor_attributes->update_main_thread_ref_count(workers.at(0), original_tensor_ref_count);
    device_tensor = tt::tt_metal::set_tensor_id(device_tensor);
    GraphTracker::instance().track_function_end(device_tensor);
    return device_tensor;
}

Tensor tensor_cpu(const Tensor& input_tensor, bool blocking, uint8_t cq_id) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::cpu", input_tensor, blocking);
    auto workers = input_tensor.get_workers(blocking);
    if (not workers.size()) {
        // Tensor is on host and does not have a worker group.
        // Return immediately. If this is a result of .cpu() called twice,
        // tensor accessors will stall until tensor is populated.
        auto output = tt::tt_metal::set_tensor_id(input_tensor);
        GraphTracker::instance().track_function_end(output);
        return output;
    }
    TT_FATAL(
        validate_worker_modes(workers), "All device threads/workers must be running in the same mode (ASYNC or SYNC)");
    Tensor host_tensor({}, workers.size());
    uint32_t original_tensor_ref_count = input_tensor.tensor_attributes->record_main_thread_ref_count();
    for (int worker_index = 0; worker_index < workers.size(); worker_index++) {
        auto target_device = workers[worker_index];
        target_device->push_work([host_tensor, blocking, target_device, input_tensor, workers, worker_index, cq_id]() mutable {
            TT_ASSERT(
                input_tensor.storage_type() == StorageType::DEVICE or input_tensor.storage_type() == StorageType::MULTI_DEVICE,
                "Can only use worker queue for cpu call if tensor is on device.");
            auto shard = get_shard_for_device(input_tensor, target_device);
            shard = tensor_impl::to_host_wrapper(shard, blocking, cq_id);
            insert_buffer_and_shape_for_device(target_device, shard, host_tensor, worker_index);
            uint32_t num_workers_completed = (host_tensor.tensor_attributes->num_workers_completed)++;
            if (not num_workers_completed) {
                host_tensor.set_shape(input_tensor.get_shape());
                host_tensor.set_dtype(input_tensor.get_dtype());
                host_tensor.set_layout(input_tensor.get_layout());
                host_tensor.set_tile(input_tensor.get_tile());
                host_tensor.tensor_attributes->metadata_populated = true;
            }
        });
    }

    if (blocking) {
        SynchronizeWorkerThreads(workers);
    }
    // Update main_thread_ref_count for tensor after pushing to queue.
    input_tensor.tensor_attributes->update_main_thread_ref_count(workers.at(0), original_tensor_ref_count);
    host_tensor = tt::tt_metal::set_tensor_id(host_tensor);
    GraphTracker::instance().track_function_end(host_tensor);
    return host_tensor;
}

Tensor tensor_cpu_sharded(const Tensor& input_tensor) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::cpu_sharded", input_tensor);
    auto output = tensor_impl::to_host_sharded_wrapper(input_tensor);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor tensor_to(const Tensor& input_tensor, Layout target_layout, Device* worker) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::to", input_tensor, target_layout, worker);
    // Only push layout conversion to worker if running in async mode
    if (worker and worker->get_worker_mode() == WorkExecutorMode::ASYNCHRONOUS) {
        // Tensor can be using borrowed storage. If so, when running in async mode, copy this tensor to owned storage.
        Tensor async_safe_tensor = copy_borrowed_tensor_in_async_mode(worker, input_tensor);
        Tensor tensor_modified_layout = Tensor({}, 1);
        worker->push_work([async_safe_tensor, tensor_modified_layout, target_layout]() mutable {
            TT_ASSERT(
                async_safe_tensor.storage_type() == StorageType::OWNED or
                async_safe_tensor.storage_type() == StorageType::BORROWED &&
                    "to(layout) must be called on host tensors with a single buffer when a single worker is specified");
            auto local_tensor = tensor_impl::to_layout_wrapper(async_safe_tensor, target_layout);
            // Populate modified layout tensor
            tensor_modified_layout.populate_buffers_and_metadata(local_tensor);
        });
        tensor_modified_layout = tt::tt_metal::set_tensor_id(tensor_modified_layout);
        GraphTracker::instance().track_function_end(tensor_modified_layout);
        return tensor_modified_layout;
    }
    // Running without worker threads (non-async)
    TT_ASSERT(
        input_tensor.storage_type() != StorageType::DEVICE or
        input_tensor.storage_type() != StorageType::MULTI_DEVICE && "Bring tensor to host before converting to target layout");
    auto output = tensor_impl::to_layout_wrapper(input_tensor, target_layout);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor tensor_to(const Tensor& input_tensor, Layout target_layout, distributed::MeshDevice* mesh_device) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::to", input_tensor, target_layout, mesh_device);
    if (mesh_device) {
        auto workers = ttnn::distributed::distribute_tensor_to_mesh(input_tensor, *mesh_device);
        TT_FATAL(
            validate_worker_modes(workers),
            "All device threads/workers must be running in the same mode (ASYNC or SYNC)");

        std::optional<DistributedTensorConfig> distributed_config = std::nullopt;
        if (std::holds_alternative<MultiDeviceHostStorage>(input_tensor.get_storage())) {
            auto& host_storage = std::get<MultiDeviceHostStorage>(input_tensor.get_storage());
            distributed_config = host_storage.strategy;
        }
        Tensor tensor_modified_layout = Tensor({}, workers.size(), distributed_config);
        for (int worker_index = 0; worker_index < workers.size(); ++worker_index) {
            auto& worker = workers[worker_index];
            worker->push_work([input_tensor, tensor_modified_layout, target_layout, worker, worker_index]() mutable {
                TT_ASSERT(
                    input_tensor.storage_type() == StorageType::OWNED || input_tensor.storage_type() == StorageType::BORROWED ||
                    input_tensor.storage_type() == StorageType::MULTI_DEVICE_HOST &&
                        "to(layout) must be called on host tensors with MULTI_DEVICE_HOST_STORAGE when multiple "
                        "workers "
                        "are specified");
                ;
                auto shard = get_shard_for_device(input_tensor, worker, worker_index);
                shard = tensor_impl::to_layout_wrapper(shard, target_layout);
                insert_buffer_and_shape_for_device(worker, shard, tensor_modified_layout, worker_index);
                uint32_t num_workers_completed = (tensor_modified_layout.tensor_attributes->num_workers_completed)++;
                if (not num_workers_completed) {
                    tensor_modified_layout.set_shape(input_tensor.get_shape());
                    tensor_modified_layout.set_dtype(input_tensor.get_dtype());
                    tensor_modified_layout.set_layout(target_layout);
                    tensor_modified_layout.set_tile(input_tensor.get_tile());
                    tensor_modified_layout.tensor_attributes->metadata_populated = true;
                };
            });
        }
        tensor_modified_layout = tt::tt_metal::set_tensor_id(tensor_modified_layout);
        GraphTracker::instance().track_function_end(tensor_modified_layout);
        return tensor_modified_layout;
    }
    // Running without worker threads (non-async)
    TT_ASSERT(
        input_tensor.storage_type() != StorageType::DEVICE or
        input_tensor.storage_type() != StorageType::MULTI_DEVICE && "Bring tensor to host before converting to target layout");
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

Tensor tensor_pad(const Tensor& input_tensor, const tt::tt_metal::LegacyShape& output_tensor_shape, const ttnn::SimpleShape& input_tensor_start, float pad_value) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::pad", input_tensor, output_tensor_shape, input_tensor_start, pad_value);
    TT_ASSERT(
        input_tensor.storage_type() == StorageType::OWNED or input_tensor.storage_type() == StorageType::MULTI_DEVICE_HOST or
        input_tensor.storage_type() == StorageType::BORROWED && "Tensor must be on host for padding");
    TT_ASSERT(input_tensor.get_layout() == Layout::ROW_MAJOR && "Tensor layout must be ROW_MAJOR for padding");

    auto input_shape = input_tensor.get_legacy_shape();
    auto dimensions_pads = std::vector<Padding::PadDimension>();
    for (auto index = 0; index < input_shape.rank(); index++) {
        auto front = input_tensor_start[index];
        auto back = output_tensor_shape[index] - (input_tensor_start[index] + input_shape[index]);
        dimensions_pads.push_back(Padding::PadDimension{.front = front, .back = back});
    }
    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
    auto output_shape_with_padding = tt::tt_metal::LegacyShape(output_tensor_shape, padding);

    auto output = tensor_impl::pad_wrapper(input_tensor, output_shape_with_padding, input_tensor_start, pad_value);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor tensor_unpad(const Tensor& input_tensor, const ttnn::SimpleShape& output_tensor_start, const ttnn::SimpleShape& output_tensor_end) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::unpad", input_tensor, output_tensor_start, output_tensor_end);
    TT_ASSERT(input_tensor.get_layout() == Layout::ROW_MAJOR && "Tensor layout must be ROW_MAJOR for unpadding");
    auto output = tensor_impl::unpad_wrapper(input_tensor, output_tensor_start, output_tensor_end);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor tensor_pad_to_tile(const Tensor& input_tensor, float pad_value)  {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::pad_to_tile", input_tensor, pad_value);
    uint32_t height = input_tensor.get_legacy_shape()[-2];
    uint32_t width = input_tensor.get_legacy_shape()[-1];
    uint32_t padded_height = round_up(height, constants::TILE_HEIGHT);
    uint32_t padded_width = round_up(width, constants::TILE_WIDTH);

    std::vector<uint32_t> shape;
    std::vector<uint32_t> padded_shape;
    std::vector<uint32_t> input_tensor_start;

    for (auto index = 0; index < input_tensor.get_legacy_shape().rank() - 2; index++) {
        shape.push_back(input_tensor.get_legacy_shape().without_padding()[index]);
        padded_shape.push_back(input_tensor.get_legacy_shape()[index]);
        input_tensor_start.push_back(0);
    }

    shape.push_back(height);
    shape.push_back(width);
    padded_shape.push_back(padded_height);
    padded_shape.push_back(padded_width);
    input_tensor_start.push_back(0);
    input_tensor_start.push_back(0);

    auto output = input_tensor.pad(tt::tt_metal::LegacyShape(shape, padded_shape), ttnn::SimpleShape{std::move(input_tensor_start)}, pad_value);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor tensor_unpad_from_tile(const Tensor& input_tensor, const ttnn::SimpleShape& output_tensor_shape) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::unpad_from_tile", input_tensor, output_tensor_shape);

    for (auto index = 0; index < input_tensor.get_legacy_shape().rank() - 2; index++) {
        TT_ASSERT(
            input_tensor.get_legacy_shape().without_padding()[index] == output_tensor_shape[index],
            "Input shape must match output shape apart from last 2 dims");
    }
    TT_ASSERT(
        input_tensor.get_legacy_shape()[-2] % constants::TILE_HEIGHT == 0 && input_tensor.get_legacy_shape()[-1] % constants::TILE_WIDTH == 0,
        "Last 2 dims of input shape must be multiples of 32");
    TT_ASSERT(
        input_tensor.get_legacy_shape()[-2] - constants::TILE_HEIGHT < output_tensor_shape[-2] &&
            input_tensor.get_legacy_shape()[-1] - constants::TILE_WIDTH < output_tensor_shape[-1],
        "Last 2 dims of output must be within range to have been padded to input");
    std::vector<uint32_t> output_tensor_start{};
    std::vector<uint32_t> output_tensor_end{};
    for (auto index = 0; index < input_tensor.get_legacy_shape().rank(); index++) {
        output_tensor_start.push_back(0);
        output_tensor_end.push_back(output_tensor_shape[index]);
    }
    auto output = input_tensor.unpad(ttnn::SimpleShape(std::move(output_tensor_start)), ttnn::SimpleShape(std::move(output_tensor_end)));
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor tensor_reshape(const Tensor& input_tensor, const ttnn::Shape& new_shape) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::reshape", input_tensor, new_shape);
    const auto& new_padded_shape = new_shape.padded_shape();
    TT_ASSERT(
        input_tensor.volume() == new_padded_shape.volume(),
        "{} != {}",
        input_tensor.volume(),
        new_padded_shape.volume());
    if (input_tensor.get_layout() == Layout::TILE) {
        TT_ASSERT(
            new_padded_shape[-2] % constants::TILE_HEIGHT == 0 && new_padded_shape[-1] % constants::TILE_WIDTH == 0 &&
            "Expected a multiple of 32 for H, W (or -1 evaluating to such) in Tensor::reshape()!");
    }
    auto output = std::visit(
        [&input_tensor, &new_shape](auto&& storage) -> Tensor {
            using T = std::decay_t<decltype(storage)>;
            const auto& tensor = input_tensor;
            if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                auto updated_storage = std::get<T>(tensor.get_storage());
                for (int i = 0; i < updated_storage.shapes.size(); i++) {
                    updated_storage.shapes[i] = new_shape;
                }
                return Tensor(updated_storage, new_shape, tensor.get_dtype(), tensor.get_layout());
            }
            if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                MultiDeviceStorage updated_storage = std::get<T>(tensor.get_storage());
                std::unordered_map<int, ttnn::Shape> new_shapes;

                for (auto device_id : updated_storage.ordered_device_ids) {
                    new_shapes.insert({device_id, new_shape});
                }
                updated_storage.shapes = new_shapes;
                return Tensor(updated_storage, new_shape, tensor.get_dtype(), tensor.get_layout());
            }
            if constexpr (std::is_same_v<T, DeviceStorage>) {
                if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
                    if (tensor.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
                        DeviceStorage device_storage = std::get<T>(tensor.get_storage());
                        DeviceBuffer device_buffer = device_storage.get_buffer();
                        device_buffer->set_page_size(new_shape[-1] * tensor.element_size());
                        device_storage.insert_buffer(device_buffer);
                        return Tensor(device_storage, new_shape, tensor.get_dtype(), tensor.get_layout());
                    } else {
                        DeviceStorage device_storage = std::get<T>(tensor.get_storage());
                        DeviceBuffer device_buffer = device_storage.get_buffer();
                        ShardSpecBuffer shard_spec_buffer = device_buffer->shard_spec();

                        auto shard_spec = shard_spec_buffer.tensor_shard_spec;
                        auto shard_shape = shard_spec.shape;

                        uint32_t mul_div = new_shape[-1] > shard_shape[1] ?
                                        (new_shape[-1] / shard_shape[1]) :
                                        (shard_shape[1] / new_shape[-1]);
                        shard_spec.shape[0] = new_shape[-1] > shard_shape[1] ? shard_shape[0] / mul_div : shard_shape[0] * mul_div;
                        shard_spec.shape[1] = new_shape[-1];

                        shard_spec_buffer.page_shape = {1, new_shape[-1]};
                        shard_spec_buffer.tensor2d_shape = {shard_spec.shape[0], 1};
                        shard_spec_buffer.set_shard_spec(shard_spec);

                        device_buffer->set_shard_spec(shard_spec_buffer);
                        device_storage.insert_buffer(device_buffer);

                        return Tensor(device_storage, new_shape, tensor.get_dtype(), tensor.get_layout());
                    }
                } else {
                    return Tensor(tensor.get_storage(), new_shape, tensor.get_dtype(), tensor.get_layout());
                }
            } else {
                return Tensor(tensor.get_storage(), new_shape, tensor.get_dtype(), tensor.get_layout());
            }
        },
        input_tensor.get_storage());
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor tensor_reshape(const Tensor& input_tensor, const ttnn::SimpleShape& new_shape) {
    return tensor_reshape(input_tensor, ttnn::Shape(new_shape.as_vector()));
}

}
