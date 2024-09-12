// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_utils.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "tt_metal/tt_stl/reflection.hpp"
#include <type_traits>
#include <optional>

namespace tt::tt_metal::operation {

template <typename T>
struct is_optional : std::false_type {};

template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};

template <typename T>
constexpr bool is_optional_v = is_optional<T>::value;

template <class T>
Tensor* get_tensor(T& maybe_tensor) {
    Tensor* output_tensor = nullptr;
    if constexpr (is_optional_v<T>) {
        if (maybe_tensor.has_value())
            output_tensor = &maybe_tensor.value();
    } else {
        output_tensor = &maybe_tensor;
    }
    return output_tensor;
}

void check_output(auto& output_tensors, const std::vector<Device *>& workers) {
    for (auto& output_tensor_like : output_tensors) {
        auto output_tensor = get_tensor(output_tensor_like);
        if (!output_tensor) {
            continue;
        }
        TT_FATAL(
            output_tensor->workers.size(),
            "Worker threads must be specified for outputs populated by launch_op. This API can only be used for "
            "creating output tensors on device.");
        TT_FATAL(
            output_tensor->workers == workers,
            "Worker threads must be consistent across all outputs populated by launch_op.");
    }
}

auto& get_workers(auto& output_tensors) {
    for (auto& output_tensor_like : output_tensors) {
        Tensor* output_tensor = get_tensor(output_tensor_like);
        if (output_tensor) {
            return output_tensor->workers;
        }
    }
    TT_THROW("Workers not found in output tensors.");
}


template<class Callable, class OutputType>
void launch_op(
    Callable&& op_func,
    const Tensors input_tensors,
    OutputType& output_tensors,
    const OptionalConstTensors optional_input_tensors,
    const OptionalTensors optional_output_tensors,
    bool enable_autoformat_device) {
    // Send host side op compile and run to the worker queue
    // Assert to ensure that worker threads are specified.
    ZoneScopedN("LaunchOp");
    auto& workers = get_workers(output_tensors);
    std::size_t workers_size = workers.size();
    if (not enable_autoformat_device and workers.empty() or not workers.at(0)->in_main_thread()) {
        // Run in main thread or immediately in worker thread
        output_tensors = op_func(input_tensors, optional_input_tensors, optional_output_tensors);
        return;
    }
    check_output(output_tensors, workers);
    validate_worker_modes(workers);
    // Record ref counts for all tensors before pushing to worker queue.
    std::vector<uint32_t> input_tensor_ref_count(input_tensors.size());
    std::vector<uint32_t> optional_input_tensor_ref_count(optional_input_tensors.size());
    std::vector<uint32_t> output_tensor_ref_count(output_tensors.size());
    std::vector<uint32_t> optional_output_tensor_ref_count(optional_output_tensors.size());

    std::vector<Tensor> async_safe_input_tensors(input_tensors.size());
    std::vector<std::optional<const Tensor>> async_safe_optional_input_tensors = {};
    std::unordered_set<uint32_t> cross_worker_input_tensor_idx = {};
    std::unordered_set<uint32_t> cross_worker_optional_input_tensor_idx = {};
    // When running on a single device, input tensors can be using borrowed storage. If so, when running in async mode,
    // copy borrowed tensors to owned storage.
    TT_FATAL(workers.size(), "At least one worker should exist");
    for (int i = 0; i < input_tensors.size(); i++) {
        async_safe_input_tensors[i] = copy_borrowed_tensor_in_async_mode(workers[0], input_tensors[i]);
        input_tensor_ref_count[i] = async_safe_input_tensors[i].tensor_attributes->record_main_thread_ref_count();
    }
    for (int i = 0; i < optional_input_tensors.size(); i++) {
        if (optional_input_tensors[i].has_value()) {
            async_safe_optional_input_tensors.push_back(
                copy_borrowed_tensor_in_async_mode(workers[0], optional_input_tensors[i].value()));
            optional_input_tensor_ref_count[i] =
                async_safe_optional_input_tensors[i].value().tensor_attributes->record_main_thread_ref_count();
        } else {
            async_safe_optional_input_tensors.push_back(std::nullopt);
            optional_input_tensor_ref_count[i] = 0;
        }
    }
    for (int i = 0; i < output_tensors.size(); i++) {
        auto output_tensor = get_tensor(output_tensors[i]);
        if (output_tensor) {
            output_tensor_ref_count[i] = output_tensor->tensor_attributes->record_main_thread_ref_count();
        }

    }
    for (int i = 0; i < optional_output_tensors.size(); i++) {
        if (optional_output_tensors[i].has_value()) {
            optional_output_tensor_ref_count[i] =
                optional_output_tensors[i].value().tensor_attributes->record_main_thread_ref_count();
        } else {
            optional_output_tensor_ref_count[i] = 0;
        }
    }
    // Check if this op dispatch step relies on tensors from other workers.
    // If so, mark them in use by current worker. Tensors shared across workers
    // are only supported when each tensor is tied to a single device/worker
    // (example all-gather).
    if (workers_size == 1) {
        // Single worker per tensor and.
        for (int i = 0; i < async_safe_input_tensors.size(); i++) {
            if (async_safe_input_tensors[i].get_workers().size() and
                async_safe_input_tensors[i].get_workers()[0] != workers[0]) {
                // This input has a worker assigned that doesn't match the worker of the output being created (its
                // shared).
                async_safe_input_tensors[i].tensor_attributes->num_sibling_workers_sharing_tensor++;
                cross_worker_input_tensor_idx.insert(i);
            }
        }
        for (int i = 0; i < async_safe_optional_input_tensors.size(); i++) {
            if (async_safe_optional_input_tensors[i].has_value() and
                async_safe_optional_input_tensors[i].value().get_workers().size() and
                async_safe_optional_input_tensors[i].value().get_workers()[0] != workers[0]) {
                async_safe_optional_input_tensors[i].value().tensor_attributes->num_sibling_workers_sharing_tensor++;
                cross_worker_optional_input_tensor_idx.insert(i);
            }
        }
    }

    {
        ZoneScopedN("PushOpToWorkers");
        auto work_lambda = std::make_shared<std::function<void(Device*)>>(
            [workers_size,
             op_func,
             optional_output_tensors,
             async_safe_optional_input_tensors,
             inputs = async_safe_input_tensors,
             outputs = output_tensors,
             shared_input_idx = cross_worker_input_tensor_idx,
             shared_optional_input_idx = cross_worker_optional_input_tensor_idx](Device* target_device) mutable {
                std::vector<Tensor> input_shards = std::vector<Tensor>(inputs.size(), Tensor());
                std::vector<std::optional<const Tensor>> optional_input_shards = {};
                std::vector<std::optional<Tensor>> optional_output_shards(optional_output_tensors.size());
                // Initialize all optional_outputs to std::nullopt
                {
                    ZoneScopedN("CreateShards");
                    for (int i = 0; i < input_shards.size(); i++) {
                        input_shards[i] = get_shard_for_device(inputs[i], target_device);
                    }

                    for (auto& input : async_safe_optional_input_tensors) {
                        if (input.has_value()) {
                            optional_input_shards.push_back(get_shard_for_device(input.value(), target_device));
                        } else {
                            optional_input_shards.push_back(std::nullopt);
                        }
                    }

                    for (std::size_t optional_output_idx = 0; optional_output_idx < optional_output_tensors.size();
                         optional_output_idx++) {
                        if (optional_output_tensors[optional_output_idx].has_value()) {
                            optional_output_shards[optional_output_idx] = get_shard_for_device(
                                optional_output_tensors[optional_output_idx].value(), target_device);
                        }
                    }
                }

                auto local_tensors = op_func(input_shards, optional_input_shards, optional_output_shards);

                {
                    ZoneScopedN("OpPostProcess");
                    // Release shared ownership of tensors belonging to other workers.
                    // If the workers for this tensor are stalled to deallocate
                    for (auto& shared_input : shared_input_idx) {
                        inputs[shared_input].tensor_attributes->num_sibling_workers_sharing_tensor--;
                    }

                    for (auto& shared_optional_input : shared_optional_input_idx) {
                        async_safe_optional_input_tensors[shared_optional_input]
                            .value()
                            .tensor_attributes->num_sibling_workers_sharing_tensor--;
                    }

                    for (int i = 0; i < local_tensors.size(); i++) {
                        auto output_tensor = get_tensor(outputs[i]);
                        auto local_tensor = get_tensor(local_tensors[i]);
                        // not sure if it the case but in my opinion it should not happen
                        // both output and local tensor should be presented or absent
                        TT_ASSERT((output_tensor != nullptr && local_tensor != nullptr) || (local_tensor == nullptr && output_tensor == nullptr));
                        if (!output_tensor || !local_tensor) {
                            continue;
                        }
                        if (std::holds_alternative<OwnedStorage>(local_tensor->tensor_attributes->storage)) {
                            TT_ASSERT(
                                output_tensor->tensor_attributes->dynamic_storage,
                                "launch_with_autoformat must be used if output tensor for op can be placed on host.");
                            // Make this a host side tensor - Set storage = Owned and clear workers
                            output_tensor->tensor_attributes->storage = OwnedStorage();
                            output_tensor->workers = {};
                        } else {
                            output_tensor->tensor_attributes->dynamic_storage = false;
                        }
                        insert_buffer_and_shape_for_device(target_device, *local_tensor, *output_tensor);
                        int num_workers_completed = (output_tensor->tensor_attributes->num_workers_completed)++;
                        if (not num_workers_completed) {
                            output_tensor->tensor_attributes->shape = local_tensor->tensor_attributes->shape;
                            output_tensor->tensor_attributes->dtype = local_tensor->tensor_attributes->dtype;
                            output_tensor->tensor_attributes->layout = local_tensor->tensor_attributes->layout;
                            output_tensor->tensor_attributes->metadata_populated = true;
                        }
                    }
                }
            });

        for (auto target_device : workers) {
            target_device->push_work(std::make_shared<std::function<void()>>(
                [target_device, work_lambda]() mutable { (*work_lambda)(target_device); }));
        }
    }

    // Update ref counts of all tensors after push was performed (done only in main thread).
    for (int i = 0; i < async_safe_input_tensors.size(); i++) {
        async_safe_input_tensors[i].tensor_attributes->update_main_thread_ref_count(
            workers[0], input_tensor_ref_count[i]);
    }
    for (int i = 0; i < async_safe_optional_input_tensors.size(); i++) {
        if (async_safe_optional_input_tensors[i].has_value()) {
            async_safe_optional_input_tensors[i].value().tensor_attributes->update_main_thread_ref_count(
                workers[0], optional_input_tensor_ref_count[i]);
        }
    }
    for (int i = 0; i < output_tensors.size(); i++) {
        auto output_tensor = get_tensor(output_tensors[i]);
        if (!output_tensor) {
            continue;
        }
        output_tensor->tensor_attributes->update_main_thread_ref_count(workers[0], output_tensor_ref_count[i]);
    }
    for (int i = 0; i < optional_output_tensors.size(); i++) {
        if (optional_output_tensors[i].has_value()) {
            optional_output_tensors[i].value().tensor_attributes->update_main_thread_ref_count(
                workers[0], optional_output_tensor_ref_count[i]);
        }
    }
}
}
