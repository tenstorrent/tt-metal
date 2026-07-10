// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention_all_gather_async_device_operation.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::experimental::prim {
void RingAttentionAllGatherAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensor;
    TT_FATAL(
        !input_tensors.empty(), "Error, Input tensor size should be greater than 0 but has {}", input_tensors.size());

    const auto& first_input_tensor = input_tensors[0];
    const auto& dtype = first_input_tensor.dtype();
    const auto& memory_config = first_input_tensor.memory_config();

    // Validate all input tensors
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        const auto& input_tensor = input_tensors[i];

        TT_FATAL(input_tensor.layout() == Layout::TILE, "Input tensor {} must be tiled", i);
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor {} must be on device", i);
        TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor {} must be allocated in buffers on device", i);

        TT_FATAL(
            input_tensor.dtype() == dtype,
            "All input tensors must have the same dtype. Input tensor {} has dtype {} but expected {}",
            i,
            input_tensor.dtype(),
            dtype);

        TT_FATAL(
            input_tensor.memory_config() == memory_config,
            "All input tensors must have the same memory config. Input tensor {} has different memory config",
            i);
    }

    TT_FATAL(
        operation_attributes.num_links > 0,
        "Error, num_links should be more than 0 but has {}",
        operation_attributes.num_links);

    TT_FATAL(
        memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Unsupported memory layout {}.",
        memory_config.memory_layout());

    // Validate output tensors if provided
    const auto& output_tensors = tensor_args.persistent_output_buffer;
    if (!output_tensors.empty()) {
        TT_FATAL(
            output_tensors.size() == input_tensors.size(),
            "Number of output tensors ({}) must match number of input tensors ({})",
            output_tensors.size(),
            input_tensors.size());

        for (size_t i = 0; i < output_tensors.size(); ++i) {
            if (output_tensors[i].has_value()) {
                const auto& output_tensor = output_tensors[i].value();

                TT_FATAL(output_tensor.layout() == Layout::TILE, "Output tensor {} must be tiled", i);
                TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Output tensor {} must be on device", i);

                TT_FATAL(
                    output_tensor.dtype() == dtype,
                    "Output tensor {} dtype should match input tensors but has {}",
                    i,
                    output_tensor.dtype());

                TT_FATAL(
                    output_tensor.memory_config() == operation_attributes.output_mem_config,
                    "Output tensor {} memory config should match output_mem_config",
                    i);

                // Check output tensor shape. The gather dimension may be larger than the populated prefix.
                auto output_shape = output_tensor.logical_shape();
                auto expected_output_shape = input_tensors[i].logical_shape();
                expected_output_shape[operation_attributes.dim] *= operation_attributes.ring_size;

                for (int d = 0; d < static_cast<int>(output_shape.rank()); ++d) {
                    if (d == operation_attributes.dim) {
                        TT_FATAL(
                            output_shape[d] >= expected_output_shape[d],
                            "Output tensor {} gather dim {} too small: got {}, expected >= {} "
                            "(= input_dim * ring_size {})",
                            i,
                            d,
                            output_shape[d],
                            expected_output_shape[d],
                            operation_attributes.ring_size);
                    } else {
                        TT_FATAL(
                            output_shape[d] == expected_output_shape[d],
                            "Output tensor {} non-gather dim {} mismatch: got {}, expected {}",
                            i,
                            d,
                            output_shape[d],
                            expected_output_shape[d]);
                    }
                }
            }
        }
    }
}

RingAttentionAllGatherAsyncDeviceOperation::spec_return_value_t
RingAttentionAllGatherAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensor;
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.logical_shape();
    std::vector<tt::tt_metal::TensorSpec> output_specs;
    output_specs.reserve(input_tensors.size());
    for (const auto& input_item : input_tensors) {
        auto shape = input_item.logical_shape();
        shape[operation_attributes.dim] *= operation_attributes.ring_size;
        output_specs.push_back(tt::tt_metal::TensorSpec(
            shape,
            TensorLayout(
                input_tensor.dtype(),
                input_tensor.tensor_spec().page_config(),
                operation_attributes.output_mem_config)));
    }
    return output_specs;
}

RingAttentionAllGatherAsyncDeviceOperation::tensor_return_value_t
RingAttentionAllGatherAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    std::vector<Tensor> output_tensors;
    const auto& persistent_output_buffer = tensor_args.persistent_output_buffer;
    if (!persistent_output_buffer.empty() && persistent_output_buffer[0].has_value()) {
        output_tensors.reserve(persistent_output_buffer.size());
        for (const auto& buffer : persistent_output_buffer) {
            TT_FATAL(buffer.has_value(), "If using optional output tensors, all output tensors must have a value");
            output_tensors.emplace_back(buffer.value());
        }
        return output_tensors;
    }
    const auto& input_tensors = tensor_args.input_tensor;
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    output_tensors.reserve(output_specs.size());
    for (const auto& output_spec : output_specs) {
        output_tensors.emplace_back(create_device_tensor(output_spec, input_tensors[0].device()));
    }
    return output_tensors;
}

std::tuple<RingAttentionAllGatherAsyncParams, RingAttentionAllGatherAsyncInputs>
ring_attention_all_gather_async_build_operation_args(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& persistent_output_buffer,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    const auto& mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(),
        "all-gather invoked with cluster_axis API without 2D mesh, which is currently unsupported");
    uint32_t ring_size = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    int32_t rank = input_tensors[0].logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    std::vector<std::optional<Tensor>> optional_output_tensors;
    optional_output_tensors.reserve(persistent_output_buffer.size());
    for (const auto& buffer : persistent_output_buffer) {
        optional_output_tensors.push_back(buffer);
    }

    return {
        RingAttentionAllGatherAsyncParams{
            {},
            gather_dim,
            num_links,
            ring_size,
            memory_config.value_or(input_tensors[0].memory_config()),
            topology,
            multi_device_global_semaphore,
            sub_device_id,
            cluster_axis,
        },
        RingAttentionAllGatherAsyncInputs{
            .input_tensor = input_tensors, .persistent_output_buffer = optional_output_tensors}};
}

std::vector<tt::tt_metal::DynamicRuntimeArg> RingAttentionAllGatherAsyncDeviceOperation::get_dynamic_runtime_args(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Re-apply the hash-excluded out_ready GlobalSemaphore L1 addresses to the cached program on every
    // dispatch (the non-Buffer analog of the BufferBinding fast path). RingAttentionAllGatherAsyncParams
    // excludes `semaphore` from the program-cache key (see attribute_values), so a cache hit with a
    // different / reallocated GlobalSemaphore set would otherwise reuse the address baked at the first
    // miss. The factory bakes these same four slots on the cache-miss build; both paths use the shared
    // ring_attention_all_gather_async_dynamic constants so the slot layout cannot drift. The addresses
    // are mesh-uniform, so they are coord-independent (mesh_dispatch_coordinate is unused) and this
    // per-coord call re-emits an identical arg set for each program in the workload.
    namespace dyn = ring_attention_all_gather_async_dynamic;

    const auto& semaphore = operation_attributes.semaphore;
    // The factory dereferences semaphore.at(kForwardSemaphoreIdx / kBackwardSemaphoreIdx) unconditionally
    // on the cache-miss build, so a cache hit implies both are present; guard defensively regardless.
    const uint32_t max_semaphore_idx =
        dyn::kForwardSemaphoreIdx > dyn::kBackwardSemaphoreIdx ? dyn::kForwardSemaphoreIdx : dyn::kBackwardSemaphoreIdx;
    if (semaphore.size() <= max_semaphore_idx) {
        return {};
    }
    const auto forward_sem_addr = static_cast<uint32_t>(semaphore[dyn::kForwardSemaphoreIdx].address());
    const auto backward_sem_addr = static_cast<uint32_t>(semaphore[dyn::kBackwardSemaphoreIdx].address());

    // Re-derive the sender worker cores exactly as build_ring_attention_all_gather_program_descriptor()
    // does: it calls ring_attention_all_gather_async_multi_core_with_workers_helper without a
    // core_grid_offset or core_allocation_strategy, so choose_worker_cores runs with CoreCoord(0, 0) and
    // ROW_MAJOR. All inputs are hashed structural params (num_links, sub_device_id) or the device, so the
    // core set is stable across cache hits (no freeze hazard).
    auto* mesh_device = tensor_args.input_tensor[0].device();
    [[maybe_unused]] const auto& [sender_worker_core_range, sender_worker_cores] = ttnn::ccl::choose_worker_cores(
        operation_attributes.num_links,
        dyn::kNumSendersPerLink,
        mesh_device,
        operation_attributes.sub_device_id,
        CoreCoord(0, 0),
        std::nullopt,
        ttnn::ccl::CoreAllocationStrategy::ROW_MAJOR);

    std::vector<tt::tt_metal::DynamicRuntimeArg> dynamic_args;
    dynamic_args.reserve(static_cast<std::size_t>(operation_attributes.num_links) * dyn::kNumSendersPerLink * 2);
    for (uint32_t link = 0; link < operation_attributes.num_links; ++link) {
        // Mirror the factory's per-link core assignment: pair slot 1 == forward sender, slot 0 == backward.
        const CoreCoord forward_core = sender_worker_cores[(link * dyn::kNumSendersPerLink) + 1];
        const CoreCoord backward_core = sender_worker_cores[link * dyn::kNumSendersPerLink];

        // Forward reader + writer bake semaphore[kForwardSemaphoreIdx].
        dynamic_args.push_back(
            {dyn::kReaderForwardKernelIdx, forward_core, dyn::kReaderSemaphoreArg, forward_sem_addr});
        dynamic_args.push_back(
            {dyn::kWriterForwardKernelIdx, forward_core, dyn::kWriterSemaphoreArg, forward_sem_addr});
        // Backward reader + writer bake semaphore[kBackwardSemaphoreIdx].
        dynamic_args.push_back(
            {dyn::kReaderBackwardKernelIdx, backward_core, dyn::kReaderSemaphoreArg, backward_sem_addr});
        dynamic_args.push_back(
            {dyn::kWriterBackwardKernelIdx, backward_core, dyn::kWriterSemaphoreArg, backward_sem_addr});
    }
    return dynamic_args;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> ring_attention_all_gather_async(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& persistent_output_buffer,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    auto [params, inputs] = experimental::prim::ring_attention_all_gather_async_build_operation_args(
        input_tensors,
        persistent_output_buffer,
        dim,
        multi_device_global_semaphore,
        cluster_axis,
        mesh_device,
        topology,
        num_links,
        memory_config,
        sub_device_id);
    return ttnn::device_operation::launch<experimental::prim::RingAttentionAllGatherAsyncDeviceOperation>(
        params, inputs);
}

}  // namespace ttnn::prim
