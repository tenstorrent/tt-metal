// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention_all_gather_async_device_operation.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::operations::experimental::ccl {

RingAttentionAllGatherAsyncDeviceOperation::program_factory_t
RingAttentionAllGatherAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Only one program factory available
    return program::RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory{};
}

void RingAttentionAllGatherAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void RingAttentionAllGatherAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensor;
    TT_FATAL(
        !input_tensors.empty(), "Error, Input tensor size should be greater than 0 but has {}", input_tensors.size());

    const auto& first_input_tensor = input_tensors[0];
    const auto& dtype = first_input_tensor.dtype();
    const auto& memory_config = first_input_tensor.memory_config();
    const auto& input_shape = first_input_tensor.logical_shape();

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

        TT_FATAL(
            input_tensor.logical_shape() == input_shape,
            "All input tensors must have the same shape. Input tensor {} has different shape",
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
}

RingAttentionAllGatherAsyncDeviceOperation::spec_return_value_t
RingAttentionAllGatherAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensor;
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.logical_shape();
    shape[operation_attributes.dim] *= operation_attributes.ring_size;

    // Need to determine output memory config - this should come from operation_attributes
    // For now, using input memory config as fallback
    MemoryConfig output_mem_config = input_tensor.memory_config();

    std::vector<ttnn::TensorSpec> output_specs;
    output_specs.reserve(input_tensors.size());
    for (uint32_t i = 0; i < input_tensors.size(); i++) {
        output_specs.push_back(TensorSpec(
            shape, TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_mem_config)));
    }
    return output_specs;
}

RingAttentionAllGatherAsyncDeviceOperation::tensor_return_value_t
RingAttentionAllGatherAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensor;
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);

    std::vector<Tensor> output_tensors;
    output_tensors.reserve(output_specs.size());
    for (const auto& output_spec : output_specs) {
        output_tensors.emplace_back(create_device_tensor(output_spec, input_tensors[0].device()));
    }
    return output_tensors;
}

tt::stl::hash::hash_t RingAttentionAllGatherAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensor;
    auto input_shape = input_tensors[0].padded_shape();
    auto input_memory_layout = input_tensors[0].layout();
    auto input_dtype = input_tensors[0].dtype();
    auto input_memory_config = input_tensors[0].memory_config();

    // Need to determine output_mem_config - this should be in operation_attributes
    // For now, using input memory config as fallback
    MemoryConfig output_mem_config = input_tensors[0].memory_config();

    return tt::tt_metal::operation::hash_operation<RingAttentionAllGatherAsyncDeviceOperation>(
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        output_mem_config,
        operation_attributes.topology,
        operation_attributes.sub_device_id.has_value(),
        operation_attributes.sub_device_id.has_value()
            ? input_tensors[0].device()->worker_cores(
                  tt::tt_metal::HalProgrammableCoreType::TENSIX, operation_attributes.sub_device_id.value())
            : CoreRangeSet(CoreRange({0, 0}, {0, 0})),
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

std::tuple<
    RingAttentionAllGatherAsyncDeviceOperation::operation_attributes_t,
    RingAttentionAllGatherAsyncDeviceOperation::tensor_args_t>
RingAttentionAllGatherAsyncDeviceOperation::invoke(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& persistent_output_buffer,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    return {
        operation_attributes_t{
            mesh_device.get_devices(),
            dim,
            num_links,
            ring_size,
            memory_config.value_or(input_tensors[0].memory_config()),
            topology,
            multi_device_global_semaphore,
            sub_device_id,
            cluster_axis,
        },
        tensor_args_t{.input_tensor = input_tensors}};
}

}  // namespace ttnn::operations::experimental::ccl
