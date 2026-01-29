// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "selective_reduce_combine_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::ccl::moe {

SelectiveReduceCombineDeviceOperation::program_factory_t SelectiveReduceCombineDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return UnifiedSelectReduce{};
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
void SelectiveReduceCombineDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        operation_attributes.axis.has_value() && operation_attributes.axis.value() == 1,
        "Only cluster axis==1 is supported");
}

void SelectiveReduceCombineDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {}
#pragma clang diagnostic pop

SelectiveReduceCombineDeviceOperation::spec_return_value_t SelectiveReduceCombineDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.dense_input_tensor;
    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    const auto num_devices = mesh_view.num_devices();

    const auto hidden_size = operation_attributes.hidden_size;

    const uint32_t batch_size = operation_attributes.batch_size;
    const uint32_t seq_size = operation_attributes.seq_size;

    const uint32_t experts = operation_attributes.experts;

    const auto& axis = operation_attributes.axis;
    const uint32_t replicate_dim = axis.has_value() ? mesh_device->shape()[!axis.value()] : 1;

    const uint32_t total_tokens_per_device = batch_size * seq_size * replicate_dim / num_devices;

    auto output_shape = ttnn::Shape({total_tokens_per_device, experts, hidden_size});

    auto mem_config = operation_attributes.output_memory_config;
    return TensorSpec(
        Shape(output_shape), TensorLayout(input_tensor.dtype(), PageConfig(Layout::ROW_MAJOR), mem_config));
}

SelectiveReduceCombineDeviceOperation::tensor_return_value_t
SelectiveReduceCombineDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return tensor_args.optional_output_tensor.value_or(
        create_device_tensor(output_spec, tensor_args.dense_input_tensor.device()));
}

}  // namespace ttnn::operations::ccl::moe

namespace ttnn::prim {
ttnn::Tensor selective_reduce_combine(
    const ttnn::Tensor& dense_input_tensor,
    const ttnn::Tensor& dense_metadata_tensor,
    const ttnn::Tensor& dense_token_maps_tensor,
    const ttnn::Tensor& dense_token_counts_tensor,
    const uint32_t hidden_size,
    const uint32_t batch_size,
    const uint32_t seq_size,
    const uint32_t select_experts_k,
    const uint32_t experts,
    const std::optional<uint32_t>& axis,
    tt::tt_fabric::Topology topology,
    uint32_t num_links,
    const uint32_t num_token_parallel_cores,
    const uint32_t num_data_parallel_cores,
    const CoreRangeSet worker_core_range_set,
    const CoreRangeSet mux_core_range_set,
    const ttnn::MemoryConfig& output_memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    const std::optional<GlobalSemaphore>& optional_cross_device_semaphore) {
    using OperationType = ttnn::operations::ccl::moe::SelectiveReduceCombineDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .hidden_size = hidden_size,
            .batch_size = batch_size,
            .seq_size = seq_size,
            .select_experts_k = select_experts_k,
            .experts = experts,
            .num_links = num_links,
            .axis = axis,
            .topology = topology,
            .num_token_parallel_cores = num_token_parallel_cores,
            .num_data_parallel_cores = num_data_parallel_cores,
            .worker_core_range_set = worker_core_range_set,
            .mux_core_range_set = mux_core_range_set,
            .output_memory_config = output_memory_config,
            .optional_cross_device_semaphore = optional_cross_device_semaphore},
        OperationType::tensor_args_t{
            .dense_input_tensor = dense_input_tensor,
            .dense_metadata_tensor = dense_metadata_tensor,
            .dense_token_maps_tensor = dense_token_maps_tensor,
            .dense_token_counts_tensor = dense_token_counts_tensor,
            .optional_output_tensor = optional_output_tensor});
}
}  // namespace ttnn::prim
