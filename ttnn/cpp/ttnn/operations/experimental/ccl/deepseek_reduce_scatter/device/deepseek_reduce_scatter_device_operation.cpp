// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_reduce_scatter_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::ccl::deepseek_reduce_scatter::detail {

DeepseekReduceScatterDeviceOperation::program_factory_t DeepseekReduceScatterDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return DeepseekReduceScatterMeshWorkloadFactory{};
}

void DeepseekReduceScatterDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Lightweight validation for cache hits
    const std::vector<ttnn::Tensor>& input_tensors = tensor_args.input_tensors;
    for (const ttnn::Tensor& input_tensor : input_tensors) {
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
        TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must have a buffer");
    }
}

void DeepseekReduceScatterDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    const std::vector<ttnn::Tensor>& input_tensors = tensor_args.input_tensors;
    const ttnn::MemoryConfig& output_memory_config = operation_attributes.output_memory_config;

    // hardcoded constants
    const uint32_t ring_size = 8;

    // validate input tensors
    TT_FATAL(
        input_tensors.size() == ring_size,
        "deepseek_reduce_scatter requires 8 input tensors, but has {}",
        input_tensors.size());
    const ttnn::Tensor& input_tensor_0 = input_tensors.at(0);

    // TODO: (GR) fix since equality operator not provided
    // for (const ttnn::Tensor& input_tensor : input_tensors) {
    //     TT_FATAL(input_tensor == input_tensor_0, "deepseek_reduce_scatter requires all input tensors to be
    //     identical");
    // }

    TT_FATAL(
        input_tensor_0.buffer()->page_size() % input_tensor_0.buffer()->alignment() == 0,
        "deepseek_reduce_scatter currently requires aligned pages");
    TT_FATAL(
        input_tensor_0.buffer()->num_pages() % 2 == 0,
        "deepseek_reduce_scatter hardcoded to operate on slices with multiple of 2 number of pages");

    // TODO: (GR) input shard spec

    // validate output memory config
    TT_FATAL(!output_memory_config.is_sharded(), "deepseek_reduce_scatter only supports interleaved output tensor");
}

spec_return_value_t DeepseekReduceScatterDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const std::vector<ttnn::Tensor>& input_tensors = tensor_args.input_tensors;

    const auto& intermediate_shape = input_tensors.at(0).logical_shape();
    ttnn::MemoryConfig intermediate_memory_config = input_tensors.at(0).memory_config();

    const auto& output_shape = input_tensors.at(0).logical_shape();

    return {
        TensorSpec(
            intermediate_shape,
            TensorLayout(
                input_tensors.at(0).dtype(),
                input_tensors.at(0).tensor_spec().page_config(),
                intermediate_memory_config)),
        TensorSpec(
            output_shape,
            TensorLayout(
                input_tensors.at(0).dtype(),
                input_tensors.at(0).tensor_spec().page_config(),
                operation_attributes.output_memory_config)),
    };
}

tensor_return_value_t DeepseekReduceScatterDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const std::vector<ttnn::Tensor>& input_tensors = tensor_args.input_tensors;

    std::vector<ttnn::TensorSpec> tensor_specs = compute_output_specs(operation_attributes, tensor_args);
    const ttnn::TensorSpec& intermediate_tensor_spec = tensor_specs.at(0);
    const ttnn::TensorSpec& output_tensor_spec = tensor_specs.at(1);

    return {
        create_device_tensor(intermediate_tensor_spec, input_tensors.at(0).device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_tensors.at(0).device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_tensors.at(0).device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_tensors.at(0).device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_tensors.at(0).device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_tensors.at(0).device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_tensors.at(0).device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_tensors.at(0).device()),  // intermediate
        create_device_tensor(output_tensor_spec, input_tensors.at(0).device()),        // output
    };
}

tt::stl::hash::hash_t DeepseekReduceScatterDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "DeepseekReduceScatterDeviceOperation::compute_program_hash is called");

    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return tt::tt_metal::operation::hash_operation<DeepseekReduceScatterDeviceOperation>(
        operation_attributes.output_memory_config,
        operation_attributes.num_links,
        operation_attributes.cluster_axis,
        tensor_args,
        program_factory.index());
}

}  // namespace ttnn::operations::experimental::ccl::deepseek_reduce_scatter::detail

namespace ttnn::prim {

ttnn::operations::experimental::ccl::deepseek_reduce_scatter::detail::DeepseekReduceScatterDeviceOperation::
    tensor_return_value_t
    deepseek_reduce_scatter(
        const std::vector<ttnn::Tensor>& input_tensors,
        const ttnn::MemoryConfig& output_memory_config,
        uint32_t num_links,
        std::optional<uint32_t> cluster_axis) {
    using OperationType =
        ttnn::operations::experimental::ccl::deepseek_reduce_scatter::detail::DeepseekReduceScatterDeviceOperation;

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{std::move(output_memory_config), num_links, cluster_axis},
        OperationType::tensor_args_t{input_tensors});
}

}  // namespace ttnn::prim
