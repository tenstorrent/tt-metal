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
    TT_FATAL(input_tensors.at(0).storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input_tensors.at(0).buffer() != nullptr, "Input tensor must have a buffer");
}

void DeepseekReduceScatterDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const std::vector<ttnn::Tensor>& input_tensors = tensor_args.input_tensors;

    // Common validation
    deepseek_reduce_scatter_common_validates(
        input_tensors,
        operation_attributes.output_memory_config,
        operation_attributes.num_links,
        /* ring size */ 8);
}

spec_return_value_t DeepseekReduceScatterDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const std::vector<ttnn::Tensor>& input_tensors = tensor_args.input_tensors;
    const ttnn::Tensor& input_slice_0_tensor = input_tensors.at(0);

    const auto& intermediate_shape = input_slice_0_tensor.logical_shape();
    ttnn::MemoryConfig intermediate_memory_config = input_slice_0_tensor.memory_config();

    const auto& output_shape = input_slice_0_tensor.logical_shape();

    return {
        TensorSpec(
            intermediate_shape,
            TensorLayout(
                input_slice_0_tensor.dtype(),
                input_slice_0_tensor.tensor_spec().page_config(),
                intermediate_memory_config)),
        TensorSpec(
            output_shape,
            TensorLayout(
                input_slice_0_tensor.dtype(),
                input_slice_0_tensor.tensor_spec().page_config(),
                operation_attributes.output_memory_config)),
    };
}

tensor_return_value_t DeepseekReduceScatterDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const std::vector<ttnn::Tensor>& input_tensors = tensor_args.input_tensors;
    const ttnn::Tensor& input_slice_0_tensor = input_tensors.at(0);

    std::vector<ttnn::TensorSpec> tensor_specs = compute_output_specs(operation_attributes, tensor_args);
    const ttnn::TensorSpec& intermediate_tensor_spec = tensor_specs.at(0);
    const ttnn::TensorSpec& output_tensor_spec = tensor_specs.at(1);

    return {
        create_device_tensor(intermediate_tensor_spec, input_slice_0_tensor.device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_slice_0_tensor.device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_slice_0_tensor.device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_slice_0_tensor.device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_slice_0_tensor.device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_slice_0_tensor.device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_slice_0_tensor.device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_slice_0_tensor.device()),  // intermediate
        create_device_tensor(output_tensor_spec, input_slice_0_tensor.device()),        // output
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

// Common validation function implementation
void deepseek_reduce_scatter_common_validates(
    const std::vector<ttnn::Tensor>& input_tensors,
    const ttnn::MemoryConfig& output_memory_config,
    uint32_t num_links,
    uint32_t ring_size) {
    const ttnn::Tensor& input_slice_0_tensor = input_tensors.at(0);
    uint32_t page_size = input_slice_0_tensor.buffer()->page_size();

    TT_FATAL(
        page_size % input_slice_0_tensor.buffer()->alignment() == 0,
        "reduce_scatter_minimal_async currently requires aligned pages");

    TT_FATAL(input_slice_0_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input_slice_0_tensor.buffer() != nullptr, "Input tensor must be allocated in buffers on device");
    TT_FATAL(num_links > 0, "num_links must be greater than 0");

    const auto& rank = input_slice_0_tensor.logical_shape().rank();
    TT_FATAL(rank == 4, "deepseek_reduce_scatter op is hardcoded for rank 4 tensors, but has rank {}", rank);

    const auto& input_shape = input_slice_0_tensor.padded_shape();
    TT_FATAL(input_shape[3] % ring_size == 0, "Dimension 3 must be divisible by ring_size");
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

    auto operation_attributes =
        OperationType::operation_attributes_t{std::move(output_memory_config), num_links, cluster_axis};

    auto tensor_args = OperationType::tensor_args_t{input_tensors};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
