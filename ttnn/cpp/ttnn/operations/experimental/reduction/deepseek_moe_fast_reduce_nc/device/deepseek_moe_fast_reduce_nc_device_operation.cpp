// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include "ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/device/deepseek_moe_fast_reduce_nc_device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

DeepseekMoEFastReduceNCDeviceOperation::program_factory_t
DeepseekMoEFastReduceNCDeviceOperation::select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
    return DeepseekMoEFastReduceNCProgramFactory{};
}

void DeepseekMoEFastReduceNCDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must have a buffer");
}

void DeepseekMoEFastReduceNCDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t input_rank = input_shape.rank();
    const uint32_t reduction_dim = operation_attributes.dim;

    // hardcoded constants
    const uint32_t num_split_tensors = 8;
    const uint32_t split_dim = input_rank - 1;

    // validate tensor
    operations::check_tensor(
        input_tensor, "DeepseekMoEFastReduceNC", "input", {DataType::BFLOAT16, DataType::BFLOAT8_B});
    TT_FATAL(input_tensor.layout() == ttnn::Layout::TILE, "input tensor must be tiled");

    // validate rank
    TT_FATAL(input_rank > 2, "input tensor rank must be greater than 2, but has {}", input_rank);

    // validate reduction dim
    TT_FATAL(
        reduction_dim <= input_rank - 3,
        "reduction dim must be between 0 and {}, but has {}",
        input_rank - 3,
        reduction_dim);

    // validate split dim
    uint32_t split_dim_size = input_shape[split_dim];
    TT_FATAL(
        split_dim_size % (num_split_tensors * tt::constants::TILE_WIDTH) == 0,
        "input tensor width must be divisible by {}",
        num_split_tensors * tt::constants::TILE_WIDTH);
}

ttnn::TensorSpec DeepseekMoEFastReduceNCDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const uint32_t reduction_dim = operation_attributes.dim;
    const ttnn::MemoryConfig& output_memory_config = operation_attributes.output_memory_config;
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;

    // hardcoded constants
    const uint32_t num_split_tensors = 8;
    const uint32_t split_dim = input_tensor.padded_shape().rank() - 1;

    auto output_shape = input_tensor.padded_shape();
    output_shape[reduction_dim] = 1;  // keepdim = true
    output_shape[split_dim] /= num_split_tensors;

    return TensorSpec(
        output_shape, TensorLayout(input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), output_memory_config));
}

std::vector<ttnn::Tensor> DeepseekMoEFastReduceNCDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;

    const ttnn::TensorSpec& output_tensor_spec = compute_output_specs(operation_attributes, tensor_args);

    return {
        create_device_tensor(output_tensor_spec, input_tensor.device()),
        create_device_tensor(output_tensor_spec, input_tensor.device()),
        create_device_tensor(output_tensor_spec, input_tensor.device()),
        create_device_tensor(output_tensor_spec, input_tensor.device()),
        create_device_tensor(output_tensor_spec, input_tensor.device()),
        create_device_tensor(output_tensor_spec, input_tensor.device()),
        create_device_tensor(output_tensor_spec, input_tensor.device()),
        create_device_tensor(output_tensor_spec, input_tensor.device()),
    };
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<ttnn::Tensor> deepseek_moe_fast_reduce_nc(
    const ttnn::Tensor& input_tensor,
    uint32_t dim,
    const ttnn::MemoryConfig& output_memory_config,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::experimental::prim::DeepseekMoEFastReduceNCDeviceOperation;

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{dim, output_memory_config, compute_kernel_config},
        OperationType::tensor_args_t{input_tensor});
}

}  // namespace ttnn::prim
