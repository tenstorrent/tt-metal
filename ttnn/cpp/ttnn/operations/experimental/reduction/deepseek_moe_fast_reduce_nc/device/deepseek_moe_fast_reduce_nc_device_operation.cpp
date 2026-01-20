// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include "ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/device/deepseek_moe_fast_reduce_nc_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::reduction::deepseek_moe_fast_reduce_nc::detail {

DeepseekMoEFastReduceNCDeviceOperation::program_factory_t
DeepseekMoEFastReduceNCDeviceOperation::select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
    return DeepseekMoEFastReduceNCProgramFactory{};
}

void DeepseekMoEFastReduceNCDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must have a buffer");
}

void DeepseekMoEFastReduceNCDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    // hardcoded constants
    const uint32_t num_split_tensors = 8;

    const uint32_t reduction_dim = operation_attributes.reduction_dim;
    const uint32_t split_dim = operation_attributes.split_dim;
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t input_rank = input_shape.rank();

    // validate tensor
    check_tensor(input_tensor, "DeepseekMoEFastReduceNC", "input", {DataType::BFLOAT16, DataType::BFLOAT8_B});

    // validate rank
    TT_FATAL(input_rank > 2, "input tensor rank must be greater than 2, but has {}", input_rank);

    // validate reduction dim
    TT_FATAL(
        reduction_dim <= tt::tt_metal::MAX_NUM_DIMENSIONS - 2,
        "reduction dim must be between 0 and {}, but has {}",
        tt::tt_metal::MAX_NUM_DIMENSIONS - 2,
        reduction_dim);
    TT_FATAL(
        reduction_dim < input_rank,
        "reduction dim must be smaller than input tensor rank {}, but has {}",
        input_rank,
        reduction_dim);

    // validate split dim
    uint32_t split_dim_size = input_shape[split_dim];
    TT_FATAL(
        split_dim < input_rank,
        "split dim must be smaller than input tensor rank {}, but has {}",
        input_rank,
        split_dim);
    TT_FATAL(split_dim_size % num_split_tensors == 0, "split dim must be divisible by {}", num_split_tensors);
    if (split_dim == input_rank - 1) {
        TT_FATAL(
            (split_dim_size / num_split_tensors) % tt::constants::TILE_WIDTH == 0,
            "split size must be divisible by tile width {}",
            tt::constants::TILE_WIDTH);
    } else if (split_dim == input_rank - 2) {
        TT_FATAL(
            (split_dim_size / num_split_tensors) % tt::constants::TILE_HEIGHT == 0,
            "split size must be divisible by tile height {}",
            tt::constants::TILE_HEIGHT);
    }

    // dim likeness
    TT_FATAL(
        reduction_dim != split_dim, "reduction dim {} must be different than split dim {}", reduction_dim, split_dim);
}

spec_return_value_t DeepseekMoEFastReduceNCDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const uint32_t reduction_dim = operation_attributes.reduction_dim;
    const uint32_t split_dim = operation_attributes.split_dim;
    const ttnn::MemoryConfig& output_memory_config = operation_attributes.output_memory_config;
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;

    // hardcoded constants
    const uint32_t num_split_tensors = 8;

    auto output_shape = input_tensor.padded_shape();
    output_shape[reduction_dim] = 1;  // keepdim = true
    output_shape[split_dim] /= num_split_tensors;

    return TensorSpec(output_shape, TensorLayout(input_tensor.dtype(), PageConfig(Layout::TILE), output_memory_config));
}

tensor_return_value_t DeepseekMoEFastReduceNCDeviceOperation::create_output_tensors(
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

}  // namespace ttnn::operations::experimental::reduction::deepseek_moe_fast_reduce_nc::detail

namespace ttnn::prim {

ttnn::operations::experimental::reduction::deepseek_moe_fast_reduce_nc::detail::DeepseekMoEFastReduceNCDeviceOperation::
    tensor_return_value_t
    deepseek_moe_fast_reduce_nc(
        const ttnn::Tensor& input_tensor,
        uint32_t reduction_dim,
        uint32_t split_dim,
        const ttnn::MemoryConfig& output_memory_config,
        const ttnn::DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::operations::experimental::reduction::deepseek_moe_fast_reduce_nc::detail::
        DeepseekMoEFastReduceNCDeviceOperation;

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            reduction_dim, split_dim, std::move(output_memory_config), compute_kernel_config},
        OperationType::tensor_args_t{input_tensor});
}

}  // namespace ttnn::prim
