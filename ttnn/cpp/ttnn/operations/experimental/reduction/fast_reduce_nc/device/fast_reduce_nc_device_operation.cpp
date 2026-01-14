// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_device_operation.hpp"
#include "ttnn/operations/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_program_factory.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::reduction::detail {

FastReduceNCDeviceOperation::program_factory_t FastReduceNCDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::FastReduceNCProgramFactory{};
}

void FastReduceNCDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void FastReduceNCDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& preallocated_output = tensor_args.preallocated_output;

    // validate tensor
    check_tensor(input, "FastReduceNC", "input", {DataType::BFLOAT16, DataType::BFLOAT8_B});
    if (preallocated_output.has_value()) {
        check_tensor(preallocated_output.value(), "FastReduceNC", "output", {DataType::BFLOAT16, DataType::BFLOAT8_B});
    }

    // validate input dim
    const auto input_rank = input.logical_shape().rank();
    TT_FATAL(
        (args.dim >= 0 && args.dim <= tt::tt_metal::MAX_NUM_DIMENSIONS - 2),
        "dim must be between 0 and {}.",
        tt::tt_metal::MAX_NUM_DIMENSIONS - 2);
    TT_FATAL((args.dim < input_rank), "dim must be smaller than input tensor rank {}.", input_rank);
}

TensorSpec FastReduceNCDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto& input = tensor_args.input;
    const auto& input_shape = input.padded_shape();

    // keepdim=true
    auto output_shape = input_shape;
    // last 2-dim
    output_shape[args.dim] = 1;
    return TensorSpec(output_shape, TensorLayout(input.dtype(), PageConfig(Layout::TILE), args.output_mem_config));
}

Tensor FastReduceNCDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }

    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::experimental::reduction::detail

namespace ttnn::prim {

ttnn::operations::experimental::reduction::detail::FastReduceNCDeviceOperation::tensor_return_value_t fast_reduce_nc(
    const Tensor& input,
    const int32_t& dim,
    const std::optional<const Tensor>& output,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::operations::experimental::reduction::detail::FastReduceNCDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .dim = dim, .output_mem_config = output_mem_config, .compute_kernel_config = compute_kernel_config};
    auto tensor_args = OperationType::tensor_args_t{.input = input, .preallocated_output = output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
