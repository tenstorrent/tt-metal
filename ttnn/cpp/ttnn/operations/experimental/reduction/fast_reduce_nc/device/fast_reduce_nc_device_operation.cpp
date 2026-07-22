// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_program_factory.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/reduction/reduce_op_validation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::prim {
void FastReduceNCDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& preallocated_output = tensor_args.preallocated_output;

    // FLOAT32 is allowed so multi-stage Sum chains can carry FP32 between stages.
    operations::check_tensor(
        input, "FastReduceNC", "input", {DataType::BFLOAT16, DataType::BFLOAT8_B, DataType::FLOAT32});
    if (preallocated_output.has_value()) {
        operations::check_tensor(
            preallocated_output.value(),
            "FastReduceNC",
            "output",
            {DataType::BFLOAT16, DataType::BFLOAT8_B, DataType::FLOAT32});
        TT_FATAL(
            preallocated_output.value().logical_shape().rank() == input.logical_shape().rank(),
            "FastReduceNC preallocated output rank {} must match input rank {}",
            preallocated_output.value().logical_shape().rank(),
            input.logical_shape().rank());
    }

    // validate input dim
    const auto input_rank = input.logical_shape().rank();
    TT_FATAL(
        (args.dim >= 0 && args.dim <= ttnn::MAX_NUM_DIMENSIONS - 2),
        "dim must be between 0 and {}.",
        ttnn::MAX_NUM_DIMENSIONS - 2);
    TT_FATAL((args.dim < input_rank), "dim must be smaller than input tensor rank {}.", input_rank);
    TT_FATAL(
        input_rank <= ttnn::MAX_NUM_DIMENSIONS,
        "FastReduceNC input rank {} exceeds maximum {}",
        input_rank,
        ttnn::MAX_NUM_DIMENSIONS);

    ttnn::prim::ReduceOpDeviceGridValidationOptions input_grid_opts{};
    if (args.sub_core_grids.has_value()) {
        input_grid_opts.sub_grid_contained_in_device_grid = &args.sub_core_grids.value();
        input_grid_opts.sub_grid_label = "sub_core_grids";
    }
    input_grid_opts.shard_grid_contained_in_device_grid = &input.memory_config();
    input_grid_opts.memory_config_label = "input";

    std::optional<tt::tt_metal::TensorSpec> input_spec_nd_shard = std::nullopt;
    if (input.memory_config().nd_shard_spec().has_value()) {
        input_spec_nd_shard = input.tensor_spec();
    }
    ttnn::prim::validate_reduce_op_tensor(input, "FastReduceNC", "input", &input_grid_opts, input_spec_nd_shard);

    ttnn::prim::ReduceOpDeviceGridValidationOptions output_grid_opts{};
    output_grid_opts.shard_grid_contained_in_device_grid = &args.output_mem_config;
    output_grid_opts.memory_config_label = "output";
    std::optional<tt::tt_metal::TensorSpec> output_spec_nd_shard = std::nullopt;
    if (args.output_mem_config.nd_shard_spec().has_value() || args.output_mem_config.shard_spec().has_value()) {
        output_spec_nd_shard = compute_output_specs(args, tensor_args);
    }
    ttnn::prim::validate_reduce_op_tensor(input, "FastReduceNC", "output", &output_grid_opts, output_spec_nd_shard);

    if (preallocated_output.has_value()) {
        ttnn::prim::validate_reduce_op_tensor(preallocated_output.value(), "FastReduceNC", "preallocated_output");
    }
}

tt::tt_metal::TensorSpec FastReduceNCDeviceOperation::compute_output_specs(
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
    const auto output_dtype = args.output_dtype.value_or(input.dtype());
    return tt::tt_metal::TensorSpec(
        output_shape,
        operations::TensorLayout(output_dtype, operations::PageConfig(Layout::TILE), args.output_mem_config));
}

Tensor FastReduceNCDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }

    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor fast_reduce_nc(
    const Tensor& input,
    const int32_t& dim,
    const std::optional<const Tensor>& output,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<DataType>& output_dtype) {
    using OperationType = ttnn::experimental::prim::FastReduceNCDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .dim = dim,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
        .sub_core_grids = sub_core_grids,
        .output_dtype = output_dtype};
    auto tensor_args = OperationType::tensor_args_t{.input = input, .preallocated_output = output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
