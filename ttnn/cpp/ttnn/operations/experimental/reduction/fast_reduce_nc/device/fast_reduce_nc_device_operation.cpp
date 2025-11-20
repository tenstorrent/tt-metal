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

spec_return_value_t FastReduceNCDeviceOperation::compute_output_specs(
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

tensor_return_value_t FastReduceNCDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }

    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t FastReduceNCDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;

    // Hash operation attributes (all affect program structure)
    // Hash specific tensor properties that affect program structure (dtype, memory_config, shape, shard specs, tile)
    // rather than the whole tensor to avoid including runtime-only properties like buffer addresses
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<FastReduceNCDeviceOperation>(
        args,                          // Includes dim, output_mem_config, compute_kernel_config
        input_tensor.dtype(),          // Affects CB data format
        input_tensor.memory_config(),  // Affects shard distribution logic
        input_tensor.padded_shape(),  // Affects num_reduce_input_tile, num_output_tiles, input_granularity, core groups
        input_tensor.nd_shard_spec(),        // Affects shard distribution logic
        input_tensor.tensor_spec().tile());  // Affects shard compatibility check

    // If preallocated output is provided, hash its properties that affect program structure
    if (tensor_args.preallocated_output.has_value()) {
        const auto& output_tensor = tensor_args.preallocated_output.value();
        hash = tt::stl::hash::hash_objects(
            hash,
            output_tensor.dtype(),                // Affects CB data format
            output_tensor.memory_config(),        // Affects buffer_distribution_spec and shard distribution
            output_tensor.nd_shard_spec(),        // Affects shard distribution logic
            output_tensor.tensor_spec().tile());  // Affects shard compatibility check
        // Note: buffer_distribution_spec is derived from memory_config and shape, which are now hashed
        // so we don't need to access buffer() here
    }

    return hash;
}

std::tuple<FastReduceNCDeviceOperation::operation_attributes_t, FastReduceNCDeviceOperation::tensor_args_t>
FastReduceNCDeviceOperation::invoke(
    const Tensor& input,
    const int32_t& dim,
    const std::optional<const Tensor>& output,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    operation_attributes_t operation_attributes{
        .dim = dim, .output_mem_config = output_mem_config, .compute_kernel_config = compute_kernel_config};

    tensor_args_t tensor_args{
        .input = input, .preallocated_output = output ? std::optional<Tensor>(*output) : std::nullopt};

    return {operation_attributes, tensor_args};
}

}  // namespace ttnn::operations::experimental::reduction::detail
