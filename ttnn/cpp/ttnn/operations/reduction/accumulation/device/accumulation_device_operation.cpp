// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "accumulation_device_operation.hpp"
#include "ttnn/operations/reduction/reduce_op_validation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include <enchantum/enchantum.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
void AccumulationDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor{tensor_args.input_tensor};
    const auto& input_shape{input_tensor.logical_shape()};
    const auto& optional_out{tensor_args.opt_output};
    auto out_memory_config{optional_out.has_value() ? optional_out->memory_config() : attributes.output_memory_config};

    if (optional_out.has_value()) {
        const auto& preallocated_output_shape = optional_out.value().logical_shape();
        TT_FATAL(
            input_shape == preallocated_output_shape,
            "The shapes of the input and the preallocated tensors are not equal.\n"
            "Input tensor's shape: {}\n"
            "Preallocated tensor's shape: {}",
            input_shape,
            preallocated_output_shape);
    }

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "ttnn accumulation operations (cumprod, cumsum) require input to be on a Tenstorrent device. "
        "The input tensor is stored on {}.",
        enchantum::to_string(input_tensor.storage_type()));

    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "ttnn accumulation operations (cumprod, cumsum) require to be allocated in buffers on the device. "
        "The buffer is null.");

    TT_FATAL(
        !input_tensor.is_sharded(),
        "ttnn accumulation operations (cumprod, cumsum) do not support sharded input tensors.");

    TT_FATAL(
        input_tensor.layout() == Layout::TILE,
        "The provided input tensor has a non-tile layout: {}.",
        enchantum::to_string(input_tensor.layout()));

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "ttnn accumulation operations (cumprod, cumsum) require the memory layout of the input tensor to be "
        "interleaved. Instead, it is {}.",
        enchantum::to_string(input_tensor.memory_config().memory_layout()));
    ReduceOpDeviceGridValidationOptions acc_grid_opts;
    acc_grid_opts.shard_grid_contained_in_device_grid = &out_memory_config;
    acc_grid_opts.memory_config_label = "output";
    validate_reduce_op_tensor(
        input_tensor, "Accumulation", "output", &acc_grid_opts, compute_output_specs(attributes, tensor_args));
    if (optional_out.has_value()) {
        validate_reduce_op_tensor(optional_out.value(), "Accumulation", "preallocated_output");
    }

    {
        const int32_t logical_rank = input_tensor.logical_shape().rank();
        const int32_t acc_dim = attributes.dim;
        TT_FATAL(
            acc_dim < logical_rank, "Accumulation dim {} must be less than logical rank {}", acc_dim, logical_rank);
    }
}

AccumulationDeviceOperation::spec_return_value_t AccumulationDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.opt_output.has_value()) {
        return tensor_args.opt_output->tensor_spec();
    }

    auto output_layout{Layout::TILE};
    if (attributes.output_memory_config.is_sharded()) {
        output_layout = tensor_args.input_tensor.layout();
    }

    const DataType dtype =
        tensor_args.opt_output
            ? tensor_args.opt_output->dtype()
            : ((attributes.dtype == DataType::INVALID) ? tensor_args.input_tensor.dtype() : attributes.dtype);

    const auto output_shape{tensor_args.input_tensor.logical_shape()};
    return TensorSpec{output_shape, TensorLayout{dtype, output_layout, attributes.output_memory_config}};
}

AccumulationDeviceOperation::tensor_return_value_t AccumulationDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.opt_output.has_value()) {
        // a copy of a Python object (referencing to the same tensor though) is returned here
        return *tensor_args.opt_output;
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

ttnn::Tensor accumulation(
    const Tensor& input_tensor,
    const int32_t& dim,
    const std::optional<DataType>& dtype,
    const bool& reverse_order,
    std::optional<Tensor> optional_out,
    const std::optional<MemoryConfig>& memory_config,
    AccumulationOp op) {
    using OperationType = AccumulationDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            (dim < 0) ? (dim + input_tensor.logical_shape().rank()) : dim,
            dtype.has_value() ? dtype.value()
                              : (optional_out.has_value() ? optional_out->dtype() : input_tensor.dtype()),
            memory_config.has_value()
                ? *memory_config
                : (optional_out.has_value() ? optional_out->memory_config() : input_tensor.memory_config()),
            reverse_order,
            op},
        OperationType::tensor_args_t{input_tensor, std::move(optional_out)});
}

}  // namespace ttnn::prim
