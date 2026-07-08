// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "prod_nc_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/reduction/reduce_op_validation.hpp"

namespace ttnn::prim {
void ProdNcDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL((args.dim >= 0 && args.dim <= 3), "dim should be 0 - 3");
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;

    auto input_shape = input.padded_shape();
    TT_FATAL((input_shape.rank() == 4), "rank should be 4, got rank: {}", input_shape.rank());
    const auto& output_shape = output.padded_shape();
    auto input_shape_wo_padding = input.logical_shape();

    if (args.dim == 0 || args.dim == 1) {
        input_shape[args.dim] = 1;
        input_shape_wo_padding[args.dim] = 1;
    }

    for (int i = 0; i < input_shape.rank(); ++i) {
        TT_FATAL(
            input_shape[i] == output_shape[i],
            "Input and output shapes must match at dimension {}, got input: {} vs output: {}",
            i,
            input_shape[i],
            output_shape[i]);
    }

    TT_FATAL(
        input.dtype() == tt::tt_metal::DataType::BFLOAT16 || input.dtype() == tt::tt_metal::DataType::FLOAT32 ||
            input.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Error - unsupported data type for prod, expected BFLOAT16, FLOAT32 or BFLOAT8_B but got {}.",
        input.dtype());

    const auto& out_memory_config = output.memory_config();
    ReduceOpDeviceGridValidationOptions prod_nc_grid_opts;
    prod_nc_grid_opts.shard_grid_contained_in_device_grid = &out_memory_config;
    prod_nc_grid_opts.memory_config_label = "output";
    validate_reduce_op_tensor(input, "Prod_nc", "input");
    validate_reduce_op_tensor(output, "Prod_nc", "output", &prod_nc_grid_opts, compute_output_specs(args, tensor_args));
}

ProdNcDeviceOperation::spec_return_value_t ProdNcDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // Inplace operation - return output tensor's spec
    return tensor_args.output.tensor_spec();
}

ProdNcDeviceOperation::tensor_return_value_t ProdNcDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // Inplace operation - return output tensor
    return tensor_args.output;
}

ttnn::Tensor prod_nc(const ttnn::Tensor& input, const ttnn::Tensor& output, int64_t dim) {
    using OperationType = ProdNcDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.dim = dim},
        OperationType::tensor_args_t{.input = input, .output = output});
}

}  // namespace ttnn::prim
