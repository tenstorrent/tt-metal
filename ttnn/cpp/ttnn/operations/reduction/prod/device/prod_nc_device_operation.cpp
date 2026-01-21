// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "prod_nc_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

ProdNcDeviceOperation::program_factory_t ProdNcDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return ProdNcProgramFactory{};
}

void ProdNcDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

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

    // prod supports only bfloat16, per ttnn/cpp/ttnn/operations/reduction/prod/prod_nanobind.hpp
    TT_FATAL(
        input.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "Error - unsupported data type for prod, expected BFLOAT16 but got {}.",
        input.dtype());
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
