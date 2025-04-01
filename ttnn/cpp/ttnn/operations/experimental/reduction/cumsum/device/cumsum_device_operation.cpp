// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cumsum_device_operation.hpp"

#include "tt-metalium/assert.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::reduction {

CumSumDeviceOperation::program_factory_t CumSumDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Scaffolding / WIP => only single core program for now
    return CumSumDeviceOperation::SingleCore();
}

void CumSumDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // TODO: Verify `dim` parameter (`-input.dims <= dim < input.dim`)
    const auto& input_tensor = tensor_args.input_tensor;
    TT_FATAL(
        args.dim < input_tensor.get_logical_shape().size() && args.dim >= -input_tensor.get_logical_shape().size(),
        "Specified dim argument exceeds tensor dimensions");

    if (tensor_args.preallocated_output.has_value()) {
        // Check if preallocated tensor matches output specs

        // Check tensor specs:
        // 1) If that's the case => OK, no need to allocate memory
        // 2) If empty tensor => reallocate memory
        // 3) Otherwise =>  error (pytorch behaviour is to reallocate but this has been deprecated as of pytorch 2.8)
        TT_FATAL(
            tensor_args.preallocated_output->tensor_spec() == compute_output_specs(args, tensor_args),
            "Preallocated output tensor mismatch expected specs");
    }
}

void CumSumDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

CumSumDeviceOperation::spec_return_value_t CumSumDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return TensorSpec(
        tensor_args.input_tensor.get_logical_shape(),
        tt::tt_metal::TensorLayout(
            args.dtype, tensor_args.input_tensor.get_layout(), tensor_args.input_tensor.memory_config()));
}

CumSumDeviceOperation::tensor_return_value_t CumSumDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }

    // otherwise, create output tensor
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

std::tuple<CumSumDeviceOperation::operation_attributes_t, CumSumDeviceOperation::tensor_args_t>
CumSumDeviceOperation::invoke(
    const Tensor& input_tensor,
    int64_t dim,
    std::optional<ttnn::DataType> dtype,
    std::optional<Tensor> preallocated_output) {
    // Scaffold => return copy of input tensor
    return {
        operation_attributes_t{.dim = dim, .dtype = dtype.value_or(input_tensor.dtype())},
        tensor_args_t{.input_tensor = input_tensor, .preallocated_output = preallocated_output}};
}

}  // namespace ttnn::operations::experimental::reduction
