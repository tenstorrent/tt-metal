// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cumsum_device_operation.hpp"

#include "tt-metalium/assert.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::reduction {

CumSumDeviceOperation::program_factory_t CumSumDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Scaffolding / WIP => only single core program for now
    return CumSumDeviceOperation::ProgramFactory();
}

void CumSumDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Verify `dim` parameter (`-input.dims <= dim < input.dim`)
    // Note: `args.dim` can be negative (but tensor rank() is unsigned) which is why we cast to int64_t
    const auto& input_tensor = tensor_args.input_tensor;
    const int64_t tensor_rank = static_cast<int64_t>(input_tensor.logical_shape().rank());
    TT_FATAL(
        (tensor_rank == 0 && args.dim == 0)  // input tensor can have a dim of 0 (c.f. torch implementation)
            || (args.dim < tensor_rank && args.dim >= -tensor_rank),
        "Specified dim ({}) argument exceeds tensor dimensions ({})\n",
        args.dim,
        input_tensor.logical_shape().rank());

    if (tensor_args.preallocated_output.has_value()) {
        // Make sure preallocated tensor specs match expected specs
        // If that's the case => OK, no need to allocate memory
        // Otherwise =>  error (pytorch behaviour is to reallocate but this has been deprecated as of pytorch 2.6)

        const auto& preallocated_tensor_specs = tensor_args.preallocated_output->tensor_spec();
        const auto& expected_tensor_specs = compute_output_specs(args, tensor_args);

        TT_FATAL(
            preallocated_tensor_specs.data_type() == expected_tensor_specs.data_type(),
            "Preallocated tensor data type ({}) mismatch expected data type ({})",
            preallocated_tensor_specs.data_type(),
            expected_tensor_specs.data_type());

        TT_FATAL(
            preallocated_tensor_specs.logical_shape() == expected_tensor_specs.logical_shape(),
            "Preallocated tensor shape ({}) mismatch expected tensor shape ({})",
            preallocated_tensor_specs.logical_shape(),
            expected_tensor_specs.logical_shape());
    }
}

void CumSumDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

operation::Hash CumSumDeviceOperation::compute_program_hash(
    const operation_attributes_t& op_args, const tensor_args_t& tensor_args) {
    return operation::hash_operation<CumSumDeviceOperation>(
        select_program_factory(op_args, tensor_args).index(),
        op_args.dim,
        op_args.dtype,
        op_args.flip,
        tensor_args.input_tensor.logical_shape(),
        tensor_args.input_tensor.dtype(),
        tensor_args.input_tensor.memory_config(),
        tensor_args.input_tensor.layout(),
        tensor_args.preallocated_output.has_value() ? tensor_args.preallocated_output.value().logical_shape() : Shape{},
        tensor_args.preallocated_output.has_value() ? tensor_args.preallocated_output.value().dtype() : DataType{},
        tensor_args.preallocated_output.has_value() ? tensor_args.preallocated_output.value().memory_config()
                                                    : MemoryConfig{});
}

CumSumDeviceOperation::spec_return_value_t CumSumDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return TensorSpec(
        tensor_args.input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            args.dtype,
            tt::tt_metal::PageConfig(tensor_args.input_tensor.layout()),
            tensor_args.input_tensor.memory_config()));
}

CumSumDeviceOperation::tensor_return_value_t CumSumDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }

    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

std::tuple<CumSumDeviceOperation::operation_attributes_t, CumSumDeviceOperation::tensor_args_t>
CumSumDeviceOperation::invoke(
    const Tensor& input_tensor,
    int64_t dim,
    std::optional<ttnn::DataType> dtype,
    std::optional<Tensor> preallocated_output,
    const bool& flip) {
    return {
        operation_attributes_t{.dim = dim, .dtype = dtype.value_or(input_tensor.dtype()), .flip = flip},
        tensor_args_t{.input_tensor = input_tensor, .preallocated_output = std::move(preallocated_output)}};
}

}  // namespace ttnn::operations::reduction
