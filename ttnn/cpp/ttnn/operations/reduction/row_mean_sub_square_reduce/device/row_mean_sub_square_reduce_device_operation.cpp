// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "row_mean_sub_square_reduce_device_operation.hpp"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::reduction::row_mean_sub_square_reduce {
using namespace tt;
using namespace tt::tt_metal;

RowMeanSubSquareReduceDeviceOperation::program_factory_t RowMeanSubSquareReduceDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::RowMeanSubSquareReduceProgramFactory{};
}

void RowMeanSubSquareReduceDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void RowMeanSubSquareReduceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    // Storage type validation
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "input_tensor tensor must be on device");
    TT_FATAL(input.buffer() != nullptr, "input_tensor tensor must be allocated");

    // Validations from spec
    TT_FATAL(input.logical_shape().rank() == 4, "Input tensor must be 4D, got rank {}", input.logical_shape().rank());
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Input must be in ROW_MAJOR layout");
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "Input dtype must be BFLOAT16, got {}", input.dtype());
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Input must be DRAM interleaved");
    TT_FATAL(input.is_allocated(), "Input tensor must be on device");
    TT_FATAL(input.logical_shape()[-1] >= 1, "Width must be at least 1, got {}", input.logical_shape()[-1]);
}

spec_return_value_t RowMeanSubSquareReduceDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    // Output shape computation
    ttnn::SmallVector<uint32_t> dims(input.logical_shape().cbegin(), input.logical_shape().cend());
    dims.back() = 32;
    ttnn::Shape output_shape(dims);
    ttnn::SmallVector<uint32_t> pdims(input.padded_shape().cbegin(), input.padded_shape().cend());
    pdims.back() = 32;
    ttnn::Shape output_padded(pdims);

    auto output_dtype = input.dtype();

    return TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            output_dtype,
            PageConfig(Layout::ROW_MAJOR),
            operation_attributes.output_mem_config,
            output_shape,
            output_padded));
}

tt::stl::hash::hash_t RowMeanSubSquareReduceDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& input_shape = input.padded_shape();

    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<RowMeanSubSquareReduceDeviceOperation>(
        operation_attributes, input.dtype(), input.memory_config(), input_shape);

    return hash;
}

tensor_return_value_t RowMeanSubSquareReduceDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}

}  // namespace ttnn::operations::reduction::row_mean_sub_square_reduce

namespace ttnn::prim {
ttnn::operations::reduction::row_mean_sub_square_reduce::RowMeanSubSquareReduceDeviceOperation::tensor_return_value_t
row_mean_sub_square_reduce(
    const Tensor& input, std::optional<DataType> output_dtype, const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType =
        ttnn::operations::reduction::row_mean_sub_square_reduce::RowMeanSubSquareReduceDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.output_dtype = output_dtype, .output_mem_config = output_mem_config},
        OperationType::tensor_args_t{.input = input});
}
}  // namespace ttnn::prim
