// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "standardize_w_rm_device_operation.hpp"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::reduction::standardize_w_rm {
using namespace tt;
using namespace tt::tt_metal;

StandardizeWRmDeviceOperation::program_factory_t StandardizeWRmDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::StandardizeWRmProgramFactory{};
}

void StandardizeWRmDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void StandardizeWRmDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Extract parameters from operation_attributes for use in validations
    const auto epsilon = operation_attributes.epsilon;

    // Extract tensors from tensor_args
    const auto& input = tensor_args.input;

    // Suppress unused variable warnings if parameters aren't used in validations
    (void)epsilon;

    // Storage type validation
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "input_tensor tensor must be on device");
    TT_FATAL(input.buffer() != nullptr, "input_tensor tensor must be allocated");

    // Validations from spec
    TT_FATAL(
        input.logical_shape().rank() >= 2,
        "Input tensor must be at least 2D, got rank {}",
        input.logical_shape().rank());
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Input tensor must be in ROW_MAJOR layout");
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input tensor must have INTERLEAVED memory layout");
    TT_FATAL(input.is_allocated(), "Input tensor must be on device");
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::FLOAT32,
        "Unsupported dtype {}",
        input.dtype());
    TT_FATAL(input.logical_shape()[-1] > 0, "Width must be positive, got {}", input.logical_shape()[-1]);
    TT_FATAL(epsilon > 0.0f, "Epsilon must be positive, got {}", epsilon);
}

StandardizeWRmDeviceOperation::spec_return_value_t StandardizeWRmDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    // Output shape computation
    ttnn::Shape output_shape = input.logical_shape();
    ttnn::SmallVector<uint32_t> pdims(input.padded_shape().cbegin(), input.padded_shape().cend());

    // Pad last two dimensions to tile boundaries (only if rank >= 2)
    if (pdims.size() >= 2) {
        pdims[pdims.size() - 2] = ((pdims[pdims.size() - 2] + 31) / 32) * 32;
        pdims[pdims.size() - 1] = ((pdims[pdims.size() - 1] + 31) / 32) * 32;
    }
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

tt::stl::hash::hash_t StandardizeWRmDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& input_shape = input.padded_shape();

    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<StandardizeWRmDeviceOperation>(
        operation_attributes, input.dtype(), input.memory_config(), input_shape);

    return hash;
}

StandardizeWRmDeviceOperation::tensor_return_value_t StandardizeWRmDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}

}  // namespace ttnn::operations::reduction::standardize_w_rm

namespace ttnn::prim {
ttnn::operations::reduction::standardize_w_rm::StandardizeWRmDeviceOperation::tensor_return_value_t standardize_w_rm(
    const Tensor& input, float epsilon, const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = ttnn::operations::reduction::standardize_w_rm::StandardizeWRmDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.epsilon = epsilon, .output_mem_config = output_mem_config},
        OperationType::tensor_args_t{.input = input});
}
}  // namespace ttnn::prim
