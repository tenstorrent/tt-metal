// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "centralize_w_rm_device_operation.hpp"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::reduction::centralize_w_rm {
using namespace tt;
using namespace tt::tt_metal;

CentralizeWRmDeviceOperation::program_factory_t CentralizeWRmDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::CentralizeWRmProgramFactory{};
}

void CentralizeWRmDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void CentralizeWRmDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Extract parameters from operation_attributes for use in validations
    (void)operation_attributes;  // Suppress unused warning

    // Extract tensors from tensor_args
    const auto& input = tensor_args.input;

    // Storage type validation
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "input_tensor tensor must be on device");
    TT_FATAL(input.buffer() != nullptr, "input_tensor tensor must be allocated");

    // Validations from spec
    TT_FATAL(input.logical_shape().rank() >= 2, "Input must be at least 2D, got rank {}", input.logical_shape().rank());
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Input must be in ROW_MAJOR layout");
    TT_FATAL(input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Input must be interleaved");
    TT_FATAL(input.is_allocated(), "Input must be on device");
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::FLOAT32,
        "Unsupported dtype {}",
        input.dtype());
    TT_FATAL(
        input.padded_shape()[-1] % 32 == 0,
        "Width must be padded to tile boundary (32), got {}",
        input.padded_shape()[-1]);
    TT_FATAL(
        input.padded_shape()[-2] % 32 == 0,
        "Height must be padded to tile boundary (32), got {}",
        input.padded_shape()[-2]);
}

CentralizeWRmDeviceOperation::spec_return_value_t CentralizeWRmDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    // Output shape computation
    ttnn::Shape output_shape = input.logical_shape();
    ttnn::Shape output_padded = input.padded_shape();

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

tt::stl::hash::hash_t CentralizeWRmDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& input_shape = input.padded_shape();

    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<CentralizeWRmDeviceOperation>(
        operation_attributes, input.dtype(), input.memory_config(), input_shape);

    return hash;
}

CentralizeWRmDeviceOperation::tensor_return_value_t CentralizeWRmDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}

}  // namespace ttnn::operations::reduction::centralize_w_rm

namespace ttnn::prim {
ttnn::operations::reduction::centralize_w_rm::CentralizeWRmDeviceOperation::tensor_return_value_t centralize_w_rm(
    const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = ttnn::operations::reduction::centralize_w_rm::CentralizeWRmDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.output_mem_config = output_mem_config},
        OperationType::tensor_args_t{.input = input});
}
}  // namespace ttnn::prim
