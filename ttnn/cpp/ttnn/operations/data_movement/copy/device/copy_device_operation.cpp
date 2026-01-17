// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "copy_program_factory.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"  // common_tm_bw_model
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::data_movement::copy {

CopyDeviceOperation::program_factory_t CopyDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return copy::program::CopyProgramFactory{};
}

void CopyDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;

    const Tensor& input_tensor_a = tensor_args.input;
    TT_FATAL(
        input_tensor_a.dtype() == DataType::BFLOAT16 or input_tensor_a.dtype() == DataType::BFLOAT8_B or
            input_tensor_a.dtype() == DataType::FLOAT32 or input_tensor_a.dtype() == DataType::BFLOAT4_B or
            input_tensor_a.dtype() == DataType::UINT32 or input_tensor_a.dtype() == DataType::INT32,
        "ttnn.copy only supports float, bfloat and int32 inputs but got {}",
        input_tensor_a.dtype());
    TT_FATAL(
        operation_attributes.output_dtype == DataType::BFLOAT16 or
            operation_attributes.output_dtype == DataType::BFLOAT8_B or
            operation_attributes.output_dtype == DataType::FLOAT32 or
            operation_attributes.output_dtype == DataType::BFLOAT4_B or
            operation_attributes.output_dtype == DataType::UINT32 or
            operation_attributes.output_dtype == DataType::INT32,
        "ttnn.copy only supports float, bfloat and int32 output tensors but got {}",
        operation_attributes.output_dtype);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to copy need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to copy need to be allocated in buffers on device!");

    // Determine the actual output dtype based on preallocated output if present
    DataType output_dtype = operation_attributes.output_dtype;
    if (tensor_args.preallocated_output.has_value()) {
        const Tensor& out_tensor = tensor_args.preallocated_output.value();
        TT_FATAL(
            out_tensor.logical_shape() == input_tensor_a.logical_shape() &&
                out_tensor.padded_shape() == input_tensor_a.padded_shape(),
            "Input tensor shape {}/{} does not match output tensor shape {}/{}",
            input_tensor_a.logical_shape(),
            input_tensor_a.padded_shape(),
            out_tensor.logical_shape(),
            out_tensor.padded_shape());
        TT_FATAL(
            input_tensor_a.layout() == out_tensor.layout(),
            "Input tensor layout ({}) must equal output tensor layout ({})",
            input_tensor_a.layout(),
            out_tensor.layout());
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == out_tensor.memory_config().memory_layout(),
            "Input tensor memory layout ({}) must equal output tensor memory layout ({})",
            input_tensor_a.memory_config().memory_layout(),
            out_tensor.memory_config().memory_layout());
        // Use the preallocated output's dtype for subsequent validation
        output_dtype = out_tensor.dtype();
    }

    // Check if dtype conversion is supported (only on TILE layout)
    if (output_dtype != input_tensor_a.dtype()) {
        TT_FATAL(input_tensor_a.layout() == Layout::TILE, "Only tile layout supports dtype conversion");
    }
}

void CopyDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

CopyDeviceOperation::spec_return_value_t CopyDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const Tensor& input_tensor = tensor_args.input;
    return TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout::fromPaddedShape(
            operation_attributes.output_dtype,
            tt::tt_metal::PageConfig(input_tensor.layout()),
            operation_attributes.output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()));
}

tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>>
CopyDeviceOperation::create_op_performance_model(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& /*optional_input_tensors*/,
    std::vector<Tensor>& output_tensors) {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);
    const int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> result(
        input_tensors, output_tensors, ideal_dev_clock_cycles);
    return result;
}

CopyDeviceOperation::tensor_return_value_t CopyDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    const Tensor& input_tensor = tensor_args.input;
    const spec_return_value_t spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, input_tensor.device());
}

}  // namespace ttnn::operations::data_movement::copy

namespace ttnn::prim {
ttnn::operations::data_movement::copy::CopyDeviceOperation::tensor_return_value_t copy(
    const Tensor& input,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype,
    const std::optional<Tensor>& preallocated_output,
    bool backwards) {
    using OperationType = ttnn::operations::data_movement::copy::CopyDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{output_mem_config, output_dtype, backwards},
        OperationType::tensor_args_t{input, preallocated_output});
}
}  // namespace ttnn::prim
