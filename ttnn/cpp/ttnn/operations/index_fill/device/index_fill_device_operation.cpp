// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "index_fill_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::index_fill {

void IndexFillOperation::validate(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& index = tensor_args.index;
    const uint32_t dim = operation_attributes.dim;
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Index fill: Input must be on device");
    TT_FATAL(input.buffer() != nullptr, "Index fill: Input must be allocated in buffer on device");
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Index fill: Only supporting row major layout");
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Index fill: Not currently supporting sharding");
    TT_FATAL(
        operation_attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Index fill: Not currently supporting sharding");

    // Validate index tensor properties expected by the kernel.
    TT_FATAL(index.storage_type() == StorageType::DEVICE, "Index fill: Index tensor must be on device");
    TT_FATAL(index.buffer() != nullptr, "Index fill: Index tensor must be allocated in buffer on device");
    TT_FATAL(index.logical_shape().rank() == 1, "Index fill: Index tensor must be 1D");
    TT_FATAL(
        index.dtype() == DataType::UINT32,
        "Index fill: Index tensor must have UINT32 dtype; got {}",
        index.dtype());
    TT_FATAL(
        index.device() == input.device(),
        "Index fill: Index tensor must be on the same device as input");

    TT_FATAL(dim < input.logical_shape().rank() && dim >= 0, "Index fill: Invalid dimension");

    // Check that value variant type matches input tensor dtype
    const auto& value = operation_attributes.value;
    const auto input_dtype = input.dtype();
    bool is_float_dtype =
        (input_dtype == DataType::BFLOAT16 ||
         input_dtype == DataType::FLOAT32);  // Note: BFLOAT8_B and BFLOAT4_B are not supported yet (these are valid for
                                             // tilized layouts only)
    bool is_int_dtype =
        (input_dtype ==
         DataType::INT32);  // Note: UINT32, UINT16, and UINT8 are not supported yet, flagged in the program factory
    bool value_is_float = std::holds_alternative<float>(value);
    bool value_is_int = std::holds_alternative<int>(value);

    TT_FATAL(
        (is_float_dtype && value_is_float) || (is_int_dtype && value_is_int),
        "Index fill: Value type must match input tensor dtype. Got {} input dtype with {} value type",
        input_dtype,
        value_is_float ? "float" : "int");
}
void IndexFillOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
}

IndexFillOperation::spec_return_value_t IndexFillOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return TensorSpec(
        tensor_args.input.logical_shape(),
        tensor_args.input.tensor_spec().tensor_layout().with_memory_config(operation_attributes.memory_config));
}
IndexFillOperation::tensor_return_value_t IndexFillOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}
}  // namespace ttnn::operations::index_fill

namespace ttnn::prim {
ttnn::Tensor index_fill(
    const Tensor& input,
    const uint32_t dim,
    const Tensor& index,
    const std::variant<float, int> value,
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::operations::index_fill::IndexFillOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{dim, value, memory_config.value_or(input.memory_config())},
        OperationType::tensor_args_t{input, index});
}
}  // namespace ttnn::prim
