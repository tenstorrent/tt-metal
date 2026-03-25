// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "forward_substitution_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::forward_substitution {

void ForwardSubstitutionOperation::validate(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& shape = input.logical_shape();

    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Forward substitution: Input must be on device");
    TT_FATAL(input.buffer() != nullptr, "Forward substitution: Input must be allocated in buffer on device");
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Forward substitution: Only ROW_MAJOR layout supported");
    TT_FATAL(input.dtype() == DataType::FLOAT32, "Forward substitution: Only FLOAT32 dtype supported");
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Forward substitution: Only INTERLEAVED memory layout supported");
    TT_FATAL(
        operation_attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Forward substitution: Only INTERLEAVED output memory layout supported");

    // Must be at least 2D with square last two dims
    TT_FATAL(shape.rank() >= 2, "Forward substitution: Input must be at least 2D, got rank {}", shape.rank());
    const uint32_t C = shape[-1];
    TT_FATAL(C >= 1, "Forward substitution: matrix dimension C must be >= 1, got {}", C);
    TT_FATAL(
        shape[-2] == C,
        "Forward substitution: Last two dims must be equal (square matrix), got [{}, {}]",
        shape[-2],
        C);

    // Matrix + CB overhead must fit in L1.
    // Total L1 usage: cb_in (2 * C * 4) + cb_work (C * C * 4) + cb_temp (C * 4)
    const uint32_t total_l1_bytes = C * C * sizeof(float) + 3 * C * sizeof(float);
    TT_FATAL(
        total_l1_bytes <= 256 * 1024,
        "Forward substitution: Matrix + buffers too large for L1. C={}, needs {} bytes, max 256KB",
        C,
        total_l1_bytes);

    // Row size must be 32-byte aligned for NOC transfers
    TT_FATAL(
        (C * sizeof(float)) % 32 == 0,
        "Forward substitution: row size must be 32-byte aligned. C={} gives {} bytes (need multiple of 32)",
        C,
        C * sizeof(float));

    // Verify padded_shape matches logical_shape (ROW_MAJOR has no padding)
    TT_FATAL(
        input.padded_shape() == shape,
        "Forward substitution: padded_shape must equal logical_shape for ROW_MAJOR tensors");
}

void ForwardSubstitutionOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
}

ForwardSubstitutionOperation::spec_return_value_t ForwardSubstitutionOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return TensorSpec(
        tensor_args.input.logical_shape(),
        tensor_args.input.tensor_spec().tensor_layout().with_memory_config(operation_attributes.memory_config));
}

ForwardSubstitutionOperation::tensor_return_value_t ForwardSubstitutionOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}

}  // namespace ttnn::operations::forward_substitution

namespace ttnn::prim {
ttnn::Tensor forward_substitution(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::operations::forward_substitution::ForwardSubstitutionOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{memory_config.value_or(input.memory_config())},
        OperationType::tensor_args_t{input});
}
}  // namespace ttnn::prim
