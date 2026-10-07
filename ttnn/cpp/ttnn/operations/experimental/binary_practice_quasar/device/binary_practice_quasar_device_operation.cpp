// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_practice_quasar_device_operation.hpp"

namespace ttnn::operations::experimental::binary_practice {

void BinaryPracticeQuasarDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& inputs) {
    const Tensor& a = inputs.a;
    const Tensor& b = inputs.b;
    for (const Tensor* t : {&a, &b}) {
        TT_FATAL(t->storage_type() == StorageType::DEVICE, "binary_practice_quasar: inputs must be on device");
        TT_FATAL(t->layout() == Layout::TILE, "binary_practice_quasar: inputs must be TILE layout");
        TT_FATAL(t->dtype() == DataType::BFLOAT16, "binary_practice_quasar: inputs must be BFLOAT16");
        TT_FATAL(
            t->logical_shape().rank() == 2, "binary_practice_quasar: inputs must be 2D, got {}", t->logical_shape());
    }
    TT_FATAL(
        a.padded_shape() == b.padded_shape(),
        "binary_practice_quasar: shapes must match (no broadcast), got {} and {}",
        a.logical_shape(),
        b.logical_shape());
    // All or nothing: both inputs interleaved, or both sharded with the same shard spec. The output takes
    // a's memory config, so it follows the inputs either way.
    TT_FATAL(
        a.is_sharded() == b.is_sharded(),
        "binary_practice_quasar: inputs must be both interleaved or both sharded, got {} and {}",
        a.memory_config(),
        b.memory_config());
    if (a.is_sharded()) {
        TT_FATAL(
            a.memory_config() == b.memory_config(),
            "binary_practice_quasar: sharded inputs must have the same memory config, got {} and {}",
            a.memory_config(),
            b.memory_config());
    }
}

BinaryPracticeQuasarDeviceOperation::spec_return_value_t BinaryPracticeQuasarDeviceOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& inputs) {
    return tt::tt_metal::TensorSpec(
        inputs.a.logical_shape(),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), params.output_memory_config));
}

BinaryPracticeQuasarDeviceOperation::tensor_return_value_t BinaryPracticeQuasarDeviceOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& inputs) {
    return create_device_tensor(compute_output_specs(params, inputs), inputs.a.device());
}

}  // namespace ttnn::operations::experimental::binary_practice

namespace ttnn::prim {
Tensor binary_practice_quasar(const Tensor& a, const Tensor& b) {
    using OperationType = ttnn::operations::experimental::binary_practice::BinaryPracticeQuasarDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.output_memory_config = a.memory_config()},
        OperationType::tensor_args_t{.a = a, .b = b});
}
}  // namespace ttnn::prim
