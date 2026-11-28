// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_bf8_device_operation.hpp"

#include "rotary_bf8_table.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::llama31_8b {

void RotaryBf8Operation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        operation_attributes.table_row < rotary_bf8_table::ENTRIES.size(),
        "llama31_8b.rotary_bf8: table row {} out of range",
        operation_attributes.table_row);
    for (const Tensor* t :
         {&tensor_args.input, &tensor_args.cos_cache, &tensor_args.sin_cache, &tensor_args.trans_mat}) {
        TT_FATAL(t->storage_type() == StorageType::DEVICE, "llama31_8b.rotary_bf8: tensors must be on device");
        TT_FATAL(t->is_allocated(), "llama31_8b.rotary_bf8: tensors must be allocated");
    }
}

RotaryBf8Operation::spec_return_value_t RotaryBf8Operation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tt::tt_metal::TensorSpec(
        tensor_args.input.logical_shape(),
        tt::tt_metal::TensorLayout(
            operation_attributes.output_dtype,
            tt::tt_metal::PageConfig(Layout::TILE),
            operation_attributes.output_memory_config));
}

RotaryBf8Operation::tensor_return_value_t RotaryBf8Operation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::experimental::llama31_8b

namespace ttnn::prim {
ttnn::Tensor llama31_8b_rotary_bf8(
    const Tensor& input,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const Tensor& trans_mat,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::uint32_t table_row) {
    using OperationType = ttnn::experimental::llama31_8b::RotaryBf8Operation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{output_dtype, output_memory_config, compute_kernel_config, table_row},
        OperationType::tensor_args_t{input, cos_cache, sin_cache, trans_mat});
}
}  // namespace ttnn::prim
