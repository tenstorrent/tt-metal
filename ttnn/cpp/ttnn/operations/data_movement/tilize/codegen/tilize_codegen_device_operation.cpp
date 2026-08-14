// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "tilize_codegen_supported.hpp"

namespace ttnn::prim {

void TilizeCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // supported_by_codegen() is a scope predicate over layout/dtype/memory config, all of which
    // answer for a host tensor too, so the native op's structural preconditions must be asserted
    // here as well -- otherwise create_output_tensors()/the program factory reach input.device()
    // and input.buffer() first and fail on a null instead of on the established error.
    TT_FATAL(tensor_args.input.storage_type() == ttnn::StorageType::DEVICE, "Operands to tilize need to be on device!");
    TT_FATAL(tensor_args.input.buffer() != nullptr, "Operands to tilize need to be allocated in buffers on device!");
    TT_FATAL(
        ttnn::operations::data_movement::tilize_codegen::supported_by_codegen(
            tensor_args.input, operation_attributes.output_mem_config, operation_attributes.output_dtype),
        "TilizeCodegenDeviceOperation invoked for a case not supported by the codegen implementation");
}

TilizeCodegenDeviceOperation::spec_return_value_t TilizeCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;
    const auto& input_tensor = tensor_args.input;
    return TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout(
            operation_attributes.output_dtype,
            PageConfig(Layout::TILE, operation_attributes.tile),
            operation_attributes.output_mem_config));
}

TilizeCodegenDeviceOperation::tensor_return_value_t TilizeCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

Tensor tilize_codegen(
    const Tensor& input,
    tt::tt_metal::MemoryConfig output_mem_config,
    tt::tt_metal::DataType output_dtype,
    tt::tt_metal::Tile tile) {
    return ttnn::device_operation::launch<TilizeCodegenDeviceOperation>(
        TilizeCodegenOperationAttributes{
            .output_mem_config = std::move(output_mem_config), .output_dtype = output_dtype, .tile = tile},
        TilizeCodegenTensorArgs{.input = input});
}

}  // namespace ttnn::prim
