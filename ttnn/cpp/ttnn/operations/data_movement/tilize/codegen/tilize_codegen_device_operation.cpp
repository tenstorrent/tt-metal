// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "tilize_codegen_supported.hpp"

namespace ttnn::prim {

void TilizeCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // supported_by_codegen() answers for a host tensor too, so the structural preconditions are
    // asserted first; otherwise output allocation fails on a null device instead of this error.
    TT_FATAL(tensor_args.input.storage_type() == ttnn::StorageType::DEVICE, "Operands to tilize need to be on device!");
    TT_FATAL(tensor_args.input.is_allocated(), "Operands to tilize need to be allocated in buffers on device!");
    TT_FATAL(
        ttnn::operations::data_movement::tilize_codegen::supported_by_codegen(
            tensor_args.input, operation_attributes.output_mem_config),
        "TilizeCodegenDeviceOperation invoked for a case not supported by the codegen implementation");
}

TilizeCodegenDeviceOperation::spec_return_value_t TilizeCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;
    return TensorSpec(
        tensor_args.input.logical_shape(),
        TensorLayout(
            operation_attributes.output_dtype, PageConfig(Layout::TILE), operation_attributes.output_mem_config));
}

TilizeCodegenDeviceOperation::tensor_return_value_t TilizeCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

Tensor tilize_codegen(
    const Tensor& input,
    tt::tt_metal::MemoryConfig output_mem_config,
    tt::tt_metal::DataType output_dtype,
    bool use_multicore,
    bool use_low_perf,
    bool preserve_logical_shape,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::device_operation::launch<TilizeCodegenDeviceOperation>(
        TilizeCodegenOperationAttributes{
            .output_mem_config = std::move(output_mem_config),
            .output_dtype = output_dtype,
            .use_multicore = use_multicore,
            .use_low_perf = use_low_perf,
            .preserve_logical_shape = preserve_logical_shape,
            .sub_core_grids = sub_core_grids},
        TilizeCodegenTensorArgs{.input = input});
}

}  // namespace ttnn::prim
