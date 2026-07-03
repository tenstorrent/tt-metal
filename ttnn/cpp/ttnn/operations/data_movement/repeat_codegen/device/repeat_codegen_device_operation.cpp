// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat_codegen/device/repeat_codegen_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

RepeatCodegenDeviceOperation::program_factory_t RepeatCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return RepeatCodegenProgramFactory{};
}

void RepeatCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& input = tensor_args.input;
    TT_FATAL(input.storage_type() == tt::tt_metal::StorageType::DEVICE, "repeat_codegen: input must be on device");
    TT_FATAL(input.buffer() != nullptr, "repeat_codegen: input must be allocated");
    TT_FATAL(input.layout() == tt::tt_metal::Layout::TILE, "repeat_codegen: TILE layout only");
    TT_FATAL(input.dtype() == tt::tt_metal::DataType::BFLOAT16, "repeat_codegen: bfloat16 only");
    TT_FATAL(operation_attributes.m_tile_page_size_bytes > 0, "repeat_codegen: tile page params not set");
    TT_FATAL(operation_attributes.m_repeat_dim >= 0, "repeat_codegen: repeat_dim must be >= 0");
}

RepeatCodegenDeviceOperation::spec_return_value_t RepeatCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Reproduces repeat's interleaved output spec exactly so the A/B compares like
    // for like: shape[dim] *= reps; dtype/layout/mem_config unchanged.
    const auto& input = tensor_args.input;
    auto output_shape = input.logical_shape();
    output_shape[operation_attributes.m_repeat_dim] *= operation_attributes.m_num_repeats;
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            input.dtype(), tt::tt_metal::PageConfig(input.layout()), operation_attributes.m_output_mem_config));
}

RepeatCodegenDeviceOperation::tensor_return_value_t RepeatCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

RepeatCodegenDeviceOperation::tensor_return_value_t repeat_codegen(
    const Tensor& input,
    uint32_t num_repeats,
    int32_t repeat_dim,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    uint32_t tile_higher_pages,
    uint32_t tile_rep_dim_pages,
    uint32_t tile_lower_pages,
    uint32_t tile_page_size_bytes) {
    using OperationType = RepeatCodegenDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .m_num_repeats = num_repeats,
            .m_output_mem_config = output_mem_config,
            .m_tile_higher_pages = tile_higher_pages,
            .m_tile_rep_dim_pages = tile_rep_dim_pages,
            .m_tile_lower_pages = tile_lower_pages,
            .m_tile_page_size_bytes = tile_page_size_bytes,
            .m_repeat_dim = repeat_dim},
        OperationType::tensor_args_t{.input = input});
}

}  // namespace ttnn::prim
