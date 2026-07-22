// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_supported.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

namespace {

// operation_attributes_t.rep_dim is stored left-padded to this rank (see
// repeat_interleave.cpp's codegen dispatch helper, the single writer of this field), regardless
// of the input tensor's real rank, so it must be recovered before indexing a real-rank shape.
constexpr uint32_t kRepDimPadRank = 4;

uint32_t recover_rep_dim(uint32_t padded_rep_dim, uint32_t ndim) { return padded_rep_dim - (kRepDimPadRank - ndim); }

}  // namespace

RepeatInterleaveCodegenDeviceOperation::program_factory_t
RepeatInterleaveCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return RepeatInterleaveCodegenProgramFactory{};
}

void RepeatInterleaveCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& input_tensor = tensor_args.input;
    TT_FATAL(
        input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
        "Operands to repeat_interleave (codegen) need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands need to be allocated in buffers on device!");

    const uint32_t rep_dim = recover_rep_dim(operation_attributes.rep_dim, input_tensor.logical_shape().rank());
    TT_FATAL(
        ttnn::operations::data_movement::supported_by_codegen(
            input_tensor,
            operation_attributes.num_repeats,
            static_cast<int32_t>(rep_dim),
            operation_attributes.output_mem_config),
        "repeat_interleave (codegen): validate_on_program_cache_miss rejected an input/attribute "
        "combination supported_by_codegen() does not support");
}

RepeatInterleaveCodegenDeviceOperation::spec_return_value_t
RepeatInterleaveCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    auto output_shape = input_tensor.logical_shape();
    const uint32_t rep_dim = recover_rep_dim(operation_attributes.rep_dim, output_shape.rank());
    output_shape[rep_dim] *= operation_attributes.num_repeats;
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(),
            tt::tt_metal::PageConfig(input_tensor.layout()),
            operation_attributes.output_mem_config));
}

RepeatInterleaveCodegenDeviceOperation::tensor_return_value_t
RepeatInterleaveCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

RepeatInterleaveCodegenDeviceOperation::tensor_return_value_t repeat_interleave_codegen(
    const Tensor& input,
    uint32_t rep_dim,
    uint32_t num_repeats,
    uint32_t lower_pages,
    uint32_t rep_dim_pages,
    uint32_t total_out_pages,
    uint32_t stick_size,
    uint32_t stick_size_out,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = RepeatInterleaveCodegenDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .rep_dim = rep_dim,
            .num_repeats = num_repeats,
            .lower_pages = lower_pages,
            .rep_dim_pages = rep_dim_pages,
            .total_out_pages = total_out_pages,
            .stick_size = stick_size,
            .stick_size_out = stick_size_out,
            .output_mem_config = output_mem_config},
        OperationType::tensor_args_t{.input = input});
}

}  // namespace ttnn::prim
