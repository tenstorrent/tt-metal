// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute_codegen_device_operation.hpp"

#include <algorithm>

#include "permute_codegen_supported.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::data_movement {

PermuteCodegenDeviceOperation::program_factory_t PermuteCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    // dims[-1] == rank - 1: last dim unchanged, row-invariant no-compute path.
    if (operation_attributes.dims[operation_attributes.rank - 1] == operation_attributes.rank - 1) {
        return RowInvariant{};
    }
    return BlockedGeneric{};
}

void PermuteCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        permute_codegen::supported_by_codegen(
            tensor_args.input_tensor, ttsl::Span<const uint32_t>(attributes.dims.data(), attributes.rank)),
        "PermuteCodegenDeviceOperation: input is not supported by the codegen implementation");
}

void PermuteCodegenDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& /*tensor_args*/) {}

PermuteCodegenDeviceOperation::spec_return_value_t PermuteCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    const auto input_shape = input_tensor.logical_shape();

    ttsl::SmallVector<uint32_t> output_shape_vec(attributes.rank);
    for (uint32_t i = 0; i < attributes.rank; ++i) {
        output_shape_vec[i] = input_shape[attributes.dims[i]];
    }

    return TensorSpec(
        Shape(std::move(output_shape_vec)),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), attributes.output_mem_config));
}

PermuteCodegenDeviceOperation::tensor_return_value_t PermuteCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

ttsl::hash::hash_t PermuteCodegenDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    const auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return tt::tt_metal::operation::hash_operation<PermuteCodegenDeviceOperation>(
        operation_attributes.rank,
        operation_attributes.dims,
        operation_attributes.output_strides,
        operation_attributes.num_rows,
        operation_attributes.aligned_stick_bytes,
        operation_attributes.elem_size,
        operation_attributes.num_blocks_total,
        operation_attributes.output_mem_config,
        program_factory.index(),
        input_tensor.tensor_spec(),
        input_tensor.padded_shape(),
        output_spec,
        output_spec.padded_shape());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<PermuteCodegenDeviceOperation::tensor_return_value_t>
PermuteCodegenDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*op_attr*/, const tensor_args_t& inputs, const Tensor& output) {
    const auto& input_tensor = inputs.input_tensor;
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output, false, 0, true);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::PermuteCodegenDeviceOperation::tensor_return_value_t permute_codegen(
    const Tensor& /*input_tensor*/,
    const ttsl::SmallVector<uint32_t>& /*dims*/,
    const std::optional<MemoryConfig>& /*memory_config*/,
    std::optional<Tensor> /*optional_output_tensor*/) {
    // Phase 4a: compute operation_attributes_t (cache_key_fields, incl. output_strides / num_rows /
    // aligned_stick_bytes / elem_size / num_blocks_total per manifest) from spec.py's host math,
    // then ttnn::device_operation::launch<PermuteCodegenDeviceOperation>(...).
    TT_THROW("ttnn::prim::permute_codegen not yet implemented (phase 4a)");
}
}  // namespace ttnn::prim
