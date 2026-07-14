// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat/codegen/repeat_codegen_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/repeat/codegen/repeat_codegen_supported.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

RepeatCodegenDeviceOperation::program_factory_t RepeatCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return RepeatCodegenProgramFactory{};
}

void RepeatCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& input = tensor_args.input;
    TT_FATAL(input.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operands to repeat need to be on device!");
    TT_FATAL(input.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(
        ttnn::operations::data_movement::repeat_codegen::supported_by_codegen(
            input, operation_attributes.rep_dim, operation_attributes.num_repeats),
        "Input is not supported by RepeatCodegen");
}

RepeatCodegenDeviceOperation::spec_return_value_t RepeatCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    auto output_shape = input.logical_shape();
    output_shape[operation_attributes.rep_dim] *= operation_attributes.num_repeats;
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            input.dtype(), tt::tt_metal::PageConfig(input.layout()), operation_attributes.output_mem_config));
}

RepeatCodegenDeviceOperation::tensor_return_value_t RepeatCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> RepeatCodegenDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input_tensor = tensor_args.input;
    int ideal_dev_clock_cycles = operations::data_movement::common_tm_bw_model(input_tensor, output_tensor);
    return {{input_tensor}, output_tensor, ideal_dev_clock_cycles};
}

RepeatCodegenDeviceOperation::tensor_return_value_t repeat_codegen(
    const Tensor& input, const RepeatCodegenParams& params) {
    using OperationType = RepeatCodegenDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(params, OperationType::tensor_args_t{.input = input});
}

}  // namespace ttnn::prim
