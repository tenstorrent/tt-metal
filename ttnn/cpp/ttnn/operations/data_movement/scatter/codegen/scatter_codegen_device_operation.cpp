// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_codegen_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include "scatter_codegen_supported.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {
using namespace tt::tt_metal;

ScatterCodegenDeviceOperation::program_factory_t ScatterCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& /*tensor_args*/) {
    return ScatterCodegenProgramFactoryInterleaved{};
}

void ScatterCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& tensor_args) {
    TT_FATAL(
        ttnn::operations::data_movement::scatter::supported_by_codegen(
            tensor_args.input_tensor, tensor_args.index_tensor, tensor_args.src_tensor),
        "scatter_codegen: input/index/src tensors are not supported by the codegen prim (see "
        "supported_by_codegen())");
}

ScatterCodegenDeviceOperation::spec_return_value_t ScatterCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value().tensor_spec();
    }
    return tt::tt_metal::TensorSpec(
        tensor_args.input_tensor.logical_shape(),
        TensorLayout(
            tensor_args.input_tensor.dtype(),
            PageConfig(tensor_args.input_tensor.layout()),
            attributes.output_mem_config));
}

ScatterCodegenDeviceOperation::tensor_return_value_t ScatterCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value();
    }
    return create_device_tensor(compute_output_specs(attributes, tensor_args), tensor_args.input_tensor.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<ScatterCodegenDeviceOperation::tensor_return_value_t>
ScatterCodegenDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& inputs, const Tensor& output) {
    const auto& input_tensor = inputs.input_tensor;
    int ideal_dev_clock_cycles = ttnn::operations::data_movement::common_tm_bw_model(input_tensor, output);
    return tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t>(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
}

Tensor scatter_codegen(
    ScatterCodegenParams params,
    const Tensor& input_tensor,
    const Tensor& index_tensor,
    const Tensor& src_tensor,
    const std::optional<Tensor>& output_tensor) {
    return ttnn::device_operation::launch<ScatterCodegenDeviceOperation>(
        std::move(params), ScatterCodegenInputs{input_tensor, index_tensor, src_tensor, output_tensor});
}

}  // namespace ttnn::prim
