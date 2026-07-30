// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_device_operation.hpp"

#include <tt_stl/assert.hpp>
#include "tilize_codegen_supported.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

TilizeCodegenDeviceOperation::program_factory_t TilizeCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return TilizeCodegenProgramFactory{};
}

void TilizeCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t&) {
    TT_FATAL(supported_by_codegen(operation_attributes), "tilize: inputs not supported by the codegen implementation");
}

TilizeCodegenDeviceOperation::spec_return_value_t TilizeCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    return tt::tt_metal::TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout(
            operation_attributes.output_dtype, PageConfig(Layout::TILE), operation_attributes.output_mem_config));
}

TilizeCodegenDeviceOperation::tensor_return_value_t TilizeCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

void TilizeCodegenDeviceOperation::override_runtime_arguments(
    tt::tt_metal::Program& /*program*/,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    TT_THROW("TilizeCodegenDeviceOperation::override_runtime_arguments is not yet implemented");
}

Tensor tilize_codegen(const Tensor& input_tensor, const TilizeCodegenParams& params) {
    return ttnn::device_operation::launch<TilizeCodegenDeviceOperation>(params, TilizeCodegenInputs{input_tensor});
}

}  // namespace ttnn::prim
