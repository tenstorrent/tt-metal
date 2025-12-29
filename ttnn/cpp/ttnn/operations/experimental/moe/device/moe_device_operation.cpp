// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_device_operation.hpp"

namespace ttnn::operations::experimental::moe {

MoEDeviceOperation::program_factory_t MoEDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::MoEProgramFactory{};
}

void MoEDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MoEDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {}

MoEDeviceOperation::spec_return_value_t MoEDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    return TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), input_tensor.memory_config()));
}

MoEDeviceOperation::tensor_return_value_t MoEDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

std::tuple<MoEDeviceOperation::operation_attributes_t, MoEDeviceOperation::tensor_args_t> MoEDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& w0_tensor,
    const Tensor& w1_tensor,
    const Tensor& w2_tensor,
    const Tensor& output_tensor) {
    return {
        operation_attributes_t{},
        tensor_args_t{
            .input_tensor = input_tensor,
            .w0_tensor = w0_tensor,
            .w1_tensor = w1_tensor,
            .w2_tensor = w2_tensor,
            .output_tensor = output_tensor}};
}

}  // namespace ttnn::operations::experimental::moe
