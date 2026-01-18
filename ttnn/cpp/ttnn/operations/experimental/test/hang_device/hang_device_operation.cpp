// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "hang_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/hal.hpp>

namespace ttnn::prim {

ExecuteTestHangDeviceOperation::program_factory_t ExecuteTestHangDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return SingleCore{};
}

void ExecuteTestHangDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& /*tensor_args*/) {}

void ExecuteTestHangDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t& /*tensor_args*/) {}

ExecuteTestHangDeviceOperation::spec_return_value_t ExecuteTestHangDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return TensorSpec(
        tensor_args.tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            tensor_args.tensor.dtype(), tt::tt_metal::PageConfig(tensor_args.tensor.layout()), MemoryConfig{}));
}

ExecuteTestHangDeviceOperation::tensor_return_value_t ExecuteTestHangDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.tensor.device());
}

std::tuple<ExecuteTestHangDeviceOperation::operation_attributes_t, ExecuteTestHangDeviceOperation::tensor_args_t>
ExecuteTestHangDeviceOperation::invoke(const Tensor& input_tensor) {
    return {operation_attributes_t{}, tensor_args_t{input_tensor}};
}
}  // namespace ttnn::prim
