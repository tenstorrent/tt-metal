// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim::qsr {

SoftmaxDeviceOperation::program_factory_t SoftmaxDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return SoftmaxProgramFactory{};
}

void SoftmaxDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input;
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to softmax need to be allocated in buffers on device!");
    TT_FATAL(a.layout() == Layout::TILE, "Input tensor must have TILE layout, got: {}", a.layout());
    TT_FATAL(!a.is_sharded(), "Quasar softmax does not support sharded input.");
    TT_FATAL(!operation_attributes.output_mem_config.is_sharded(), "Quasar softmax does not support sharded output.");
    TT_FATAL(
        a.dtype() == DataType::BFLOAT16,
        "Quasar softmax: input must be BFLOAT16 (no BFLOAT8_B on Quasar; FLOAT32 not implemented), got: {}",
        a.dtype());
    TT_FATAL(
        a.logical_shape()[-1] % tt::constants::TILE_WIDTH == 0,
        "Quasar softmax: the last dimension ({}) must be a multiple of {} (no padding mask yet)",
        a.logical_shape()[-1],
        tt::constants::TILE_WIDTH);
}

TensorSpec SoftmaxDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return TensorSpec(
        input.logical_shape(),
        TensorLayout(input.dtype(), PageConfig(input.layout()), operation_attributes.output_mem_config));
}

Tensor SoftmaxDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

Tensor softmax(
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool numeric_stable) {
    return ttnn::device_operation::launch<SoftmaxDeviceOperation>(
        SoftmaxParams{
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config,
            .numeric_stable = numeric_stable,
        },
        SoftmaxInputs{.input = input_tensor});
}

}  // namespace ttnn::prim::qsr
