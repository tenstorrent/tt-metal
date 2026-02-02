// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_rmsnorm_pre_all_gather_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

FusedRMSNormPreAllGatherDeviceOperation::program_factory_t
FusedRMSNormPreAllGatherDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return FusedRMSNormPreAllGatherProgramFactory{};
}

void FusedRMSNormPreAllGatherDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void FusedRMSNormPreAllGatherDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    using namespace tt::constants;

    const auto& tensor = tensor_args.input_tensor;

    TT_FATAL(tensor.layout() == Layout::TILE, "Input tensor must have TILE layout, got: {}", tensor.layout());
    TT_FATAL(
        tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input tensor must use INTERLEAVED memory layout, got: {}",
        tensor.memory_config().memory_layout());
    TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "Input tensor must be BFLOAT16, got: {}", tensor.dtype());
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device, got: {}", tensor.storage_type());
    TT_FATAL(tensor.buffer() != nullptr, "Input tensor must be allocated in device buffers (buffer is null)");
}

TensorSpec FusedRMSNormPreAllGatherDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;
    using namespace tt::constants;

    const auto& input_tensor = tensor_args.input_tensor;

    auto output_shape = input_tensor.logical_shape();
    uint32_t num_tiles_w = 1;  // RMSNorm only
    output_shape[-1] = num_tiles_w * TILE_WIDTH;

    return TensorSpec(output_shape, TensorLayout(args.dtype, PageConfig(Layout::TILE), input_tensor.memory_config()));
}

Tensor FusedRMSNormPreAllGatherDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor fused_rmsnorm_pre_all_gather(
    const Tensor& input_tensor, tt::tt_metal::DataType dtype, const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::experimental::prim::FusedRMSNormPreAllGatherDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .dtype = dtype,
        .compute_kernel_config = compute_kernel_config,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input_tensor,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
