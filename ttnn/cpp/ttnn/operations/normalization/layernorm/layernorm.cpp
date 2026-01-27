// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm.hpp"
#include <optional>

#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "device/layernorm_device_operation.hpp"
#include "ttnn/operations/experimental/parallel/device/parallel_device_operation_types.hpp"
#include "ttnn/device.hpp"

namespace ttnn::operations::normalization {

using LayerNormDeviceOp = ttnn::prim::LayerNormDeviceOperation;

ttnn::Tensor ExecuteLayerNorm::invoke(
    const ttnn::Tensor& input_tensor,
    float epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
    auto rank = input_tensor.logical_shape().rank();

    // For 0D tensors
    TT_FATAL(rank > 0, "LayerNorm operation not supported for 0D tensors. (rank={})", rank);

    // For 0V tensors
    if (input_tensor.logical_volume() == 0) [[unlikely]] {
        return ttnn::clone(input_tensor, /*dtype=*/std::nullopt, output_memory_config, compute_kernel_config);
    }

    auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch()
                                                                   : ttnn::GetDefaultDevice()->arch();
    bool approx_mode = false;
    bool fp32_acc = true;
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, approx_mode, fp32_acc);

    return ttnn::prim::layer_norm(
        input_tensor,
        epsilon,
        weight,
        bias,
        residual_input_tensor,
        output_memory_config,
        program_config.value_or(ttnn::prim::create_program_config(input_tensor.shard_spec())),
        kernel_config_val);
}

ttnn::experimental::prim::BranchDescriptor ExecuteLayerNorm::branch(
    const ttnn::Tensor& input_tensor,
    const tt::tt_metal::CoreRangeSet& cores,
    float epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());

    auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch()
                                                                   : ttnn::GetDefaultDevice()->arch();
    // Match the defaults used by invoke(): approx_mode=false, fp32_acc=true
    bool approx_mode = false;
    bool fp32_acc = true;
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, approx_mode, fp32_acc);

    LayerNormDeviceOp::operation_attributes_t op_attrs{
        .norm_type = ttnn::prim::LayerNormType::LAYERNORM,
        .distributed_norm_stage = ttnn::prim::DistributedLayerNormStage::NOT_DISTRIBUTED,
        .eps = epsilon,
        .output_mem_config = output_memory_config,
        .program_config = program_config.value_or(ttnn::prim::create_program_config(input_tensor.shard_spec())),
        .compute_kernel_config = kernel_config_val};

    LayerNormDeviceOp::tensor_args_t tensor_args{
        .input = input_tensor, .residual_input_tensor = residual_input_tensor, .weight = weight, .bias = bias};

    return ttnn::experimental::prim::create_branch<LayerNormDeviceOp>(cores, op_attrs, tensor_args);
}

}  // namespace ttnn::operations::normalization
