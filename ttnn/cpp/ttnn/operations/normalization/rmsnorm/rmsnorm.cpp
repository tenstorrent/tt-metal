// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm.hpp"

#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_common.hpp"
#include "ttnn/operations/normalization/layernorm/layernorm.hpp"
#include "ttnn/device.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn {

DeviceComputeKernelConfig rmsnorm_default_compute_config(tt::ARCH arch) {
    bool approx_mode = true;
    bool fp32_acc = false;
    return init_device_compute_kernel_config(arch, std::nullopt, MathFidelity::HiFi4, approx_mode, fp32_acc);
}

Tensor rms_norm(
    const Tensor& input_tensor,
    float epsilon,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& bias,
    const std::optional<const Tensor>& residual_input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const prim::LayerNormProgramConfig>& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
    auto rank = input_tensor.logical_shape().size();

    // ROW_MAJOR, 0V, 0D handled here before prepare_norm() which fatals on these cases.
    TT_FATAL(
        input_tensor.layout() != Layout::ROW_MAJOR,
        "ttnn::rms_norm does not support ROW_MAJOR input tensors. Use TILE layout.");

    if (input_tensor.logical_volume() == 0) [[unlikely]] {
        return ttnn::clone(input_tensor, /*dtype=*/std::nullopt, output_memory_config, compute_kernel_config);
    }

    if (rank == 0) [[unlikely]] {
        auto result = ttnn::divide(
            input_tensor, ttnn::abs(input_tensor, output_memory_config), /*alpha=*/std::nullopt, output_memory_config);

        if (weight.has_value()) {
            result = ttnn::multiply(result, weight.value(), /*alpha=*/std::nullopt, output_memory_config);
        }
        if (bias.has_value()) {
            result = ttnn::add(result, bias.value(), /*alpha=*/std::nullopt, output_memory_config);
        }
        return result;
    }

    auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch()
                                                                   : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = compute_kernel_config.value_or(rmsnorm_default_compute_config(arch));
    return ttnn::prim::layer_norm(
        input_tensor,
        epsilon,
        weight,
        bias,
        residual_input_tensor,
        output_memory_config,
        program_config.value_or(ttnn::prim::create_layernorm_program_config(
            input_tensor.shard_spec(),
            input_tensor.tensor_spec().tile().get_height(),
            input_tensor.tensor_spec().tile().get_width())),
        kernel_config_val,
        std::nullopt,  // dtype
        prim::LayerNormType::RMSNORM);
}

ttnn::device_operation::OpDescriptorResult<prim::LayerNormDeviceOperation> rms_norm_descriptor(
    const Tensor& input_tensor,
    float epsilon,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& bias,
    const std::optional<const Tensor>& residual_input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const prim::LayerNormProgramConfig>& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch()
                                                                   : ttnn::GetDefaultDevice()->arch();
    auto [attrs, args] = prepare_norm(
        input_tensor,
        epsilon,
        prim::LayerNormType::RMSNORM,
        rmsnorm_default_compute_config(arch),
        weight,
        bias,
        residual_input_tensor,
        memory_config,
        program_config,
        compute_kernel_config);

    return ttnn::device_operation::create_op_descriptor<prim::LayerNormDeviceOperation>(attrs, args);
}

}  // namespace ttnn
