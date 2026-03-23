// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "dit_rms_norm_unary_fused.hpp"

#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_common.hpp"
#include "ttnn/device.hpp"

namespace ttnn::experimental {

ttnn::Tensor dit_rms_norm_unary_fused(
    const ttnn::Tensor& input_tensor,
    float epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation) {
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());

    auto rank = input_tensor.logical_shape().size();
    TT_FATAL(rank > 0 && input_tensor.logical_volume() >= 0, "Input tensor must have rank > 0 and logical volume >= 0");

    auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch()
                                                                   : ttnn::GetDefaultDevice()->arch();
    const bool approx_mode = true;
    const bool fp32_acc = false;
    auto kernel_config_val = compute_kernel_config.value_or(
        init_device_compute_kernel_config(arch, std::nullopt, MathFidelity::HiFi4, approx_mode, fp32_acc));

    if (!compute_kernel_config.has_value()) {
        kernel_config_val.fp32_dest_acc_en = (input_tensor.dtype() == DataType::FLOAT32);
    }

    return ttnn::prim::layer_norm(
        input_tensor,
        epsilon,
        weight,
        bias,
        residual_input_tensor,
        output_memory_config,
        program_config.value_or(ttnn::prim::create_layernorm_program_config(input_tensor.shard_spec())),
        kernel_config_val,
        std::nullopt,  // dtype
        ttnn::prim::LayerNormType::RMSNORM,
        ttnn::prim::DistributedLayerNormStage::NOT_DISTRIBUTED,
        std::nullopt,  // stats
        std::nullopt,  // recip_tensor
        activation);
}

}  // namespace ttnn::experimental
