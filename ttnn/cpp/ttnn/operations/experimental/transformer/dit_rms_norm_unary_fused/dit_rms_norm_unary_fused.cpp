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
    const bool is_fp32_input = input_tensor.dtype() == DataType::FLOAT32;
    // Due to hardware bug (#38306), HiFi4 + fp32_dest_acc_en produces incorrect results on Wormhole B0.
    // fp32_dest_acc_en will be True for FLOAT32 inputs (set below), so use HiFi3 as default on Wormhole B0.
    const auto is_wormhole = arch == tt::ARCH::WORMHOLE_B0;
    const auto default_fidelity = (is_wormhole && is_fp32_input) ? MathFidelity::HiFi3 : MathFidelity::HiFi4;
    auto kernel_config_val = compute_kernel_config.value_or(
        init_device_compute_kernel_config(arch, std::nullopt, default_fidelity, approx_mode, fp32_acc));

    if (!compute_kernel_config.has_value()) {
        kernel_config_val.fp32_dest_acc_en = (input_tensor.dtype() == DataType::FLOAT32);
    }

    // Warn if user explicitly passed HiFi4 + fp32_dest_acc_en on Wormhole B0 (hw bug #38306).
    if (is_wormhole && compute_kernel_config.has_value() && compute_kernel_config->fp32_dest_acc_en &&
        compute_kernel_config->math_fidelity == MathFidelity::HiFi4) {
        log_warning(
            tt::LogOp,
            "On Wormhole with fp32 accumulation, output accuracy can be worse with HiFi4 than HiFi3 due to a hardware "
            "bug. "
            "Prefer using HiFi3 with fp32 accumulation on Wormhole.");
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
