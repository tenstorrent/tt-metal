// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm.hpp"

#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_common.hpp"
#include "ttnn/device.hpp"

namespace ttnn {

DeviceComputeKernelConfig rmsnorm_default_compute_config(tt::ARCH arch) {
    // Quasar's simulator does not support fp32 accumulation of bf16 inputs yet.
    bool quasar = arch == tt::ARCH::QUASAR;
    bool approx_mode = quasar;
    bool fp32_acc = !quasar;
    return init_device_compute_kernel_config(
        arch, std::nullopt, tt::tt_metal::MathFidelity::HiFi4, approx_mode, fp32_acc);
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

    TT_FATAL(
        input_tensor.layout() != Layout::ROW_MAJOR,
        "ttnn::rms_norm does not support ROW_MAJOR input tensors. Use TILE layout.");

    // For 0V tensors
    if (input_tensor.logical_volume() == 0) [[unlikely]] {
        return ttnn::clone(input_tensor, /*dtype=*/std::nullopt, output_memory_config, compute_kernel_config);
    }

    // For 0D tensors
    if (rank == 0) [[unlikely]] {
        auto result = ttnn::multiply(
            input_tensor,
            ttnn::rsqrt(
                ttnn::add(
                    ttnn::square(input_tensor, output_memory_config),
                    epsilon,
                    /*output_dtype=*/std::nullopt,
                    output_memory_config),
                /*fast_and_approximate_mode=*/false,
                output_memory_config),
            /*output_dtype=*/std::nullopt,
            output_memory_config);

        if (weight.has_value()) {
            result = ttnn::multiply(result, weight.value(), /*output_dtype=*/std::nullopt, output_memory_config);
        }
        if (bias.has_value()) {
            result = ttnn::add(result, bias.value(), /*output_dtype=*/std::nullopt, output_memory_config);
        }
        return result;
    }

    auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch()
                                                                   : ttnn::GetDefaultDevice()->arch();
    // fp32 accumulation can overflow L1 with a ROW_MAJOR weight or bias, and DEST with subblock_w > 4. Those cases
    // use HiFi4, math_approx_mode=true, fp32_dest_acc_en=false, the default of rms_norm_pre/post_all_gather.
    const bool row_major_affine = (weight.has_value() && weight->layout() == Layout::ROW_MAJOR) ||
                                  (bias.has_value() && bias->layout() == Layout::ROW_MAJOR);
    const auto* sharded_config = program_config.has_value()
                                     ? std::get_if<prim::LayerNormShardedMultiCoreProgramConfig>(&*program_config)
                                     : nullptr;
    const bool wide_subblock = sharded_config != nullptr && sharded_config->subblock_w > 4;
    auto kernel_config_val = compute_kernel_config.value_or(
        row_major_affine || wide_subblock
            ? init_device_compute_kernel_config(arch, std::nullopt, tt::tt_metal::MathFidelity::HiFi4, true, false)
            : rmsnorm_default_compute_config(arch));
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

}  // namespace ttnn
