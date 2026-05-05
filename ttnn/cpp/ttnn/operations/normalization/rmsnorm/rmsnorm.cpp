// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm.hpp"

#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/device.hpp"

namespace ttnn {

DeviceComputeKernelConfig rmsnorm_default_compute_config(tt::ARCH arch) {
    bool approx_mode = true;
    bool fp32_acc = false;
    return init_device_compute_kernel_config(arch, std::nullopt, tt::tt_metal::MathFidelity::HiFi4, approx_mode, fp32_acc);
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
    (void)program_config;
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
        auto result = ttnn::divide(
            input_tensor,
            ttnn::abs(input_tensor, output_memory_config),
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
    (void)arch;
    (void)epsilon;
    (void)weight;
    (void)bias;
    (void)residual_input_tensor;
    (void)output_memory_config;
    (void)compute_kernel_config;
    // TODO(nuked-op layernorm): restore real ttnn::prim::layer_norm-backed RMSNorm path.
    return input_tensor;
}

}  // namespace ttnn
