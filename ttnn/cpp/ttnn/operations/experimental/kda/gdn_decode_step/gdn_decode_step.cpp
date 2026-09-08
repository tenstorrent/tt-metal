// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "gdn_decode_step.hpp"

#include <cmath>

#include "device/gdn_decode_step_device_operation.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor gdn_decode_step(
    const ttnn::Tensor& qkv,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& g,
    const ttnn::Tensor& state,
    const ttnn::Tensor& weight,
    uint32_t num_value_heads,
    uint32_t num_key_heads,
    uint32_t key_dim,
    uint32_t value_dim,
    std::optional<float> scale,
    float l2_epsilon,
    float norm_epsilon,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    DataType output_dtype,
    const std::optional<std::vector<ttnn::Tensor>>& conv_states,
    const std::optional<std::vector<ttnn::Tensor>>& conv_taps,
    uint32_t qkvz_dim) {
    TT_FATAL(
        qkv.storage_type() == StorageType::DEVICE && qkv.buffer() != nullptr,
        "gdn_decode_step: qkv must be an allocated device tensor");
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        qkv.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    const float scale_value = scale.value_or(1.0f / std::sqrt(static_cast<float>(key_dim)));
    return ttnn::experimental::prim::gdn_decode_step(
        qkv,
        beta,
        g,
        state,
        weight,
        num_value_heads,
        num_key_heads,
        key_dim,
        value_dim,
        scale_value,
        l2_epsilon,
        norm_epsilon,
        output_memory_config,
        kernel_config,
        output_dtype,
        conv_states.value_or(std::vector<ttnn::Tensor>{}),
        conv_taps.value_or(std::vector<ttnn::Tensor>{}),
        qkvz_dim);
}

}  // namespace ttnn::experimental::kda
