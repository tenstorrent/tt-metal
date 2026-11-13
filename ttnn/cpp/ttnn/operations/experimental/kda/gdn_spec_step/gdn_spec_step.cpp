// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "gdn_spec_step.hpp"

#include <cmath>

#include "device/gdn_spec_step_device_operation.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor gdn_spec_step(
    const ttnn::Tensor& qkvzab,
    const std::optional<ttnn::Tensor>& win_a,
    const std::optional<ttnn::Tensor>& win_b,
    const ttnn::Tensor& ring,
    const ttnn::Tensor& ctrl,
    const ttnn::Tensor& taps,
    const ttnn::Tensor& dt_bias,
    const ttnn::Tensor& neg_exp_A,
    const ttnn::Tensor& weight,
    uint32_t num_value_heads,
    uint32_t num_key_heads,
    uint32_t key_dim,
    uint32_t value_dim,
    uint32_t T,
    uint32_t B,
    uint32_t qkvz_dim,
    uint32_t conv_kernel,
    std::optional<float> scale,
    float l2_epsilon,
    float norm_epsilon,
    uint32_t hnew_depth,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    DataType output_dtype,
    bool fast_mode,
    bool fp32_state,
    uint32_t max_users,
    uint32_t value_blocks,
    uint32_t kernel_opts,
    const std::optional<ttnn::Tensor>& ring_page_idx,
    const std::optional<ttnn::Tensor>& ring_pool2,
    const std::optional<ttnn::Tensor>& conv_hist) {
    TT_FATAL(
        qkvzab.storage_type() == StorageType::DEVICE && qkvzab.buffer() != nullptr,
        "gdn_spec_step: qkvzab must be an allocated device tensor");
    TT_FATAL(hnew_depth == 2, "gdn_spec_step: hnew_depth must be 2 (got {})", hnew_depth);
    TT_FATAL(
        max_users == 0 || max_users == B,
        "gdn_spec_step: max_users must be 0 or B (got {}, B = {}); ring and ctrl are laid out for exactly B users",
        max_users,
        B);
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        qkvzab.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    const float scale_value = scale.value_or(1.0f / std::sqrt(static_cast<float>(key_dim)));
    return ttnn::experimental::prim::gdn_spec_step(
        qkvzab,
        win_a,
        win_b,
        ring,
        ctrl,
        taps,
        dt_bias,
        neg_exp_A,
        weight,
        num_value_heads,
        num_key_heads,
        key_dim,
        value_dim,
        T,
        B,
        conv_kernel,
        qkvz_dim,
        scale_value,
        l2_epsilon,
        norm_epsilon,
        output_memory_config,
        kernel_config,
        output_dtype,
        fast_mode,
        fp32_state,
        value_blocks,
        kernel_opts,
        ring_page_idx,
        ring_pool2,
        conv_hist);
}

}  // namespace ttnn::experimental::kda
