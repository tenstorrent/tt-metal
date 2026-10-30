// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_scores/attn_res_scores.hpp"

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_scores/device/attn_res_scores_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::attn_res_scores {

std::array<ttnn::Tensor, 3> attn_res_scores(
    const ttnn::Tensor& stats,
    uint32_t num_sites,
    float inv_hidden_size,
    float eps,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    TT_FATAL(
        stats.storage_type() == StorageType::DEVICE,
        "Input tensor storage type must be DEVICE but got {}",
        stats.storage_type());

    // HiFi4 with fp32 dest accumulation, as the read's own derivation runs: the chain
    // is rsqrt, a product, a max and exp, and a bf16 dest would round between each.
    auto kernel_config_val = init_device_compute_kernel_config(
        stats.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);

    return ttnn::prim::attn_res_scores(
        stats, num_sites, inv_hidden_size, eps, memory_config.value_or(stats.memory_config()), kernel_config_val);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::attn_res_scores
