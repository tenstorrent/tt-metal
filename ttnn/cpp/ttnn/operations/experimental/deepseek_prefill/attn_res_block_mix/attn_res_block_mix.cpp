// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/attn_res_block_mix.hpp"

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/device/attn_res_block_mix_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::attn_res_block_mix {

std::array<ttnn::Tensor, 3> attn_res_block_mix(
    const std::vector<ttnn::Tensor>& sealed,
    const ttnn::Tensor& stats,
    uint32_t num_sites,
    float inv_hidden_size,
    float eps,
    uint32_t num_closed_sites,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    TT_FATAL(!sealed.empty(), "AttnResBlockMix needs at least one sealed snapshot tensor");
    // HiFi4 with fp32 dest accumulation: the scoring chain is rsqrt, a product, a max and
    // exp, and the mixture MACs every candidate into one accumulator per site.
    auto kernel_config_val = init_device_compute_kernel_config(
        stats.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);

    return ttnn::prim::attn_res_block_mix(
        sealed,
        stats,
        num_sites,
        inv_hidden_size,
        eps,
        num_closed_sites,
        memory_config.value_or(sealed.front().memory_config()),
        kernel_config_val);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::attn_res_block_mix
