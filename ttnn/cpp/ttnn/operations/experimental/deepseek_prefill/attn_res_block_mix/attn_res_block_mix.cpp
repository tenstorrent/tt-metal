// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/attn_res_block_mix.hpp"

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/device/attn_res_block_mix_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::attn_res_block_mix {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, std::optional<ttnn::Tensor>> attn_res_block_mix(
    const std::vector<ttnn::Tensor>& sealed,
    const ttnn::Tensor& stats,
    uint32_t num_sites,
    float eps,
    uint32_t num_closed_sites,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    TT_FATAL(!sealed.empty(), "AttnResBlockMix needs at least one sealed snapshot tensor");
    // Fidelity reaches only the mixture's bf16 MACs, the one FPU product: HiFi2 keeps a bf16
    // weight's full mantissa. Scoring is SFPU in fp32, and dest accumulates in fp32.
    auto kernel_config_val = init_device_compute_kernel_config(
        stats.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);

    auto outputs = ttnn::prim::attn_res_block_mix(
        sealed,
        stats,
        num_sites,
        eps,
        num_closed_sites,
        memory_config.value_or(sealed.front().memory_config()),
        kernel_config_val);
    std::optional<ttnn::Tensor> closed;
    if (outputs.size() > 3) {
        closed = std::move(outputs[3]);
    }
    return {std::move(outputs[0]), std::move(outputs[1]), std::move(outputs[2]), std::move(closed)};
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::attn_res_block_mix
