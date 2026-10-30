// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_stats/attn_res_block_stats.hpp"

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_stats/device/attn_res_block_stats_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::attn_res_block_stats {

ttnn::Tensor attn_res_block_stats(
    const std::vector<ttnn::Tensor>& sealed,
    const ttnn::Tensor& queries,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    TT_FATAL(!sealed.empty(), "AttnResBlockStats needs at least one sealed snapshot tensor");
    // HiFi4 with fp32 dest accumulation: at default fidelity the bf16 mantissa is cut going
    // into the multiply, which costs the dots and the sum of squares an order of magnitude.
    auto kernel_config_val = init_device_compute_kernel_config(
        queries.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);

    return ttnn::prim::attn_res_block_stats(
        sealed, queries, memory_config.value_or(queries.memory_config()), kernel_config_val);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::attn_res_block_stats
