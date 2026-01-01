// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_combine.hpp"

#include "device/routed_combine_device_operation.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

Tensor routed_combine(
    const Tensor& expert_rows,
    const Tensor& combine_index,
    const Tensor& combine_weights,
    const Tensor& bias,
    uint32_t num_tokens,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    // Without a config: HiFi4, so the weights' products are exact in the fp32 DST.
    const auto kernel_config = init_device_compute_kernel_config(
        expert_rows.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);
    return ttnn::prim::routed_combine(
        expert_rows,
        combine_index,
        combine_weights,
        bias,
        num_tokens,
        memory_config.value_or(expert_rows.memory_config()),
        kernel_config);
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
