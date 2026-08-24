// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/fused_qkv_sdpa.hpp"
#include "ttnn/operations/transformer/sdpa/device/fused_qkv_sdpa_device_operation.hpp"
#include <tt-metalium/hal.hpp>

namespace ttnn::transformer {

ttnn::Tensor fused_qkv_sdpa(
    const ttnn::Tensor& qkv,
    uint32_t num_heads,
    const std::optional<ttnn::Tensor>& attn_mask,
    std::optional<float> scale,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto kernel_config = init_device_compute_kernel_config(
        tt::tt_metal::hal::get_arch(),
        compute_kernel_config,
        /*default_fidelity=*/MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false);

    return ttnn::prim::fused_qkv_sdpa(
        qkv,
        attn_mask,
        num_heads,
        scale,
        memory_config.value_or(qkv.memory_config()),
        std::move(program_config),
        kernel_config);
}

}  // namespace ttnn::transformer
