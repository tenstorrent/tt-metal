// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_causal_conv.hpp"
#include "device/kda_causal_conv_device_operation.hpp"

namespace ttnn::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> kda_causal_conv1d_split(
    const ttnn::Tensor& input,
    const ttnn::Tensor& state,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        input.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/true,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false);
    auto outputs = ttnn::prim::kda_causal_conv1d_split(
        input, state, tap0, tap1, tap2, tap3, q_width, k_width, v_width, output_memory_config, kernel_config);
    return {outputs[0], outputs[1], outputs[2]};
}

}  // namespace ttnn::transformer
