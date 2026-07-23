// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/kda_recurrent/kda_recurrent.hpp"

#include "ttnn/device.hpp"
#include "ttnn/operations/transformer/kda_recurrent/device/kda_recurrent_device_operation.hpp"

namespace ttnn::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor> kda_recurrent_step(
    const ttnn::Tensor& q_scaled,
    const ttnn::Tensor& k_unit,
    const ttnn::Tensor& v,
    const ttnn::Tensor& decay,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& state,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    auto output_memory_config = memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
    auto kernel_config = init_device_compute_kernel_config(
        q_scaled.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);

    auto results =
        ttnn::prim::kda_recurrent_step(q_scaled, k_unit, v, decay, beta, state, output_memory_config, kernel_config);
    return {results[0], results[1]};
}

}  // namespace ttnn::transformer
