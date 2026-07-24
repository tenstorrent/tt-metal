// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/flash_kda/flash_kda.hpp"
#include "ttnn/operations/transformer/flash_kda/device/flash_kda_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/device.hpp"

namespace ttnn::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor> flash_kda(
    const ttnn::Tensor& S_prev,
    const ttnn::Tensor& g,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& q,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    auto mc = memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
    auto kc = init_device_compute_kernel_config(
        S_prev.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);

    auto results = ttnn::prim::flash_kda(S_prev, g, k, v, beta, q, mc, kc);

    return {results[0], results[1]};
}

}  // namespace ttnn::transformer
