// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_affine_composition.hpp"

#include "device/kda_affine_composition_device_operation.hpp"

namespace ttnn::transformer {

std::pair<ttnn::Tensor, ttnn::Tensor> kda_affine_compose(
    const ttnn::Tensor& transform_a,
    const ttnn::Tensor& transform_b,
    uint32_t groups_per_head,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        transform_a.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    return ttnn::prim::kda_affine_compose(
        transform_a, transform_b, groups_per_head, output_memory_config, kernel_config);
}

}  // namespace ttnn::transformer
