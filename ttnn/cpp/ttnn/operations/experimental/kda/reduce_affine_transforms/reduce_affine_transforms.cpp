// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "reduce_affine_transforms.hpp"

#include "device/reduce_affine_transforms_device_operation.hpp"

namespace ttnn::experimental::kda {

std::pair<ttnn::Tensor, ttnn::Tensor> reduce_affine_transforms(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    uint32_t groups_per_head,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(
        a.storage_type() == StorageType::DEVICE && a.buffer() != nullptr,
        "reduce_affine_transforms: a must be an allocated device tensor");
    TT_FATAL(
        b.storage_type() == StorageType::DEVICE && b.buffer() != nullptr,
        "reduce_affine_transforms: b must be an allocated device tensor");
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        a.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    return ttnn::experimental::prim::reduce_affine_transforms(
        a, b, groups_per_head, output_memory_config, kernel_config);
}

}  // namespace ttnn::experimental::kda
