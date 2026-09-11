// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "affine_exclusive_scan.hpp"

#include "device/affine_exclusive_scan_device_operation.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor affine_exclusive_scan(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const ttnn::Tensor& initial_state,
    uint32_t groups_per_head,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(
        a.storage_type() == StorageType::DEVICE && a.buffer() != nullptr,
        "affine_exclusive_scan: a must be an allocated device tensor");
    TT_FATAL(
        b.storage_type() == StorageType::DEVICE && b.buffer() != nullptr,
        "affine_exclusive_scan: b must be an allocated device tensor");
    TT_FATAL(
        initial_state.storage_type() == StorageType::DEVICE && initial_state.buffer() != nullptr,
        "affine_exclusive_scan: initial_state must be an allocated device tensor");
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        a.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    return ttnn::experimental::prim::affine_exclusive_scan(
        a, b, initial_state, groups_per_head, output_memory_config, kernel_config);
}

}  // namespace ttnn::experimental::kda
