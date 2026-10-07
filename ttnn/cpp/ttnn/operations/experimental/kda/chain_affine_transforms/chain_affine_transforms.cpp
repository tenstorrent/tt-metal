// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chain_affine_transforms.hpp"

#include "device/chain_affine_transforms_device_operation.hpp"

namespace ttnn::experimental::kda {

std::pair<ttnn::Tensor, ttnn::Tensor> chain_affine_transforms(
    const ttnn::Tensor& transforms,
    const ttnn::Tensor& initial_state,
    const ttnn::Tensor& actual_start,
    uint32_t local_rows,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    uint32_t sequence_parallel_axis) {
    TT_FATAL(
        transforms.storage_type() == StorageType::DEVICE && transforms.buffer() != nullptr,
        "chain_affine_transforms: transforms must be an allocated device tensor");
    TT_FATAL(
        initial_state.storage_type() == StorageType::DEVICE && initial_state.buffer() != nullptr,
        "chain_affine_transforms: initial_state must be an allocated device tensor");
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        transforms.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    return ttnn::experimental::prim::chain_affine_transforms(
        transforms,
        initial_state,
        output_memory_config,
        kernel_config,
        actual_start,
        sequence_parallel_axis,
        local_rows);
}

}  // namespace ttnn::experimental::kda
