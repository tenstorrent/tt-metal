// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "reduce_affine_transforms.hpp"

#include "device/reduce_affine_transforms_device_operation.hpp"

namespace ttnn::experimental::kda {

std::pair<ttnn::Tensor, ttnn::Tensor> reduce_affine_transforms(
    const ttnn::Tensor& summaries,
    uint32_t groups_per_head,
    const ttnn::Tensor& actual_start,
    uint32_t local_rows,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    uint32_t sequence_parallel_axis,
    const std::optional<Tensor>& actual_end) {
    TT_FATAL(
        summaries.storage_type() == StorageType::DEVICE && summaries.is_allocated(),
        "reduce_affine_transforms: summaries must be an allocated device tensor");
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        summaries.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    return ttnn::experimental::prim::reduce_affine_transforms(
        summaries,
        groups_per_head,
        output_memory_config,
        kernel_config,
        actual_start,
        sequence_parallel_axis,
        local_rows,
        actual_end);
}

}  // namespace ttnn::experimental::kda
