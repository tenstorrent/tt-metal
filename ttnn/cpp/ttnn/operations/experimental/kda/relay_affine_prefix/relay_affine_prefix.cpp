// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "relay_affine_prefix.hpp"

#include "device/relay_affine_prefix_device_operation.hpp"

namespace ttnn::experimental::kda {

std::tuple<ttnn::Tensor, ttnn::Tensor> relay_affine_prefix(
    const ttnn::Tensor& packed,
    const ttnn::Tensor& initial_state,
    const ttnn::Tensor& actual_start,
    const ttnn::Tensor& entry_landing,
    const ttnn::Tensor& state_staging,
    const ttnn::Tensor& final_landing,
    const std::vector<tt::tt_metal::GlobalSemaphore>& semaphores,
    uint32_t local_rows,
    uint32_t num_links,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    uint32_t sequence_parallel_axis,
    const std::optional<ttnn::MemoryConfig>& entry_memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(
        packed.storage_type() == StorageType::DEVICE && packed.is_allocated(),
        "relay_affine_prefix: packed must be an allocated device tensor");
    TT_FATAL(semaphores.size() == 2, "relay_affine_prefix: needs two global semaphores, got {}", semaphores.size());
    const auto arch = packed.device()->arch();
    // The hop's matmul runs at the affine prefix's fidelity, accumulating in an FP32 DST.
    const auto kernel_config = ttnn::init_device_compute_kernel_config(
        arch,
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    return ttnn::experimental::prim::relay_affine_prefix(
        packed,
        initial_state,
        actual_start,
        entry_landing,
        state_staging,
        final_landing,
        entry_memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        kernel_config,
        {static_cast<uint32_t>(semaphores[0].address()), static_cast<uint32_t>(semaphores[1].address())},
        sequence_parallel_axis,
        local_rows,
        num_links,
        cores);
}

}  // namespace ttnn::experimental::kda
