// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "entry_correction.hpp"

#include "device/entry_correction_device_operation.hpp"

namespace ttnn::experimental::kda {

std::tuple<ttnn::Tensor, ttnn::Tensor> entry_correction(
    const ttnn::Tensor& zero_output,
    const ttnn::Tensor& identity_output,
    const ttnn::Tensor& final_a,
    const ttnn::Tensor& final_b,
    const ttnn::Tensor& entry,
    const ttnn::Tensor& tail_entry,
    const ttnn::Tensor& actual_start,
    uint32_t local_rows,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const std::optional<ttnn::Tensor>& actual_end,
    uint32_t sequence_parallel_axis,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(
        final_b.storage_type() == StorageType::DEVICE && final_b.buffer() != nullptr,
        "entry_correction: final_b must be an allocated device tensor");
    // The products accumulate in the FP32 DST at the scan's fidelity.
    const auto kernel_config = init_device_compute_kernel_config(
        final_b.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    return ttnn::experimental::prim::entry_correction(
        zero_output,
        identity_output,
        final_a,
        final_b,
        entry,
        tail_entry,
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        kernel_config,
        actual_start,
        actual_end,
        sequence_parallel_axis,
        local_rows,
        cores);
}

}  // namespace ttnn::experimental::kda
