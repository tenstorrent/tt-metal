// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sigmoid_gated_rms_norm.hpp"

#include "device/sigmoid_gated_rms_norm_device_operation.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor sigmoid_gated_rms_norm(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& weight,
    uint32_t num_heads,
    float epsilon,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    DataType output_dtype) {
    TT_FATAL(
        input.storage_type() == StorageType::DEVICE && input.buffer() != nullptr,
        "sigmoid_gated_rms_norm: input must be an allocated device tensor");
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        input.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/true,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false,
        /*default_dst_full_sync_en=*/false,
        ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE);
    return ttnn::experimental::prim::sigmoid_gated_rms_norm(
        input, gate, weight, num_heads, epsilon, output_memory_config, kernel_config, output_dtype);
}

}  // namespace ttnn::experimental::kda
