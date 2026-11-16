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
    DataType output_dtype,
    uint32_t gate_column_offset) {
    TT_FATAL(
        input.storage_type() == StorageType::DEVICE && input.is_allocated(),
        "sigmoid_gated_rms_norm: input must be an allocated device tensor");
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto arch = input.device()->arch();
    // Due to hardware bug (#38306), HiFi4 + fp32_dest_acc_en can sometime produce incorrect results on Wormhole.
    // Use HiFi3 when fp32_dest_acc_en is True on Wormhole (less likely to give bad results).
    const auto math_fidelity =
        arch == tt::ARCH::WORMHOLE_B0 ? tt::tt_metal::MathFidelity::HiFi3 : tt::tt_metal::MathFidelity::HiFi4;
    ttnn::verify_numerical_configuration(arch, compute_kernel_config);
    const auto kernel_config = init_device_compute_kernel_config(
        arch,
        compute_kernel_config,
        math_fidelity,
        /*default_approx_mode=*/true,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false,
        /*default_dst_full_sync_en=*/false,
        ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE);
    return ttnn::experimental::prim::sigmoid_gated_rms_norm(
        input, gate, weight, num_heads, epsilon, output_memory_config, kernel_config, output_dtype, gate_column_offset);
}

}  // namespace ttnn::experimental::kda
