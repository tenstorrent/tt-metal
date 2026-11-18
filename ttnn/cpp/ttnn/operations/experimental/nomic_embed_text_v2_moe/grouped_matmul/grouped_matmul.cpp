// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_matmul.hpp"

#include "device/grouped_matmul_device_operation.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

Tensor grouped_matmul(
    const Tensor& input_tensor,
    const Tensor& weight,
    const Tensor& expert_counts,
    tt::tt_metal::DataType dtype,
    bool transpose_weight,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation,
    tt::tt_metal::Layout output_layout,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    bool per_expert_lanes) {
    // Without a config: HiFi4 and fp32 accumulation, the most precise setting.
    const auto kernel_config = init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);
    return ttnn::prim::grouped_matmul(
        input_tensor,
        weight,
        expert_counts,
        dtype,
        transpose_weight,
        activation,
        output_layout,
        memory_config.value_or(input_tensor.memory_config()),
        kernel_config,
        per_expert_lanes);
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
