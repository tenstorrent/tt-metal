// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/deepseek_moe_fast_reduce_nc.hpp"
#include "ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/deepseek_moe_fast_reduce_nc_fused.hpp"
#include "ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/deepseek_moe_fast_reduce_nc_fused_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::reduction {

std::vector<ttnn::Tensor> deepseek_moe_fast_reduce_nc_fused(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    int32_t reduce_dim,
    uint64_t split_size,
    uint32_t cluster_axis,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    const std::optional<ttnn::Tensor>& scores_tensor,
    uint32_t num_shared_experts,
    float shared_expert_scale,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    if (!scores_tensor.has_value()) {
        // Fallback path: expert_indices / expert_mapping / cluster_axis are unused without scores.
        return deepseek_moe_fast_reduce_nc(
            input_tensor, reduce_dim, split_size, output_memory_config, compute_kernel_config);
    }

    const auto arch = input_tensor.device()->arch();
    // Due to hardware bug (#38306), HiFi4 + fp32_dest_acc_en can sometime produce incorrect results on Wormhole.
    // Use HiFi3 when fp32_dest_acc_en is True on Wormhole (less likely to give bad results).
    const auto math_fidelity =
        arch == tt::ARCH::WORMHOLE_B0 ? tt::tt_metal::MathFidelity::HiFi3 : tt::tt_metal::MathFidelity::HiFi4;
    ttnn::verify_numerical_configuration(arch, compute_kernel_config);
    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(init_device_compute_kernel_config(
        arch,
        std::nullopt,
        math_fidelity,
        /* default_approx_mode */ false,
        /* default_fp32_acc */ true));

    uint32_t rank = input_tensor.padded_shape().rank();
    uint32_t normalized_dim = (reduce_dim < 0) ? reduce_dim + rank : (uint32_t)reduce_dim;
    return ttnn::prim::deepseek_moe_fast_reduce_nc_fused(
        input_tensor,
        scores_tensor.value(),
        expert_indices_tensor,
        expert_mapping_tensor,
        normalized_dim,
        split_size,
        cluster_axis,
        output_memory_config,
        num_shared_experts,
        shared_expert_scale,
        config);
}

}  // namespace ttnn::experimental::reduction
