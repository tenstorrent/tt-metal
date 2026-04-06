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
    int32_t reduce_dim,
    uint64_t split_size,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    const std::optional<ttnn::Tensor>& scores_tensor,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    if (!scores_tensor.has_value()) {
        return deepseek_moe_fast_reduce_nc(
            input_tensor, reduce_dim, split_size, output_memory_config, compute_kernel_config);
    }

    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        std::nullopt,
        tt::tt_metal::MathFidelity::HiFi4,
        /* default_approx_mode */ false,
        /* default_fp32_acc */ true));

    uint32_t rank = input_tensor.padded_shape().rank();
    uint32_t normalized_dim = (reduce_dim < 0) ? reduce_dim + rank : (uint32_t)reduce_dim;
    return ttnn::prim::deepseek_moe_fast_reduce_nc_fused(
        input_tensor, scores_tensor.value(), normalized_dim, split_size, output_memory_config, config);
}

}  // namespace ttnn::experimental::reduction
