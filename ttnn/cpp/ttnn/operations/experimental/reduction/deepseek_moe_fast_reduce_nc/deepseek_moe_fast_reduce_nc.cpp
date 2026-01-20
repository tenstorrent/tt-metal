// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/deepseek_moe_fast_reduce_nc.hpp"
#include "ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/device/deepseek_moe_fast_reduce_nc_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::reduction {

std::vector<ttnn::Tensor> DeepseekMoEFastReduceNCOperation::invoke(
    const ttnn::Tensor& input_tensor,
    int32_t reduction_dim,
    int32_t split_dim,
    const ttnn::MemoryConfig& output_memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    ttnn::DeviceComputeKernelConfig config = init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /* default_approx_mode */ false,
        /* default_fp32_acc */ true);

    uint32_t rank = input_tensor.padded_shape().rank();
    uint32_t normalized_reduction_dim = (reduction_dim < 0) ? reduction_dim + rank : (uint32_t)reduction_dim;
    uint32_t normalized_split_dim = (split_dim < 0) ? split_dim + rank : (uint32_t)split_dim;
    return ttnn::prim::deepseek_moe_fast_reduce_nc(
        input_tensor, normalized_reduction_dim, normalized_split_dim, output_memory_config, config);
}

}  // namespace ttnn::operations::experimental::reduction
