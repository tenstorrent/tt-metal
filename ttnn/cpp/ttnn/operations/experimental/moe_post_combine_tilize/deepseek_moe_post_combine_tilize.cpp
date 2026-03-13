// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/operations/experimental/deepseek_moe_post_combine_tilize/deepseek_moe_post_combine_tilize.hpp"
#include "ttnn/operations/experimental/deepseek_moe_post_combine_tilize/device/deepseek_moe_post_combine_tilize_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek_moe_post_combine_tilize {

std::vector<ttnn::Tensor> DeepseekMoEPostCombineTilizeOperation::invoke(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    uint64_t split_size,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        std::nullopt,
        MathFidelity::HiFi4,
        /* default_approx_mode */ false,
        /* default_fp32_acc */ true));

    uint32_t rank = input_tensor.padded_shape().rank();
    uint32_t normalized_dim = (dim < 0) ? dim + rank : (uint32_t)dim;
    return ttnn::prim::deepseek_moe_post_combine_tilize(
        input_tensor, normalized_dim, split_size, output_memory_config, config);
}

}  // namespace ttnn::operations::experimental::deepseek_moe_post_combine_tilize
