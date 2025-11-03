// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::experimental::transformer {

struct ExecuteFusedRMSNormPostAllGather {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& stats,
        float epsilon = 1e-5,
        uint32_t num_heads = 1,
        const std::optional<const ttnn::Tensor>& weight = std::nullopt,
        const std::optional<const ttnn::Tensor>& transformation_mat = std::nullopt,
        const std::optional<const ttnn::Tensor>& rope_cos = std::nullopt,
        const std::optional<const ttnn::Tensor>& rope_sin = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<const DataType>& dtype = std::nullopt);
};

}  // namespace operations::experimental::transformer

namespace experimental {
constexpr auto wan_fused_rmsnorm_post_allgather = ttnn::register_operation<
    "ttnn::experimental::wan_fused_rmsnorm_post_allgather",
    ttnn::operations::experimental::transformer::ExecuteFusedRMSNormPostAllGather>();

}  // namespace experimental
}  // namespace ttnn
