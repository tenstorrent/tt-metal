// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::experimental::transformer {

struct ExecuteDitLayerNormPreAllGather {
    // Computes Welford stats (sum and sumsq) over the last dim for LayerNorm.
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        DataType dtype = DataType::BFLOAT16,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::experimental::transformer

namespace experimental {
constexpr auto dit_layernorm_pre_allgather = ttnn::register_operation<
    "ttnn::experimental::dit_layernorm_pre_allgather",
    ttnn::operations::experimental::transformer::ExecuteDitLayerNormPreAllGather>();

}  // namespace experimental
}  // namespace ttnn
