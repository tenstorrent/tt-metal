// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteFusedRMSNormPreAllGather {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        DataType dtype = DataType::BFLOAT16,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {
constexpr auto wan_fused_rmsnorm_pre_allgather = ttnn::register_operation<
    "ttnn::experimental::wan_fused_rmsnorm_pre_allgather",
    ttnn::operations::experimental::ccl::ExecuteFusedRMSNormPreAllGather>();

}  // namespace experimental
}  // namespace ttnn
