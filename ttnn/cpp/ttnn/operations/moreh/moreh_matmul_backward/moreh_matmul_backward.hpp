// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
namespace ttnn::operations::moreh::moreh_matmul_backward {
struct MorehMatmulBackward {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& output_grad,
        const Tensor& input,
        const Tensor& other,
        const std::vector<bool>& are_required_outputs,
        const std::optional<const Tensor>& input_grad,
        const std::optional<const Tensor>& other_grad,
        const std::optional<ttnn::MemoryConfig>& output_mem_config,
        const std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_matmul_backward

namespace ttnn {
constexpr auto moreh_matmul_backward = ttnn::register_operation<
    "ttnn::moreh_matmul_backward",
    ttnn::operations::moreh::moreh_matmul_backward::MorehMatmulBackward>();
}
