// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
namespace ttnn::operations::moreh::moreh_matmul {
struct MorehMatmul {
    static Tensor invoke(const Tensor &input,
                         const Tensor &other,
                         bool transpose_input,
                         bool transpose_other,
                         const std::optional<Tensor> &output,
                         const std::optional<const Tensor> bias,
                         const std::optional<MemoryConfig> &memory_config,
                         const std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_matmul

namespace ttnn {
constexpr auto moreh_matmul =
    ttnn::register_operation_with_auto_launch_op<"ttnn::moreh_matmul",
                                                 ttnn::operations::moreh::moreh_matmul::MorehMatmul>();
}  // namespace ttnn
