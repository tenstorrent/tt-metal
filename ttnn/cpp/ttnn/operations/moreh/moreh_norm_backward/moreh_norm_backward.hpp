// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_norm_backward {
struct MorehNormBackward {
    static Tensor invoke(
        const Tensor& input,
        const Tensor& output,
        const Tensor& output_grad,
        float p,
        std::optional<std::variant<int64_t, ttnn::SmallVector<int64_t>>> dim,
        bool keepdim,
        const std::optional<Tensor>& input_grad,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_norm_backward

namespace ttnn {
constexpr auto moreh_norm_backward = ttnn::
    register_operation<"ttnn::moreh_norm_backward", ttnn::operations::moreh::moreh_norm_backward::MorehNormBackward>();
}
