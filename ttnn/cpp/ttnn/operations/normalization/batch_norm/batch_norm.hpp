// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::normalization {
struct BatchNorm {
    static Tensor invoke(
        const Tensor& input,
        std::optional<Tensor> running_mean = std::nullopt,
        std::optional<Tensor> running_var = std::nullopt,
        const bool training = false,
        const float eps = 1e-05,
        const float momentum = 0.1,
        std::optional<Tensor> weight = std::nullopt,
        std::optional<Tensor> bias = std::nullopt,
        std::optional<Tensor> output = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};
}  // namespace operations::normalization

constexpr auto batch_norm =
    ttnn::register_operation_with_auto_launch_op<"ttnn::batch_norm", ttnn::operations::normalization::BatchNorm>();
}  // namespace ttnn
