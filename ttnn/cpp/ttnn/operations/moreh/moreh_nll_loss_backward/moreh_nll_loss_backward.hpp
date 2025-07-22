
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss_backward {

struct MorehNllLossBackward {
    static Tensor invoke(
        const Tensor& target_tensor,
        const Tensor& output_grad_tensor,
        bool reduction_mean,
        const std::optional<Tensor>& weight_tensor,
        const std::optional<Tensor>& input_grad_tensor,
        const std::optional<Tensor>& divisor_tensor,
        int32_t ignore_index,
        const std::optional<ttnn::MemoryConfig>& memory_config,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config);
};

}  // namespace ttnn::operations::moreh::moreh_nll_loss_backward

namespace ttnn {
constexpr auto moreh_nll_loss_backward = ttnn::register_operation<
    "ttnn::moreh_nll_loss_backward",
    operations::moreh::moreh_nll_loss_backward::MorehNllLossBackward>();
}  // namespace ttnn
